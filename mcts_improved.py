"""Drop-in alternative for ``run_mcts`` with four sample-efficiency improvements.

All four behaviors are gated behind CLI toggles (default ON) so each can be
ablated independently against the original ``run_mcts``:

1. **Reward normalization** (`--mcts_improved_reward_norm`)
   Running min-max normalize raw rollout scores to [0,1] before backprop.
   Fixes UCB exploration when rewards are not in [0,1] (HPSv3 ~10–13,
   ImageReward ~[-2,2]). Without this, exploration term ≪ exploit mean and
   the tree latches onto the first decent rollout.

2. **x0-pred bootstrap rollout** (`--mcts_improved_x0_bootstrap`)
   After expanding a child node, decode the model's current x0 prediction
   directly and use that as the value estimate — no further rollout to
   terminal. ~5–10× wallclock speedup on 28-step backends.

3. **UCB1-Tuned** (`--mcts_improved_ucb_tuned`)
   Variance-aware UCB. Tracks per-arm sum-of-squares to estimate empirical
   variance V_n; bonus becomes `c·sqrt(log(N)/n · min(1/4, V_n + sqrt(2·log(N)/n)))`.
   Same NFE, allocates more samples to high-variance arms.

4. **Step-dependent rollout policy** (`--mcts_improved_late_step_baseline`)
   When `--mcts_improved_x0_bootstrap` is OFF (so we still rollout), bias
   late-step rollouts toward the baseline action — late-step CFG choice has
   minimal reward effect, so uniform-random wastes budget there.

The underlying tree shape, key-step branching, integrated-noise actions, and
fresh-noise-sampling logic are unchanged from ``run_mcts``.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np
import torch
from PIL import Image

import sampling_unified_sd35 as su
from sampling_unified_sd35 import (
    MCTSNode,
    SearchResult,
    UnifiedRewardScorer,
    _expand_child,
    _final_decode_tensor,
    _parse_key_steps,
    _resolve_mcts_fresh_noise_steps,
    _resolve_noise_inject_steps,
    _run_segment,
    decode_to_pil,
    expand_emb_with_interp,
    make_latents,
    score_image,
    step_schedule,
)


# ── CLI ─────────────────────────────────────────────────────────────────────


def add_mcts_improved_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mcts_improved_reward_norm", action="store_true", default=True,
                        help="Running min/max normalize rewards before backprop.")
    parser.add_argument("--mcts_improved_no_reward_norm",
                        dest="mcts_improved_reward_norm", action="store_false")

    parser.add_argument("--mcts_improved_x0_bootstrap", action="store_true", default=True,
                        help="Skip terminal rollout; use x0-pred decode at expanded node as value.")
    parser.add_argument("--mcts_improved_no_x0_bootstrap",
                        dest="mcts_improved_x0_bootstrap", action="store_false")

    parser.add_argument("--mcts_improved_ucb_tuned", action="store_true", default=True,
                        help="Use UCB1-Tuned variance-aware bonus instead of UCB1.")
    parser.add_argument("--mcts_improved_no_ucb_tuned",
                        dest="mcts_improved_ucb_tuned", action="store_false")

    parser.add_argument("--mcts_improved_late_step_baseline", action="store_true", default=True,
                        help="In rollouts (when not bootstrapping), late-step actions default to baseline.")
    parser.add_argument("--mcts_improved_no_late_step_baseline",
                        dest="mcts_improved_late_step_baseline", action="store_false")

    parser.add_argument("--mcts_improved_late_step_frac", type=float, default=0.7,
                        help="Steps past this fraction of n_key are forced to baseline action in rollouts.")


# ── Improved MCTS node with sum-of-squares bookkeeping ──────────────────────


class MCTSNodeV2(MCTSNode):
    __slots__ = ("action_value_sq",)

    def __init__(self, step: int, dx: torch.Tensor, latents: torch.Tensor | None):
        super().__init__(step, dx, latents)
        self.action_value_sq: dict[tuple[int, float, float], float] = {}

    def ucb_tuned(self, action: tuple, c: float) -> float:
        n = self.action_visits.get(action, 0)
        if n == 0:
            return float("inf")
        mean = self.action_values[action] / n
        sq = self.action_value_sq.get(action, 0.0) / n
        var = max(0.0, sq - mean * mean)
        log_term = math.log(max(self.visits, 1)) / n
        bonus = c * math.sqrt(log_term * min(0.25, var + math.sqrt(2.0 * log_term)))
        return mean + bonus

    def best_ucb_tuned(self, actions: list, c: float) -> tuple:
        return max(actions, key=lambda a: self.ucb_tuned(a, c))


# ── Running min/max reward normalizer ───────────────────────────────────────


class _RewardNormalizer:
    """Stateful running [min, max] normalizer.

    First few rollouts (until span > eps) return raw scores so UCB doesn't
    divide by ~0. After warmup, returns (x - min) / (max - min) ∈ [0, 1].
    """

    def __init__(self, eps: float = 1e-3):
        self.lo = float("inf")
        self.hi = -float("inf")
        self.eps = float(eps)

    def update(self, x: float) -> None:
        x = float(x)
        if x < self.lo:
            self.lo = x
        if x > self.hi:
            self.hi = x

    def normalize(self, x: float) -> float:
        span = self.hi - self.lo
        if not math.isfinite(span) or span < self.eps:
            return float(x)
        return float((float(x) - self.lo) / span)


# ── Driver ──────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_mcts_improved(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    emb: su.EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    variants: list[str],
    seed: int,
) -> SearchResult:
    """Improved-MCTS variant of :func:`sampling_unified_sd35.run_mcts`.

    Same signature, same CLI args (plus ``--mcts_improved_*`` toggles).
    Drop-in replacement that holds prompt + cfg search space identical to
    the original; only the *value-estimation* and *backprop* paths differ.
    """
    del variants

    family = getattr(args, "mcts_interp_family", "none")
    n_interp = getattr(args, "mcts_n_interp", 1)
    if family != "none":
        emb = expand_emb_with_interp(emb, family, n_interp)

    use_reward_norm = bool(getattr(args, "mcts_improved_reward_norm", True))
    use_x0_bootstrap = bool(getattr(args, "mcts_improved_x0_bootstrap", True))
    use_ucb_tuned = bool(getattr(args, "mcts_improved_ucb_tuned", True))
    use_late_baseline = bool(getattr(args, "mcts_improved_late_step_baseline", True))
    late_frac = float(getattr(args, "mcts_improved_late_step_frac", 0.7))

    corr_strengths = list(getattr(args, "correction_strengths", [0.0]))
    base_actions = [
        (vi, cfg, cs)
        for vi in range(len(emb.cond_text))
        for cfg in args.cfg_scales
        for cs in corr_strengths
    ]
    if not base_actions:
        raise RuntimeError("MCTS-improved requires non-empty action space.")

    total_steps = int(args.steps)
    use_euler = bool(getattr(args, "euler_sampler", False))
    x0_sampler = bool(getattr(args, "x0_sampler", False))

    latents0 = make_latents(ctx, seed, args.height, args.width, emb.cond_text[0].dtype)
    dx0 = torch.zeros_like(latents0)
    sched = step_schedule(
        ctx.device, latents0.dtype, args.steps,
        getattr(args, "sigmas", None), euler=use_euler,
    )
    _, t0_4d, _ = sched[0]
    start_latents = (1.0 - t0_4d) * dx0 + t0_4d * latents0 if not use_euler else latents0

    key_steps = _parse_key_steps(args)
    if key_steps is None:
        key_steps = list(range(int(args.steps)))
    if 0 not in key_steps:
        key_steps = [0] + key_steps
    key_steps = sorted(set(key_steps))
    n_key = len(key_steps)
    key_segments = [
        (key_steps[i], key_steps[i + 1] if i + 1 < n_key else int(args.steps))
        for i in range(n_key)
    ]
    late_step_threshold = int(round(late_frac * n_key))

    fresh_noise_steps = _resolve_mcts_fresh_noise_steps(args, int(args.steps), key_steps=key_steps)

    baseline_cfg = float(getattr(args, "baseline_cfg", args.cfg_scales[0]))
    baseline_action = _find_baseline_action(base_actions, baseline_cfg)

    root = MCTSNodeV2(0, dx0, start_latents)
    normalizer = _RewardNormalizer()
    best_global_score = -float("inf")
    best_global_dx: torch.Tensor | None = None
    best_global_latents: torch.Tensor | None = None
    best_global_path: list[tuple] = []

    print(
        f"  mcts-improved: sims={args.n_sims} actions={len(base_actions)} "
        f"steps={args.steps} key_steps={key_steps} ({n_key} branch points) "
        f"flags=[norm={use_reward_norm} x0={use_x0_bootstrap} "
        f"ucb_tuned={use_ucb_tuned} late_baseline={use_late_baseline}]"
    )

    for sim in range(int(args.n_sims)):
        node = root
        path: list[tuple[MCTSNodeV2, tuple]] = []
        action: tuple | None = None

        # ── Selection ──────────────────────────────────────────────────────
        while not node.is_leaf(n_key):
            untried = node.untried_actions(base_actions)
            if untried:
                action = untried[np.random.randint(len(untried))]
                break
            if use_ucb_tuned:
                action = node.best_ucb_tuned(base_actions, args.ucb_c)
            else:
                action = node.best_ucb(base_actions, args.ucb_c)
            path.append((node, action))
            node = node.children[action]

        if action is None:
            raise RuntimeError("MCTS-improved failed to pick an action.")

        # ── Expansion ──────────────────────────────────────────────────────
        if not node.is_leaf(n_key):
            if action not in node.children:
                seg_start, seg_end = key_segments[node.step]
                child_lat, child_dx = _run_segment(
                    args, ctx, emb, reward_model, prompt,
                    node.latents, node.dx, action, sched, seg_start, seg_end,
                    noise_explore_steps=fresh_noise_steps, eps_bank=None,
                )
                node.children[action] = MCTSNodeV2(node.step + 1, child_dx, child_lat)
            path.append((node, action))
            node = node.children[action]

        # ── Value estimate (x0-pred bootstrap OR full rollout) ─────────────
        if use_x0_bootstrap:
            # Decode current x0_pred (or running latents for Euler) directly.
            tensor_for_decode = _final_decode_tensor(node.latents, node.dx, use_euler)
            img = decode_to_pil(ctx, tensor_for_decode)
            value = float(score_image(reward_model, prompt, img))
            rollout_dx = node.dx
            rollout_latents = node.latents
            rollout_actions: list[tuple] = []
        else:
            rollout_dx = node.dx
            rollout_latents = node.latents
            rollout_key_idx = node.step
            rollout_actions = []
            while rollout_key_idx < n_key:
                if use_late_baseline and rollout_key_idx >= late_step_threshold:
                    rollout_action = baseline_action
                else:
                    rollout_action = base_actions[np.random.randint(len(base_actions))]
                rollout_actions.append(rollout_action)
                seg_start, seg_end = key_segments[rollout_key_idx]
                rollout_latents, rollout_dx = _run_segment(
                    args, ctx, emb, reward_model, prompt,
                    rollout_latents, rollout_dx, rollout_action, sched, seg_start, seg_end,
                    noise_explore_steps=fresh_noise_steps, eps_bank=None,
                )
                rollout_key_idx += 1
            tensor_for_decode = _final_decode_tensor(rollout_latents, rollout_dx, use_euler)
            img = decode_to_pil(ctx, tensor_for_decode)
            value = float(score_image(reward_model, prompt, img))

        # Track best raw score globally (always raw, never normalized).
        if value > best_global_score:
            best_global_score = value
            best_global_dx = tensor_for_decode.clone()
            best_global_latents = rollout_latents
            best_global_path = [a for _, a in path] + list(rollout_actions)

        # ── Backprop ───────────────────────────────────────────────────────
        if use_reward_norm:
            normalizer.update(value)
            backprop_value = normalizer.normalize(value)
        else:
            backprop_value = value

        for pnode, paction in path:
            pnode.visits += 1
            pnode.action_visits[paction] = pnode.action_visits.get(paction, 0) + 1
            pnode.action_values[paction] = pnode.action_values.get(paction, 0.0) + backprop_value
            pnode.action_value_sq[paction] = (
                pnode.action_value_sq.get(paction, 0.0) + backprop_value * backprop_value
            )

        if (sim + 1) % 10 == 0 or sim == 0:
            tag = "x0-bootstrap" if use_x0_bootstrap else "rollout"
            print(
                f"    sim {sim + 1:3d}/{args.n_sims} {tag} value={value:.4f} "
                f"best_global={best_global_score:.4f} "
                f"norm_range=[{normalizer.lo if math.isfinite(normalizer.lo) else 'inf'},"
                f"{normalizer.hi if math.isfinite(normalizer.hi) else '-inf'}]"
            )

    # ── Exploit pass (greedy by mean value through the tree) ───────────────
    exploit_path: list[tuple] = []
    node = root
    for _ in range(n_key):
        a = node.best_exploit(base_actions)
        if a is None:
            break
        exploit_path.append(a)
        if a not in node.children:
            break
        node = node.children[a]

    replay_lat, replay_dx = start_latents, dx0
    for key_idx, exploit_action in enumerate(exploit_path):
        seg_start, seg_end = key_segments[key_idx]
        replay_lat, replay_dx = _run_segment(
            args, ctx, emb, reward_model, prompt,
            replay_lat, replay_dx, exploit_action, sched, seg_start, seg_end,
            noise_explore_steps=fresh_noise_steps, eps_bank=None,
        )
    for key_idx in range(len(exploit_path), n_key):
        seg_start, seg_end = key_segments[key_idx]
        replay_lat, replay_dx = _run_segment(
            args, ctx, emb, reward_model, prompt,
            replay_lat, replay_dx, baseline_action, sched, seg_start, seg_end,
            noise_explore_steps=fresh_noise_steps, eps_bank=None,
        )

    exploit_img = decode_to_pil(ctx, _final_decode_tensor(replay_lat, replay_dx, use_euler))
    exploit_score = float(score_image(reward_model, prompt, exploit_img))

    if exploit_score >= best_global_score or best_global_dx is None:
        selected_img = exploit_img
        selected_score = exploit_score
        selected_path = list(exploit_path)
        selected_source = "exploit"
    else:
        selected_img = decode_to_pil(ctx, best_global_dx)
        selected_score = best_global_score
        selected_path = list(best_global_path)
        selected_source = "best_global"

    diagnostics: dict[str, Any] = {
        "method": "mcts_improved",
        "source": selected_source,
        "exploit_score": float(exploit_score),
        "best_global_score": float(best_global_score),
        "n_sims": int(args.n_sims),
        "flags": {
            "reward_norm": use_reward_norm,
            "x0_bootstrap": use_x0_bootstrap,
            "ucb_tuned": use_ucb_tuned,
            "late_step_baseline": use_late_baseline,
        },
        "norm_range": [
            float(normalizer.lo) if math.isfinite(normalizer.lo) else None,
            float(normalizer.hi) if math.isfinite(normalizer.hi) else None,
        ],
    }
    return SearchResult(
        image=selected_img,
        score=selected_score,
        actions=[(int(a[0]), float(a[1]), float(a[2])) for a in selected_path],
        diagnostics=diagnostics,
    )


# ── Helpers ─────────────────────────────────────────────────────────────────


def _find_baseline_action(actions: list[tuple], baseline_cfg: float) -> tuple:
    """Pick the action closest to (variant=0, cfg=baseline_cfg, cs=0)."""
    best = actions[0]
    best_delta = float("inf")
    for a in actions:
        vi, cfg, cs = int(a[0]), float(a[1]), float(a[2])
        delta = abs(cfg - baseline_cfg) + 100.0 * abs(cs) + 50.0 * vi
        if delta < best_delta:
            best_delta = delta
            best = a
    return best
