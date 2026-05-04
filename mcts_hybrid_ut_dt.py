"""Hybrid MCTS: U_t / D_t latent-displacement priors + reward norm + x0-bootstrap.

This is a deliberate cherry-pick of two existing modules:

  * From ``sampling_unified_sd35_lookahead_reweighting``: the per-action
    *trajectory feedback* prior. For each child node we measure
       u_t = ‖child.latents − parent.latents‖_rms     (latent displacement)
       d_t = ‖child.dx‖_rms                            (running x0_pred magnitude)
    Then the per-action prior is softmax over
       phi(a) = w_u · (−log(1 + u_t/u_ref))  +  w_d · tanh(log(1 + d_t/d_ref))
    (Small u_t and large d_t are favored — small latent change combined
    with a confident running x0_pred indicates a settled trajectory.)
    Selection is **PUCT** with this prior:
       PUCT(a) = Q(a) + c_puct · π(a) · √N / (1 + n(a))

  * From ``mcts_improved``: reward normalization + x0-pred bootstrap +
    late-step baseline rollout.

What's intentionally dropped from the lookahead module:
  - adaptive CFG-bank widening (`lookahead_cfg_width_*`)
  - multi-mode prior switching (`lookahead_prior_mode`)
  - sparse noise refine post-pass
  - per-step reference distributions
  - all the prior_mode={signal, progress_prompt, trajectory_feedback} variants
    other than the trajectory_feedback formula itself.

The point is to keep the **U_t/D_t prior** because that's the part that
encoded real signal about the local denoising trajectory, while throwing
away the rest of the lookahead complexity.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np
import torch
from PIL import Image

import sampling_unified_sd35 as su
from mcts_improved import MCTSNodeV2, _RewardNormalizer, _find_baseline_action
from sampling_unified_sd35 import (
    SearchResult,
    UnifiedRewardScorer,
    _final_decode_tensor,
    _parse_key_steps,
    _resolve_mcts_fresh_noise_steps,
    _run_segment,
    decode_to_pil,
    expand_emb_with_interp,
    make_latents,
    score_image,
    step_schedule,
)
from sampling_unified_sd35_lookahead_reweighting import _compute_u_t, _rms_tensor


# ── CLI ─────────────────────────────────────────────────────────────────────


def add_mcts_hybrid_args(parser: argparse.ArgumentParser) -> None:
    # U_t / D_t prior knobs (subset of lookahead module's flags).
    parser.add_argument("--mcts_hybrid_u_t_def", choices=["latent_delta_rms", "latent_rms", "dx_rms"],
                        default="latent_delta_rms")
    parser.add_argument("--mcts_hybrid_d_t_def", choices=["dx_rms", "latent_rms"],
                        default="dx_rms",
                        help="Definition of D_t signal (running x0_pred magnitude by default).")
    parser.add_argument("--mcts_hybrid_u_ref", type=float, default=1.0)
    parser.add_argument("--mcts_hybrid_d_ref", type=float, default=1.0)
    parser.add_argument("--mcts_hybrid_w_u", type=float, default=1.0)
    parser.add_argument("--mcts_hybrid_w_d", type=float, default=1.0)
    parser.add_argument("--mcts_hybrid_tau", type=float, default=0.35)
    parser.add_argument("--mcts_hybrid_c_puct", type=float, default=1.20)

    # Inherited mcts_improved tricks (defaults all ON).
    parser.add_argument("--mcts_hybrid_reward_norm", action="store_true", default=True)
    parser.add_argument("--mcts_hybrid_no_reward_norm",
                        dest="mcts_hybrid_reward_norm", action="store_false")
    parser.add_argument("--mcts_hybrid_x0_bootstrap", action="store_true", default=True)
    parser.add_argument("--mcts_hybrid_no_x0_bootstrap",
                        dest="mcts_hybrid_x0_bootstrap", action="store_false")
    parser.add_argument("--mcts_hybrid_late_step_baseline", action="store_true", default=True)
    parser.add_argument("--mcts_hybrid_no_late_step_baseline",
                        dest="mcts_hybrid_late_step_baseline", action="store_false")
    parser.add_argument("--mcts_hybrid_late_step_frac", type=float, default=0.7)

    # ── Prior-mode ablation (cell selector) ─────────────────────────────────
    parser.add_argument(
        "--mcts_hybrid_prior_mode",
        choices=["ut_dt", "uniform", "random"],
        default="ut_dt",
        help="Action prior over expanded children:\n"
             "  ut_dt  : softmax of (w_u·u_score + w_d·d_score)/τ — full U_t/D_t prior.\n"
             "          Set w_u=0 for only_d, w_d=0 for only_u.\n"
             "  uniform: π(a) = 1/|A_expanded| — PUCT bonus with no prior signal.\n"
             "  random : π(a) ∝ Dirichlet(1) — fresh per-call random weights."
    )


# ── Node with U_t / D_t ─────────────────────────────────────────────────────


class HybridNode(MCTSNodeV2):
    __slots__ = ("u_t", "d_t", "parent_latents")

    def __init__(self, step: int, dx: torch.Tensor, latents: torch.Tensor | None,
                 *, u_t: float = 0.0, d_t: float = 0.0,
                 parent_latents: torch.Tensor | None = None):
        super().__init__(step, dx, latents)
        self.u_t = float(u_t)
        self.d_t = float(d_t)
        self.parent_latents = parent_latents

    def puct(self, action: tuple, prior: float, c_puct: float) -> float:
        n = self.action_visits.get(action, 0)
        if n > 0:
            q = self.action_values[action] / n
        else:
            q = 0.0
        bonus = c_puct * float(prior) * math.sqrt(max(self.visits, 1)) / (1 + n)
        return q + bonus

    def best_puct(self, actions: list, prior_map: dict, c_puct: float) -> tuple:
        return max(actions, key=lambda a: self.puct(a, prior_map.get(a, 0.0), c_puct))


# ── U_t / D_t prior over actions ────────────────────────────────────────────


def _trajectory_prior(
    node: HybridNode,
    actions: list[tuple],
    *,
    u_ref: float,
    d_ref: float,
    w_u: float,
    w_d: float,
    tau: float,
    prior_mode: str = "ut_dt",
) -> dict[tuple, float]:
    """Per-action prior π(a) over EXPANDED children only.

    prior_mode:
      ut_dt   : π(a) ∝ exp(phi(a) / tau)    where phi = w_u·u_score + w_d·d_score
      uniform : π(a) = 1 / |A_expanded|     (no signal — but PUCT bonus shape)
      random  : π(a) ∝ Dirichlet(1)         (fresh random per call)

    For unexpanded actions, prior = uniform residual mass / count, so PUCT
    still gives them a fair "first visit" allocation through the bonus term.
    """
    eps = 1e-6
    n = len(actions)
    if n <= 0:
        return {}
    expanded_children: dict[tuple, HybridNode] = {}
    for a in actions:
        ch = node.children.get(a)
        if isinstance(ch, HybridNode):
            expanded_children[a] = ch

    if not expanded_children:
        return {a: 1.0 / float(n) for a in actions}

    # Branch on prior_mode.
    if prior_mode == "uniform":
        ne = len(expanded_children)
        weights = np.full((ne,), 1.0 / float(ne), dtype=np.float64)
    elif prior_mode == "random":
        ne = len(expanded_children)
        # Dirichlet(1) → uniform on simplex; uses default RNG so each call differs.
        weights = np.random.dirichlet(np.ones(ne, dtype=np.float64))
    else:
        # ut_dt (default).
        phis: dict[tuple, float] = {}
        for a, ch in expanded_children.items():
            u = max(eps, float(ch.u_t))
            d = max(eps, float(ch.d_t))
            u_score = -math.log1p(u / max(eps, u_ref))
            d_score = math.tanh(math.log1p(d / max(eps, d_ref)))
            phis[a] = w_u * u_score + w_d * d_score
        log_w = np.array([phis[a] / max(eps, tau) for a in expanded_children])
        log_w = log_w - float(np.max(log_w))
        weights = np.exp(np.clip(log_w, -50.0, 50.0))
        weights = weights / max(eps, float(np.sum(weights)))

    n_unexpanded = n - len(expanded_children)
    # Allocate 1/n to each unexpanded; rescale expanded to (1 - unexpanded_share).
    unexpanded_share = (n_unexpanded / float(n)) if n_unexpanded > 0 else 0.0
    expanded_share = 1.0 - unexpanded_share

    out: dict[tuple, float] = {}
    for i, a in enumerate(expanded_children):
        out[a] = float(weights[i]) * expanded_share
    for a in actions:
        if a not in out:
            out[a] = 1.0 / float(n) * (n_unexpanded / float(max(1, n_unexpanded))) if n_unexpanded > 0 else 0.0
            if n_unexpanded > 0:
                out[a] = 1.0 / float(n_unexpanded) * unexpanded_share
    return out


# ── Driver ──────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_mcts_hybrid_ut_dt(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    emb: su.EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    variants: list[str],
    seed: int,
) -> SearchResult:
    del variants

    family = getattr(args, "mcts_interp_family", "none")
    n_interp = getattr(args, "mcts_n_interp", 1)
    if family != "none":
        emb = expand_emb_with_interp(emb, family, n_interp)

    u_t_def = str(getattr(args, "mcts_hybrid_u_t_def", "latent_delta_rms"))
    d_t_def = str(getattr(args, "mcts_hybrid_d_t_def", "dx_rms"))
    u_ref = float(getattr(args, "mcts_hybrid_u_ref", 1.0))
    d_ref = float(getattr(args, "mcts_hybrid_d_ref", 1.0))
    w_u = float(getattr(args, "mcts_hybrid_w_u", 1.0))
    w_d = float(getattr(args, "mcts_hybrid_w_d", 1.0))
    tau = float(getattr(args, "mcts_hybrid_tau", 0.35))
    c_puct = float(getattr(args, "mcts_hybrid_c_puct", 1.20))

    use_reward_norm = bool(getattr(args, "mcts_hybrid_reward_norm", True))
    use_x0_bootstrap = bool(getattr(args, "mcts_hybrid_x0_bootstrap", True))
    use_late_baseline = bool(getattr(args, "mcts_hybrid_late_step_baseline", True))
    late_frac = float(getattr(args, "mcts_hybrid_late_step_frac", 0.7))
    prior_mode = str(getattr(args, "mcts_hybrid_prior_mode", "ut_dt"))

    corr_strengths = list(getattr(args, "correction_strengths", [0.0]))
    base_actions = [
        (vi, cfg, cs)
        for vi in range(len(emb.cond_text))
        for cfg in args.cfg_scales
        for cs in corr_strengths
    ]
    if not base_actions:
        raise RuntimeError("mcts_hybrid requires non-empty action space.")

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

    root = HybridNode(0, dx0, start_latents, parent_latents=None)
    normalizer = _RewardNormalizer()
    best_global_score = -float("inf")
    best_global_dx: torch.Tensor | None = None

    print(
        f"  mcts-hybrid-ut-dt: sims={args.n_sims} actions={len(base_actions)} "
        f"steps={args.steps} key_steps={key_steps} "
        f"prior_mode={prior_mode} (w_u={w_u} w_d={w_d}) tau={tau} c_puct={c_puct} "
        f"flags=[norm={use_reward_norm} x0={use_x0_bootstrap} late_baseline={use_late_baseline}]"
    )

    for sim in range(int(args.n_sims)):
        node = root
        path: list[tuple[HybridNode, tuple]] = []
        action: tuple | None = None

        # ── Selection (PUCT with U_t/D_t prior) ────────────────────────────
        while not node.is_leaf(n_key):
            untried = node.untried_actions(base_actions)
            if untried:
                action = untried[np.random.randint(len(untried))]
                break
            prior = _trajectory_prior(
                node, base_actions,
                u_ref=u_ref, d_ref=d_ref,
                w_u=w_u, w_d=w_d, tau=tau,
                prior_mode=prior_mode,
            )
            action = node.best_puct(base_actions, prior, c_puct)
            path.append((node, action))
            node = node.children[action]

        if action is None:
            raise RuntimeError("mcts_hybrid failed to pick an action.")

        # ── Expansion ──────────────────────────────────────────────────────
        if not node.is_leaf(n_key):
            if action not in node.children:
                seg_start, seg_end = key_segments[node.step]
                parent_latents = node.latents
                child_lat, child_dx = _run_segment(
                    args, ctx, emb, reward_model, prompt,
                    node.latents, node.dx, action, sched, seg_start, seg_end,
                    noise_explore_steps=fresh_noise_steps, eps_bank=None,
                )
                # Compute U_t (latent displacement) and D_t (x0_pred magnitude).
                u_t = _compute_u_t(u_t_def, parent_latents, child_lat, child_dx)
                d_t = _compute_u_t(d_t_def, parent_latents, child_lat, child_dx)
                node.children[action] = HybridNode(
                    node.step + 1, child_dx, child_lat,
                    u_t=u_t, d_t=d_t, parent_latents=parent_latents,
                )
            path.append((node, action))
            node = node.children[action]

        # ── Value estimate ─────────────────────────────────────────────────
        if use_x0_bootstrap:
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

        if value > best_global_score:
            best_global_score = value
            best_global_dx = tensor_for_decode.clone()

        # ── Backprop (reward-normalized) ───────────────────────────────────
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
                f"best_global={best_global_score:.4f}"
            )

    # ── Exploit (greedy-by-mean through tree) ──────────────────────────────
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
        selected_path = []
        selected_source = "best_global"

    diagnostics: dict[str, Any] = {
        "method": "mcts_hybrid_ut_dt",
        "source": selected_source,
        "exploit_score": float(exploit_score),
        "best_global_score": float(best_global_score),
        "n_sims": int(args.n_sims),
        "prior_mode": prior_mode,
        "u_t_def": u_t_def,
        "d_t_def": d_t_def,
        "u_ref": u_ref, "d_ref": d_ref,
        "w_u": w_u, "w_d": w_d, "tau": tau, "c_puct": c_puct,
        "flags": {
            "reward_norm": use_reward_norm,
            "x0_bootstrap": use_x0_bootstrap,
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
