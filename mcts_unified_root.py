"""Unified MCTS where the seed pool sits at the root layer.

A single tree replaces the two-phase bon_mcts (BoN prescreen → top-K refine).
Layer 0 of the tree is `n_seeds` "seed arms"; each arm's first visit runs the
full denoise (= what bon_mcts called "prescreen"). UCB1 selects which seed
to descend into; subsequent visits to the same seed run a regular MCTS sim
within that seed's subtree.

Cost is the same as bon_mcts at matched total_sims: 8 mandatory first visits
+ (total_sims − 8) UCB-driven sims, where good seeds get more budget and
bad seeds get little. No hard top-K cutoff.

Default total_sims = `args.n_sims` (so the existing `--n_sims` flag plumbs
through naturally).
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


def add_unified_root_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--unified_root_n_seeds",
        type=int,
        default=8,
        help="Number of seed-arms at the tree root. First visit to each seed "
             "is a full denoise; subsequent visits run MCTS in that seed's subtree.",
    )
    parser.add_argument(
        "--unified_root_seed_stride",
        type=int,
        default=1,
        help="Stride between candidate seeds (seeds = base_seed + i*stride for i in 0..N).",
    )
    parser.add_argument(
        "--unified_root_seed_offset",
        type=int,
        default=0,
        help="Additive offset before generating the seed pool.",
    )
    parser.add_argument(
        "--unified_root_seed_ucb_c",
        type=float,
        default=1.0,
        help="UCB1 exploration constant for SEED selection at the root layer.",
    )


# ── Driver ──────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_mcts_unified_root(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    emb: su.EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    variants: list[str],
    seed: int,
) -> SearchResult:
    """Single-tree MCTS with seed pool at the root."""
    del variants

    family = getattr(args, "mcts_interp_family", "none")
    n_interp = getattr(args, "mcts_n_interp", 1)
    if family != "none":
        emb = expand_emb_with_interp(emb, family, n_interp)

    n_seeds = max(1, int(getattr(args, "unified_root_n_seeds", 8)))
    seed_stride = max(1, int(getattr(args, "unified_root_seed_stride", 1)))
    seed_offset = int(getattr(args, "unified_root_seed_offset", 0))
    seed_ucb_c = float(getattr(args, "unified_root_seed_ucb_c", 1.0))
    action_ucb_c = float(getattr(args, "ucb_c", 1.0))
    total_sims = max(1, int(getattr(args, "n_sims", 30)))

    corr_strengths = list(getattr(args, "correction_strengths", [0.0]))
    base_actions = [
        (vi, cfg, cs)
        for vi in range(len(emb.cond_text))
        for cfg in args.cfg_scales
        for cs in corr_strengths
    ]
    if not base_actions:
        raise RuntimeError("unified_root requires non-empty action space.")

    use_euler = bool(getattr(args, "euler_sampler", False))
    x0_sampler = bool(getattr(args, "x0_sampler", False))

    # Seed pool.
    seed_pool = [int(seed + seed_offset + i * seed_stride) for i in range(n_seeds)]

    # Build a per-seed initial state lazily — only when the seed is first visited.
    sched_template_dtype = emb.cond_text[0].dtype
    sched = None  # built once on first use
    n_key = 0
    key_steps: list[int] = []
    key_segments: list[tuple[int, int]] = []
    fresh_noise_steps: set[int] = set()

    baseline_cfg = float(getattr(args, "baseline_cfg", args.cfg_scales[0]))

    # Per-seed subtrees keyed by seed.
    seed_roots: dict[int, MCTSNode] = {}
    seed_start_latents: dict[int, torch.Tensor] = {}
    seed_dx0: dict[int, torch.Tensor] = {}

    # Root (layer 0) seed-arm stats.
    root_visits = 0
    seed_arm_visits: dict[int, int] = {s: 0 for s in seed_pool}
    seed_arm_values: dict[int, float] = {s: 0.0 for s in seed_pool}

    best_global_score = -float("inf")
    best_global_dx: torch.Tensor | None = None
    best_global_seed = seed_pool[0]

    def _ensure_schedule() -> None:
        nonlocal sched, n_key, key_steps, key_segments, fresh_noise_steps
        if sched is not None:
            return
        # Build schedule once using a sentinel latent for shape.
        sentinel = make_latents(ctx, seed_pool[0], args.height, args.width, sched_template_dtype)
        sched = step_schedule(
            ctx.device, sentinel.dtype, args.steps,
            getattr(args, "sigmas", None), euler=use_euler,
        )
        ks = _parse_key_steps(args)
        if ks is None:
            ks = list(range(int(args.steps)))
        if 0 not in ks:
            ks = [0] + ks
        key_steps = sorted(set(ks))
        n_key = len(key_steps)
        key_segments[:] = [
            (key_steps[i], key_steps[i + 1] if i + 1 < n_key else int(args.steps))
            for i in range(n_key)
        ]
        fresh_noise_steps |= _resolve_mcts_fresh_noise_steps(args, int(args.steps), key_steps=key_steps)

    def _initialize_seed_root(s: int) -> None:
        """First-visit setup: make seed-s's noise, run full denoise to score, init subtree."""
        _ensure_schedule()
        latents0 = make_latents(ctx, s, args.height, args.width, sched_template_dtype)
        dx0 = torch.zeros_like(latents0)
        _, t0_4d, _ = sched[0]
        start_latents = (1.0 - t0_4d) * dx0 + t0_4d * latents0 if not use_euler else latents0
        seed_dx0[s] = dx0
        seed_start_latents[s] = start_latents
        seed_roots[s] = MCTSNode(0, dx0, start_latents)

    def _ucb_seed(s: int) -> float:
        n = seed_arm_visits.get(s, 0)
        if n <= 0:
            return float("inf")  # forces a first visit
        mean = seed_arm_values[s] / n
        return mean + seed_ucb_c * math.sqrt(math.log(max(root_visits, 1)) / n)

    def _select_seed() -> int:
        """At the root, pick the seed maximizing UCB1."""
        return max(seed_pool, key=_ucb_seed)

    def _full_denoise_score(s: int) -> tuple[float, torch.Tensor]:
        """First-visit cost: full baseline denoise from seed s, return (score, final_tensor)."""
        # Use the existing run_baseline so we share all the correct sampler logic.
        img, score = su.run_baseline(
            args, ctx, emb, reward_model, prompt, s, cfg_scale=float(baseline_cfg),
        )
        # We don't have direct access to dx after run_baseline (it returns image/score),
        # but the subtree root MCTSNode is already initialized in _initialize_seed_root
        # with dx0 = zeros. The "full trajectory" here serves only as the first-visit
        # value estimate; subsequent MCTS sims build their own trajectories from dx0.
        del img  # free memory
        return float(score), torch.zeros_like(seed_dx0[s])

    def _run_one_subtree_sim(s: int) -> tuple[float, torch.Tensor]:
        """Standard MCTS sim within seed s's subtree. Returns (rollout_score, final_tensor)."""
        node = seed_roots[s]
        path: list[tuple[MCTSNode, tuple]] = []
        action: tuple | None = None

        # Selection.
        while not node.is_leaf(n_key):
            untried = node.untried_actions(base_actions)
            if untried:
                action = untried[np.random.randint(len(untried))]
                break
            action = node.best_ucb(base_actions, action_ucb_c)
            path.append((node, action))
            node = node.children[action]

        if action is None:
            # Tree is fully explored — re-decode and return current state's score.
            tensor = _final_decode_tensor(node.latents, node.dx, use_euler)
            img = decode_to_pil(ctx, tensor)
            return float(score_image(reward_model, prompt, img)), tensor.clone()

        # Expansion.
        if not node.is_leaf(n_key):
            if action not in node.children:
                seg_start, seg_end = key_segments[node.step]
                child_lat, child_dx = _run_segment(
                    args, ctx, emb, reward_model, prompt,
                    node.latents, node.dx, action, sched, seg_start, seg_end,
                    noise_explore_steps=fresh_noise_steps, eps_bank=None,
                )
                node.children[action] = MCTSNode(node.step + 1, child_dx, child_lat)
            path.append((node, action))
            node = node.children[action]

        # Random rollout to terminal.
        rollout_dx = node.dx
        rollout_lat = node.latents
        rollout_idx = node.step
        while rollout_idx < n_key:
            action_r = base_actions[np.random.randint(len(base_actions))]
            seg_start, seg_end = key_segments[rollout_idx]
            rollout_lat, rollout_dx = _run_segment(
                args, ctx, emb, reward_model, prompt,
                rollout_lat, rollout_dx, action_r, sched, seg_start, seg_end,
                noise_explore_steps=fresh_noise_steps, eps_bank=None,
            )
            rollout_idx += 1

        tensor = _final_decode_tensor(rollout_lat, rollout_dx, use_euler)
        img = decode_to_pil(ctx, tensor)
        rollout_score = float(score_image(reward_model, prompt, img))

        # Backprop within subtree.
        for pnode, paction in path:
            pnode.visits += 1
            pnode.action_visits[paction] = pnode.action_visits.get(paction, 0) + 1
            pnode.action_values[paction] = pnode.action_values.get(paction, 0.0) + rollout_score

        return rollout_score, tensor.clone()

    print(
        f"  unified-root: total_sims={total_sims} n_seeds={n_seeds} "
        f"seed_ucb_c={seed_ucb_c} action_ucb_c={action_ucb_c} "
        f"actions={len(base_actions)} steps={args.steps}"
    )

    # ── Main loop ────────────────────────────────────────────────────────
    for sim in range(int(total_sims)):
        s = _select_seed()
        is_first_visit = (seed_arm_visits.get(s, 0) == 0)

        if is_first_visit:
            # Initialize the subtree, then do a full denoise to score this seed.
            _initialize_seed_root(s)
            value, tensor = _full_denoise_score(s)
        else:
            # Subtree already initialized; run an MCTS sim inside it.
            value, tensor = _run_one_subtree_sim(s)

        # Update root-layer (seed) stats.
        root_visits += 1
        seed_arm_visits[s] = seed_arm_visits.get(s, 0) + 1
        seed_arm_values[s] = seed_arm_values.get(s, 0.0) + value

        if value > best_global_score:
            best_global_score = value
            best_global_dx = tensor
            best_global_seed = s

        if (sim + 1) % 5 == 0 or sim == 0:
            tag = "first-visit" if is_first_visit else "subtree-sim"
            print(
                f"    sim {sim + 1:3d}/{total_sims} {tag} seed={s} value={value:.4f} "
                f"best_global={best_global_score:.4f}"
            )

    # ── Exploit: pick the seed with highest mean value, then greedy through subtree ──
    best_seed = max(seed_pool, key=lambda s: (
        (seed_arm_values[s] / seed_arm_visits[s]) if seed_arm_visits.get(s, 0) > 0 else -float("inf")
    ))

    if best_seed in seed_roots:
        node = seed_roots[best_seed]
        exploit_path: list[tuple] = []
        for _ in range(n_key):
            a = node.best_exploit(base_actions)
            if a is None:
                break
            exploit_path.append(a)
            if a not in node.children:
                break
            node = node.children[a]

        # Replay deterministically.
        replay_lat = seed_start_latents[best_seed]
        replay_dx = seed_dx0[best_seed]
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
                replay_lat, replay_dx, base_actions[0], sched, seg_start, seg_end,
                noise_explore_steps=fresh_noise_steps, eps_bank=None,
            )
        exploit_img = decode_to_pil(ctx, _final_decode_tensor(replay_lat, replay_dx, use_euler))
        exploit_score = float(score_image(reward_model, prompt, exploit_img))
    else:
        exploit_path = []
        exploit_img = None
        exploit_score = -float("inf")

    if exploit_score >= best_global_score or best_global_dx is None:
        selected_img = exploit_img
        selected_score = exploit_score
        selected_path = list(exploit_path)
        selected_seed = best_seed
        selected_source = "exploit"
    else:
        selected_img = decode_to_pil(ctx, best_global_dx)
        selected_score = best_global_score
        selected_path = []
        selected_seed = best_global_seed
        selected_source = "best_global"

    diagnostics: dict[str, Any] = {
        "method": "unified_root",
        "source": selected_source,
        "exploit_score": float(exploit_score),
        "best_global_score": float(best_global_score),
        "best_global_seed": int(best_global_seed),
        "selected_seed": int(selected_seed),
        "total_sims": int(total_sims),
        "n_seeds": int(n_seeds),
        "seed_pool": [int(s) for s in seed_pool],
        "seed_arm_visits": {str(k): int(v) for k, v in seed_arm_visits.items()},
        "seed_arm_means": {
            str(k): float(seed_arm_values[k] / max(1, seed_arm_visits[k]))
            for k in seed_pool
        },
    }
    if selected_img is None:
        # Shouldn't happen, but guard against it.
        raise RuntimeError("unified_root produced no result.")

    return SearchResult(
        image=selected_img,
        score=selected_score,
        actions=[(int(a[0]), float(a[1]), float(a[2])) for a in selected_path],
        diagnostics=diagnostics,
    )
