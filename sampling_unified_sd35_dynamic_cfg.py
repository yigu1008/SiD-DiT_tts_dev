"""
Standalone SD3.5 unified runner with adaptive per-node CFG discretization for MCTS.

This file keeps the original sampling_unified_sd35 behavior for non-MCTS methods,
and only replaces run_mcts with a dynamic-CFG version.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np
import torch

import sampling_unified_sd35 as su


class DynamicMCTSNode:
    __slots__ = (
        "step",
        "dx",
        "latents",
        "parent",
        "incoming_action",
        "children",
        "visits",
        "action_visits",
        "action_values",
    )

    def __init__(
        self,
        step: int,
        dx: torch.Tensor | None,
        latents: torch.Tensor | None,
        parent: "DynamicMCTSNode | None" = None,
        incoming_action: tuple[int, float, float] | None = None,
    ) -> None:
        self.step = int(step)
        self.dx = dx
        self.latents = latents
        self.parent = parent
        self.incoming_action = incoming_action
        self.children: dict[tuple[int, float, float], DynamicMCTSNode] = {}
        self.visits = 0
        self.action_visits: dict[tuple[int, float, float], int] = {}
        self.action_values: dict[tuple[int, float, float], float] = {}

    def is_leaf(self, max_steps: int) -> bool:
        return self.step >= int(max_steps)

    def untried_actions(self, actions: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
        return [a for a in actions if a not in self.action_visits]

    def ucb(self, action: tuple[int, float, float], c: float) -> float:
        n = self.action_visits.get(action, 0)
        if n <= 0:
            return float("inf")
        mean = float(self.action_values.get(action, 0.0)) / float(n)
        return mean + float(c) * math.sqrt(math.log(max(self.visits, 1)) / float(n))

    def best_ucb(self, actions: list[tuple[int, float, float]], c: float) -> tuple[int, float, float]:
        return max(actions, key=lambda action: self.ucb(action, c))

    def best_exploit(self, actions: list[tuple[int, float, float]]) -> tuple[int, float, float] | None:
        best: tuple[int, float, float] | None = None
        best_v = -float("inf")
        for action in actions:
            n = int(self.action_visits.get(action, 0))
            if n <= 0:
                continue
            value = float(self.action_values.get(action, 0.0)) / float(n)
            if value > best_v:
                best_v = value
                best = action
        return best


def _cfg_delta_for_depth(depth: int) -> float:
    if int(depth) <= 0:
        return 0.50
    if int(depth) == 1:
        return 0.25
    return 0.125


def _parse_cfg_args(argv: list[str] | None) -> argparse.Namespace:
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--mcts_cfg_mode", choices=["adaptive", "fixed"], default="adaptive")
    extra.add_argument("--mcts_cfg_root_bank", nargs="+", type=float, default=[1.0, 1.5, 2.0])
    extra.add_argument("--mcts_cfg_anchors", nargs="+", type=float, default=[1.0, 2.0])
    extra.add_argument("--mcts_cfg_min_parent_visits", type=int, default=3)
    extra.add_argument("--mcts_cfg_round_ndigits", type=int, default=6)
    extra.add_argument("--mcts_cfg_log_action_topk", type=int, default=12)
    cfg_args, remaining = extra.parse_known_args(argv)
    base_args = su.parse_args(remaining)
    for k, v in vars(cfg_args).items():
        setattr(base_args, k, v)
    return base_args


def _cfg_stats(node: DynamicMCTSNode) -> list[dict[str, Any]]:
    grouped: dict[float, dict[str, float]] = {}
    for action, visits in node.action_visits.items():
        v = int(visits)
        if v <= 0:
            continue
        cfg = float(action[1])
        slot = grouped.get(cfg)
        if slot is None:
            slot = {"visits": 0.0, "q_sum": 0.0}
            grouped[cfg] = slot
        slot["visits"] += float(v)
        slot["q_sum"] += float(node.action_values.get(action, 0.0))
    rows: list[dict[str, Any]] = []
    for cfg in sorted(grouped.keys()):
        visits = max(1.0, float(grouped[cfg]["visits"]))
        q_sum = float(grouped[cfg]["q_sum"])
        rows.append(
            {
                "cfg": float(cfg),
                "visits": int(round(visits)),
                "q_sum": float(q_sum),
                "q_mean": float(q_sum / visits),
            }
        )
    return rows


def _action_stats(node: DynamicMCTSNode, topk: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action, visits in node.action_visits.items():
        v = int(visits)
        if v <= 0:
            continue
        q_sum = float(node.action_values.get(action, 0.0))
        rows.append(
            {
                "variant_idx": int(action[0]),
                "cfg": float(action[1]),
                "correction_strength": float(action[2]),
                "visits": int(v),
                "q_sum": float(q_sum),
                "q_mean": float(q_sum / float(v)),
            }
        )
    rows.sort(key=lambda x: (float(x["q_mean"]), int(x["visits"])), reverse=True)
    return rows[: max(1, int(topk))]


def run_mcts_dynamic_cfg(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    emb: su.EmbeddingContext,
    reward_model: su.UnifiedRewardScorer,
    prompt: str,
    variants: list[str],
    seed: int,
) -> su.SearchResult:
    del variants
    family = getattr(args, "mcts_interp_family", "none")
    n_interp = getattr(args, "mcts_n_interp", 1)
    if family != "none":
        emb = su.expand_emb_with_interp(emb, family, n_interp)
        print(f"  mcts: interp={family} n_interp={n_interp} total_variants={len(emb.cond_text)}")

    corr_strengths = [float(x) for x in getattr(args, "correction_strengths", [0.0])]
    if len(corr_strengths) <= 0:
        corr_strengths = [0.0]
    prompt_actions: list[tuple[int, float]] = []
    for vi in range(len(emb.cond_text)):
        for cs in corr_strengths:
            prompt_actions.append((int(vi), float(cs)))

    if len(prompt_actions) <= 0:
        raise RuntimeError("MCTS requires at least one prompt action.")

    global_cfg = [float(x) for x in getattr(args, "cfg_scales", [1.0])]
    if len(global_cfg) <= 0:
        global_cfg = [float(getattr(args, "baseline_cfg", 1.0))]
    cfg_min = float(min(global_cfg))
    cfg_max = float(max(global_cfg))
    cfg_mode = str(getattr(args, "mcts_cfg_mode", "adaptive")).strip().lower()
    cfg_root_bank = [float(x) for x in getattr(args, "mcts_cfg_root_bank", [1.0, 1.5, 2.0])]
    cfg_anchors = [float(x) for x in getattr(args, "mcts_cfg_anchors", [1.0, 2.0])]
    cfg_min_parent_visits = max(1, int(getattr(args, "mcts_cfg_min_parent_visits", 3)))
    cfg_round_ndigits = max(1, int(getattr(args, "mcts_cfg_round_ndigits", 6)))
    action_topk = max(1, int(getattr(args, "mcts_cfg_log_action_topk", 12)))

    def clamp_cfg(cfg: float) -> float:
        return float(round(float(np.clip(float(cfg), cfg_min, cfg_max)), cfg_round_ndigits))

    def dedup_cfg(values: list[float]) -> list[float]:
        out: list[float] = []
        seen: set[float] = set()
        for val in values:
            v = clamp_cfg(float(val))
            if v not in seen:
                out.append(float(v))
                seen.add(float(v))
        if len(out) <= 0:
            out = [clamp_cfg(float(getattr(args, "baseline_cfg", global_cfg[0])))]
        return out

    def make_action(variant_idx: int, cfg: float, corr_strength: float) -> tuple[int, float, float]:
        return (int(variant_idx), float(clamp_cfg(cfg)), float(corr_strength))

    root_cfg_values = dedup_cfg(cfg_root_bank + cfg_anchors)
    fixed_cfg_values = dedup_cfg(global_cfg + cfg_anchors)
    default_cfg = clamp_cfg(float(getattr(args, "baseline_cfg", global_cfg[0])))
    default_action = make_action(prompt_actions[0][0], default_cfg, prompt_actions[0][1])

    def parent_best_cfg(parent: DynamicMCTSNode | None) -> float | None:
        if parent is None:
            return None
        best_cfg = None
        best_mean = -float("inf")
        for action, visits in parent.action_visits.items():
            v = int(visits)
            if v <= 0:
                continue
            mean = float(parent.action_values.get(action, 0.0)) / float(v)
            if mean > best_mean:
                best_mean = mean
                best_cfg = float(action[1])
        return None if best_cfg is None else float(best_cfg)

    def node_cfg_candidates(node: DynamicMCTSNode) -> dict[str, Any]:
        if cfg_mode == "fixed":
            return {
                "cfg_candidates": [float(x) for x in fixed_cfg_values],
                "center": float(np.mean(fixed_cfg_values)),
                "delta": 0.0,
                "center_source": "fixed_global",
                "parent_cfg_visits": int(sum(node.parent.action_visits.values())) if node.parent is not None else 0,
            }

        if int(node.step) <= 0:
            return {
                "cfg_candidates": [float(x) for x in root_cfg_values],
                "center": float(np.mean(root_cfg_values)),
                "delta": float(_cfg_delta_for_depth(0)),
                "center_source": "root_coarse",
                "parent_cfg_visits": 0,
            }

        delta = float(_cfg_delta_for_depth(int(node.step)))
        parent = node.parent
        parent_visits = int(sum(parent.action_visits.values())) if parent is not None else 0
        center = None
        center_source = "baseline_fallback"

        if parent is not None and parent_visits >= cfg_min_parent_visits:
            wsum = 0.0
            total = 0
            for action, visits in parent.action_visits.items():
                v = int(visits)
                if v <= 0:
                    continue
                wsum += float(action[1]) * float(v)
                total += int(v)
            if total > 0:
                center = float(wsum / float(total))
                center_source = "parent_visit_weighted"

        if center is None:
            best_cfg = parent_best_cfg(parent)
            if best_cfg is not None:
                center = float(best_cfg)
                center_source = "parent_best_sparse"

        if center is None and node.incoming_action is not None:
            center = float(node.incoming_action[1])
            center_source = "incoming_cfg_fallback"

        if center is None:
            center = float(default_cfg)

        cfg_vals = dedup_cfg([float(center - delta), float(center), float(center + delta)] + cfg_anchors)
        return {
            "cfg_candidates": [float(x) for x in cfg_vals],
            "center": float(center),
            "delta": float(delta),
            "center_source": str(center_source),
            "parent_cfg_visits": int(parent_visits),
        }

    def node_actions(node: DynamicMCTSNode) -> tuple[dict[str, Any], list[tuple[int, float, float]]]:
        meta = node_cfg_candidates(node)
        out: list[tuple[int, float, float]] = []
        for vi, cs in prompt_actions:
            for cfg in meta["cfg_candidates"]:
                out.append(make_action(int(vi), float(cfg), float(cs)))
        if len(out) <= 0:
            out = [default_action]
        return meta, out

    latents0 = su.make_latents(ctx, seed, args.height, args.width, emb.cond_text[0].dtype)
    dx0 = torch.zeros_like(latents0)
    sched = su.step_schedule(ctx.device, latents0.dtype, args.steps, getattr(args, "sigmas", None))
    _, t0_4d = sched[0]
    start_latents = (1.0 - t0_4d) * dx0 + t0_4d * latents0
    root = DynamicMCTSNode(0, dx0, start_latents, parent=None, incoming_action=None)

    best_global_score = -float("inf")
    best_global_dx = None
    best_global_path: list[tuple[int, float, float]] = []
    node_cfg_logs: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []

    rng = np.random.default_rng(int(seed) + 2026)
    n_sims = max(1, int(getattr(args, "n_sims", 50)))
    ucb_c = float(getattr(args, "ucb_c", 1.41))
    log_every = 10
    print(f"  mcts: sims={n_sims} steps={int(args.steps)} cfg_mode={cfg_mode} cfg_range=[{cfg_min:.3f},{cfg_max:.3f}]")

    for sim in range(n_sims):
        node = root
        path: list[tuple[DynamicMCTSNode, tuple[int, float, float]]] = []
        sim_logs: list[dict[str, Any]] = []

        while not node.is_leaf(args.steps):
            cfg_meta, candidates = node_actions(node)
            untried = node.untried_actions(candidates)
            if len(untried) > 0:
                action = untried[int(rng.integers(0, len(untried)))]
                mode = "expand"
                break
            action = node.best_ucb(candidates, ucb_c)
            mode = "ucb"
            path.append((node, action))
            if action in node.children:
                node = node.children[action]
            else:
                break

            # This log row is for traversed nodes before expansion/rollout.
            sim_logs.append(
                {
                    "sim": int(sim),
                    "step": int(path[-1][0].step),
                    "selection_mode": str(mode),
                    "cfg_candidates": [float(x) for x in cfg_meta["cfg_candidates"]],
                    "cfg_center": float(cfg_meta["center"]),
                    "cfg_delta": float(cfg_meta["delta"]),
                    "cfg_center_source": str(cfg_meta["center_source"]),
                    "parent_cfg_visits": int(cfg_meta["parent_cfg_visits"]),
                    "chosen_cfg": float(path[-1][1][1]),
                    "chosen_action": {
                        "variant_idx": int(path[-1][1][0]),
                        "cfg": float(path[-1][1][1]),
                        "correction_strength": float(path[-1][1][2]),
                    },
                    "node_visits_before": int(path[-1][0].visits),
                    "node_cfg_stats_before": _cfg_stats(path[-1][0]),
                    "node_action_stats_before": _action_stats(path[-1][0], action_topk),
                }
            )

        if not node.is_leaf(args.steps):
            cfg_meta, _ = node_actions(node)
            if action not in node.children:
                child_dx, child_lat = su._expand_child(args, ctx, emb, node, action, sched, reward_model, prompt)
                node.children[action] = DynamicMCTSNode(
                    step=node.step + 1,
                    dx=child_dx,
                    latents=child_lat,
                    parent=node,
                    incoming_action=action,
                )
            path.append((node, action))
            sim_logs.append(
                {
                    "sim": int(sim),
                    "step": int(node.step),
                    "selection_mode": "expand",
                    "cfg_candidates": [float(x) for x in cfg_meta["cfg_candidates"]],
                    "cfg_center": float(cfg_meta["center"]),
                    "cfg_delta": float(cfg_meta["delta"]),
                    "cfg_center_source": str(cfg_meta["center_source"]),
                    "parent_cfg_visits": int(cfg_meta["parent_cfg_visits"]),
                    "chosen_cfg": float(action[1]),
                    "chosen_action": {
                        "variant_idx": int(action[0]),
                        "cfg": float(action[1]),
                        "correction_strength": float(action[2]),
                    },
                    "node_visits_before": int(node.visits),
                    "node_cfg_stats_before": _cfg_stats(node),
                    "node_action_stats_before": _action_stats(node, action_topk),
                }
            )
            node = node.children[action]

        rollout_dx = node.dx
        rollout_latents = node.latents
        rollout_step = node.step
        rollout_node = node
        while rollout_step < int(args.steps):
            _, roll_candidates = node_actions(rollout_node)
            variant_idx, cfg, cs = roll_candidates[int(rng.integers(0, len(roll_candidates)))]
            t_flat, t_4d = sched[rollout_step]
            flow = su.transformer_step(args, ctx, rollout_latents, emb, variant_idx, t_flat, cfg)
            rollout_dx = su._pred_x0(rollout_latents, t_4d, flow, args.x0_sampler)
            if float(cs) > 0.0:
                rollout_dx = su.apply_reward_correction(ctx, rollout_dx, prompt, reward_model, float(cs), cfg=float(cfg))
            rollout_step += 1
            if rollout_step < int(args.steps):
                _, next_t_4d = sched[rollout_step]
                noise = torch.randn_like(rollout_dx)
                rollout_latents = (1.0 - next_t_4d) * rollout_dx + next_t_4d * noise
                rollout_node = DynamicMCTSNode(
                    step=rollout_step,
                    dx=rollout_dx,
                    latents=rollout_latents,
                    parent=rollout_node,
                    incoming_action=(int(variant_idx), float(cfg), float(cs)),
                )

        rollout_img = su.decode_to_pil(ctx, rollout_dx)
        rollout_score = su.score_image(reward_model, prompt, rollout_img)
        if rollout_score > best_global_score:
            best_global_score = float(rollout_score)
            best_global_dx = rollout_dx.clone()
            best_global_path = [a for _, a in path]

        for pnode, paction in path:
            pnode.visits += 1
            pnode.action_visits[paction] = pnode.action_visits.get(paction, 0) + 1
            pnode.action_values[paction] = pnode.action_values.get(paction, 0.0) + float(rollout_score)

        for row in sim_logs:
            row["rollout_score"] = float(rollout_score)
        node_cfg_logs.extend(sim_logs)

        if (sim + 1) % log_every == 0 or sim == 0:
            root_meta, root_candidates = node_actions(root)
            top_rows = _action_stats(root, action_topk)
            history.append(
                {
                    "sim": int(sim + 1),
                    "best_score": float(best_global_score),
                    "root_visits": int(root.visits),
                    "cfg_mode": str(cfg_mode),
                    "root_cfg_candidates": [float(x) for x in root_meta["cfg_candidates"]],
                    "root_cfg_center": float(root_meta["center"]),
                    "root_cfg_delta": float(root_meta["delta"]),
                    "root_top_actions": top_rows,
                    "root_candidate_count": int(len(root_candidates)),
                }
            )
            print(f"    sim {sim + 1:3d}/{n_sims} best={best_global_score:.4f}")

    exploit_path: list[tuple[int, float, float]] = []
    node = root
    for _ in range(int(args.steps)):
        _, candidates = node_actions(node)
        action = node.best_exploit(candidates)
        if action is None:
            break
        exploit_path.append(action)
        if action in node.children:
            node = node.children[action]
        else:
            break

    replay_dx = dx0
    replay_lat = start_latents
    for step_idx, (variant_idx, cfg, cs) in enumerate(exploit_path):
        t_flat, t_4d = sched[step_idx]
        flow = su.transformer_step(args, ctx, replay_lat, emb, variant_idx, t_flat, cfg)
        replay_dx = su._pred_x0(replay_lat, t_4d, flow, args.x0_sampler)
        if float(cs) > 0.0:
            replay_dx = su.apply_reward_correction(ctx, replay_dx, prompt, reward_model, float(cs), cfg=float(cfg))
        if step_idx + 1 < int(args.steps):
            _, next_t_4d = sched[step_idx + 1]
            noise = torch.randn_like(replay_dx)
            replay_lat = (1.0 - next_t_4d) * replay_dx + next_t_4d * noise

    exploit_img = su.decode_to_pil(ctx, replay_dx)
    exploit_score = su.score_image(reward_model, prompt, exploit_img)

    out_img = exploit_img
    out_score = float(exploit_score)
    out_actions = exploit_path
    if exploit_score < best_global_score and best_global_dx is not None:
        out_img = su.decode_to_pil(ctx, best_global_dx)
        out_score = float(best_global_score)
        out_actions = list(best_global_path)

    diagnostics = {
        "mcts_cfg_mode": str(cfg_mode),
        "cfg_range": [float(cfg_min), float(cfg_max)],
        "cfg_root_bank": [float(x) for x in root_cfg_values],
        "cfg_global_bank": [float(x) for x in fixed_cfg_values],
        "cfg_anchors": [float(x) for x in dedup_cfg(cfg_anchors)],
        "cfg_min_parent_visits": int(cfg_min_parent_visits),
        "cfg_delta_by_depth": {"0": 0.50, "1": 0.25, "2+": 0.125},
        "history": history,
        "node_cfg_logs": node_cfg_logs,
        "final_cfg_trajectory": [float(a[1]) for a in out_actions],
        "exploit_cfg_trajectory": [float(a[1]) for a in exploit_path],
        "best_global_cfg_trajectory": [float(a[1]) for a in best_global_path],
    }
    return su.SearchResult(
        image=out_img,
        score=float(out_score),
        actions=[(int(v), float(c), float(cs)) for v, c, cs in out_actions],
        diagnostics=diagnostics,
    )


def run(args: argparse.Namespace) -> None:
    original_run_mcts = su.run_mcts
    su.run_mcts = run_mcts_dynamic_cfg
    try:
        su.run(args)
    finally:
        su.run_mcts = original_run_mcts


def main(argv: list[str] | None = None) -> None:
    args = _parse_cfg_args(argv)
    run(su.normalize_paths(args))


if __name__ == "__main__":
    main()

