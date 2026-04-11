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
    extra.add_argument(
        "--mcts_cfg_step_anchor_count",
        type=int,
        default=2,
        help="How many anchors to sample from timestep root bank C_t(0).",
    )
    extra.add_argument("--mcts_cfg_min_parent_visits", type=int, default=3)
    extra.add_argument("--mcts_cfg_round_ndigits", type=int, default=6)
    extra.add_argument("--mcts_cfg_log_action_topk", type=int, default=12)
    extra.add_argument(
        "--mcts_key_mode",
        choices=["all", "manual", "count", "stride"],
        default="count",
        help="How to choose branching key steps. "
             "all=every step, manual=--mcts_key_steps, "
             "count=evenly-spaced key steps using --mcts_key_step_count "
             "(or --mcts_key_default_count when unset), "
             "stride=every K steps using --mcts_key_step_stride.",
    )
    extra.add_argument(
        "--mcts_key_step_stride",
        type=int,
        default=0,
        help="Stride K for --mcts_key_mode stride. 0 disables stride mode.",
    )
    extra.add_argument(
        "--mcts_key_default_count",
        type=int,
        default=2,
        help="Fallback key-step count when --mcts_key_mode count and --mcts_key_step_count<=0.",
    )
    cfg_args, remaining = extra.parse_known_args(argv)
    base_args = su.parse_args(remaining)
    for k, v in vars(cfg_args).items():
        setattr(base_args, k, v)
    return base_args


def _sanitize_key_steps(steps: list[int], total_steps: int) -> list[int]:
    total = max(1, int(total_steps))
    out = sorted(set(int(s) for s in steps if 0 <= int(s) < total))
    if 0 not in out:
        out = [0] + out
    if len(out) <= 0:
        out = [0]
    return out


def _key_steps_by_count(total_steps: int, count: int) -> list[int]:
    total = max(1, int(total_steps))
    c = max(1, int(count))
    if c >= total:
        return list(range(total))
    if c == 1:
        return [0]
    idx = sorted(set(int(round(i * (total - 1) / (c - 1))) for i in range(c)))
    return _sanitize_key_steps(idx, total)


def _resolve_key_steps(args: argparse.Namespace, total_steps: int) -> tuple[list[int], dict[str, Any]]:
    total = max(1, int(total_steps))
    mode = str(getattr(args, "mcts_key_mode", "count")).strip().lower()

    if mode == "all":
        ks = list(range(total))
        return ks, {"mode": "all", "count": int(len(ks))}

    if mode == "manual":
        raw = str(getattr(args, "mcts_key_steps", "")).strip()
        if not raw:
            raise RuntimeError("--mcts_key_mode manual requires --mcts_key_steps")
        parsed = su._parse_key_steps(args)
        if parsed is None:
            ks = list(range(total))
            return ks, {"mode": "manual", "raw": raw, "fallback": "all_steps", "count": int(len(ks))}
        ks = _sanitize_key_steps(parsed, total)
        return ks, {"mode": "manual", "raw": raw, "count": int(len(ks))}

    if mode == "stride":
        stride = int(getattr(args, "mcts_key_step_stride", 0))
        if stride <= 0:
            raise RuntimeError("--mcts_key_mode stride requires --mcts_key_step_stride > 0")
        ks = list(range(0, total, stride))
        if (total - 1) not in ks:
            ks.append(total - 1)
        ks = _sanitize_key_steps(ks, total)
        return ks, {"mode": "stride", "stride": int(stride), "count": int(len(ks))}

    # default: count
    count = int(getattr(args, "mcts_key_step_count", 0))
    default_count = max(1, int(getattr(args, "mcts_key_default_count", 2)))
    if count <= 0:
        count = default_count
        source = "default"
    else:
        source = "explicit"
    ks = _key_steps_by_count(total, count)
    return ks, {"mode": "count", "count": int(len(ks)), "requested_count": int(count), "count_source": str(source)}


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
    cfg_step_anchor_count = max(0, int(getattr(args, "mcts_cfg_step_anchor_count", 2)))
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

    root_cfg_values_step0 = dedup_cfg(cfg_root_bank + cfg_anchors)
    fixed_cfg_values = dedup_cfg(global_cfg + cfg_anchors)
    default_cfg = clamp_cfg(float(getattr(args, "baseline_cfg", global_cfg[0])))
    default_action = make_action(prompt_actions[0][0], default_cfg, prompt_actions[0][1])
    cfg_root_center = float(np.mean(root_cfg_values_step0))

    def timestep_root_bank(step: int) -> dict[str, Any]:
        t = max(0, int(step))
        # C_t(0): coarse at t=0 from explicit root bank, then narrowed by timestep delta.
        if t <= 0:
            bank_vals = dedup_cfg(cfg_root_bank + cfg_anchors)
            source = "explicit_root_bank_t0"
        else:
            dt = float(_cfg_delta_for_depth(t))
            bank_vals = dedup_cfg([float(cfg_root_center - dt), float(cfg_root_center), float(cfg_root_center + dt)] + cfg_anchors)
            source = "timestep_root_bank"
        return {
            "step": int(t),
            "values": [float(x) for x in bank_vals],
            "source": str(source),
        }

    def pick_step_anchors(step_root: list[float], extra_anchors: list[float], k: int) -> list[float]:
        merged = dedup_cfg([float(x) for x in step_root] + [float(x) for x in extra_anchors])
        if k <= 0 or len(merged) <= 0:
            return []
        if len(merged) <= k:
            return [float(x) for x in merged]
        # Pick spread anchors (e.g. min/max for k=2).
        idx = np.linspace(0, len(merged) - 1, num=k, dtype=int).tolist()
        return [float(merged[i]) for i in sorted(set(int(i) for i in idx))]

    def visit_weighted_center(action_visits: dict[tuple[int, float, float], int]) -> tuple[float | None, int]:
        total = 0
        wsum = 0.0
        for action, visits in action_visits.items():
            v = int(visits)
            if v <= 0:
                continue
            total += int(v)
            wsum += float(action[1]) * float(v)
        if total <= 0:
            return None, 0
        return float(wsum / float(total)), int(total)

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
            step_root = timestep_root_bank(int(node.step))
            return {
                "cfg_candidates": [float(x) for x in fixed_cfg_values],
                "center": float(np.mean(fixed_cfg_values)),
                "delta": 0.0,
                "center_source": "fixed_global",
                "step": int(node.step),
                "step_root_bank": [float(x) for x in step_root["values"]],
                "step_root_anchors": [],
                "parent_cfg_visits": int(sum(node.parent.action_visits.values())) if node.parent is not None else 0,
            }

        step = int(node.step)
        step_root = timestep_root_bank(step)
        step_root_values = [float(x) for x in step_root["values"]]
        step_root_anchors = pick_step_anchors(step_root_values, cfg_anchors, cfg_step_anchor_count)

        if step <= 0:
            return {
                "cfg_candidates": [float(x) for x in step_root_values],
                "center": float(np.mean(step_root_values)),
                "delta": float(_cfg_delta_for_depth(0)),
                "center_source": "root_timestep_bank",
                "step": int(step),
                "step_root_bank": [float(x) for x in step_root_values],
                "step_root_anchors": [float(x) for x in step_root_anchors],
                "parent_cfg_visits": 0,
            }

        delta = float(_cfg_delta_for_depth(step))
        parent = node.parent
        parent_visits = int(sum(parent.action_visits.values())) if parent is not None else 0
        center = None
        center_source = "root_bank_fallback"

        # Local timestep center c_t(n): visit-weighted from node stats when available.
        local_center, local_visits = visit_weighted_center(node.action_visits)
        if local_center is not None and local_visits >= cfg_min_parent_visits:
            center = float(local_center)
            center_source = "node_visit_weighted"

        if center is None and parent is not None and parent_visits > 0:
            parent_center, parent_total = visit_weighted_center(parent.action_visits)
            if parent_center is not None and parent_total >= cfg_min_parent_visits:
                center = float(parent_center)
                center_source = "parent_visit_weighted_sparse"

        if center is None:
            best_cfg = parent_best_cfg(parent)
            if best_cfg is not None:
                center = float(best_cfg)
                center_source = "parent_best_sparse"

        if center is None and node.incoming_action is not None:
            center = float(node.incoming_action[1])
            center_source = "incoming_cfg_fallback"

        if center is None:
            center = float(np.mean(step_root_values)) if len(step_root_values) > 0 else float(default_cfg)
            center_source = "step_root_mean_fallback"

        # C_t(n)=DedupClamp({c_t-delta_t,c_t,c_t+delta_t} U anchors_from_C_t(0)).
        cfg_vals = dedup_cfg(
            [float(center - delta), float(center), float(center + delta)]
            + [float(x) for x in step_root_anchors]
        )
        return {
            "cfg_candidates": [float(x) for x in cfg_vals],
            "center": float(center),
            "delta": float(delta),
            "center_source": str(center_source),
            "step": int(step),
            "step_root_bank": [float(x) for x in step_root_values],
            "step_root_anchors": [float(x) for x in step_root_anchors],
            "parent_cfg_visits": int(parent_visits),
            "node_cfg_visits": int(local_visits),
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
    use_euler = getattr(args, "euler_sampler", False)
    sched = su.step_schedule(ctx.device, latents0.dtype, args.steps, getattr(args, "sigmas", None), euler=use_euler)
    total_steps = int(len(sched))
    _, t0_4d, _ = sched[0]
    if use_euler:
        start_latents = latents0
    else:
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

    # --- Key-step branching setup ---
    key_steps, key_step_meta = _resolve_key_steps(args, total_steps)
    n_key = len(key_steps)
    key_segments: list[tuple[int, int]] = []
    for i in range(n_key):
        seg_start = key_steps[i]
        seg_end = key_steps[i + 1] if i + 1 < n_key else int(total_steps)
        key_segments.append((seg_start, seg_end))

    log_every = 10
    print(
        f"  mcts: sims={n_sims} steps={int(total_steps)} key_steps={key_steps} ({n_key} branch points) "
        f"key_mode={key_step_meta.get('mode', 'count')} "
        f"cfg_mode={cfg_mode} cfg_range=[{cfg_min:.3f},{cfg_max:.3f}]"
    )

    for sim in range(n_sims):
        node = root
        path: list[tuple[DynamicMCTSNode, tuple[int, float, float]]] = []
        sim_logs: list[dict[str, Any]] = []

        while not node.is_leaf(n_key):
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
                    "cfg_step": int(cfg_meta.get("step", path[-1][0].step)),
                    "cfg_step_root_bank": [float(x) for x in cfg_meta.get("step_root_bank", [])],
                    "cfg_step_root_anchors": [float(x) for x in cfg_meta.get("step_root_anchors", [])],
                    "parent_cfg_visits": int(cfg_meta["parent_cfg_visits"]),
                    "node_cfg_visits": int(cfg_meta.get("node_cfg_visits", 0)),
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

        if not node.is_leaf(n_key):
            cfg_meta, _ = node_actions(node)
            if action not in node.children:
                seg_start, seg_end = key_segments[node.step]
                child_lat, child_dx = su._run_segment(
                    args, ctx, emb, reward_model, prompt,
                    node.latents, node.dx, action, sched, seg_start, seg_end)
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
                    "cfg_step": int(cfg_meta.get("step", node.step)),
                    "cfg_step_root_bank": [float(x) for x in cfg_meta.get("step_root_bank", [])],
                    "cfg_step_root_anchors": [float(x) for x in cfg_meta.get("step_root_anchors", [])],
                    "parent_cfg_visits": int(cfg_meta["parent_cfg_visits"]),
                    "node_cfg_visits": int(cfg_meta.get("node_cfg_visits", 0)),
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
        rollout_key_idx = node.step
        rollout_node = node
        while rollout_key_idx < n_key:
            _, roll_candidates = node_actions(rollout_node)
            variant_idx, cfg, cs = roll_candidates[int(rng.integers(0, len(roll_candidates)))]
            seg_start, seg_end = key_segments[rollout_key_idx]
            rollout_action = (int(variant_idx), float(cfg), float(cs))
            rollout_latents, rollout_dx = su._run_segment(
                args, ctx, emb, reward_model, prompt,
                rollout_latents, rollout_dx, rollout_action, sched, seg_start, seg_end)
            rollout_key_idx += 1
            if rollout_key_idx < n_key:
                rollout_node = DynamicMCTSNode(
                    step=rollout_key_idx,
                    dx=rollout_dx,
                    latents=rollout_latents,
                    parent=rollout_node,
                    incoming_action=rollout_action,
                )

        rollout_final = su._final_decode_tensor(rollout_latents, rollout_dx, use_euler)
        rollout_img = su.decode_to_pil(ctx, rollout_final)
        rollout_score = su.score_image(reward_model, prompt, rollout_img)
        if rollout_score > best_global_score:
            best_global_score = float(rollout_score)
            best_global_dx = rollout_final.clone()
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
                    "root_cfg_step_root_bank": [float(x) for x in root_meta.get("step_root_bank", [])],
                    "root_cfg_step_root_anchors": [float(x) for x in root_meta.get("step_root_anchors", [])],
                    "root_top_actions": top_rows,
                    "root_candidate_count": int(len(root_candidates)),
                }
            )
            print(f"    sim {sim + 1:3d}/{n_sims} best={best_global_score:.4f}")

    exploit_path: list[tuple[int, float, float]] = []
    node = root
    for _ in range(n_key):
        _, candidates = node_actions(node)
        action = node.best_exploit(candidates)
        if action is None:
            break
        exploit_path.append(action)
        if action in node.children:
            node = node.children[action]
        else:
            break

    # Replay key-step policy exactly through denoising segments.
    replay_dx = dx0
    replay_lat = start_latents
    for key_idx, action in enumerate(exploit_path):
        seg_start, seg_end = key_segments[key_idx]
        replay_lat, replay_dx = su._run_segment(
            args, ctx, emb, reward_model, prompt,
            replay_lat, replay_dx, action, sched, seg_start, seg_end,
        )
    # Fill missing tail with baseline action if exploit path is short.
    fallback_action = make_action(prompt_actions[0][0], default_cfg, prompt_actions[0][1])
    for key_idx in range(len(exploit_path), n_key):
        seg_start, seg_end = key_segments[key_idx]
        replay_lat, replay_dx = su._run_segment(
            args, ctx, emb, reward_model, prompt,
            replay_lat, replay_dx, fallback_action, sched, seg_start, seg_end,
        )

    exploit_img = su.decode_to_pil(ctx, su._final_decode_tensor(replay_lat, replay_dx, use_euler))
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
        "mcts_key_mode": str(key_step_meta.get("mode", "count")),
        "mcts_key_step_meta": key_step_meta,
        "mcts_key_steps": [int(x) for x in key_steps],
        "mcts_key_segments": [[int(a), int(b)] for a, b in key_segments],
        "cfg_range": [float(cfg_min), float(cfg_max)],
        "cfg_root_bank_step0": [float(x) for x in root_cfg_values_step0],
        "cfg_global_bank": [float(x) for x in fixed_cfg_values],
        "cfg_anchors": [float(x) for x in dedup_cfg(cfg_anchors)],
        "cfg_step_anchor_count": int(cfg_step_anchor_count),
        "cfg_min_parent_visits": int(cfg_min_parent_visits),
        "cfg_delta_by_depth": {"0": 0.50, "1": 0.25, "2+": 0.125},
        "cfg_step_root_banks": {
            str(step): timestep_root_bank(step)["values"]
            for step in range(int(total_steps))
        },
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
