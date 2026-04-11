"""
Standalone SD3.5 runner with lookahead reweighting for MCTS action priors.

This file keeps the base sampling_unified_sd35 behavior for non-MCTS search and
adds a separate MCTS variant with:
  - u_t instrumentation
  - heuristic action logits + softmax priors
  - rollout prior sampling (optional)
  - PUCT tree prior (optional)
  - optional u_t-adaptive local CFG-bank width

It also provides an A-F ablation runner plus a standard MCTS comparison mode.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from typing import Any

import numpy as np
import torch

import sampling_unified_sd35 as su


class LookaheadMCTSNode:
    __slots__ = (
        "step",
        "dx",
        "latents",
        "parent",
        "incoming_action",
        "u_t",
        "children",
        "visits",
        "action_visits",
        "action_values",
    )

    def __init__(
        self,
        step: int,
        dx: torch.Tensor,
        latents: torch.Tensor | None,
        *,
        parent: "LookaheadMCTSNode | None" = None,
        incoming_action: tuple[int, float, float] | None = None,
        u_t: float = 0.0,
    ) -> None:
        self.step = int(step)
        self.dx = dx
        self.latents = latents
        self.parent = parent
        self.incoming_action = incoming_action
        self.u_t = float(u_t)
        self.children: dict[tuple[int, float, float], LookaheadMCTSNode] = {}
        self.visits = 0
        self.action_visits: dict[tuple[int, float, float], int] = {}
        self.action_values: dict[tuple[int, float, float], float] = {}

    def is_leaf(self, max_steps: int) -> bool:
        return self.step >= int(max_steps)

    def untried_actions(self, actions: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
        return [a for a in actions if a not in self.action_visits]

    def ucb(self, action: tuple[int, float, float], c: float) -> float:
        n = int(self.action_visits.get(action, 0))
        if n <= 0:
            return float("inf")
        mean = float(self.action_values.get(action, 0.0)) / float(n)
        return mean + float(c) * math.sqrt(math.log(max(self.visits, 1)) / float(n))

    def best_ucb(self, actions: list[tuple[int, float, float]], c: float) -> tuple[int, float, float]:
        return max(actions, key=lambda action: self.ucb(action, c))

    def puct(self, action: tuple[int, float, float], prior: float, c_puct: float) -> float:
        n = int(self.action_visits.get(action, 0))
        if n > 0:
            q = float(self.action_values.get(action, 0.0)) / float(n)
        else:
            q = 0.0
        bonus = float(c_puct) * float(prior) * math.sqrt(float(max(1, self.visits))) / float(1 + n)
        return float(q + bonus)

    def best_puct(
        self,
        actions: list[tuple[int, float, float]],
        prior_map: dict[tuple[int, float, float], float],
        c_puct: float,
    ) -> tuple[int, float, float]:
        return max(actions, key=lambda a: self.puct(a, float(prior_map.get(a, 0.0)), c_puct))

    def best_exploit(self, actions: list[tuple[int, float, float]]) -> tuple[int, float, float] | None:
        best: tuple[int, float, float] | None = None
        best_v = -float("inf")
        for action in actions:
            n = int(self.action_visits.get(action, 0))
            if n <= 0:
                continue
            mean = float(self.action_values.get(action, 0.0)) / float(n)
            if mean > best_v:
                best_v = mean
                best = action
        return best


def _action_to_dict(action: tuple[int, float, float]) -> dict[str, Any]:
    return {
        "variant_idx": int(action[0]),
        "cfg": float(action[1]),
        "correction_strength": float(action[2]),
    }


def _dedup_float_list(values: list[float], ndigits: int = 6) -> list[float]:
    out: list[float] = []
    seen: set[float] = set()
    for val in values:
        v = float(round(float(val), int(ndigits)))
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _pick_anchor_cfgs(cfg_values: list[float], baseline_cfg: float, k: int) -> list[float]:
    if len(cfg_values) <= 0 or k <= 0:
        return []
    vals = sorted(float(x) for x in cfg_values)
    if len(vals) <= k:
        return _dedup_float_list(vals + [float(baseline_cfg)])
    idxs = np.linspace(0, len(vals) - 1, num=int(k), dtype=int).tolist()
    picked = [float(vals[int(i)]) for i in idxs]
    picked.append(float(baseline_cfg))
    return _dedup_float_list(picked)


def _rms_tensor(t: torch.Tensor) -> float:
    if t is None or t.numel() <= 0:
        return 0.0
    x = t.detach().float()
    return float(torch.sqrt(torch.mean(x * x)).item())


def _compute_u_t(
    u_def: str,
    parent_latents: torch.Tensor | None,
    child_latents: torch.Tensor | None,
    child_dx: torch.Tensor | None,
) -> float:
    key = str(u_def).strip().lower()
    if key == "latent_delta_rms":
        if parent_latents is None or child_latents is None:
            return 0.0
        return _rms_tensor(child_latents - parent_latents)
    if key == "latent_rms":
        return _rms_tensor(child_latents if child_latents is not None else parent_latents)
    if key == "dx_rms":
        return _rms_tensor(child_dx)
    return 0.0


def _softmax_prior(logits: np.ndarray, tau: float) -> np.ndarray:
    if logits.size <= 0:
        return np.zeros((0,), dtype=np.float64)
    t = max(1e-6, float(tau))
    shifted = (logits - float(np.max(logits))) / t
    shifted = np.clip(shifted, -50.0, 50.0)
    e = np.exp(shifted)
    s = float(np.sum(e))
    if not np.isfinite(s) or s <= 0.0:
        return np.full((logits.size,), 1.0 / float(max(1, logits.size)), dtype=np.float64)
    return (e / s).astype(np.float64)


def _zscore(x: float, values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    sd = float(arr.std())
    if sd <= 1e-8:
        return 0.0
    return float((float(x) - float(arr.mean())) / sd)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument(
        "--lookahead_mode",
        choices=[
            "standard",
            "instrumentation",
            "rollout_prior",
            "tree_prior",
            "rollout_tree_prior",
            "rollout_tree_prior_adaptive_cfg",
            "adaptive_cfg_width",
        ],
        default="rollout_prior",
    )
    extra.add_argument("--lookahead_run_ablations", action="store_true", default=False)
    extra.add_argument("--lookahead_ablation_set", nargs="+", default=["A", "B", "C", "D", "E", "F"])
    extra.add_argument("--lookahead_include_standard", action=argparse.BooleanOptionalAction, default=True)
    extra.add_argument(
        "--lookahead_u_t_def",
        choices=["latent_delta_rms", "latent_rms", "dx_rms"],
        default="latent_delta_rms",
    )
    extra.add_argument(
        "--lookahead_u_t_defs_for_f",
        nargs="+",
        default=["latent_delta_rms", "latent_rms", "dx_rms"],
    )
    extra.add_argument("--lookahead_tau", type=float, default=0.35)
    extra.add_argument("--lookahead_c_puct", type=float, default=1.20)
    extra.add_argument("--lookahead_u_ref", type=float, default=0.0, help="Optional fixed u_t normalization ref.")

    extra.add_argument("--lookahead_w_cfg", type=float, default=1.0)
    extra.add_argument("--lookahead_w_variant", type=float, default=0.25)
    extra.add_argument("--lookahead_w_cs", type=float, default=0.10)
    extra.add_argument("--lookahead_w_q", type=float, default=0.20)
    extra.add_argument("--lookahead_w_explore", type=float, default=0.05)

    extra.add_argument("--lookahead_cfg_width_min", type=int, default=3)
    extra.add_argument("--lookahead_cfg_width_max", type=int, default=7)
    extra.add_argument("--lookahead_cfg_anchor_count", type=int, default=2)
    extra.add_argument("--lookahead_min_visits_for_center", type=int, default=3)
    extra.add_argument("--lookahead_log_action_topk", type=int, default=-1)

    parsed_extra, remaining = extra.parse_known_args(argv)
    args = su.parse_args(remaining)
    for key, value in vars(parsed_extra).items():
        setattr(args, key, value)
    return args


def _mode_flags(mode: str) -> dict[str, bool]:
    m = str(mode).strip().lower()
    return {
        "use_rollout_prior": m in {"rollout_prior", "rollout_tree_prior", "rollout_tree_prior_adaptive_cfg"},
        "use_tree_prior": m in {"tree_prior", "rollout_tree_prior", "rollout_tree_prior_adaptive_cfg"},
        "adaptive_cfg_width": m in {"adaptive_cfg_width", "rollout_tree_prior_adaptive_cfg"},
        "instrumentation_only": m in {"instrumentation"},
    }


def _ablation_plan(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    wanted = {str(x).strip().upper() for x in getattr(args, "lookahead_ablation_set", [])}
    plan: list[tuple[str, dict[str, Any]]] = []
    if bool(getattr(args, "lookahead_include_standard", True)):
        plan.append(("standard", {"lookahead_mode": "standard"}))

    if "A" in wanted:
        plan.append(("A_instrumentation_only", {"lookahead_mode": "instrumentation"}))
    if "B" in wanted:
        plan.append(("B_rollout_prior_only", {"lookahead_mode": "rollout_prior"}))
    if "C" in wanted:
        plan.append(("C_tree_prior_only", {"lookahead_mode": "tree_prior"}))
    if "D" in wanted:
        plan.append(("D_rollout_plus_tree_prior", {"lookahead_mode": "rollout_tree_prior"}))
    if "E" in wanted:
        plan.append(("E_adaptive_cfg_width_only", {"lookahead_mode": "adaptive_cfg_width"}))
    if "F" in wanted:
        for u_def in [str(x).strip().lower() for x in getattr(args, "lookahead_u_t_defs_for_f", [])]:
            if u_def not in {"latent_delta_rms", "latent_rms", "dx_rms"}:
                continue
            plan.append(
                (
                    f"F_u_def_{u_def}",
                    {
                        "lookahead_mode": "rollout_tree_prior",
                        "lookahead_u_t_def": str(u_def),
                    },
                )
            )
    return plan


def run_mcts_lookahead(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    emb: su.EmbeddingContext,
    reward_model: su.UnifiedRewardScorer,
    prompt: str,
    variants: list[str],
    seed: int,
) -> su.SearchResult:
    del variants
    flags = _mode_flags(str(getattr(args, "lookahead_mode", "rollout_prior")))
    use_rollout_prior = bool(flags["use_rollout_prior"])
    use_tree_prior = bool(flags["use_tree_prior"])
    adaptive_cfg_width = bool(flags["adaptive_cfg_width"])

    family = getattr(args, "mcts_interp_family", "none")
    n_interp = int(getattr(args, "mcts_n_interp", 1))
    if family != "none":
        emb = su.expand_emb_with_interp(emb, family, n_interp)
        print(f"  mcts: interp={family} n_interp={n_interp} total_variants={len(emb.cond_text)}")

    cfg_values = _dedup_float_list([float(x) for x in getattr(args, "cfg_scales", [1.0])], ndigits=6)
    if len(cfg_values) <= 0:
        cfg_values = [float(getattr(args, "baseline_cfg", 1.0))]
    cfg_min = float(min(cfg_values))
    cfg_max = float(max(cfg_values))
    cfg_span = max(1e-6, float(cfg_max - cfg_min))

    corr_strengths = [float(x) for x in getattr(args, "correction_strengths", [0.0])]
    if len(corr_strengths) <= 0:
        corr_strengths = [0.0]
    cs_min = float(min(corr_strengths))
    cs_max = float(max(corr_strengths))
    cs_span = max(1e-6, float(cs_max - cs_min))

    prompt_actions: list[tuple[int, float]] = []
    for vi in range(len(emb.cond_text)):
        for cs in corr_strengths:
            prompt_actions.append((int(vi), float(cs)))
    if len(prompt_actions) <= 0:
        raise RuntimeError("MCTS requires at least one prompt action.")

    global_actions: list[tuple[int, float, float]] = []
    for vi, cs in prompt_actions:
        for cfg in cfg_values:
            global_actions.append((int(vi), float(cfg), float(cs)))
    if len(global_actions) <= 0:
        raise RuntimeError("MCTS requires non-empty action space.")

    root_cfg_anchors = _pick_anchor_cfgs(
        cfg_values,
        baseline_cfg=float(getattr(args, "baseline_cfg", cfg_values[0])),
        k=max(0, int(getattr(args, "lookahead_cfg_anchor_count", 2))),
    )

    latents0 = su.make_latents(ctx, seed, args.height, args.width, emb.cond_text[0].dtype)
    dx0 = torch.zeros_like(latents0)
    use_euler = getattr(args, "euler_sampler", False)
    sched = su.step_schedule(ctx.device, latents0.dtype, args.steps, getattr(args, "sigmas", None), euler=use_euler)
    _, t0_4d, _ = sched[0]
    if use_euler:
        start_latents = latents0  # Euler: start from pure noise
    else:
        start_latents = (1.0 - t0_4d) * dx0 + t0_4d * latents0  # SiD: re-noise
    root = LookaheadMCTSNode(step=0, dx=dx0, latents=start_latents, parent=None, incoming_action=None, u_t=0.0)

    u_values_seen: list[float] = []
    best_global_score = -float("inf")
    best_global_dx = None
    best_global_path: list[tuple[int, float, float]] = []
    history: list[dict[str, Any]] = []
    node_logs: list[dict[str, Any]] = []

    rng = np.random.default_rng(int(seed) + 4046)
    n_sims = max(1, int(getattr(args, "n_sims", 50)))
    ucb_c = float(getattr(args, "ucb_c", 1.41))
    c_puct = float(getattr(args, "lookahead_c_puct", 1.20))
    tau = float(getattr(args, "lookahead_tau", 0.35))
    topk = int(getattr(args, "lookahead_log_action_topk", -1))
    min_visits_for_center = max(1, int(getattr(args, "lookahead_min_visits_for_center", 3)))
    cfg_w_min = max(1, int(getattr(args, "lookahead_cfg_width_min", 3)))
    cfg_w_max = max(cfg_w_min, int(getattr(args, "lookahead_cfg_width_max", 7)))
    cfg_w_max = min(cfg_w_max, len(cfg_values))
    if cfg_w_min > len(cfg_values):
        cfg_w_min = len(cfg_values)

    def current_u_ref() -> float:
        fixed_ref = float(getattr(args, "lookahead_u_ref", 0.0))
        if fixed_ref > 0.0:
            return max(1e-6, fixed_ref)
        if len(u_values_seen) <= 0:
            return 1.0
        arr = np.asarray(u_values_seen, dtype=np.float64)
        q = float(np.percentile(arr, 75))
        if np.isfinite(q) and q > 1e-8:
            return q
        m = float(np.mean(np.abs(arr)))
        return max(1e-6, m if m > 1e-8 else 1.0)

    def visit_weighted_cfg_center(node: LookaheadMCTSNode) -> tuple[float | None, int]:
        total = 0
        wsum = 0.0
        for action, visits in node.action_visits.items():
            v = int(visits)
            if v <= 0:
                continue
            total += int(v)
            wsum += float(action[1]) * float(v)
        if total <= 0:
            return None, 0
        return float(wsum / float(total)), int(total)

    def parent_best_cfg(node: LookaheadMCTSNode) -> float | None:
        parent = node.parent
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

    def cfg_width_from_u(u_t: float) -> int:
        if not adaptive_cfg_width:
            return int(len(cfg_values))
        u_ref = current_u_ref()
        ratio = float(np.clip(float(u_t) / max(1e-6, u_ref), 0.0, 2.0))
        frac = float(np.clip(ratio / 2.0, 0.0, 1.0))
        raw = float(cfg_w_min) + (float(cfg_w_max - cfg_w_min) * frac)
        width = int(round(raw))
        width = max(cfg_w_min, min(cfg_w_max, width))
        if width % 2 == 0 and width < cfg_w_max:
            width += 1
        elif width % 2 == 0 and width > cfg_w_min:
            width -= 1
        return int(max(1, width))

    def node_candidates(node: LookaheadMCTSNode) -> tuple[dict[str, Any], list[tuple[int, float, float]]]:
        if not adaptive_cfg_width or int(node.step) <= 0 or len(cfg_values) <= 1:
            cfg_bank = [float(x) for x in cfg_values]
            center = float(np.mean(cfg_bank))
            center_source = "global_cfg_bank"
            width = int(len(cfg_bank))
        else:
            center = None
            center_source = "fallback_baseline_cfg"
            local_center, local_visits = visit_weighted_cfg_center(node)
            if local_center is not None and local_visits >= min_visits_for_center:
                center = float(local_center)
                center_source = "node_visit_weighted"

            if center is None:
                parent = node.parent
                if parent is not None:
                    parent_center, parent_visits = visit_weighted_cfg_center(parent)
                    if parent_center is not None and parent_visits >= min_visits_for_center:
                        center = float(parent_center)
                        center_source = "parent_visit_weighted"

            if center is None:
                best_cfg = parent_best_cfg(node)
                if best_cfg is not None:
                    center = float(best_cfg)
                    center_source = "parent_best_sparse"

            if center is None and node.incoming_action is not None:
                center = float(node.incoming_action[1])
                center_source = "incoming_cfg_fallback"

            if center is None:
                center = float(getattr(args, "baseline_cfg", cfg_values[0]))

            width = cfg_width_from_u(node.u_t)
            nearest = sorted(cfg_values, key=lambda c: (abs(float(c) - float(center)), float(c)))
            cfg_bank = sorted(_dedup_float_list([float(x) for x in nearest[:width]] + root_cfg_anchors))

        actions = [
            (int(vi), float(cfg), float(cs))
            for vi, cs in prompt_actions
            for cfg in cfg_bank
        ]
        meta = {
            "step_idx": int(node.step),
            "u_t": float(node.u_t),
            "cfg_bank": [float(x) for x in cfg_bank],
            "cfg_bank_width": int(width),
            "cfg_center": float(center),
            "cfg_center_source": str(center_source),
            "u_ref": float(current_u_ref()),
            "candidate_count": int(len(actions)),
            "adaptive_cfg_width": bool(adaptive_cfg_width),
        }
        return meta, actions

    def compute_action_logits(
        node: LookaheadMCTSNode,
        candidates: list[tuple[int, float, float]],
    ) -> np.ndarray:
        if len(candidates) <= 0:
            return np.zeros((0,), dtype=np.float64)

        w_cfg = float(getattr(args, "lookahead_w_cfg", 1.0))
        w_variant = float(getattr(args, "lookahead_w_variant", 0.25))
        w_cs = float(getattr(args, "lookahead_w_cs", 0.10))
        w_q = float(getattr(args, "lookahead_w_q", 0.20))
        w_explore = float(getattr(args, "lookahead_w_explore", 0.05))

        u_ratio = float(np.clip(float(node.u_t) / max(1e-6, current_u_ref()), 0.0, 2.0))
        u01 = float(np.clip(u_ratio, 0.0, 1.0))
        cfg_target = float(cfg_min + (cfg_span * u01))
        cs_target = float(cs_min + (cs_span * (1.0 - u01)))

        by_variant: dict[int, list[float]] = {}
        for action, visits in node.action_visits.items():
            v = int(visits)
            if v <= 0:
                continue
            mean = float(node.action_values.get(action, 0.0)) / float(v)
            by_variant.setdefault(int(action[0]), []).append(float(mean))
        variant_means = {k: float(np.mean(vals)) for k, vals in by_variant.items() if len(vals) > 0}
        variant_pool = [float(x) for x in variant_means.values()]

        action_q_means: dict[tuple[int, float, float], float] = {}
        for action in candidates:
            n = int(node.action_visits.get(action, 0))
            if n > 0:
                action_q_means[action] = float(node.action_values.get(action, 0.0)) / float(n)
            else:
                action_q_means[action] = 0.0
        q_pool = [float(x) for x in action_q_means.values()]

        out = np.zeros((len(candidates),), dtype=np.float64)
        for i, action in enumerate(candidates):
            vi, cfg, cs = int(action[0]), float(action[1]), float(action[2])
            cfg_score = -abs(cfg - cfg_target) / cfg_span
            cs_score = 0.0 if cs_span <= 1e-6 else (-abs(cs - cs_target) / cs_span)
            variant_score = _zscore(float(variant_means.get(vi, 0.0)), variant_pool)
            q_score = _zscore(float(action_q_means.get(action, 0.0)), q_pool)
            n = int(node.action_visits.get(action, 0))
            explore_score = 1.0 / math.sqrt(float(1 + n))
            out[i] = (
                (w_cfg * cfg_score)
                + (w_variant * variant_score)
                + (w_cs * cs_score)
                + (w_q * q_score)
                + (w_explore * explore_score)
            )
        return out

    def sample_with_prior(
        candidates: list[tuple[int, float, float]],
        prior: np.ndarray,
    ) -> tuple[int, float, float]:
        if len(candidates) <= 0:
            raise RuntimeError("Cannot sample from empty candidates.")
        if prior.size != len(candidates):
            return candidates[int(rng.integers(0, len(candidates)))]
        if not np.all(np.isfinite(prior)) or float(np.sum(prior)) <= 0.0:
            return candidates[int(rng.integers(0, len(candidates)))]
        idx = int(rng.choice(len(candidates), p=prior))
        return candidates[idx]

    def select_untried_with_optional_prior(
        node: LookaheadMCTSNode,
        candidates: list[tuple[int, float, float]],
        prior: np.ndarray,
    ) -> tuple[int, float, float]:
        untried_idx = [i for i, action in enumerate(candidates) if action not in node.action_visits]
        if len(untried_idx) <= 0:
            return candidates[int(rng.integers(0, len(candidates)))]
        if not (use_rollout_prior or use_tree_prior):
            return candidates[int(untried_idx[int(rng.integers(0, len(untried_idx)))])]
        p = np.asarray([float(prior[i]) for i in untried_idx], dtype=np.float64)
        s = float(np.sum(p))
        if (not np.isfinite(s)) or s <= 0.0:
            return candidates[int(untried_idx[int(rng.integers(0, len(untried_idx)))])]
        p = p / s
        picked_local = int(rng.choice(len(untried_idx), p=p))
        return candidates[int(untried_idx[picked_local])]

    def log_candidates(
        node: LookaheadMCTSNode,
        candidates: list[tuple[int, float, float]],
        logits: np.ndarray,
        prior: np.ndarray,
    ) -> tuple[list[dict[str, Any]], list[float], list[float]]:
        rows: list[dict[str, Any]] = []
        if len(candidates) <= 0:
            return rows, [], []
        order = list(range(len(candidates)))
        if topk > 0 and len(order) > topk:
            order = sorted(order, key=lambda i: float(prior[i]), reverse=True)[:topk]
        out_logits: list[float] = []
        out_prior: list[float] = []
        for i in order:
            action = candidates[i]
            visits = int(node.action_visits.get(action, 0))
            q_mean = float(node.action_values.get(action, 0.0)) / float(visits) if visits > 0 else 0.0
            row = _action_to_dict(action)
            row["visits"] = int(visits)
            row["q_mean"] = float(q_mean)
            rows.append(row)
            out_logits.append(float(logits[i]))
            out_prior.append(float(prior[i]))
        return rows, out_logits, out_prior

    def append_decision_log(
        logs: list[dict[str, Any]],
        *,
        sim_idx: int,
        phase: str,
        node: LookaheadMCTSNode,
        candidate_meta: dict[str, Any],
        candidates: list[tuple[int, float, float]],
        logits: np.ndarray,
        prior: np.ndarray,
        chosen_action: tuple[int, float, float],
        selection_mode: str,
        preview_reward: float | None,
    ) -> None:
        rows, part_logits, part_prior = log_candidates(node, candidates, logits, prior)
        logs.append(
            {
                "sim": int(sim_idx),
                "phase": str(phase),
                "step_idx": int(node.step),
                "u_t": float(node.u_t),
                "u_ref": float(candidate_meta.get("u_ref", current_u_ref())),
                "selection_mode": str(selection_mode),
                "candidate_count": int(candidate_meta.get("candidate_count", len(candidates))),
                "cfg_bank": [float(x) for x in candidate_meta.get("cfg_bank", [])],
                "cfg_bank_width": int(candidate_meta.get("cfg_bank_width", len(candidate_meta.get("cfg_bank", [])))),
                "cfg_center": float(candidate_meta.get("cfg_center", 0.0)),
                "cfg_center_source": str(candidate_meta.get("cfg_center_source", "unknown")),
                "adaptive_cfg_width": bool(candidate_meta.get("adaptive_cfg_width", False)),
                "candidate_actions": rows,
                "action_logits": [float(x) for x in part_logits],
                "softmax_prior": [float(x) for x in part_prior],
                "chosen_action": _action_to_dict(chosen_action),
                "preview_reward": None if preview_reward is None else float(preview_reward),
            }
        )

    # --- Key-step branching setup ---
    key_steps = su._parse_key_steps(args)
    if key_steps is None:
        key_steps = list(range(int(args.steps)))
    if 0 not in key_steps:
        key_steps = [0] + key_steps
    key_steps = sorted(set(key_steps))
    n_key = len(key_steps)
    key_segments: list[tuple[int, int]] = []
    for i in range(n_key):
        seg_start = key_steps[i]
        seg_end = key_steps[i + 1] if i + 1 < n_key else int(args.steps)
        key_segments.append((seg_start, seg_end))

    log_every = 10
    print(
        f"  mcts: sims={n_sims} actions={len(global_actions)} steps={int(args.steps)} "
        f"key_steps={key_steps} ({n_key} branch points) "
        f"lookahead_mode={args.lookahead_mode} u_t_def={args.lookahead_u_t_def}"
    )
    for sim in range(n_sims):
        node = root
        path: list[tuple[LookaheadMCTSNode, tuple[int, float, float]]] = []
        sim_logs: list[dict[str, Any]] = []

        action: tuple[int, float, float] | None = None
        candidate_meta: dict[str, Any] | None = None
        candidates: list[tuple[int, float, float]] = []
        logits = np.zeros((0,), dtype=np.float64)
        prior = np.zeros((0,), dtype=np.float64)

        while not node.is_leaf(n_key):
            candidate_meta, candidates = node_candidates(node)
            logits = compute_action_logits(node, candidates)
            prior = _softmax_prior(logits, tau=tau)
            untried = node.untried_actions(candidates)
            if len(untried) > 0:
                action = select_untried_with_optional_prior(node, candidates, prior)
                break

            if use_tree_prior:
                prior_map = {action_i: float(prior[i]) for i, action_i in enumerate(candidates)}
                action = node.best_puct(candidates, prior_map, c_puct)
                selection_mode = "puct_u_prior"
            else:
                action = node.best_ucb(candidates, ucb_c)
                selection_mode = "ucb"
            append_decision_log(
                sim_logs,
                sim_idx=sim,
                phase="tree_select",
                node=node,
                candidate_meta=candidate_meta,
                candidates=candidates,
                logits=logits,
                prior=prior,
                chosen_action=action,
                selection_mode=selection_mode,
                preview_reward=None,
            )
            path.append((node, action))
            if action in node.children:
                node = node.children[action]
            else:
                break

        if action is None:
            raise RuntimeError("Tree search failed to pick an action.")

        if not node.is_leaf(n_key):
            if action not in node.children:
                seg_start, seg_end = key_segments[node.step]
                child_lat, child_dx = su._run_segment(
                    args, ctx, emb, reward_model, prompt,
                    node.latents, node.dx, action, sched, seg_start, seg_end)
                child_u = _compute_u_t(
                    str(getattr(args, "lookahead_u_t_def", "latent_delta_rms")),
                    parent_latents=node.latents,
                    child_latents=child_lat,
                    child_dx=child_dx,
                )
                if np.isfinite(child_u) and child_u > 0.0:
                    u_values_seen.append(float(child_u))
                node.children[action] = LookaheadMCTSNode(
                    step=node.step + 1,
                    dx=child_dx,
                    latents=child_lat,
                    parent=node,
                    incoming_action=action,
                    u_t=float(child_u),
                )
            path.append((node, action))
            if candidate_meta is None:
                candidate_meta, candidates = node_candidates(node)
                logits = compute_action_logits(node, candidates)
                prior = _softmax_prior(logits, tau=tau)
            append_decision_log(
                sim_logs,
                sim_idx=sim,
                phase="tree_expand",
                node=node,
                candidate_meta=candidate_meta,
                candidates=candidates,
                logits=logits,
                prior=prior,
                chosen_action=action,
                selection_mode="expand",
                preview_reward=None,
            )
            node = node.children[action]

        rollout_dx = node.dx
        rollout_latents = node.latents
        rollout_key_idx = node.step
        rollout_node = node
        while rollout_key_idx < n_key:
            r_meta, r_candidates = node_candidates(rollout_node)
            r_logits = compute_action_logits(rollout_node, r_candidates)
            r_prior = _softmax_prior(r_logits, tau=tau)
            if use_rollout_prior:
                variant_idx, cfg, cs = sample_with_prior(r_candidates, r_prior)
                roll_mode = "rollout_prior"
            else:
                variant_idx, cfg, cs = r_candidates[int(rng.integers(0, len(r_candidates)))]
                roll_mode = "rollout_uniform"

            append_decision_log(
                sim_logs,
                sim_idx=sim,
                phase="rollout",
                node=rollout_node,
                candidate_meta=r_meta,
                candidates=r_candidates,
                logits=r_logits,
                prior=r_prior,
                chosen_action=(variant_idx, cfg, cs),
                selection_mode=roll_mode,
                preview_reward=None,
            )

            seg_start, seg_end = key_segments[rollout_key_idx]
            rollout_action = (int(variant_idx), float(cfg), float(cs))
            next_latents, rollout_dx = su._run_segment(
                args, ctx, emb, reward_model, prompt,
                rollout_latents, rollout_dx, rollout_action, sched, seg_start, seg_end)
            child_u = _compute_u_t(
                str(getattr(args, "lookahead_u_t_def", "latent_delta_rms")),
                parent_latents=rollout_node.latents,
                child_latents=next_latents,
                child_dx=rollout_dx,
            )
            if np.isfinite(child_u) and child_u > 0.0:
                u_values_seen.append(float(child_u))
            rollout_key_idx += 1
            if rollout_key_idx < n_key:
                rollout_node = LookaheadMCTSNode(
                    step=rollout_key_idx,
                    dx=rollout_dx,
                    latents=next_latents,
                    parent=rollout_node,
                    incoming_action=rollout_action,
                    u_t=float(child_u),
                )
            rollout_latents = next_latents

        rollout_final = su._final_decode_tensor(rollout_latents, rollout_dx, use_euler)
        rollout_img = su.decode_to_pil(ctx, rollout_final)
        rollout_score = float(su.score_image(reward_model, prompt, rollout_img))
        if rollout_score > best_global_score:
            best_global_score = float(rollout_score)
            best_global_dx = rollout_final.clone()
            best_global_path = [a for _, a in path]

        for pnode, paction in path:
            pnode.visits += 1
            pnode.action_visits[paction] = int(pnode.action_visits.get(paction, 0) + 1)
            pnode.action_values[paction] = float(pnode.action_values.get(paction, 0.0) + float(rollout_score))

        for row in sim_logs:
            row["final_reward"] = float(rollout_score)
        node_logs.extend(sim_logs)

        if (sim + 1) % log_every == 0 or sim == 0:
            root_meta, root_candidates = node_candidates(root)
            root_logits = compute_action_logits(root, root_candidates)
            root_prior = _softmax_prior(root_logits, tau=tau)
            order = sorted(range(len(root_candidates)), key=lambda i: float(root_prior[i]), reverse=True)[:8]
            root_top = []
            for i in order:
                action = root_candidates[i]
                n = int(root.action_visits.get(action, 0))
                q_mean = float(root.action_values.get(action, 0.0)) / float(n) if n > 0 else 0.0
                root_top.append(
                    {
                        **_action_to_dict(action),
                        "prior": float(root_prior[i]),
                        "visits": int(n),
                        "q_mean": float(q_mean),
                    }
                )
            history.append(
                {
                    "sim": int(sim + 1),
                    "best_score": float(best_global_score),
                    "root_visits": int(root.visits),
                    "root_u_t": float(root.u_t),
                    "root_candidate_count": int(len(root_candidates)),
                    "root_cfg_bank": [float(x) for x in root_meta.get("cfg_bank", [])],
                    "root_top_actions": root_top,
                }
            )
            print(f"    sim {sim + 1:3d}/{n_sims} best={best_global_score:.4f}")

    exploit_path: list[tuple[int, float, float]] = []
    node = root
    for _ in range(n_key):
        _, candidates = node_candidates(node)
        action = node.best_exploit(candidates)
        if action is None:
            break
        exploit_path.append(action)
        if action in node.children:
            node = node.children[action]
        else:
            break

    replay_lat = start_latents
    replay_dx = dx0
    for key_idx, exploit_action in enumerate(exploit_path):
        seg_start, seg_end = key_segments[key_idx]
        replay_lat, replay_dx = su._run_segment(
            args, ctx, emb, reward_model, prompt,
            replay_lat, replay_dx, exploit_action, sched, seg_start, seg_end)
    # Fill remaining key steps with fallback
    for key_idx in range(len(exploit_path), n_key):
        fallback = global_actions[0]
        seg_start, seg_end = key_segments[key_idx]
        replay_lat, replay_dx = su._run_segment(
            args, ctx, emb, reward_model, prompt,
            replay_lat, replay_dx, fallback, sched, seg_start, seg_end)

    exploit_img = su.decode_to_pil(ctx, su._final_decode_tensor(replay_lat, replay_dx, use_euler))
    exploit_score = float(su.score_image(reward_model, prompt, exploit_img))

    out_img = exploit_img
    out_score = float(exploit_score)
    out_actions = exploit_path
    if exploit_score < best_global_score and best_global_dx is not None:
        out_img = su.decode_to_pil(ctx, best_global_dx)
        out_score = float(best_global_score)
        out_actions = list(best_global_path)

    u_arr = np.asarray(u_values_seen, dtype=np.float64) if len(u_values_seen) > 0 else np.asarray([], dtype=np.float64)
    diagnostics = {
        "lookahead_mode": str(getattr(args, "lookahead_mode", "rollout_prior")),
        "key_steps": [int(x) for x in key_steps],
        "n_key": int(n_key),
        "lookahead_flags": {
            "use_rollout_prior": bool(use_rollout_prior),
            "use_tree_prior": bool(use_tree_prior),
            "adaptive_cfg_width": bool(adaptive_cfg_width),
        },
        "u_t_def": str(getattr(args, "lookahead_u_t_def", "latent_delta_rms")),
        "u_t_stats": {
            "count": int(u_arr.size),
            "mean": float(u_arr.mean()) if u_arr.size > 0 else 0.0,
            "std": float(u_arr.std()) if u_arr.size > 0 else 0.0,
            "min": float(u_arr.min()) if u_arr.size > 0 else 0.0,
            "max": float(u_arr.max()) if u_arr.size > 0 else 0.0,
            "u_ref_final": float(current_u_ref()),
        },
        "cfg_range": [float(cfg_min), float(cfg_max)],
        "cfg_values": [float(x) for x in cfg_values],
        "cfg_root_anchors": [float(x) for x in root_cfg_anchors],
        "history": history,
        "lookahead_node_logs": node_logs,
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


def _run_single(args: argparse.Namespace) -> None:
    mode = str(getattr(args, "lookahead_mode", "rollout_prior")).strip().lower()
    if mode == "standard":
        su.run(args)
        return
    original_run_mcts = su.run_mcts
    su.run_mcts = run_mcts_lookahead
    try:
        su.run(args)
    finally:
        su.run_mcts = original_run_mcts


def _collect_mode_stats(summary_path: str) -> dict[str, Any]:
    if not os.path.exists(summary_path):
        return {
            "count": 0,
            "mean_baseline": 0.0,
            "mean_mcts": 0.0,
            "mean_delta": 0.0,
            "best_mcts": 0.0,
        }
    with open(summary_path, encoding="utf-8") as f:
        rows = json.load(f)
    baseline = [float(r["baseline_IR"]) for r in rows if "baseline_IR" in r]
    mcts = [float(r["mcts_IR"]) for r in rows if "mcts_IR" in r]
    delta = [float(r["delta_IR"]) for r in rows if "delta_IR" in r]
    return {
        "count": int(len(mcts)),
        "mean_baseline": float(np.mean(baseline)) if baseline else 0.0,
        "mean_mcts": float(np.mean(mcts)) if mcts else 0.0,
        "mean_delta": float(np.mean(delta)) if delta else 0.0,
        "best_mcts": float(np.max(mcts)) if mcts else 0.0,
    }


def _run_ablation_suite(args: argparse.Namespace) -> None:
    if str(getattr(args, "search_method", "mcts")).lower() != "mcts":
        print("[lookahead] forcing --search_method mcts for ablation suite")
    base_out = args.out_dir
    os.makedirs(base_out, exist_ok=True)
    plan = _ablation_plan(args)
    if len(plan) <= 0:
        raise RuntimeError("No ablation modes selected.")

    rows: list[dict[str, Any]] = []
    for mode_name, override in plan:
        run_args = argparse.Namespace(**vars(copy.deepcopy(args)))
        run_args.search_method = "mcts"
        run_args.out_dir = os.path.join(base_out, mode_name)
        os.makedirs(run_args.out_dir, exist_ok=True)
        for key, value in override.items():
            setattr(run_args, key, value)
        print(
            f"[lookahead-ablation] mode={mode_name} "
            f"lookahead_mode={run_args.lookahead_mode} u_t_def={run_args.lookahead_u_t_def} "
            f"out_dir={run_args.out_dir}"
        )
        _run_single(run_args)
        stats = _collect_mode_stats(os.path.join(run_args.out_dir, "summary.json"))
        rows.append(
            {
                "mode": str(mode_name),
                "lookahead_mode": str(getattr(run_args, "lookahead_mode", "standard")),
                "u_t_def": str(getattr(run_args, "lookahead_u_t_def", "latent_delta_rms")),
                **stats,
            }
        )

    rows.sort(key=lambda r: float(r["mean_mcts"]), reverse=True)
    tsv_path = os.path.join(base_out, "lookahead_reweighting_ablation_summary.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("mode\tlookahead_mode\tu_t_def\tcount\tmean_baseline\tmean_mcts\tmean_delta\tbest_mcts\n")
        for row in rows:
            f.write(
                f"{row['mode']}\t{row['lookahead_mode']}\t{row['u_t_def']}\t{int(row['count'])}\t"
                f"{float(row['mean_baseline']):.6f}\t{float(row['mean_mcts']):.6f}\t"
                f"{float(row['mean_delta']):+.6f}\t{float(row['best_mcts']):.6f}\n"
            )
    json_path = os.path.join(base_out, "lookahead_reweighting_ablation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print("\nLookahead Ablation Summary")
    print(f"{'mode':<30} {'mean_mcts':>10} {'mean_delta':>10} {'n':>5}")
    print("-" * 62)
    for row in rows:
        print(
            f"{row['mode']:<30} {float(row['mean_mcts']):>10.4f} "
            f"{float(row['mean_delta']):>+10.4f} {int(row['count']):>5d}"
        )
    print("-" * 62)
    print(f"TSV:  {tsv_path}")
    print(f"JSON: {json_path}")


def run(args: argparse.Namespace) -> None:
    if bool(getattr(args, "lookahead_run_ablations", False)):
        _run_ablation_suite(args)
    else:
        _run_single(args)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run(su.normalize_paths(args))


if __name__ == "__main__":
    main()
