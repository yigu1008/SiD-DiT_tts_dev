"""
Per-step adaptive CFG search via decoded-x0 reward scoring.

At each (or every k-th) denoising step:
  1. The caller supplies (flow_u, flow_c) from a single transformer call.
  2. We recombine: flow_w = flow_u + w * (flow_c - flow_u) for each w in
     a candidate CFG grid (no extra forward passes).
  3. Each flow_w is converted to x0_pred, decoded by the VAE, and scored
     by enabled evaluators (currently imagereward + hpsv3).
  4. Per-evaluator scores are z-normalized across candidates, weighted by
     a timestep-progress-dependent schedule, optionally gated by score
     dispersion (confidence), then combined with smoothness penalties.
  5. The argmax CFG is returned and the caller runs one scheduler step.

This module is deliberately pure: it imports torch + numpy but no diffusion
samplers and no reward models. All model interaction is through callables
passed by the caller, so the same algorithm works for SD3.5 and FLUX.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from PIL import Image


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class DynamicCfgX0Config:
    """User-facing knobs. Defaults match the spec's recommended starting point."""

    enabled: bool = False
    cfg_grid: list[float] = field(default_factory=lambda: [2.0, 3.5, 5.0, 6.5, 8.0])
    score_start_frac: float = 0.25
    score_end_frac: float = 1.0
    score_every: int = 2
    evaluators: list[str] = field(default_factory=lambda: ["imagereward", "hpsv3"])
    weight_schedule: str = "piecewise"  # "piecewise" | "smooth"
    prompt_type: str = "general"        # "general" | "text" | "counting"
    confidence_gating: bool = True
    cfg_smooth_weight: float = 0.05
    high_cfg_penalty: float = 0.02
    cfg_soft_max: float = 7.5
    cfg_min: float = 0.0
    cfg_max: float = 12.0
    decode_x0_pred: bool = True
    reward_batch_size: int = 8
    log_dynamic_cfg: bool = True
    log_path: str | None = None
    add_local_neighborhood: bool = False
    neighborhood_deltas: list[float] = field(default_factory=lambda: [-1.5, -0.75, 0.75, 1.5])

    @classmethod
    def from_args(cls, args: Any) -> "DynamicCfgX0Config":
        """Build config from argparse Namespace. Only fields with attrs override defaults."""
        cfg = cls()
        # Direct overrides
        for name in (
            "enabled", "cfg_grid", "score_start_frac", "score_end_frac", "score_every",
            "evaluators", "weight_schedule", "prompt_type", "confidence_gating",
            "cfg_smooth_weight", "high_cfg_penalty", "cfg_soft_max", "cfg_min", "cfg_max",
            "decode_x0_pred", "reward_batch_size", "log_dynamic_cfg", "log_path",
            "add_local_neighborhood", "neighborhood_deltas",
        ):
            cli = f"dynamic_cfg_x0_{name}" if name != "enabled" else "dynamic_cfg_x0"
            if hasattr(args, cli):
                val = getattr(args, cli)
                if val is not None:
                    setattr(cfg, name, val)
        return cfg


# ── Progress and scheduling ─────────────────────────────────────────────────


def progress_from_sigma(sigma_i: float, sigma_min: float, sigma_max: float) -> float:
    """p in [0,1]: 0 at sigma_max (pure noise), 1 at sigma_min (clean)."""
    span = float(sigma_max) - float(sigma_min)
    if span <= 1e-8:
        return 0.0
    return float(np.clip(1.0 - (float(sigma_i) - float(sigma_min)) / span, 0.0, 1.0))


def progress_from_step(step_idx: int, num_steps: int) -> float:
    """Fallback: linear in step index."""
    if num_steps <= 1:
        return 0.0
    return float(step_idx) / float(num_steps - 1)


def should_score_step(
    step_idx: int,
    progress: float,
    cfg: DynamicCfgX0Config,
) -> bool:
    if not cfg.enabled:
        return False
    if progress < cfg.score_start_frac - 1e-8:
        return False
    if progress > cfg.score_end_frac + 1e-8:
        return False
    every = max(1, int(cfg.score_every))
    return (int(step_idx) % every) == 0


# ── Evaluator weight schedules ──────────────────────────────────────────────


# Piecewise weights from the spec. Keys = prompt_type. Each entry maps
# (low_p_band, mid_p_band, high_p_band) -> dict[evaluator -> weight].
# Spec used 3-evaluator schedule (clip / quality / special). For Pass 1
# we expose only imagereward (text-aligned, quality-leaning) and hpsv3
# (preference, quality-leaning) — the early band biases imagereward (more
# CLIP-text-aware), late band biases hpsv3 (preference).
_PIECEWISE_BANDS: dict[str, list[tuple[float, dict[str, float]]]] = {
    "general": [
        (0.30, {"imagereward": 0.70, "hpsv3": 0.30}),
        (0.70, {"imagereward": 0.50, "hpsv3": 0.50}),
        (1.01, {"imagereward": 0.30, "hpsv3": 0.70}),
    ],
    "text": [
        (0.30, {"imagereward": 0.65, "hpsv3": 0.35}),
        (0.70, {"imagereward": 0.55, "hpsv3": 0.45}),
        (1.01, {"imagereward": 0.40, "hpsv3": 0.60}),
    ],
    "counting": [
        (0.30, {"imagereward": 0.65, "hpsv3": 0.35}),
        (0.70, {"imagereward": 0.55, "hpsv3": 0.45}),
        (1.01, {"imagereward": 0.45, "hpsv3": 0.55}),
    ],
}


def _piecewise_weights(progress: float, prompt_type: str) -> dict[str, float]:
    bands = _PIECEWISE_BANDS.get(prompt_type, _PIECEWISE_BANDS["general"])
    for upper, w in bands:
        if progress < upper:
            return dict(w)
    return dict(bands[-1][1])


def _smooth_weights(progress: float, prompt_type: str) -> dict[str, float]:
    """Linear interp between band centers (0.15 / 0.50 / 0.85). Same prompt_type families."""
    p = float(np.clip(progress, 0.0, 1.0))
    bands = _PIECEWISE_BANDS.get(prompt_type, _PIECEWISE_BANDS["general"])
    centers = [0.15, 0.50, 0.85]
    keys = sorted({k for _, w in bands for k in w.keys()})
    if p <= centers[0]:
        return dict(bands[0][1])
    if p >= centers[-1]:
        return dict(bands[-1][1])
    for i in range(len(centers) - 1):
        if centers[i] <= p <= centers[i + 1]:
            t = (p - centers[i]) / max(1e-8, (centers[i + 1] - centers[i]))
            out = {}
            for k in keys:
                a = float(bands[i][1].get(k, 0.0))
                b = float(bands[i + 1][1].get(k, 0.0))
                out[k] = (1.0 - t) * a + t * b
            return out
    return dict(bands[-1][1])


def evaluator_weights(progress: float, cfg: DynamicCfgX0Config) -> dict[str, float]:
    """Look up base weights and renormalize over cfg.evaluators (drops disabled)."""
    if cfg.weight_schedule == "smooth":
        raw = _smooth_weights(progress, cfg.prompt_type)
    else:
        raw = _piecewise_weights(progress, cfg.prompt_type)
    enabled = [e for e in cfg.evaluators if e in raw]
    if not enabled:
        return {}
    sub = {k: float(raw[k]) for k in enabled}
    s = sum(sub.values())
    if s <= 1e-8:
        eq = 1.0 / len(enabled)
        return {k: eq for k in enabled}
    return {k: v / s for k, v in sub.items()}


# ── Candidate generation ────────────────────────────────────────────────────


def generate_cfg_candidates(
    cfg: DynamicCfgX0Config,
    w_prev: float | None,
) -> list[float]:
    cands: list[float] = list(cfg.cfg_grid)
    if cfg.add_local_neighborhood and w_prev is not None:
        for d in cfg.neighborhood_deltas:
            cands.append(float(w_prev) + float(d))
    clipped = [float(np.clip(c, cfg.cfg_min, cfg.cfg_max)) for c in cands]
    seen: set[float] = set()
    out: list[float] = []
    for c in sorted(clipped):
        key = round(float(c), 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(c))
    return out


# ── x0-pred helper (flow matching) ──────────────────────────────────────────


def x0_pred_from_flow(
    latents: torch.Tensor,
    sigma_4d: torch.Tensor,
    flow: torch.Tensor,
    x0_sampler: bool = False,
) -> torch.Tensor:
    """For flow models: x0 = latents - sigma * flow.
    For x0-prediction models (SenseFlow x0_sampler): flow already IS x0.
    """
    if x0_sampler:
        return flow
    return latents - sigma_4d * flow


# ── Score normalization ─────────────────────────────────────────────────────


def zscore_normalize(
    raw_scores: dict[float, float],
    eps: float = 1e-6,
    min_std: float = 1e-3,
) -> dict[float, float]:
    """Per-evaluator z-score across CFG candidates. Returns 0s if std too small."""
    if not raw_scores:
        return {}
    vals = np.array([float(raw_scores[k]) for k in raw_scores], dtype=np.float64)
    mu = float(vals.mean())
    sd = float(vals.std())
    if sd < min_std:
        return {k: 0.0 for k in raw_scores}
    return {k: float((float(raw_scores[k]) - mu) / (sd + eps)) for k in raw_scores}


def confidence_gate(
    raw_per_eval: dict[str, dict[float, float]],
    base_weights: dict[str, float],
    min_dispersion: float = 1e-3,
) -> dict[str, float]:
    """Multiply each evaluator's base weight by its score dispersion (std), then renormalize.
    If all evaluators are non-dispersive, fall back to base."""
    if not base_weights:
        return {}
    confs: dict[str, float] = {}
    for ev in base_weights.keys():
        rows = raw_per_eval.get(ev, {})
        if not rows:
            confs[ev] = 0.0
            continue
        vals = np.array([float(v) for v in rows.values()], dtype=np.float64)
        confs[ev] = float(vals.std())
    weighted = {ev: float(base_weights[ev]) * float(confs.get(ev, 0.0)) for ev in base_weights}
    s = sum(weighted.values())
    if s < min_dispersion:
        return dict(base_weights)
    return {ev: v / s for ev, v in weighted.items()}


# ── Score combination ───────────────────────────────────────────────────────


def combine_scores(
    norm_per_eval: dict[str, dict[float, float]],
    weights: dict[str, float],
    candidates: Sequence[float],
    w_prev: float | None,
    cfg: DynamicCfgX0Config,
) -> dict[float, float]:
    out: dict[float, float] = {}
    for w in candidates:
        s = 0.0
        for ev, wgt in weights.items():
            s += float(wgt) * float(norm_per_eval.get(ev, {}).get(w, 0.0))
        if w_prev is not None:
            s -= float(cfg.cfg_smooth_weight) * abs(float(w) - float(w_prev))
        s -= float(cfg.high_cfg_penalty) * max(float(w) - float(cfg.cfg_soft_max), 0.0)
        out[float(w)] = float(s)
    return out


def select_best_cfg(total_scores: dict[float, float]) -> float:
    return max(total_scores.items(), key=lambda kv: kv[1])[0]


# ── Candidate scoring ───────────────────────────────────────────────────────


def score_candidates(
    candidates: Sequence[float],
    flow_u: torch.Tensor,
    flow_c: torch.Tensor,
    latents: torch.Tensor,
    sigma_4d: torch.Tensor,
    x0_sampler: bool,
    decode_fn: Callable[[torch.Tensor], Image.Image],
    eval_fn: Callable[[str, str, Image.Image], float],   # (evaluator, prompt, image) -> float
    evaluators: Sequence[str],
    prompt: str,
) -> tuple[dict[str, dict[float, float]], dict[float, torch.Tensor]]:
    """Recombine → x0 → decode → score for each w. Returns (raw_per_eval, x0_per_w)."""
    raw_per_eval: dict[str, dict[float, float]] = {ev: {} for ev in evaluators}
    x0_per_w: dict[float, torch.Tensor] = {}
    for w in candidates:
        flow_w = flow_u + float(w) * (flow_c - flow_u)
        x0 = x0_pred_from_flow(latents, sigma_4d, flow_w, x0_sampler=x0_sampler)
        x0_per_w[float(w)] = x0
        img = decode_fn(x0)
        for ev in evaluators:
            try:
                s = float(eval_fn(ev, prompt, img))
            except Exception as exc:
                print(f"[dyncfg-x0] WARN: eval={ev} cfg={w:.3f} failed: {exc}")
                s = 0.0
            raw_per_eval[ev][float(w)] = float(s)
    return raw_per_eval, x0_per_w


# ── Per-candidate scoring (FLUX guidance-distillation path) ─────────────────


def score_candidates_per_call(
    candidates: Sequence[float],
    flow_for_w_fn: Callable[[float], torch.Tensor],
    latents: torch.Tensor,
    sigma_4d: torch.Tensor,
    x0_sampler: bool,
    decode_fn: Callable[[torch.Tensor], Image.Image],
    eval_fn: Callable[[str, str, Image.Image], float],
    evaluators: Sequence[str],
    prompt: str,
) -> tuple[dict[str, dict[float, float]], dict[float, torch.Tensor]]:
    """Per-candidate scoring for samplers without an analytic CFG split.

    For each w in ``candidates`` we call ``flow_for_w_fn(w)`` (one transformer
    forward per candidate), convert to x0_pred, decode and score. Used by FLUX
    backends where ``guidance_scale`` enters the network as conditioning rather
    than as a linear interpolation between (uncond, cond) outputs.
    """
    raw_per_eval: dict[str, dict[float, float]] = {ev: {} for ev in evaluators}
    x0_per_w: dict[float, torch.Tensor] = {}
    for w in candidates:
        flow_w = flow_for_w_fn(float(w))
        x0 = x0_pred_from_flow(latents, sigma_4d, flow_w, x0_sampler=x0_sampler)
        x0_per_w[float(w)] = x0
        img = decode_fn(x0)
        for ev in evaluators:
            try:
                s = float(eval_fn(ev, prompt, img))
            except Exception as exc:
                print(f"[dyncfg-x0] WARN: eval={ev} cfg={w:.3f} failed: {exc}")
                s = 0.0
            raw_per_eval[ev][float(w)] = float(s)
    return raw_per_eval, x0_per_w


def select_cfg_for_step_per_call(
    *,
    candidates: Sequence[float],
    flow_for_w_fn: Callable[[float], torch.Tensor],
    latents: torch.Tensor,
    sigma_4d: torch.Tensor,
    x0_sampler: bool,
    decode_fn: Callable[[torch.Tensor], Image.Image],
    eval_fn: Callable[[str, str, Image.Image], float],
    prompt: str,
    progress: float,
    w_prev: float | None,
    cfg: DynamicCfgX0Config,
) -> dict[str, Any]:
    """End-to-end search using per-candidate forward calls. Mirrors
    ``select_cfg_for_step`` but feeds each candidate through ``flow_for_w_fn``
    instead of recombining a single (flow_u, flow_c) split.
    """
    raw_per_eval, x0_per_w = score_candidates_per_call(
        candidates=candidates,
        flow_for_w_fn=flow_for_w_fn,
        latents=latents,
        sigma_4d=sigma_4d,
        x0_sampler=x0_sampler,
        decode_fn=decode_fn,
        eval_fn=eval_fn,
        evaluators=cfg.evaluators,
        prompt=prompt,
    )
    norm_per_eval: dict[str, dict[float, float]] = {
        ev: zscore_normalize(rows) for ev, rows in raw_per_eval.items()
    }
    base_w = evaluator_weights(progress, cfg)
    if cfg.confidence_gating:
        weights = confidence_gate(raw_per_eval, base_w)
    else:
        weights = dict(base_w)
    totals = combine_scores(norm_per_eval, weights, candidates, w_prev, cfg)
    best = select_best_cfg(totals)
    return {
        "chosen_cfg": float(best),
        "raw_scores": {ev: {float(k): float(v) for k, v in rows.items()} for ev, rows in raw_per_eval.items()},
        "norm_scores": {ev: {float(k): float(v) for k, v in rows.items()} for ev, rows in norm_per_eval.items()},
        "total_scores": {float(k): float(v) for k, v in totals.items()},
        "weights": {ev: float(v) for ev, v in weights.items()},
        "base_weights": {ev: float(v) for ev, v in base_w.items()},
        "candidates": [float(x) for x in candidates],
        "progress": float(progress),
        "x0_chosen": x0_per_w[float(best)],
    }


# ── One-shot select (SD3.5 split-recombine path) ────────────────────────────


def select_cfg_for_step(
    *,
    candidates: Sequence[float],
    flow_u: torch.Tensor,
    flow_c: torch.Tensor,
    latents: torch.Tensor,
    sigma_4d: torch.Tensor,
    x0_sampler: bool,
    decode_fn: Callable[[torch.Tensor], Image.Image],
    eval_fn: Callable[[str, str, Image.Image], float],
    prompt: str,
    progress: float,
    w_prev: float | None,
    cfg: DynamicCfgX0Config,
) -> dict[str, Any]:
    """End-to-end search for the best CFG at one step.

    Returns a dict with: chosen_cfg, raw_scores, norm_scores, total_scores,
    weights, candidates, progress (for logging) and x0_chosen tensor.
    """
    raw_per_eval, x0_per_w = score_candidates(
        candidates=candidates,
        flow_u=flow_u,
        flow_c=flow_c,
        latents=latents,
        sigma_4d=sigma_4d,
        x0_sampler=x0_sampler,
        decode_fn=decode_fn,
        eval_fn=eval_fn,
        evaluators=cfg.evaluators,
        prompt=prompt,
    )
    norm_per_eval: dict[str, dict[float, float]] = {
        ev: zscore_normalize(rows) for ev, rows in raw_per_eval.items()
    }
    base_w = evaluator_weights(progress, cfg)
    if cfg.confidence_gating:
        weights = confidence_gate(raw_per_eval, base_w)
    else:
        weights = dict(base_w)
    totals = combine_scores(norm_per_eval, weights, candidates, w_prev, cfg)
    best = select_best_cfg(totals)
    return {
        "chosen_cfg": float(best),
        "raw_scores": {ev: {float(k): float(v) for k, v in rows.items()} for ev, rows in raw_per_eval.items()},
        "norm_scores": {ev: {float(k): float(v) for k, v in rows.items()} for ev, rows in norm_per_eval.items()},
        "total_scores": {float(k): float(v) for k, v in totals.items()},
        "weights": {ev: float(v) for ev, v in weights.items()},
        "base_weights": {ev: float(v) for ev, v in base_w.items()},
        "candidates": [float(x) for x in candidates],
        "progress": float(progress),
        "x0_chosen": x0_per_w[float(best)],
    }


# ── Logging ─────────────────────────────────────────────────────────────────


class DynamicCfgLogger:
    """Append-only JSONL logger; opened lazily on first write."""

    def __init__(self, path: str | None) -> None:
        self.path = str(path) if path else None
        self._f: Any = None

    def _open(self) -> None:
        if self._f is None and self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._f = open(self.path, "a", encoding="utf-8")

    def log(self, row: dict[str, Any]) -> None:
        if not self.path:
            return
        self._open()
        if self._f is None:
            return
        self._f.write(json.dumps(row, default=float) + "\n")
        self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            try:
                self._f.close()
            finally:
                self._f = None


# ── CLI helpers ─────────────────────────────────────────────────────────────


def add_dynamic_cfg_x0_args(parser: Any) -> None:
    """Attach --dynamic_cfg_x0_* flags to an argparse parser. Idempotent — the
    caller is responsible for not double-registering when used as a mixin."""
    parser.add_argument("--dynamic_cfg_x0", action="store_true",
                        help="Enable per-step adaptive CFG search via decoded-x0 reward scoring.")
    parser.add_argument("--dynamic_cfg_x0_cfg_grid", nargs="+", type=float,
                        default=[2.0, 3.5, 5.0, 6.5, 8.0])
    parser.add_argument("--dynamic_cfg_x0_score_start_frac", type=float, default=0.25)
    parser.add_argument("--dynamic_cfg_x0_score_end_frac", type=float, default=1.0)
    parser.add_argument("--dynamic_cfg_x0_score_every", type=int, default=2)
    parser.add_argument("--dynamic_cfg_x0_evaluators", nargs="+", type=str,
                        default=["imagereward", "hpsv3"])
    parser.add_argument("--dynamic_cfg_x0_weight_schedule", choices=["piecewise", "smooth"],
                        default="piecewise")
    parser.add_argument("--dynamic_cfg_x0_prompt_type", choices=["general", "text", "counting"],
                        default="general")
    parser.add_argument("--dynamic_cfg_x0_confidence_gating", action="store_true", default=True)
    parser.add_argument("--dynamic_cfg_x0_no_confidence_gating", dest="dynamic_cfg_x0_confidence_gating",
                        action="store_false")
    parser.add_argument("--dynamic_cfg_x0_cfg_smooth_weight", type=float, default=0.05)
    parser.add_argument("--dynamic_cfg_x0_high_cfg_penalty", type=float, default=0.02)
    parser.add_argument("--dynamic_cfg_x0_cfg_soft_max", type=float, default=7.5)
    parser.add_argument("--dynamic_cfg_x0_cfg_min", type=float, default=0.0)
    parser.add_argument("--dynamic_cfg_x0_cfg_max", type=float, default=12.0)
    parser.add_argument("--dynamic_cfg_x0_log_dynamic_cfg", action="store_true", default=True)
    parser.add_argument("--dynamic_cfg_x0_log_path", type=str, default=None)
    parser.add_argument("--dynamic_cfg_x0_add_local_neighborhood", action="store_true", default=False)
