# Dynamic-CFG-x0 Code Review Bundle

Per-step adaptive classifier-free guidance via decoded-x0 reward scoring.
At each (or every k-th) denoising step we recombine `(flow_u, flow_c)` (SD3.5)
or invoke the transformer per-candidate (FLUX guidance distillation), decode
to image, score with reward models, and pick the argmax CFG.

## Manifest

| File | LOC | Role |
|------|-----|------|
| `dynamic_cfg_x0.py` | 540 | Pure algorithm module — no diffusion, no rewards |
| `sampling_unified_sd35_dynamic_cfg_x0.py` | 189 | SD3.5 wrapper (`sid` + `sd35_base`) — uses split-recombine |
| `sd35_ddp_experiment_dynamic_cfg_x0.py` | 58 | SD3.5 DDP entry-point (monkey-patches `run_baseline`) |
| `sampling_flux_unified_dynamic_cfg_x0.py` | 293 | FLUX wrapper (`flux` + `tdd_flux`) — uses per-candidate forward |
| `test_dynamic_cfg_x0.py` | 309 | Unit + smoke tests (19 tests, no GPU required) |
| `amlt/sd35_dynamic_cfg_x0_server.yaml` | 276 | AMLT job: 4 trials = `{sid, sd35_base} × {hpsv3, imagereward}` |
| `amlt/flux_dynamic_cfg_x0_server.yaml` | 279 | AMLT job: 4 trials = `{flux, tdd_flux} × {hpsv3, imagereward}` |

Plus integration into existing suite scripts:
- `hpsv2_sd35_sid_ddp_suite.sh` — env defaults (lines 265–278), method case (line 932), arg block (lines 1063–1085)
- `hpsv2_flux_schnell_ddp_suite.sh` — method case (lines 922–948)

---

## 1. Core algorithm — `dynamic_cfg_x0.py`

Pure Python module: argparse mixin, dataclass config, scheduling, evaluator
weight bands, z-score normalization, confidence gating, score combination,
two scoring entry-points (split-recombine and per-call).

```python
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
    if num_steps <= 1:
        return 0.0
    return float(step_idx) / float(num_steps - 1)


def should_score_step(step_idx: int, progress: float, cfg: DynamicCfgX0Config) -> bool:
    if not cfg.enabled:
        return False
    if progress < cfg.score_start_frac - 1e-8:
        return False
    if progress > cfg.score_end_frac + 1e-8:
        return False
    every = max(1, int(cfg.score_every))
    return (int(step_idx) % every) == 0


# ── Evaluator weight schedules ──────────────────────────────────────────────


# Piecewise weights: prompt_type -> [(upper_progress_threshold, {evaluator: w}), ...]
# Early band biases imagereward (CLIP-text-aware), late band biases hpsv3 (preference).
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
    """Linear interp between band centers (0.15 / 0.50 / 0.85)."""
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


def generate_cfg_candidates(cfg: DynamicCfgX0Config, w_prev: float | None) -> list[float]:
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
    eval_fn: Callable[[str, str, Image.Image], float],
    evaluators: Sequence[str],
    prompt: str,
) -> tuple[dict[str, dict[float, float]], dict[float, torch.Tensor]]:
    """SD3.5 path — single transformer call, recombine flow_w per candidate."""
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
    """FLUX path — one transformer forward per candidate (guidance is conditioning)."""
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
    raw_per_eval, x0_per_w = score_candidates_per_call(
        candidates=candidates, flow_for_w_fn=flow_for_w_fn,
        latents=latents, sigma_4d=sigma_4d, x0_sampler=x0_sampler,
        decode_fn=decode_fn, eval_fn=eval_fn, evaluators=cfg.evaluators, prompt=prompt,
    )
    norm_per_eval = {ev: zscore_normalize(rows) for ev, rows in raw_per_eval.items()}
    base_w = evaluator_weights(progress, cfg)
    weights = confidence_gate(raw_per_eval, base_w) if cfg.confidence_gating else dict(base_w)
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
    raw_per_eval, x0_per_w = score_candidates(
        candidates=candidates, flow_u=flow_u, flow_c=flow_c,
        latents=latents, sigma_4d=sigma_4d, x0_sampler=x0_sampler,
        decode_fn=decode_fn, eval_fn=eval_fn, evaluators=cfg.evaluators, prompt=prompt,
    )
    norm_per_eval = {ev: zscore_normalize(rows) for ev, rows in raw_per_eval.items()}
    base_w = evaluator_weights(progress, cfg)
    weights = confidence_gate(raw_per_eval, base_w) if cfg.confidence_gating else dict(base_w)
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
    parser.add_argument("--dynamic_cfg_x0_no_confidence_gating",
                        dest="dynamic_cfg_x0_confidence_gating", action="store_false")
    parser.add_argument("--dynamic_cfg_x0_cfg_smooth_weight", type=float, default=0.05)
    parser.add_argument("--dynamic_cfg_x0_high_cfg_penalty", type=float, default=0.02)
    parser.add_argument("--dynamic_cfg_x0_cfg_soft_max", type=float, default=7.5)
    parser.add_argument("--dynamic_cfg_x0_cfg_min", type=float, default=0.0)
    parser.add_argument("--dynamic_cfg_x0_cfg_max", type=float, default=12.0)
    parser.add_argument("--dynamic_cfg_x0_log_dynamic_cfg", action="store_true", default=True)
    parser.add_argument("--dynamic_cfg_x0_log_path", type=str, default=None)
    parser.add_argument("--dynamic_cfg_x0_add_local_neighborhood", action="store_true",
                        default=False)
```

---

## 2. SD3.5 wrapper — `sampling_unified_sd35_dynamic_cfg_x0.py`

Drop-in replacement for `sampling_unified_sd35.run_baseline`.
Uses `transformer_step_split` (single forward → flow_u, flow_c) per scoring step.

```python
"""SD3.5 unified runner with per-step adaptive CFG via decoded-x0 reward scoring."""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import torch
from PIL import Image

import dynamic_cfg_x0 as dcx
import sampling_unified_sd35 as su


def _make_eval_fn(reward_model: su.UnifiedRewardScorer):
    """Return (evaluator, prompt, image) -> float; auto-routes to reward server."""
    def _eval(evaluator: str, prompt: str, image: Image.Image) -> float:
        e = str(evaluator).lower().strip()
        if e == "imagereward": return float(reward_model._score_imagereward(prompt, image))
        if e == "hpsv3":       return float(reward_model._score_hpsv3(prompt, image))
        if e == "hpsv2":       return float(reward_model._score_hpsv2(prompt, image))
        if e == "pickscore":   return float(reward_model._score_pickscore(prompt, image))
        raise ValueError(f"unknown evaluator: {evaluator}")
    return _eval


def _run_baseline_dynamic_cfg_x0(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    emb: su.EmbeddingContext,
    reward_model: su.UnifiedRewardScorer,
    prompt: str,
    seed: int,
    cfg_scale: float = 1.0,
) -> tuple[Image.Image, float]:
    """Drop-in replacement for su.run_baseline with per-step adaptive CFG."""
    cfg = dcx.DynamicCfgX0Config.from_args(args)
    if not cfg.enabled:
        return su.run_baseline(args, ctx, emb, reward_model, prompt, seed, cfg_scale=cfg_scale)

    use_euler = bool(getattr(args, "euler_sampler", False))
    x0_sampler = bool(getattr(args, "x0_sampler", False))
    latents = su.make_latents(ctx, seed, args.height, args.width, emb.cond_text[0].dtype)
    sched = su.step_schedule(
        ctx.device, latents.dtype, args.steps, getattr(args, "sigmas", None), euler=use_euler
    )
    sigma_vals = [float(s[0].item()) for s in sched]
    sigma_min = float(min(sigma_vals)) if sigma_vals else 0.0
    sigma_max = float(max(sigma_vals)) if sigma_vals else 1.0

    default_cfg = float(cfg_scale) if float(cfg_scale) > 0.0 else 1.0
    w_prev: float | None = None

    log_path = cfg.log_path
    if log_path is None and getattr(args, "out_dir", None):
        log_path = os.path.join(
            str(args.out_dir),
            f"dynamic_cfg_x0_seed{int(seed)}_{abs(hash(prompt)) % (10 ** 8)}.jsonl",
        )
    logger = dcx.DynamicCfgLogger(log_path if cfg.log_dynamic_cfg else None)

    eval_fn = _make_eval_fn(reward_model)
    def _decode(x0_tensor: torch.Tensor) -> Image.Image:
        return su.decode_to_pil(ctx, x0_tensor)

    dx = torch.zeros_like(latents)

    try:
        for i, (t_flat, t_4d, dt) in enumerate(sched):
            sigma_i = float(t_flat.item())
            progress = dcx.progress_from_sigma(sigma_i, sigma_min, sigma_max)
            do_score = dcx.should_score_step(i, progress, cfg)

            if not use_euler:
                noise = latents if i == 0 else torch.randn_like(latents)
                latents = (1.0 - t_4d) * dx + t_4d * noise

            if do_score:
                t0 = time.time()
                flow_u, flow_c = su.transformer_step_split(args, ctx, latents, emb, 0, t_flat)
                candidates = dcx.generate_cfg_candidates(cfg, w_prev)
                result = dcx.select_cfg_for_step(
                    candidates=candidates, flow_u=flow_u, flow_c=flow_c,
                    latents=latents, sigma_4d=t_4d, x0_sampler=x0_sampler,
                    decode_fn=_decode, eval_fn=eval_fn, prompt=prompt,
                    progress=progress, w_prev=w_prev, cfg=cfg,
                )
                chosen = float(result["chosen_cfg"])
                flow = flow_u + chosen * (flow_c - flow_u)
                logger.log({
                    "step": int(i), "timestep": float(t_flat.item()), "sigma": float(sigma_i),
                    "progress": float(progress), "cfg_candidates": result["candidates"],
                    "chosen_cfg": chosen, "weights": result["weights"],
                    "base_weights": result["base_weights"], "raw_scores": result["raw_scores"],
                    "norm_scores": result["norm_scores"], "total_scores": result["total_scores"],
                    "elapsed_s": float(time.time() - t0),
                    "prompt_hash": int(abs(hash(prompt)) % (10 ** 12)), "seed": int(seed),
                })
                w_prev = chosen
            else:
                w_use = w_prev if w_prev is not None else default_cfg
                flow = su.transformer_step(args, ctx, latents, emb, 0, t_flat, float(w_use))

            latents, dx = su._apply_step(latents, flow, dx, t_4d, dt, use_euler, x0_sampler)

        final = latents if use_euler else dx
        image = su.decode_to_pil(ctx, final)
        return image, su.score_image(reward_model, prompt, image)
    finally:
        logger.close()


def run(args: argparse.Namespace) -> None:
    original = su.run_baseline
    su.run_baseline = _run_baseline_dynamic_cfg_x0
    try:
        su.run(args)
    finally:
        su.run_baseline = original


def _parse_with_dyncfg_flags(argv: list[str] | None) -> argparse.Namespace:
    extra = argparse.ArgumentParser(add_help=False)
    dcx.add_dynamic_cfg_x0_args(extra)
    dyn_args, remaining = extra.parse_known_args(argv)
    args = su.parse_args(remaining)
    if not hasattr(args, "x0_sampler"):
        setattr(args, "x0_sampler", False)
    for k, v in vars(dyn_args).items():
        setattr(args, k, v)
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_with_dyncfg_flags(argv)
    run(su.normalize_paths(args))


if __name__ == "__main__":
    main()
```

---

## 3. SD3.5 DDP entry-point — `sd35_ddp_experiment_dynamic_cfg_x0.py`

Thin shim that monkey-patches `sampling_unified_sd35.run_baseline` before
delegating to the existing DDP experiment harness.

```python
"""DDP runner for SD3.5 dynamic-CFG-x0 sampling.

Patches sd35_ddp_experiment so methods="baseline" routes through
sampling_unified_sd35_dynamic_cfg_x0._run_baseline_dynamic_cfg_x0
when --dynamic_cfg_x0 is set.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import dynamic_cfg_x0 as dcx
import sampling_unified_sd35 as su
import sd35_ddp_experiment as base
from sampling_unified_sd35_dynamic_cfg_x0 import _run_baseline_dynamic_cfg_x0


def _parse_dyncfg_x0_flags(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    dcx.add_dynamic_cfg_x0_args(parser)
    return parser.parse_known_args(argv)


def _make_patched_parse_args(
    original_parse_args: Callable[[], argparse.Namespace],
) -> Callable[[], argparse.Namespace]:
    def _patched_parse_args() -> argparse.Namespace:
        dyn_args, remaining = _parse_dyncfg_x0_flags(sys.argv[1:])
        original_argv = sys.argv
        sys.argv = [original_argv[0]] + remaining
        try:
            args = original_parse_args()
        finally:
            sys.argv = original_argv
        if not hasattr(args, "x0_sampler"):
            setattr(args, "x0_sampler", False)
        for k, v in vars(dyn_args).items():
            setattr(args, k, v)
        return args
    return _patched_parse_args


def main() -> None:
    original_parse_args = base.parse_args
    original_run_baseline = su.run_baseline
    base.parse_args = _make_patched_parse_args(original_parse_args)
    su.run_baseline = _run_baseline_dynamic_cfg_x0
    try:
        base.main()
    finally:
        base.parse_args = original_parse_args
        su.run_baseline = original_run_baseline


if __name__ == "__main__":
    main()
```

---

## 4. FLUX wrapper — `sampling_flux_unified_dynamic_cfg_x0.py`

Custom `run()` (does NOT use base.run's baseline+search two-pass) because
FLUX guidance distillation requires a full forward per candidate — there
is no analytic CFG split to recombine.

```python
"""FLUX runner with per-step adaptive CFG via decoded-x0 reward scoring.

Wraps sampling_flux_unified so the per-step guidance is chosen by
dynamic_cfg_x0 instead of being a fixed scalar.

Unlike SD3.5, FLUX feeds the guidance value into the transformer as a
conditioning input (guidance distillation), so there is no analytic CFG
split — each candidate guidance value requires a full forward pass. We
therefore use dynamic_cfg_x0.select_cfg_for_step_per_call, which invokes
flow_for_w_fn(w) once per candidate.

For backends without effective guidance (e.g. FLUX.1-schnell where
guidance_embeds=False) the adaptive search degenerates to evaluating
len(cfg_grid) identical forwards — set --dynamic_cfg_x0_cfg_grid to a
single value in that case to skip the redundant work.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections.abc import Callable
from typing import Any

import torch
from PIL import Image

import dynamic_cfg_x0 as dcx
import sampling_flux_unified as base


def _make_eval_fn(reward_model: Any) -> Callable[[str, str, Image.Image], float]:
    """Return (evaluator, prompt, image) -> float; HPSv3/ImageReward auto-route to server."""
    def _eval(evaluator: str, prompt: str, image: Image.Image) -> float:
        e = str(evaluator).lower().strip()
        if e == "imagereward": return float(reward_model._score_imagereward(prompt, image))
        if e == "hpsv3":       return float(reward_model._score_hpsv3(prompt, image))
        if e == "hpsv2":       return float(reward_model._score_hpsv2(prompt, image))
        if e == "pickscore":   return float(reward_model._score_pickscore(prompt, image))
        raise ValueError(f"unknown evaluator: {evaluator}")
    return _eval


def _run_dynamic_cfg_x0_flux(
    args: argparse.Namespace,
    ctx: base.FluxContext,
    reward_model: Any,
    prompt: str,
    embeds: list[base.PromptEmbed],
    seed: int,
    cfg: dcx.DynamicCfgX0Config,
    logger: dcx.DynamicCfgLogger,
) -> base.SearchResult:
    """Run one sample with per-step adaptive guidance scoring."""
    use_euler = bool(getattr(args, "euler_sampler", False))
    x0_sampler = bool(getattr(args, "x0_sampler", False))
    init_latents = base.make_initial_latents(ctx, seed, args.height, args.width, batch_size=1)
    dx = torch.zeros_like(init_latents)
    latents = init_latents.clone()
    rng = torch.Generator(device=ctx.device).manual_seed(seed + 2048)
    t_values = base.build_t_schedule(int(args.steps), getattr(args, "sigmas", None))

    embed_for_step = embeds[0]   # adaptive search varies guidance, not prompt variant

    sigma_min = float(min(t_values)) if t_values else 0.0
    sigma_max = float(max(t_values)) if t_values else 1.0
    default_g = float(getattr(args, "baseline_guidance_scale", 1.0))
    w_prev: float | None = None
    actions: list[tuple[int, float]] = []

    def _decode(x0_tensor: torch.Tensor) -> Image.Image:
        return base.decode_to_pil(ctx, x0_tensor)

    eval_fn = _make_eval_fn(reward_model)

    for step_idx, t_val in enumerate(t_values):
        t_4d = torch.tensor(float(t_val), device=ctx.device, dtype=init_latents.dtype).view(1, 1, 1, 1)
        if not use_euler:
            if step_idx == 0:
                noise = init_latents
            else:
                noise = torch.randn(
                    init_latents.shape, device=ctx.device,
                    dtype=init_latents.dtype, generator=rng,
                )
            latents = (1.0 - t_4d) * dx + t_4d * noise

        progress = dcx.progress_from_sigma(float(t_val), sigma_min, sigma_max)
        do_score = dcx.should_score_step(step_idx, progress, cfg)

        if do_score:
            t0 = time.time()
            candidates = dcx.generate_cfg_candidates(cfg, w_prev)

            def _flow_for_w(w: float, _latents=latents, _t=float(t_val), _emb=embed_for_step) -> torch.Tensor:
                return base.flux_transformer_step(ctx, _latents, _emb, _t, float(w))

            result = dcx.select_cfg_for_step_per_call(
                candidates=candidates, flow_for_w_fn=_flow_for_w,
                latents=latents, sigma_4d=t_4d, x0_sampler=x0_sampler,
                decode_fn=_decode, eval_fn=eval_fn, prompt=prompt,
                progress=progress, w_prev=w_prev, cfg=cfg,
            )
            chosen = float(result["chosen_cfg"])
            # Re-run chosen guidance once for clean flow_pred (avoids holding ~|grid| flow tensors).
            flow_pred = base.flux_transformer_step(ctx, latents, embed_for_step, float(t_val), chosen)
            logger.log({
                "step": int(step_idx), "timestep": float(t_val), "progress": float(progress),
                "cfg_candidates": result["candidates"], "chosen_cfg": chosen,
                "weights": result["weights"], "base_weights": result["base_weights"],
                "raw_scores": result["raw_scores"], "norm_scores": result["norm_scores"],
                "total_scores": result["total_scores"],
                "elapsed_s": float(time.time() - t0),
                "prompt_hash": int(abs(hash(prompt)) % (10 ** 12)), "seed": int(seed),
            })
            w_prev = chosen
        else:
            w_use = w_prev if w_prev is not None else default_g
            flow_pred = base.flux_transformer_step(ctx, latents, embed_for_step, float(t_val), float(w_use))

        actions.append((0, float(w_prev) if w_prev is not None else float(default_g)))
        dx = base._pred_x0(latents, t_4d, flow_pred, x0_sampler)
        if use_euler:
            dt = base._compute_dt(t_values, step_idx)
            latents = latents + dt * flow_pred

    image = base.decode_to_pil(ctx, base._final_decode_tensor(latents, dx, use_euler))
    score = base.score_image(reward_model, prompt, image)
    return base.SearchResult(image=image, score=float(score), actions=actions, diagnostics=None)


def run(args: argparse.Namespace) -> None:
    if args.cuda_alloc_conf and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf

    os.makedirs(args.out_dir, exist_ok=True)
    prompts = base.load_prompts(args)
    if not prompts:
        raise RuntimeError("No prompts loaded.")

    device = base.pick_device(args)
    dtype = base.resolve_dtype(args.dtype)
    ctx = base.load_pipeline(args, device=device, dtype=dtype)
    reward_model, reward_device = base.load_reward(args, pipeline_device=device)
    print(f"Loaded. device={device} dtype={args.dtype} reward_device={reward_device} "
          f"decode_device={ctx.decode_device}")

    cfg = dcx.DynamicCfgX0Config.from_args(args)
    if not cfg.enabled:
        print("[dyncfg-x0] --dynamic_cfg_x0 not set; falling back to base.run().")
        base.run(args)
        return

    log_path = cfg.log_path or os.path.join(args.out_dir, "dynamic_cfg_x0.jsonl")
    logger = dcx.DynamicCfgLogger(log_path if cfg.log_dynamic_cfg else None)
    summary: list[dict[str, Any]] = []

    try:
        for p_idx, prompt in enumerate(prompts):
            slug = f"p{p_idx:04d}"
            save_entry = args.save_first_k < 0 or p_idx < int(args.save_first_k)
            print(f"\n{'=' * 72}\n[{slug}] {prompt}\n{'=' * 72}")

            prompt_bank = base.select_prompt_bank(prompt, int(args.n_variants))
            embeds = base.encode_prompt_bank(args, ctx, prompt_bank)
            if save_entry:
                with open(os.path.join(args.out_dir, f"{slug}_variants.txt"), "w", encoding="utf-8") as f:
                    for vi, (label, text) in enumerate(prompt_bank):
                        f.write(f"v{vi}[{label}]: {text}\n")

            prompt_samples: list[dict[str, Any]] = []
            for sample_i in range(int(args.n_samples)):
                seed = int(args.seed) + sample_i
                print(f"  sample {sample_i + 1}/{args.n_samples} seed={seed}")
                result = _run_dynamic_cfg_x0_flux(
                    args=args, ctx=ctx, reward_model=reward_model, prompt=prompt,
                    embeds=embeds, seed=seed, cfg=cfg, logger=logger,
                )
                if save_entry:
                    out_path = os.path.join(args.out_dir, f"{slug}_s{sample_i}_dynamic_cfg_x0.png")
                    result.image.save(out_path)
                prompt_samples.append({
                    "seed": seed,
                    "search_score": float(result.score),
                    "baseline_score": float(result.score),
                    "delta_score": 0.0,
                    "actions": [[int(v), float(g)] for v, g in result.actions],
                    "artifacts_saved": bool(save_entry),
                })
                del result
                gc.collect()
                if ctx.device.startswith("cuda"):
                    torch.cuda.empty_cache()

            summary.append({"slug": slug, "prompt": prompt,
                            "search_method": "dynamic_cfg_x0", "samples": prompt_samples})
            del embeds
            gc.collect()
            if ctx.device.startswith("cuda"):
                torch.cuda.empty_cache()
    finally:
        logger.close()

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    rows = [s for entry in summary for s in entry["samples"]]
    search_vals = [float(r["search_score"]) for r in rows]
    aggregate = {
        "model_id": args.model_id, "search_method": "dynamic_cfg_x0",
        "n_prompts": len(prompts), "n_samples": int(args.n_samples),
        "save_first_k": int(args.save_first_k),
        "mean_search_score": float(sum(search_vals) / len(search_vals)) if search_vals else None,
    }
    aggregate_path = os.path.join(args.out_dir, "aggregate_summary.json")
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nSummary saved:   {summary_path}")
    print(f"Aggregate saved: {aggregate_path}")
    print(f"Outputs:         {os.path.abspath(args.out_dir)}")


def _parse_with_dyncfg_flags(argv: list[str] | None) -> argparse.Namespace:
    extra = argparse.ArgumentParser(add_help=False)
    dcx.add_dynamic_cfg_x0_args(extra)
    source = list(argv) if argv is not None else list(sys.argv[1:])
    dyn_args, remaining = extra.parse_known_args(source)
    args = base.parse_args(remaining)
    for k, v in vars(dyn_args).items():
        setattr(args, k, v)
    if not hasattr(args, "x0_sampler"):
        setattr(args, "x0_sampler", False)
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_with_dyncfg_flags(argv)
    run(args)


if __name__ == "__main__":
    main()
```

---

## 5. Suite-script integration

### 5a. SD3.5 — `hpsv2_sd35_sid_ddp_suite.sh`

Env defaults (lines 265–278):

```bash
DYNAMIC_CFG_X0_GRID="${DYNAMIC_CFG_X0_GRID:-2.0 3.5 5.0 6.5 8.0}"
DYNAMIC_CFG_X0_SCORE_START_FRAC="${DYNAMIC_CFG_X0_SCORE_START_FRAC:-0.25}"
DYNAMIC_CFG_X0_SCORE_END_FRAC="${DYNAMIC_CFG_X0_SCORE_END_FRAC:-1.0}"
DYNAMIC_CFG_X0_SCORE_EVERY="${DYNAMIC_CFG_X0_SCORE_EVERY:-2}"
DYNAMIC_CFG_X0_EVALUATORS="${DYNAMIC_CFG_X0_EVALUATORS:-imagereward hpsv3}"
DYNAMIC_CFG_X0_WEIGHT_SCHEDULE="${DYNAMIC_CFG_X0_WEIGHT_SCHEDULE:-piecewise}"
DYNAMIC_CFG_X0_PROMPT_TYPE="${DYNAMIC_CFG_X0_PROMPT_TYPE:-general}"
DYNAMIC_CFG_X0_CONFIDENCE_GATING="${DYNAMIC_CFG_X0_CONFIDENCE_GATING:-1}"
DYNAMIC_CFG_X0_SMOOTH_WEIGHT="${DYNAMIC_CFG_X0_SMOOTH_WEIGHT:-0.05}"
DYNAMIC_CFG_X0_HIGH_CFG_PENALTY="${DYNAMIC_CFG_X0_HIGH_CFG_PENALTY:-0.02}"
DYNAMIC_CFG_X0_CFG_SOFT_MAX="${DYNAMIC_CFG_X0_CFG_SOFT_MAX:-7.5}"
DYNAMIC_CFG_X0_CFG_MIN="${DYNAMIC_CFG_X0_CFG_MIN:-0.0}"
DYNAMIC_CFG_X0_CFG_MAX="${DYNAMIC_CFG_X0_CFG_MAX:-12.0}"
DYNAMIC_CFG_X0_ADD_LOCAL_NEIGHBORHOOD="${DYNAMIC_CFG_X0_ADD_LOCAL_NEIGHBORHOOD:-0}"
```

Method case (line 932) and arg block (lines 1063–1085):

```bash
    dynamic_cfg_x0)
      mode_arg="base"
      runner_script="${SCRIPT_DIR}/sd35_ddp_experiment_dynamic_cfg_x0.py"
      ;;
# ...
  if [[ "${runner_script}" == "${SCRIPT_DIR}/sd35_ddp_experiment_dynamic_cfg_x0.py" ]]; then
    extra+=(
      --dynamic_cfg_x0
      --dynamic_cfg_x0_cfg_grid ${DYNAMIC_CFG_X0_GRID}
      --dynamic_cfg_x0_score_start_frac "${DYNAMIC_CFG_X0_SCORE_START_FRAC}"
      --dynamic_cfg_x0_score_end_frac "${DYNAMIC_CFG_X0_SCORE_END_FRAC}"
      --dynamic_cfg_x0_score_every "${DYNAMIC_CFG_X0_SCORE_EVERY}"
      --dynamic_cfg_x0_evaluators ${DYNAMIC_CFG_X0_EVALUATORS}
      --dynamic_cfg_x0_weight_schedule "${DYNAMIC_CFG_X0_WEIGHT_SCHEDULE}"
      --dynamic_cfg_x0_prompt_type "${DYNAMIC_CFG_X0_PROMPT_TYPE}"
      --dynamic_cfg_x0_cfg_smooth_weight "${DYNAMIC_CFG_X0_SMOOTH_WEIGHT}"
      --dynamic_cfg_x0_high_cfg_penalty "${DYNAMIC_CFG_X0_HIGH_CFG_PENALTY}"
      --dynamic_cfg_x0_cfg_soft_max "${DYNAMIC_CFG_X0_CFG_SOFT_MAX}"
      --dynamic_cfg_x0_cfg_min "${DYNAMIC_CFG_X0_CFG_MIN}"
      --dynamic_cfg_x0_cfg_max "${DYNAMIC_CFG_X0_CFG_MAX}"
    )
    if [[ "${DYNAMIC_CFG_X0_CONFIDENCE_GATING:-1}" == "0" ]]; then
      extra+=(--dynamic_cfg_x0_no_confidence_gating)
    fi
    if [[ "${DYNAMIC_CFG_X0_ADD_LOCAL_NEIGHBORHOOD:-0}" == "1" ]]; then
      extra+=(--dynamic_cfg_x0_add_local_neighborhood)
    fi
  fi
```

### 5b. FLUX — `hpsv2_flux_schnell_ddp_suite.sh`

Method case (lines 922–948):

```bash
    dynamic_cfg_x0)
      dyncfg_args=(
        --runner_script "${SCRIPT_DIR}/sampling_flux_unified_dynamic_cfg_x0.py"
        --n_variants 1
        --cfg_scales ${CFG_SCALES}
        --dynamic_cfg_x0
        --dynamic_cfg_x0_cfg_grid ${DYNAMIC_CFG_X0_GRID:-1.5 2.0 2.5 3.0 3.5}
        --dynamic_cfg_x0_score_start_frac "${DYNAMIC_CFG_X0_SCORE_START_FRAC:-0.25}"
        --dynamic_cfg_x0_score_end_frac "${DYNAMIC_CFG_X0_SCORE_END_FRAC:-1.0}"
        --dynamic_cfg_x0_score_every "${DYNAMIC_CFG_X0_SCORE_EVERY:-2}"
        --dynamic_cfg_x0_evaluators ${DYNAMIC_CFG_X0_EVALUATORS:-imagereward hpsv3}
        --dynamic_cfg_x0_weight_schedule "${DYNAMIC_CFG_X0_WEIGHT_SCHEDULE:-piecewise}"
        --dynamic_cfg_x0_prompt_type "${DYNAMIC_CFG_X0_PROMPT_TYPE:-general}"
        --dynamic_cfg_x0_cfg_smooth_weight "${DYNAMIC_CFG_X0_SMOOTH_WEIGHT:-0.05}"
        --dynamic_cfg_x0_high_cfg_penalty "${DYNAMIC_CFG_X0_HIGH_CFG_PENALTY:-0.02}"
        --dynamic_cfg_x0_cfg_soft_max "${DYNAMIC_CFG_X0_CFG_SOFT_MAX:-3.5}"
        --dynamic_cfg_x0_cfg_min "${DYNAMIC_CFG_X0_CFG_MIN:-0.0}"
        --dynamic_cfg_x0_cfg_max "${DYNAMIC_CFG_X0_CFG_MAX:-5.0}"
      )
      if [[ "${DYNAMIC_CFG_X0_CONFIDENCE_GATING:-1}" == "0" ]]; then
        dyncfg_args+=(--dynamic_cfg_x0_no_confidence_gating)
      fi
      if [[ "${DYNAMIC_CFG_X0_ADD_LOCAL_NEIGHBORHOOD:-0}" == "1" ]]; then
        dyncfg_args+=(--dynamic_cfg_x0_add_local_neighborhood)
      fi
      run_flux_sharded "dynamic_cfg_x0" "ga" "${dyncfg_args[@]}"
      ;;
```

---

## 6. AMLT YAMLs (per-backend × per-search-reward grid)

Both YAMLs grid over `target_model × search_reward` for **4 trials each**:

- `sd35_dynamic_cfg_x0_server.yaml`: `{sid, sd35_base} × {hpsv3, imagereward}`
- `flux_dynamic_cfg_x0_server.yaml`:  `{flux, tdd_flux} × {hpsv3, imagereward}`

Per-backend CFG ranges (set in bash conditional inside each YAML's command body):

| backend | grid | baseline | min | max | soft_max |
|---|---|---|---|---|---|
| `sid` (SD3.5, 4 steps) | `1.0 1.25 1.5 1.75 2.0 2.25 2.5` | 1.0 | 0.5 | 3.0 | 2.5 |
| `sd35_base` (28 steps) | `3.5 4.0 4.5 5.0 5.5 6.0 7.0` | 4.5 | 2.0 | 8.0 | 6.5 |
| `flux` (FLUX.1-dev, 28 steps) | `3.5 4.0 4.5 5.0 5.5 6.0 7.0` | 4.5 | 2.0 | 8.0 | 6.5 |
| `tdd_flux` (8 steps + LoRA) | `1.0 1.5 2.0 2.5 3.0` | 2.0 | 1.0 | 3.5 | 3.0 |

`sid` and `sd35_base` grids match existing BoN-MCTS yamls exactly. `flux`
mirrors `sd35_base`'s shape (no FLUX-dev BoN-MCTS reference exists).

Reward server hosts both `hpsv3 imagereward` (so the same server config
serves both runs); the `search_reward` param picks which one drives the
adaptive search per trial via:

```bash
- export REWARD_BACKEND={search_reward}
- export REWARD_TYPE={search_reward}
- export REWARD_BACKENDS={search_reward}
- export EVAL_BACKENDS='imagereward hpsv3'      # phase-1 eval covers BOTH for cross-run comparability
- export DYNAMIC_CFG_X0_EVALUATORS='{search_reward}'
- export OUT_ROOT=/mnt/data/v-yigu/hpsv2_dyncfgx0_runs/{target_model}/{search_reward}/{run_tag}
```

Phase-2 posthoc eval = `hpsv2 pickscore` (the rewards not on the server).

---

## 7. Tests — `test_dynamic_cfg_x0.py`

19 tests, no GPU required. Run with `python test_dynamic_cfg_x0.py` or pytest.
Coverage:

- **Progress + step gating** — endpoints, zero-span, in/out-of-window stride
- **Candidate generation** — clamp + dedupe, neighborhood expansion around `w_prev`
- **Evaluator weights** — renormalization after filter, late-progress bias, smooth interp
- **Z-score normalization** — mean/std behavior, constant-input fallback
- **Confidence gating** — drops constant evaluators, falls back to base when all flat
- **Combine + select** — argmax tied-break, high-CFG penalty, smoothness bias
- **x0_pred** — flow-matching identity; x0_sampler passthrough
- **End-to-end smoke** — synthetic eval `r = -|w − 5.0|` correctly picks 5.0
- **Logger** — appends JSONL, lazy directory creation

All 19 currently pass.

---

## 8. Key design decisions

1. **Pure algorithm module.** `dynamic_cfg_x0.py` has no diffusion or reward
   imports beyond torch/numpy/PIL. All model interaction is via callables
   the wrapper supplies. This is what lets the same algorithm cover SD3.5
   split-recombine and FLUX per-call paths cleanly.

2. **Two scoring entry-points:**
   - `select_cfg_for_step` (SD3.5) — caller does ONE `transformer_step_split`
     forward and we recombine `flow_w = flow_u + w·(flow_c − flow_u)` per
     candidate. Cheap.
   - `select_cfg_for_step_per_call` (FLUX) — caller passes `flow_for_w_fn`,
     we invoke once per candidate. Expensive but unavoidable: FLUX feeds
     guidance as a transformer input, not as a CFG-interpolation factor.

3. **Single search-reward per AMLT trial.** Two separate runs per backend
   (one driven by HPSv3, one by ImageReward) so we can attribute the
   ranking gain to the search reward independently.

4. **Phase-1 eval covers both rewards.** The reward server already hosts
   both, so EVAL_BACKENDS='imagereward hpsv3' costs nothing extra during
   phase 1 and makes the two runs directly comparable on identical metric
   columns. Phase-2 posthoc fills in `hpsv2 pickscore`.

5. **Confidence gating.** Each evaluator's base weight is multiplied by its
   own score dispersion (std across candidates). If a reward returns a
   nearly flat curve over the grid, its vote shrinks toward zero — protects
   against an evaluator with no signal at the current step dragging the
   choice toward arbitrary values. Falls back to base weights if ALL
   evaluators are flat (rare, well-defined behavior).

6. **High-CFG penalty + smoothness term.** `combine_scores` subtracts
   `cfg_smooth_weight × |w − w_prev|` (encourages temporal continuity) and
   `high_cfg_penalty × max(w − cfg_soft_max, 0)` (gentle pushback above
   typical-good range). Both can be disabled with `0.0`.
