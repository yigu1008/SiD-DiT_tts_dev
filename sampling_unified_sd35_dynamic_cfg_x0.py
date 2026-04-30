"""
SD3.5 unified runner with per-step adaptive CFG via decoded-x0 reward scoring.

Wraps :func:`sampling_unified_sd35.run_baseline` so the per-step CFG is chosen
by :mod:`dynamic_cfg_x0` instead of being a fixed scalar. Falls back to the
default CFG when the step is outside the scoring window or when dynamic CFG
is disabled.

Compatible with both SiD (re-noising + x0_pred) and Euler (sd35_base).
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import torch
from PIL import Image

import dynamic_cfg_x0 as dcx
import sampling_unified_sd35 as su


# ── Reward eval bridge ──────────────────────────────────────────────────────


def _make_eval_fn(reward_model: su.UnifiedRewardScorer):
    """Return a (evaluator_name, prompt, image) -> float scoring callable.

    Routes to UnifiedRewardScorer's per-backend score methods. Server backends
    are auto-routed via REWARD_SERVER_URL by the existing _score_via_server
    plumbing in those methods.
    """
    def _eval(evaluator: str, prompt: str, image: Image.Image) -> float:
        e = str(evaluator).lower().strip()
        if e == "imagereward":
            return float(reward_model._score_imagereward(prompt, image))
        if e == "hpsv3":
            return float(reward_model._score_hpsv3(prompt, image))
        if e == "hpsv2":
            return float(reward_model._score_hpsv2(prompt, image))
        if e == "pickscore":
            return float(reward_model._score_pickscore(prompt, image))
        raise ValueError(f"unknown evaluator: {evaluator}")
    return _eval


# ── Driver ──────────────────────────────────────────────────────────────────


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

    # Choose default CFG: caller's cfg_scale unless it's the SiD/no-CFG sentinel.
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

            # Prepare noisy latents (SiD re-noises; Euler keeps the running state).
            if not use_euler:
                noise = latents if i == 0 else torch.randn_like(latents)
                latents = (1.0 - t_4d) * dx + t_4d * noise

            if do_score:
                t0 = time.time()
                flow_u, flow_c = su.transformer_step_split(args, ctx, latents, emb, 0, t_flat)
                candidates = dcx.generate_cfg_candidates(cfg, w_prev)
                result = dcx.select_cfg_for_step(
                    candidates=candidates,
                    flow_u=flow_u,
                    flow_c=flow_c,
                    latents=latents,
                    sigma_4d=t_4d,
                    x0_sampler=x0_sampler,
                    decode_fn=_decode,
                    eval_fn=eval_fn,
                    prompt=prompt,
                    progress=progress,
                    w_prev=w_prev,
                    cfg=cfg,
                )
                chosen = float(result["chosen_cfg"])
                flow = flow_u + chosen * (flow_c - flow_u)
                logger.log({
                    "step": int(i),
                    "timestep": float(t_flat.item()),
                    "sigma": float(sigma_i),
                    "progress": float(progress),
                    "cfg_candidates": result["candidates"],
                    "chosen_cfg": chosen,
                    "weights": result["weights"],
                    "base_weights": result["base_weights"],
                    "raw_scores": result["raw_scores"],
                    "norm_scores": result["norm_scores"],
                    "total_scores": result["total_scores"],
                    "elapsed_s": float(time.time() - t0),
                    "prompt_hash": int(abs(hash(prompt)) % (10 ** 12)),
                    "seed": int(seed),
                })
                w_prev = chosen
            else:
                # Use default or last chosen CFG.
                w_use = w_prev if w_prev is not None else default_cfg
                flow = su.transformer_step(args, ctx, latents, emb, 0, t_flat, float(w_use))

            # One scheduler step.
            latents, dx = su._apply_step(latents, flow, dx, t_4d, dt, use_euler, x0_sampler)

        final = latents if use_euler else dx
        image = su.decode_to_pil(ctx, final)
        return image, su.score_image(reward_model, prompt, image)
    finally:
        logger.close()


# ── Module wiring ───────────────────────────────────────────────────────────


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
