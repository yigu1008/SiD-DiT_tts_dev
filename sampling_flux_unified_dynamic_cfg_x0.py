"""FLUX runner with per-step adaptive CFG via decoded-x0 reward scoring.

Wraps :mod:`sampling_flux_unified` so the per-step guidance is chosen by
:mod:`dynamic_cfg_x0` instead of being a fixed scalar.

Unlike SD3.5, FLUX feeds the guidance value into the transformer as a
conditioning input (guidance distillation), so there is no analytic CFG
split — each candidate guidance value requires a full forward pass. We
therefore use :func:`dynamic_cfg_x0.select_cfg_for_step_per_call`, which
invokes ``flow_for_w_fn(w)`` once per candidate.

Compatible with ``flux`` (Euler), ``senseflow_flux`` (re-noise + x0_pred)
and ``tdd_flux`` (Euler with LoRA). For backends without effective
guidance (e.g. FLUX.1-schnell where ``guidance_embeds=False``) the
adaptive search degenerates to evaluating ``len(cfg_grid)`` identical
forwards — set ``--dynamic_cfg_x0_cfg_grid`` to a single value in that
case to skip the redundant work.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from collections.abc import Callable
from typing import Any

import torch
from PIL import Image

import dynamic_cfg_x0 as dcx
import sampling_flux_unified as base


# ── Reward eval bridge ──────────────────────────────────────────────────────


def _make_eval_fn(reward_model: Any) -> Callable[[str, str, Image.Image], float]:
    """Return (evaluator, prompt, image) -> float. Routes to the per-backend
    score methods; HPSv3 / ImageReward auto-route to the reward server when
    ``REWARD_SERVER_URL`` is set."""

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


# ── Per-step adaptive sampler ───────────────────────────────────────────────


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

    # Use the prompt-bank head (variant 0) — adaptive search varies guidance,
    # not the prompt variant.
    embed_for_step = embeds[0]

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
                candidates=candidates,
                flow_for_w_fn=_flow_for_w,
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
            # Re-run the chosen guidance once more to get a clean flow_pred for
            # the actual scheduler step (avoids holding ~|grid| flow tensors).
            flow_pred = base.flux_transformer_step(ctx, latents, embed_for_step, float(t_val), chosen)
            logger.log({
                "step": int(step_idx),
                "timestep": float(t_val),
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


# ── Custom run() ────────────────────────────────────────────────────────────


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
    print(
        f"Loaded. device={device} dtype={args.dtype} reward_device={reward_device} "
        f"decode_device={ctx.decode_device}"
    )

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

            summary.append({
                "slug": slug, "prompt": prompt,
                "search_method": "dynamic_cfg_x0",
                "samples": prompt_samples,
            })
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
        "model_id": args.model_id,
        "search_method": "dynamic_cfg_x0",
        "n_prompts": len(prompts),
        "n_samples": int(args.n_samples),
        "save_first_k": int(args.save_first_k),
        "mean_search_score": float(sum(search_vals) / len(search_vals)) if search_vals else None,
    }
    aggregate_path = os.path.join(args.out_dir, "aggregate_summary.json")
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nSummary saved:   {summary_path}")
    print(f"Aggregate saved: {aggregate_path}")
    print(f"Outputs:         {os.path.abspath(args.out_dir)}")


# ── Module wiring ───────────────────────────────────────────────────────────


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
