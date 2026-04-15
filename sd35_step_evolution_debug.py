#!/usr/bin/env python3
"""
Single-GPU step-evolution debugger for SD3.5 base search methods.

Focus:
- MCTS trajectory replay (chosen actions)
- SMC trajectory tracing (best particle per step + optional particle grids)

This script is intentionally separate from production runners.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import sampling_sd35_base as sb
import sampling_unified_sd35 as su


def _font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _annotate(image: Image.Image, lines: list[str], *, bg=(18, 18, 18), fg=(230, 230, 230)) -> Image.Image:
    if len(lines) <= 0:
        return image
    pad = 6
    font = _font(16)
    line_h = 22
    header_h = pad * 2 + line_h * len(lines)
    out = Image.new("RGB", (image.width, image.height + header_h), bg)
    out.paste(image, (0, header_h))
    draw = ImageDraw.Draw(out)
    y = pad
    for line in lines:
        draw.text((pad, y), line, fill=fg, font=font)
        y += line_h
    return out


def _save_gif(frames: list[Image.Image], path: str, duration_ms: int) -> None:
    if len(frames) <= 0:
        return
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=max(40, int(duration_ms)),
        loop=0,
        optimize=False,
    )


def _sanitize_slug(text: str, max_len: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_").lower()
    if not s:
        s = "prompt"
    return s[:max_len]


def _save_particle_grid(images: list[Image.Image], scores: list[float], out_path: str, cols: int = 4) -> None:
    if len(images) <= 0:
        return
    cols = max(1, int(cols))
    rows = int(math.ceil(len(images) / float(cols)))
    tile_w, tile_h = images[0].size
    header_h = 24
    grid = Image.new("RGB", (cols * tile_w, rows * (tile_h + header_h)), (12, 12, 12))
    draw = ImageDraw.Draw(grid)
    font = _font(14)
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        x0 = c * tile_w
        y0 = r * (tile_h + header_h)
        grid.paste(img, (x0, y0 + header_h))
        score = float(scores[i]) if i < len(scores) else float("nan")
        draw.text((x0 + 4, y0 + 4), f"p{i} score={score:.4f}", fill=(220, 220, 220), font=font)
    grid.save(out_path)


def _parse_debug_and_base_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Debug per-step image evolution for SD3.5 base MCTS/SMC.",
        add_help=True,
    )
    parser.add_argument("--methods", nargs="+", default=["mcts", "smc"], choices=["mcts", "smc"])
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--debug_out", default=None, help="Debug output root. Defaults to <out_dir>/step_debug_<timestamp>.")
    parser.add_argument("--save_latent_steps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_x0_steps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_smc_particle_grids", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--make_gif", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gif_ms", type=int, default=350)
    parser.add_argument("--smc_grid_cols", type=int, default=4)

    dbg_args, remaining = parser.parse_known_args(argv)

    # Parse base SD3.5 args exactly like sampling_sd35_base.py
    base_args = sb._parse_extra_args(remaining)
    base_args = sb._apply_sd35_base_defaults(base_args)
    base_args = su.normalize_paths(base_args)

    # Force a deterministic and debug-friendly default unless user overrides.
    if not hasattr(base_args, "reward_backend") or base_args.reward_backend is None:
        base_args.reward_backend = "imagereward"

    return dbg_args, base_args


def _resolve_debug_out(dbg_args: argparse.Namespace, base_args: argparse.Namespace, prompt: str) -> str:
    if dbg_args.debug_out:
        out = Path(dbg_args.debug_out).expanduser().resolve()
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        prompt_slug = _sanitize_slug(prompt)
        out = Path(base_args.out_dir).expanduser().resolve() / f"step_debug_{ts}_{prompt_slug}"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def _select_prompt(base_args: argparse.Namespace, prompt_index: int) -> tuple[str, int, list[str]]:
    prompts = su.load_prompts(base_args)
    if len(prompts) <= 0:
        raise RuntimeError("No prompts loaded.")
    idx = max(0, min(len(prompts) - 1, int(prompt_index)))
    return prompts[idx], idx, prompts


def _load_rewrite_cache(path: str | None) -> dict[str, list[str]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _action_at(actions: list[tuple[int, float, float]], step_idx: int, fallback_cfg: float) -> tuple[int, float, float]:
    if len(actions) > step_idx:
        a = actions[step_idx]
    elif len(actions) > 0:
        a = actions[-1]
    else:
        a = (0, fallback_cfg, 0.0)
    if len(a) >= 3:
        return int(a[0]), float(a[1]), float(a[2])
    if len(a) == 2:
        return int(a[0]), float(a[1]), 0.0
    return 0, float(fallback_cfg), 0.0


@torch.no_grad()
def _replay_mcts_with_step_dumps(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    emb: su.EmbeddingContext,
    reward_model: su.UnifiedRewardScorer,
    prompt: str,
    seed: int,
    actions: list[tuple[int, float, float]],
    out_dir: str,
    save_latent_steps: bool,
    save_x0_steps: bool,
    make_gif: bool,
    gif_ms: int,
) -> dict[str, Any]:
    method_dir = Path(out_dir) / "mcts"
    method_dir.mkdir(parents=True, exist_ok=True)

    latents = su.make_latents(ctx, seed, args.height, args.width, emb.cond_text[0].dtype)
    sched = su.step_schedule(ctx.device, latents.dtype, args.steps, getattr(args, "sigmas", None), euler=True)

    step_records: list[dict[str, Any]] = []
    latent_frames: list[Image.Image] = []
    x0_frames: list[Image.Image] = []

    for step_idx, (t_flat, t_4d, dt) in enumerate(sched):
        vi, cfg, cs = _action_at(actions, step_idx, float(args.baseline_cfg))
        flow = su.transformer_step(args, ctx, latents, emb, vi, t_flat, cfg)
        x0_est = latents - t_4d * flow
        latents = latents + dt * flow

        rec = {
            "step_idx": int(step_idx),
            "sigma": float(t_flat[0].item()),
            "dt": float(dt),
            "variant_idx": int(vi),
            "cfg": float(cfg),
            "correction_strength": float(cs),
        }

        if save_latent_steps:
            latent_img = su.decode_to_pil(ctx, latents)
            latent_score = float(su.score_image(reward_model, prompt, latent_img))
            rec["latent_score"] = latent_score
            latent_path = method_dir / f"step_{step_idx:03d}_latent.png"
            _annotate(
                latent_img,
                [
                    f"MCTS step={step_idx} sigma={float(t_flat[0].item()):.4f}",
                    f"action: variant={vi} cfg={cfg:.3f} cs={cs:.3f}",
                    f"ImageReward={latent_score:.4f} (decoded from latent_t)",
                ],
            ).save(latent_path)
            latent_frames.append(Image.open(latent_path).copy())

        if save_x0_steps:
            x0_img = su.decode_to_pil(ctx, x0_est)
            x0_score = float(su.score_image(reward_model, prompt, x0_img))
            rec["x0_score"] = x0_score
            x0_path = method_dir / f"step_{step_idx:03d}_x0.png"
            _annotate(
                x0_img,
                [
                    f"MCTS step={step_idx} sigma={float(t_flat[0].item()):.4f}",
                    f"action: variant={vi} cfg={cfg:.3f} cs={cs:.3f}",
                    f"ImageReward={x0_score:.4f} (decoded from x0 estimate)",
                ],
            ).save(x0_path)
            x0_frames.append(Image.open(x0_path).copy())

        step_records.append(rec)

    final_img = su.decode_to_pil(ctx, latents)
    final_score = float(su.score_image(reward_model, prompt, final_img))
    final_path = method_dir / "final_replay.png"
    _annotate(
        final_img,
        [
            "MCTS replay final",
            f"steps={int(args.steps)} score={final_score:.4f}",
        ],
    ).save(final_path)

    if make_gif and save_latent_steps and len(latent_frames) > 0:
        _save_gif(latent_frames, str(method_dir / "evolution_latent.gif"), gif_ms)
    if make_gif and save_x0_steps and len(x0_frames) > 0:
        _save_gif(x0_frames, str(method_dir / "evolution_x0.gif"), gif_ms)

    with (method_dir / "step_records.json").open("w", encoding="utf-8") as f:
        json.dump(step_records, f, indent=2)

    return {
        "method": "mcts",
        "steps": step_records,
        "final_replay_score": float(final_score),
        "final_replay_image": str(final_path),
    }


@torch.no_grad()
def _run_smc_with_step_dumps(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    emb: su.EmbeddingContext,
    reward_model: su.UnifiedRewardScorer,
    prompt: str,
    seed: int,
    out_dir: str,
    save_grids: bool,
    grid_cols: int,
    make_gif: bool,
    gif_ms: int,
) -> tuple[su.SearchResult, dict[str, Any]]:
    method_dir = Path(out_dir) / "smc"
    method_dir.mkdir(parents=True, exist_ok=True)

    k = max(2, int(args.smc_k))
    cfg = float(args.smc_cfg_scale)
    variant_idx = int(max(0, min(len(emb.cond_text) - 1, int(args.smc_variant_idx))))
    corr_strengths = list(getattr(args, "correction_strengths", [0.0]))
    smc_cs = float(corr_strengths[0]) if corr_strengths else 0.0

    particle_latents = [su.make_latents(ctx, seed + pi, args.height, args.width, emb.cond_text[0].dtype) for pi in range(k)]
    latents = torch.cat(particle_latents, dim=0)
    dx = torch.zeros_like(latents)

    use_euler = bool(getattr(args, "euler_sampler", False))
    sched = su.step_schedule(ctx.device, latents.dtype, args.steps, getattr(args, "sigmas", None), euler=use_euler)
    rng = torch.Generator(device=ctx.device).manual_seed(seed + 6001)

    start_idx = int((1.0 - float(args.resample_start_frac)) * int(args.steps))
    start_idx = max(0, min(int(args.steps) - 1, start_idx))

    log_w = torch.zeros(k, device=ctx.device, dtype=torch.float32)
    step_records: list[dict[str, Any]] = []
    best_frames: list[Image.Image] = []
    resample_count = 0

    print(
        f"  smc(debug): K={k} cfg={cfg:.2f} variant={variant_idx} "
        f"gamma={float(args.smc_gamma):.3f} ess_thr={float(args.ess_threshold):.2f} euler={use_euler}"
    )

    for step_idx, (t_flat, t_4d, dt) in enumerate(sched):
        if not use_euler:
            if step_idx == 0:
                noise = latents
            else:
                noise = torch.randn(
                    latents.shape,
                    device=latents.device,
                    dtype=latents.dtype,
                    generator=rng,
                )
            latents = (1.0 - t_4d) * dx + t_4d * noise

        next_dx_parts = []
        next_latents_parts = []
        for pi in range(k):
            flow = su.transformer_step(args, ctx, latents[pi : pi + 1], emb, variant_idx, t_flat, cfg)
            if use_euler:
                p_latents = latents[pi : pi + 1] + dt * flow
                p_dx = latents[pi : pi + 1] - t_4d * flow
                next_latents_parts.append(p_latents)
            else:
                p_dx = su._pred_x0(latents[pi : pi + 1], t_4d, flow, args.x0_sampler)
            if smc_cs > 0.0:
                p_dx = su.apply_reward_correction(ctx, p_dx, prompt, reward_model, smc_cs, cfg=cfg)
            next_dx_parts.append(p_dx)

        dx = torch.cat(next_dx_parts, dim=0)
        if use_euler:
            latents = torch.cat(next_latents_parts, dim=0)

        score_tensor = su._final_decode_tensor(latents, dx, use_euler)
        step_images = [su.decode_to_pil(ctx, score_tensor[pi : pi + 1]) for pi in range(k)]
        step_scores_list = [float(su.score_image(reward_model, prompt, img)) for img in step_images]
        step_scores = torch.tensor(step_scores_list, device=dx.device, dtype=torch.float32)

        best_idx = int(np.argmax(step_scores_list))
        best_score = float(step_scores_list[best_idx])
        best_img = step_images[best_idx]
        best_img_path = method_dir / f"step_{step_idx:03d}_best.png"
        _annotate(
            best_img,
            [
                f"SMC step={step_idx} sigma={float(t_flat[0].item()):.4f}",
                f"best particle={best_idx}/{k-1} score={best_score:.4f}",
                f"variant={variant_idx} cfg={cfg:.3f} cs={smc_cs:.3f}",
            ],
        ).save(best_img_path)
        best_frames.append(Image.open(best_img_path).copy())

        if save_grids:
            _save_particle_grid(
                step_images,
                step_scores_list,
                str(method_dir / f"step_{step_idx:03d}_particles.png"),
                cols=grid_cols,
            )

        record = {
            "step_idx": int(step_idx),
            "sigma": float(t_flat[0].item()),
            "dt": float(dt),
            "best_particle": int(best_idx),
            "best_score": float(best_score),
            "mean_score": float(np.mean(step_scores_list)),
            "min_score": float(np.min(step_scores_list)),
            "max_score": float(np.max(step_scores_list)),
            "resampled": False,
            "ess": None,
        }

        if step_idx >= start_idx:
            lam = (1.0 + float(args.smc_gamma)) ** (int(args.steps) - 1 - step_idx) - 1.0
            log_w = log_w + float(lam) * step_scores
            weights = torch.softmax(log_w, dim=0)
            ess = float(1.0 / torch.sum(weights * weights).item())
            record["ess"] = ess
            if ess < float(args.ess_threshold) * float(k):
                idx = su._systematic_resample(weights)
                dx = dx[idx].clone()
                latents = latents[idx].clone()
                log_w = torch.zeros_like(log_w)
                resample_count += 1
                record["resampled"] = True
                record["resample_indices"] = [int(v) for v in idx.detach().cpu().tolist()]

        step_records.append(record)

    final_tensor = su._final_decode_tensor(latents, dx, use_euler)
    final_images = [su.decode_to_pil(ctx, final_tensor[pi : pi + 1]) for pi in range(k)]
    final_scores = [float(su.score_image(reward_model, prompt, img)) for img in final_images]
    best_idx = int(np.argmax(final_scores))
    result = su.SearchResult(
        image=final_images[best_idx],
        score=float(final_scores[best_idx]),
        actions=[(variant_idx, cfg, smc_cs) for _ in range(int(args.steps))],
        diagnostics={
            "smc_style": "debug_step_tracking",
            "smc_k": int(k),
            "smc_cfg_scale": float(cfg),
            "smc_variant_idx": int(variant_idx),
            "smc_gamma": float(args.smc_gamma),
            "resample_start_step": int(start_idx),
            "resample_count": int(resample_count),
            "final_particle_scores": [float(v) for v in final_scores],
        },
    )

    final_path = method_dir / "final_smc.png"
    _annotate(
        result.image,
        [
            "SMC final",
            f"best particle={best_idx}/{k-1} score={result.score:.4f}",
        ],
    ).save(final_path)

    if make_gif and len(best_frames) > 0:
        _save_gif(best_frames, str(method_dir / "evolution_best_particle.gif"), gif_ms)

    with (method_dir / "step_records.json").open("w", encoding="utf-8") as f:
        json.dump(step_records, f, indent=2)

    debug_payload = {
        "method": "smc",
        "steps": step_records,
        "final_score": float(result.score),
        "final_image": str(final_path),
    }
    return result, debug_payload


def main(argv: list[str] | None = None) -> None:
    dbg_args, args = _parse_debug_and_base_args(argv)

    prompt, prompt_idx, prompts_all = _select_prompt(args, dbg_args.prompt_index)
    debug_out = _resolve_debug_out(dbg_args, args, prompt)

    # Save run config early for reproducibility.
    run_cfg = {
        "debug_args": vars(dbg_args),
        "base_args": vars(args),
        "selected_prompt_index": int(prompt_idx),
        "selected_prompt": prompt,
        "num_prompts_loaded": len(prompts_all),
    }
    with open(os.path.join(debug_out, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    print(f"[debug] output_dir={debug_out}")
    print(f"[debug] prompt_index={prompt_idx} prompt={prompt}")

    ctx = sb.load_pipeline_sd35base(args)
    reward_model = su.load_reward_model(args, ctx.device)

    rewrite_cache = _load_rewrite_cache(getattr(args, "rewrites_file", None))
    variants = su.generate_variants(args, prompt, rewrite_cache)
    emb = su.encode_variants(ctx, variants)

    with open(os.path.join(debug_out, "variants.txt"), "w", encoding="utf-8") as f:
        for i, v in enumerate(variants):
            f.write(f"v{i}: {v}\n")

    summary: dict[str, Any] = {
        "prompt": prompt,
        "prompt_index": int(prompt_idx),
        "variants": variants,
        "methods": {},
    }

    # Baseline reference
    base_img, base_score = su.run_baseline(
        args,
        ctx,
        emb,
        reward_model,
        prompt,
        args.seed,
        cfg_scale=float(args.baseline_cfg),
    )
    base_path = os.path.join(debug_out, "baseline.png")
    _annotate(base_img, [f"baseline cfg={float(args.baseline_cfg):.3f}", f"score={float(base_score):.4f}"]).save(base_path)
    summary["baseline"] = {"score": float(base_score), "image": base_path}

    for method in dbg_args.methods:
        method = str(method).strip().lower()
        if method == "mcts":
            print("[debug] running MCTS search ...")
            mcts = sb.run_mcts_sd35base(args, ctx, emb, reward_model, prompt, variants, args.seed)
            mcts_final_path = os.path.join(debug_out, "mcts", "final_search.png")
            Path(mcts_final_path).parent.mkdir(parents=True, exist_ok=True)
            _annotate(
                mcts.image,
                [
                    "MCTS search final",
                    f"score={float(mcts.score):.4f}",
                    f"steps={len(mcts.actions)}",
                ],
            ).save(mcts_final_path)

            debug_payload = _replay_mcts_with_step_dumps(
                args,
                ctx,
                emb,
                reward_model,
                prompt,
                args.seed,
                mcts.actions,
                debug_out,
                save_latent_steps=bool(dbg_args.save_latent_steps),
                save_x0_steps=bool(dbg_args.save_x0_steps),
                make_gif=bool(dbg_args.make_gif),
                gif_ms=int(dbg_args.gif_ms),
            )
            summary["methods"]["mcts"] = {
                "search_score": float(mcts.score),
                "search_actions": [[int(a[0]), float(a[1]), float(a[2])] for a in mcts.actions],
                "search_final_image": mcts_final_path,
                "replay": debug_payload,
            }

        elif method == "smc":
            print("[debug] running SMC debug rollout ...")
            smc_result, smc_debug = _run_smc_with_step_dumps(
                args,
                ctx,
                emb,
                reward_model,
                prompt,
                args.seed,
                debug_out,
                save_grids=bool(dbg_args.save_smc_particle_grids),
                grid_cols=int(dbg_args.smc_grid_cols),
                make_gif=bool(dbg_args.make_gif),
                gif_ms=int(dbg_args.gif_ms),
            )
            summary["methods"]["smc"] = {
                "score": float(smc_result.score),
                "actions": [[int(a[0]), float(a[1]), float(a[2])] for a in smc_result.actions],
                "diagnostics": smc_result.diagnostics,
                "debug": smc_debug,
            }
        else:
            raise RuntimeError(f"Unsupported method: {method}")

    summary_path = os.path.join(debug_out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[debug] done")
    print(f"[debug] summary={summary_path}")


if __name__ == "__main__":
    main()
