"""FLUX runner for Search-over-Paths (SoP) inference-time search.

Mirrors :mod:`sd35_sop_search` but for FLUX. SoP is a path-space search that
holds prompt variant and guidance fixed and searches only over noise/path
perturbations. Compatible with ``flux``, ``flux_schnell`` and ``tdd_flux``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

import sampling_flux_unified as base


# ── CLI args ────────────────────────────────────────────────────────────────


def add_sop_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sop_init_paths", type=int, default=8)
    parser.add_argument("--sop_branch_factor", type=int, default=4)
    parser.add_argument("--sop_keep_top", type=int, default=4)
    parser.add_argument("--sop_branch_every", type=int, default=1)
    parser.add_argument("--sop_start_frac", type=float, default=0.25)
    parser.add_argument("--sop_end_frac", type=float, default=1.0)
    parser.add_argument("--sop_score_decode", choices=["x0_pred", "final"], default="x0_pred")
    parser.add_argument("--sop_variant_idx", type=int, default=0)


# ── Path container ──────────────────────────────────────────────────────────


@dataclass
class _Path:
    latents: torch.Tensor
    dx: torch.Tensor
    score: float = 0.0


# ── Driver ──────────────────────────────────────────────────────────────────


@torch.no_grad()
def _run_sop_search_flux(
    args: argparse.Namespace,
    ctx: base.FluxContext,
    reward_model: Any,
    prompt: str,
    embeds: list[base.PromptEmbed],
    seed: int,
) -> base.SearchResult:
    """SoP for FLUX: K paths, branch by re-noise, prune by x0_pred reward."""
    n_init = max(1, int(getattr(args, "sop_init_paths", 8)))
    branch_M = max(1, int(getattr(args, "sop_branch_factor", 4)))
    keep_K = max(1, int(getattr(args, "sop_keep_top", 4)))
    branch_every = max(1, int(getattr(args, "sop_branch_every", 1)))
    start_frac = float(getattr(args, "sop_start_frac", 0.25))
    end_frac = float(getattr(args, "sop_end_frac", 1.0))
    score_x0 = (str(getattr(args, "sop_score_decode", "x0_pred")) == "x0_pred")
    variant_idx = int(getattr(args, "sop_variant_idx", 0))
    variant_idx = max(0, min(len(embeds) - 1, variant_idx))
    embed_for_step = embeds[variant_idx]

    use_euler = bool(getattr(args, "euler_sampler", False))
    x0_sampler = bool(getattr(args, "x0_sampler", False))
    guidance_scale = float(getattr(args, "baseline_guidance_scale", 1.0))

    init_latents = base.make_initial_latents(ctx, seed, args.height, args.width, batch_size=1)
    t_values = base.build_t_schedule(int(args.steps), getattr(args, "sigmas", None))
    sigma_min = float(min(t_values)) if t_values else 0.0
    sigma_max = float(max(t_values)) if t_values else 1.0
    span = max(1e-8, sigma_max - sigma_min)
    n_steps = len(t_values)

    # N initial paths.
    paths: list[_Path] = [
        _Path(latents=init_latents, dx=torch.zeros_like(init_latents), score=0.0)
    ]
    rng = torch.Generator(device=ctx.device).manual_seed(int(seed) + 4096)
    for _ in range(n_init - 1):
        z = torch.randn(
            init_latents.shape, device=ctx.device,
            dtype=init_latents.dtype, generator=rng,
        )
        paths.append(_Path(latents=z, dx=torch.zeros_like(init_latents), score=0.0))

    def _t4d(t_val: float) -> torch.Tensor:
        return torch.tensor(float(t_val), device=ctx.device, dtype=init_latents.dtype).view(1, 1, 1, 1)

    def _step_one(p: _Path, step_idx: int, t_val: float, t_4d: torch.Tensor) -> _Path:
        if not use_euler:
            if step_idx == 0:
                noise = p.latents
            else:
                noise = torch.randn(
                    p.latents.shape, device=ctx.device,
                    dtype=p.latents.dtype, generator=rng,
                )
            latents_in = (1.0 - t_4d) * p.dx + t_4d * noise
        else:
            latents_in = p.latents
        flow_pred = base.flux_transformer_step(ctx, latents_in, embed_for_step, float(t_val), guidance_scale)
        new_dx = base._pred_x0(latents_in, t_4d, flow_pred, x0_sampler)
        if use_euler:
            dt = base._compute_dt(t_values, step_idx)
            new_latents = latents_in + dt * flow_pred
        else:
            new_latents = latents_in
        return _Path(latents=new_latents, dx=new_dx, score=p.score)

    def _perturb(p: _Path, t_4d: torch.Tensor) -> _Path:
        new_noise = torch.randn(
            p.latents.shape, device=ctx.device,
            dtype=p.latents.dtype, generator=rng,
        )
        if not use_euler:
            new_latents = (1.0 - t_4d) * p.dx + t_4d * new_noise
        else:
            new_latents = (1.0 - t_4d) * p.latents + t_4d * new_noise
        return _Path(latents=new_latents, dx=p.dx.clone(), score=p.score)

    def _decode_x0_score(p: _Path) -> float:
        tensor = p.latents if use_euler else p.dx
        img = base.decode_to_pil(ctx, tensor)
        return float(base.score_image(reward_model, prompt, img))

    print(
        f"  sop[flux]: N={n_init} M={branch_M} K={keep_K} every={branch_every} "
        f"window=[{start_frac:.2f},{end_frac:.2f}] score={'x0' if score_x0 else 'final'} "
        f"steps={n_steps} guidance={guidance_scale:.2f} variant={variant_idx}"
    )

    actions: list[tuple[int, float]] = []
    for step_idx, t_val in enumerate(t_values):
        t_4d = _t4d(t_val)
        progress = max(0.0, min(1.0, 1.0 - (float(t_val) - sigma_min) / span))
        in_window = (progress >= start_frac - 1e-8) and (progress <= end_frac + 1e-8)
        do_branch = in_window and ((step_idx % branch_every) == 0)

        if not do_branch:
            paths = [_step_one(p, step_idx, t_val, t_4d) for p in paths]
            actions.append((variant_idx, guidance_scale))
            continue

        children: list[_Path] = []
        for p in paths:
            for _m in range(branch_M):
                p_branched = _perturb(p, t_4d)
                p_advanced = _step_one(p_branched, step_idx, t_val, t_4d)
                if score_x0:
                    p_advanced.score = _decode_x0_score(p_advanced)
                children.append(p_advanced)
        if score_x0:
            children.sort(key=lambda x: x.score, reverse=True)
        paths = children[: max(1, keep_K)]
        actions.append((variant_idx, guidance_scale))
        print(
            f"    step {step_idx + 1}/{n_steps} progress={progress:.2f} "
            f"branched K*M={len(children)} kept={len(paths)} "
            f"top_score={paths[0].score if score_x0 else float('nan'):.4f}"
        )

    # Final decode + score across surviving paths.
    best_score = -float("inf")
    best_img: Image.Image | None = None
    for p in paths:
        tensor = base._final_decode_tensor(p.latents, p.dx, use_euler)
        img = base.decode_to_pil(ctx, tensor)
        s = float(base.score_image(reward_model, prompt, img))
        if s > best_score:
            best_score = s
            best_img = img
    assert best_img is not None
    return base.SearchResult(image=best_img, score=float(best_score), actions=actions, diagnostics={
        "method": "sop",
        "n_init": n_init,
        "branch_factor": branch_M,
        "keep_top": keep_K,
        "branch_every": branch_every,
        "start_frac": start_frac,
        "end_frac": end_frac,
        "score_decode": "x0_pred" if score_x0 else "final",
    })


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

    summary: list[dict[str, Any]] = []
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
            result = _run_sop_search_flux(
                args=args, ctx=ctx, reward_model=reward_model,
                prompt=prompt, embeds=embeds, seed=seed,
            )
            if save_entry:
                out_path = os.path.join(args.out_dir, f"{slug}_s{sample_i}_sop.png")
                result.image.save(out_path)
            prompt_samples.append({
                "seed": seed,
                "search_score": float(result.score),
                "baseline_score": float(result.score),
                "delta_score": 0.0,
                "actions": [[int(v), float(g)] for v, g in result.actions],
                "artifacts_saved": bool(save_entry),
                "diagnostics": result.diagnostics,
            })
            del result
            gc.collect()
            if ctx.device.startswith("cuda"):
                torch.cuda.empty_cache()

        summary.append({"slug": slug, "prompt": prompt, "search_method": "sop", "samples": prompt_samples})
        del embeds
        gc.collect()
        if ctx.device.startswith("cuda"):
            torch.cuda.empty_cache()

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    rows = [s for entry in summary for s in entry["samples"]]
    search_vals = [float(r["search_score"]) for r in rows]
    aggregate = {
        "model_id": args.model_id,
        "search_method": "sop",
        "n_prompts": len(prompts),
        "n_samples": int(args.n_samples),
        "mean_search_score": float(sum(search_vals) / len(search_vals)) if search_vals else None,
    }
    with open(os.path.join(args.out_dir, "aggregate_summary.json"), "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
    print(f"Outputs: {os.path.abspath(args.out_dir)}")


# ── Module wiring ───────────────────────────────────────────────────────────


def _parse_with_sop_flags(argv: list[str] | None) -> argparse.Namespace:
    extra = argparse.ArgumentParser(add_help=False)
    add_sop_args(extra)
    source = list(argv) if argv is not None else list(sys.argv[1:])
    sop_args, remaining = extra.parse_known_args(source)
    args = base.parse_args(remaining)
    for k, v in vars(sop_args).items():
        setattr(args, k, v)
    if not hasattr(args, "x0_sampler"):
        setattr(args, "x0_sampler", False)
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_with_sop_flags(argv)
    run(args)


if __name__ == "__main__":
    main()
