#!/usr/bin/env python3
"""
Sweep guidance scale on SenseFlow to visualise its effect.

Usage:
  # Default: SenseFlow-Large, one prompt, CFG 0.0-5.0
  python test_senseflow_cfg_sweep.py

  # Custom prompt
  python test_senseflow_cfg_sweep.py --prompt "a cat wearing a hat"

  # Custom CFG range
  python test_senseflow_cfg_sweep.py --cfg_scales 0.0 0.5 1.0 2.0 3.0 5.0 7.0

  # SenseFlow-Medium
  python test_senseflow_cfg_sweep.py --backend senseflow_medium

  # Multiple prompts
  python test_senseflow_cfg_sweep.py --prompt_file hpsv2_subset.txt --end_index 3

  # Compare against SD3.5 base (non-distilled)
  python test_senseflow_cfg_sweep.py --backend sd35_base --cfg_scales 0.0 1.0 3.5 4.5 7.0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SenseFlow guidance scale sweep")
    p.add_argument("--backend", choices=["senseflow_large", "senseflow_medium", "sid", "sd35_base"],
                    default="senseflow_large")
    p.add_argument("--prompt", default=None, help="Single prompt to test")
    p.add_argument("--prompt_file", default=None, help="File with prompts (one per line)")
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--end_index", type=int, default=1)
    p.add_argument("--cfg_scales", nargs="+", type=float,
                    default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
                    help="Guidance scales to sweep")
    p.add_argument("--steps", type=int, default=None, help="Override denoising steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--out_dir", default="./senseflow_cfg_sweep")
    p.add_argument("--device", default="cuda")
    p.add_argument("--make_grid", action="store_true", default=True,
                    help="Save a comparison grid image per prompt")
    p.add_argument("--no_grid", dest="make_grid", action="store_false")
    return p.parse_args()


def load_prompts(args) -> list[str]:
    if args.prompt:
        return [args.prompt]
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        end = args.end_index if args.end_index > 0 else len(lines)
        return lines[args.start_index:end]
    return ["A cat sitting on a windowsill watching the rain"]


def make_grid(images: list[Image.Image], labels: list[str], prompt: str) -> Image.Image:
    """Create a comparison grid with labels."""
    from PIL import ImageDraw, ImageFont

    n = len(images)
    thumb_w, thumb_h = images[0].size
    label_h = 40
    title_h = 50
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    grid_w = cols * thumb_w
    grid_h = title_h + rows * (thumb_h + label_h)

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Title
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    title = prompt[:100] + ("..." if len(prompt) > 100 else "")
    draw.text((10, 10), title, fill=(0, 0, 0), font=font)

    for i, (img, label) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        x = c * thumb_w
        y = title_h + r * (thumb_h + label_h)
        grid.paste(img, (x, y))
        draw.text((x + 10, y + thumb_h + 5), label, fill=(0, 0, 0), font=font_small)

    return grid


def main() -> None:
    args = parse_args()

    # Import sampling_unified_sd35 for pipeline loading and stepping
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sampling_unified_sd35 as su

    # Build a minimal args namespace for su functions
    su_args = su.parse_args([
        "--backend", args.backend,
        "--cfg_scales", *[str(c) for c in args.cfg_scales],
        "--baseline_cfg", str(args.cfg_scales[0]),
        "--n_variants", "1",
        "--steps", str(args.steps) if args.steps else str(4 if "senseflow" in args.backend or args.backend == "sid" else 28),
        "--width", str(args.width),
        "--height", str(args.height),
        "--reward_backend", "imagereward",
        "--correction_strengths", "0.0",
        "--no_qwen",
    ])

    prompts = load_prompts(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Backend:    {args.backend}")
    print(f"CFG scales: {args.cfg_scales}")
    print(f"Steps:      {su_args.steps}")
    print(f"Sigmas:     {getattr(su_args, 'sigmas', None)}")
    print(f"Euler:      {getattr(su_args, 'euler_sampler', False)}")
    print(f"x0_sampler: {getattr(su_args, 'x0_sampler', False)}")
    print(f"Prompts:    {len(prompts)}")
    print(f"Output:     {out_dir}")
    print()

    # Load pipeline
    print("Loading pipeline...")
    t0 = time.time()
    dtype = torch.bfloat16 if su_args.dtype == "bfloat16" else torch.float16
    ctx = su.load_pipeline(su_args, device=args.device, dtype=dtype)
    print(f"Pipeline loaded in {time.time() - t0:.1f}s")

    if torch.cuda.is_available() and "cuda" in args.device:
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Build schedule
    sched = su.step_schedule(su_args)
    use_euler = getattr(su_args, "euler_sampler", False)
    print(f"Schedule: {len(sched)} steps, euler={use_euler}")

    for pi, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Prompt {pi}: {prompt[:80]}")
        print(f"{'='*60}")

        # Encode prompt (single variant, no Qwen rewrite)
        emb = su.encode_variants(su_args, ctx, [prompt], dtype)

        # Generate initial latents (same seed for all CFG values → fair comparison)
        latents_init = su.make_latents(ctx, args.seed, args.height, args.width, dtype)

        images = []
        labels = []

        for cfg in args.cfg_scales:
            print(f"  cfg={cfg:.1f} ...", end=" ", flush=True)
            t1 = time.time()

            latents = latents_init.clone()
            dx = latents.clone()

            for step_idx, (t_flat, t_4d, dt) in enumerate(sched):
                flow = su.transformer_step(su_args, ctx, latents, emb, 0, t_flat, cfg)

                if use_euler:
                    # Euler ODE step: x_{t+1} = x_t + dt * flow
                    latents = latents + dt * flow
                else:
                    # SiD re-noising step
                    if su_args.x0_sampler:
                        dx = flow  # transformer predicts x0 directly
                    else:
                        dx = latents - t_4d * flow  # derive x0 from flow

                    # Re-noise for next step
                    if step_idx + 1 < len(sched):
                        _, next_t_4d, _ = sched[step_idx + 1]
                        noise = torch.randn_like(dx)
                        latents = (1.0 - next_t_4d) * dx + next_t_4d * noise

            # Decode
            if use_euler:
                decode_input = latents
            else:
                decode_input = dx

            img = su.decode_to_pil(ctx, decode_input)
            elapsed = time.time() - t1
            print(f"{elapsed:.1f}s")

            images.append(img)
            labels.append(f"cfg={cfg}")

            # Save individual image
            img_path = out_dir / f"prompt{pi}_cfg{cfg:.2f}.png"
            img.save(str(img_path))

        # Save comparison grid
        if args.make_grid and len(images) > 1:
            grid = make_grid(images, labels, prompt)
            grid_path = out_dir / f"prompt{pi}_grid.png"
            grid.save(str(grid_path))
            print(f"  Grid saved: {grid_path}")

    print(f"\nAll done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
