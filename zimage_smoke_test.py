from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from reward_unified import UnifiedRewardScorer
from zimage_tts import build_embedding_bank, get_dtype, run_with_schedule, save_comparison


DEFAULT_PROMPTS = [
    "A close-up portrait of a red panda wearing a tiny backpack, natural light, detailed fur",
    "A futuristic city street in rain at night, reflections, cinematic composition",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick smoke test for ZImage TTS pipeline.")
    parser.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--prompt_file", default=None, help="Optional txt file with prompts (one per line).")
    parser.add_argument("--prompts", nargs="*", default=None, help="Inline prompts. If omitted, built-in 2 prompts are used.")
    parser.add_argument("--outdir", default="./zimage_smoke_out")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--baseline_cfg", type=float, default=0.0)
    parser.add_argument("--probe_cfg", type=float, default=1.0)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--attention", choices=["", "flash", "_flash_3"], default="")
    parser.add_argument("--compile_transformer", action="store_true")
    parser.add_argument("--reward_backend", choices=["auto", "imagereward", "hpsv2", "unified"], default="auto")
    parser.add_argument("--reward_model", default="ImageReward-v1.0")
    parser.add_argument("--reward_weights", nargs=2, type=float, default=[1.0, 1.0])
    return parser.parse_args()


def _load_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompt_file:
        prompts = [line.strip() for line in open(args.prompt_file, encoding="utf-8") if line.strip()]
    elif args.prompts:
        prompts = args.prompts
    else:
        prompts = DEFAULT_PROMPTS
    if not prompts:
        raise RuntimeError("No prompts provided.")
    return prompts[:4]


def _font(size: int = 16):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def add_label(img: Image.Image, text: str, pad: int = 38) -> Image.Image:
    w, h = img.size
    canvas = Image.new("RGB", (w, h + pad), (255, 255, 255))
    canvas.paste(img, (0, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), text, fill=(0, 0, 0), font=_font(14))
    return canvas


def make_grid(images: List[Image.Image], cols: int = 2) -> Image.Image:
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, img in enumerate(images):
        canvas.paste(img, ((i % cols) * w, (i // cols) * h))
    return canvas


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from diffusers import ZImagePipeline

    device = "cuda"
    dtype = get_dtype(args.dtype)
    print(f"Loading pipeline: {args.model}")
    pipe = ZImagePipeline.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=False).to(device)
    if args.attention:
        pipe.transformer.set_attention_backend(args.attention)
    if args.compile_transformer:
        pipe.transformer.compile()

    scorer = UnifiedRewardScorer(
        device=device,
        backend=args.reward_backend,
        image_reward_model=args.reward_model,
        unified_weights=(float(args.reward_weights[0]), float(args.reward_weights[1])),
    )
    print(f"Reward: {scorer.describe()}")

    prompts = _load_prompts(args)
    summary_rows: List[Dict[str, Any]] = []
    grid_items: List[Image.Image] = []

    for pidx, prompt in enumerate(prompts):
        slug = f"p{pidx:02d}"
        print(f"\n[{slug}] {prompt}")

        # 2 lightweight variants: original + a tiny style perturbation
        variants = [
            prompt,
            f"{prompt}, soft cinematic rim light, slight depth of field",
        ]
        with open(os.path.join(args.outdir, f"{slug}_variants.txt"), "w", encoding="utf-8") as f:
            for vi, text in enumerate(variants):
                f.write(f"v{vi}: {text}\n")

        bank = build_embedding_bank(
            pipe=pipe,
            variants=variants,
            negative_prompt=args.negative_prompt,
            max_sequence_length=args.max_sequence_length,
        )

        base_schedule: List[Tuple[int, float]] = [(0, float(args.baseline_cfg)) for _ in range(args.steps)]
        probe_schedule: List[Tuple[int, float]] = []
        for s in range(args.steps):
            if s % 2 == 0:
                probe_schedule.append((0, float(args.baseline_cfg)))
            else:
                probe_schedule.append((1, float(args.probe_cfg)))

        base = run_with_schedule(
            args=args,
            pipe=pipe,
            reward_scorer=scorer,
            prompt_for_reward=prompt,
            bank=bank,
            schedule=base_schedule,
            seed=args.seed + pidx,
            capture_intermediates=True,
            save_intermediate_dir=os.path.join(args.outdir, f"{slug}_base_steps"),
            score_intermediates=False,
        )
        probe = run_with_schedule(
            args=args,
            pipe=pipe,
            reward_scorer=scorer,
            prompt_for_reward=prompt,
            bank=bank,
            schedule=probe_schedule,
            seed=args.seed + pidx,
            capture_intermediates=True,
            save_intermediate_dir=os.path.join(args.outdir, f"{slug}_probe_steps"),
            score_intermediates=False,
        )

        base_path = os.path.join(args.outdir, f"{slug}_base.png")
        probe_path = os.path.join(args.outdir, f"{slug}_probe.png")
        comp_path = os.path.join(args.outdir, f"{slug}_comparison.png")
        base.image.save(base_path)
        probe.image.save(probe_path)
        save_comparison(comp_path, base.image, probe.image, base.score, probe.score, probe_schedule)

        base_steps_logged = len(base.intermediate_records)
        probe_steps_logged = len(probe.intermediate_records)
        ok = (base_steps_logged == args.steps) and (probe_steps_logged == args.steps)

        row = {
            "slug": slug,
            "prompt": prompt,
            "seed": args.seed + pidx,
            "base_score": float(base.score),
            "probe_score": float(probe.score),
            "delta": float(probe.score - base.score),
            "base_steps_logged": base_steps_logged,
            "probe_steps_logged": probe_steps_logged,
            "expected_steps": int(args.steps),
            "status": "PASS" if ok else "WARN",
        }
        summary_rows.append(row)
        print(
            f"  base={base.score:.4f} probe={probe.score:.4f} delta={probe.score-base.score:+.4f} "
            f"steps(base/probe)={base_steps_logged}/{probe_steps_logged} status={row['status']}"
        )

        grid_items.append(add_label(base.image, f"{slug} base cfg={args.baseline_cfg:.2f} IR={base.score:.3f}"))
        grid_items.append(add_label(probe.image, f"{slug} probe alt-schedule IR={probe.score:.3f}"))

    grid = make_grid(grid_items, cols=2)
    grid.save(os.path.join(args.outdir, "smoke_grid.png"))

    with open(os.path.join(args.outdir, "smoke_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    with open(os.path.join(args.outdir, "smoke_summary.txt"), "w", encoding="utf-8") as f:
        f.write("slug\tstatus\tbase_score\tprobe_score\tdelta\tbase_steps\tprobe_steps\n")
        for row in summary_rows:
            f.write(
                f"{row['slug']}\t{row['status']}\t{row['base_score']:.6f}\t{row['probe_score']:.6f}\t"
                f"{row['delta']:+.6f}\t{row['base_steps_logged']}\t{row['probe_steps_logged']}\n"
            )

    n_warn = sum(1 for row in summary_rows if row["status"] != "PASS")
    print(f"\nDone: {len(summary_rows)} prompts, warnings={n_warn}")
    print(f"Outputs: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
