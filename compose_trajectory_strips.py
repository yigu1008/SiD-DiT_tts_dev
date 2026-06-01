#!/usr/bin/env python3
"""Compose each prompt's per-step x_0 images into a single horizontal strip.

Reads:
    <in_dir>/prompt_<NNNN>/step_0_cfg<X.XX>_v<V>.png
    <in_dir>/prompt_<NNNN>/step_1_cfg<X.XX>_v<V>.png
    ...
    <in_dir>/prompt_<NNNN>/final.png

Writes:
    <out_dir>/prompt_<NNNN>.png   <- horizontal film strip
    <out_dir>/_all_grid.png       <- (optional) one row per prompt, all in a grid

Each panel has a small caption header: step index + CFG/variant.

Usage:
    python compose_trajectory_strips.py \
        --in_dir  /data/ygu/runs/dpg_sid_inline_<ts>/step_images_inline \
        --out_dir /data/ygu/runs/dpg_sid_inline_<ts>/trajectory_strips \
        --prompts_file /data/ygu/runs/dpg_sid_inline_<ts>/_prompts/backend_sid.txt \
        --panel_size 384 --build_grid

The optional --prompts_file annotates each strip with the prompt text.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


_STEP_RE = re.compile(r"^step_(\d+)_cfg([0-9.]+)_v(\d+)\.png$")


def _font(size: int) -> ImageFont.ImageFont:
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(cand, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _list_steps(prompt_dir: Path) -> list[tuple[int, float, int, Path]]:
    out: list[tuple[int, float, int, Path]] = []
    for fp in prompt_dir.glob("step_*.png"):
        m = _STEP_RE.match(fp.name)
        if not m:
            continue
        out.append((int(m.group(1)), float(m.group(2)), int(m.group(3)), fp))
    out.sort(key=lambda r: r[0])
    return out


def _load_resized(fp: Path, size: int) -> Image.Image:
    img = Image.open(fp).convert("RGB")
    return img.resize((size, size), Image.LANCZOS)


def _draw_caption(panel: Image.Image, lines: list[str], font: ImageFont.ImageFont,
                  pad: int = 6, bg=(0, 0, 0), fg=(255, 255, 255)) -> Image.Image:
    """Draw a black caption band at the TOP of the panel."""
    line_h = font.getbbox("Ag")[3] + 2
    band_h = len(lines) * line_h + 2 * pad
    out = Image.new("RGB", (panel.width, panel.height + band_h), bg)
    out.paste(panel, (0, band_h))
    d = ImageDraw.Draw(out)
    for i, ln in enumerate(lines):
        d.text((pad, pad + i * line_h), ln, font=font, fill=fg)
    return out


def _compose_one(prompt_dir: Path, panel_size: int, prompt_text: str | None,
                 caption_font: ImageFont.ImageFont) -> Image.Image | None:
    steps = _list_steps(prompt_dir)
    final_fp = prompt_dir / "final.png"
    if not steps and not final_fp.exists():
        return None

    panels: list[Image.Image] = []
    for step_idx, cfg, v, fp in steps:
        img = _load_resized(fp, panel_size)
        panels.append(_draw_caption(img, [f"step {step_idx}", f"cfg={cfg:.2f}  v={v}"], caption_font))
    if final_fp.exists():
        img = _load_resized(final_fp, panel_size)
        panels.append(_draw_caption(img, ["final", ""], caption_font))

    band_h = panels[0].height - panel_size  # caption band height
    strip_w = panel_size * len(panels) + (len(panels) - 1) * 6  # 6px gap
    strip_h = panels[0].height
    if prompt_text:
        # Header band above all panels with wrapped prompt text.
        header_font = caption_font
        line_h = header_font.getbbox("Ag")[3] + 2
        wrap_cols = max(40, strip_w // (header_font.getbbox("M")[2] or 8))
        # naive wrap
        words = prompt_text.split()
        lines: list[str] = []
        cur = ""
        for w in words:
            if len(cur) + len(w) + 1 <= wrap_cols:
                cur = (cur + " " + w).strip()
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        lines = lines[:3]  # cap to 3 lines
        prompt_band_h = len(lines) * line_h + 12
    else:
        lines = []
        prompt_band_h = 0

    out = Image.new("RGB", (strip_w, strip_h + prompt_band_h), (255, 255, 255))
    if lines:
        d = ImageDraw.Draw(out)
        for i, ln in enumerate(lines):
            d.text((8, 6 + i * line_h), ln, font=caption_font, fill=(0, 0, 0))
    x = 0
    for p in panels:
        out.paste(p, (x, prompt_band_h))
        x += panel_size + 6
    return out


def _read_prompts(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_dir", required=True, type=Path,
                   help="Directory containing prompt_NNNN/ subfolders.")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--panel_size", type=int, default=384,
                   help="Per-panel side length in pixels (default 384).")
    p.add_argument("--prompts_file", type=Path, default=None,
                   help="Optional: plain text file with one prompt per line; "
                        "used to caption each strip with the prompt text.")
    p.add_argument("--build_grid", action="store_true",
                   help="Also stack all strips into one big PNG grid.")
    p.add_argument("--font_size", type=int, default=14)
    args = p.parse_args()

    if not args.in_dir.is_dir():
        raise SystemExit(f"in_dir not found: {args.in_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prompt_dirs = sorted([d for d in args.in_dir.glob("prompt_*") if d.is_dir()])
    if not prompt_dirs:
        raise SystemExit(f"no prompt_NNNN/ subdirs in {args.in_dir}")

    prompts: list[str] = []
    if args.prompts_file and args.prompts_file.is_file():
        prompts = _read_prompts(args.prompts_file)

    font = _font(args.font_size)
    strips: list[Image.Image] = []
    print(f"# composing {len(prompt_dirs)} trajectories from {args.in_dir}")
    for pd in prompt_dirs:
        m = re.match(r"prompt_(\d+)$", pd.name)
        if not m:
            continue
        idx = int(m.group(1))
        text = prompts[idx] if idx < len(prompts) else None
        strip = _compose_one(pd, args.panel_size, text, font)
        if strip is None:
            print(f"  prompt {idx}: skipped (no step images)")
            continue
        out_fp = args.out_dir / f"prompt_{idx:04d}.png"
        strip.save(out_fp)
        print(f"  prompt {idx:4d} -> {out_fp}  ({strip.width}x{strip.height})")
        strips.append(strip)

    if args.build_grid and strips:
        # Stack vertically; pad widths to the max strip width.
        max_w = max(s.width for s in strips)
        total_h = sum(s.height for s in strips) + (len(strips) - 1) * 4
        grid = Image.new("RGB", (max_w, total_h), (255, 255, 255))
        y = 0
        for s in strips:
            x = (max_w - s.width) // 2
            grid.paste(s, (x, y))
            y += s.height + 4
        grid_fp = args.out_dir / "_all_grid.png"
        grid.save(grid_fp)
        print(f"# grid -> {grid_fp}  ({grid.width}x{grid.height})")

    print(f"[done] strips under {args.out_dir}")


if __name__ == "__main__":
    main()
