#!/usr/bin/env python3
"""
Build a deterministic smaller HPSv2 prompt subset from exported style files.

Input files expected in --in_dir:
  - hpsv2_prompts_anime.txt
  - hpsv2_prompts_concept-art.txt
  - hpsv2_prompts_paintings.txt
  - hpsv2_prompts_photo.txt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


DEFAULT_STYLES = ["anime", "concept-art", "paintings", "photo"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create fixed-seed HPSv2 mini prompt set.")
    p.add_argument("--in_dir", default="/data/ygu", help="Directory containing per-style HPSv2 prompt txt files.")
    p.add_argument("--styles", nargs="+", default=DEFAULT_STYLES, help="Styles to include.")
    p.add_argument("--per_style", type=int, default=64, help="Number of prompts to sample per style.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True, help="Shuffle merged prompts.")
    p.add_argument(
        "--allow_short_style",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow a style file with fewer than --per_style prompts.",
    )
    p.add_argument("--out_file", default="", help="Output txt path. Defaults to <in_dir>/hpsv2_prompts_mini_<N>_seed<seed>.txt")
    return p.parse_args()


def _load_prompts(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def main() -> None:
    args = parse_args()
    if int(args.per_style) <= 0:
        raise ValueError("--per_style must be > 0")
    if not args.styles:
        raise ValueError("--styles cannot be empty")

    in_dir = Path(args.in_dir).expanduser().resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"in_dir not found: {in_dir}")

    rng = random.Random(int(args.seed))
    merged: list[str] = []
    by_style_counts: dict[str, int] = {}

    for style in args.styles:
        style_path = in_dir / f"hpsv2_prompts_{style}.txt"
        if not style_path.exists():
            raise FileNotFoundError(f"Style prompt file not found: {style_path}")
        prompts = _load_prompts(style_path)
        if len(prompts) < int(args.per_style):
            if not args.allow_short_style:
                raise RuntimeError(
                    f"Style '{style}' has only {len(prompts)} prompts (< per_style={args.per_style}). "
                    "Use --allow_short_style or lower --per_style."
                )
            take_n = len(prompts)
        else:
            take_n = int(args.per_style)
        sampled = rng.sample(prompts, k=take_n)
        merged.extend(sampled)
        by_style_counts[style] = take_n

    if args.shuffle:
        rng.shuffle(merged)

    out_file = args.out_file.strip()
    if not out_file:
        out_file = str(in_dir / f"hpsv2_prompts_mini_{len(merged)}_seed{int(args.seed)}.txt")
    out_path = Path(out_file).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(merged) + "\n", encoding="utf-8")

    meta = {
        "seed": int(args.seed),
        "styles": [str(s) for s in args.styles],
        "per_style": int(args.per_style),
        "total_prompts": int(len(merged)),
        "by_style_counts": by_style_counts,
        "shuffle": bool(args.shuffle),
        "source_dir": str(in_dir),
        "output_file": str(out_path),
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote subset prompts: {out_path} ({len(merged)})")
    print(f"Wrote metadata:      {meta_path}")


if __name__ == "__main__":
    main()

