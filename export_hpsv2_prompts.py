#!/usr/bin/env python3
"""
Export official HPSv2 benchmark prompts to text files.

Examples:
  python export_hpsv2_prompts.py --out_dir /data/ygu
  python export_hpsv2_prompts.py --style photo --out_dir /data/ygu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export HPSv2 benchmark prompts.")
    p.add_argument(
        "--style",
        default="all",
        choices=["all", "anime", "concept-art", "paintings", "photo"],
        help="Prompt subset to export. 'all' exports merged and per-style files.",
    )
    p.add_argument(
        "--out_dir",
        default="/data/ygu",
        help="Directory to save output prompt files.",
    )
    p.add_argument(
        "--dedupe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="De-duplicate prompts while preserving first-seen order.",
    )
    return p.parse_args()


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _write_txt(path: Path, prompts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(prompt.strip() + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import hpsv2
    except Exception as exc:
        raise RuntimeError(
            "Failed to import hpsv2. Install first with: pip install hpsv2"
        ) from exc

    by_style = hpsv2.benchmark_prompts(args.style)
    if not isinstance(by_style, dict) or not by_style:
        raise RuntimeError(f"Unexpected prompt payload for style={args.style!r}")

    # Normalize and optionally dedupe each style bucket.
    normalized: dict[str, list[str]] = {}
    for style_name, prompts in by_style.items():
        cur = [str(p).strip() for p in prompts if str(p).strip()]
        if args.dedupe:
            cur = _dedupe_keep_order(cur)
        normalized[str(style_name)] = cur

    if args.style == "all":
        merged: list[str] = []
        for style_name in ("anime", "concept-art", "paintings", "photo"):
            if style_name in normalized:
                merged.extend(normalized[style_name])
        if args.dedupe:
            merged = _dedupe_keep_order(merged)

        merged_path = out_dir / "hpsv2_prompts.txt"
        _write_txt(merged_path, merged)
        print(f"Wrote merged prompts: {merged_path} ({len(merged)})")

        for style_name, prompts in normalized.items():
            style_path = out_dir / f"hpsv2_prompts_{style_name}.txt"
            _write_txt(style_path, prompts)
            print(f"Wrote style prompts:  {style_path} ({len(prompts)})")
    else:
        prompts = normalized.get(args.style, [])
        out_path = out_dir / f"hpsv2_prompts_{args.style}.txt"
        _write_txt(out_path, prompts)
        print(f"Wrote prompts: {out_path} ({len(prompts)})")

    counts_path = out_dir / "hpsv2_prompt_counts.json"
    with counts_path.open("w", encoding="utf-8") as f:
        json.dump({k: len(v) for k, v in normalized.items()}, f, indent=2)
    print(f"Wrote counts:  {counts_path}")


if __name__ == "__main__":
    main()
