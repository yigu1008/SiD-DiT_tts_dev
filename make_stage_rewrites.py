#!/usr/bin/env python3
"""Heuristic stage-split prompt rewrites — no Qwen required.

Splits the denoising trajectory into N stages (default 3: early / mid / late)
and produces a different suffix per stage. The suffix steers each MCTS
variant_idx toward what matters most at that stage of denoising:

  - early stage (high noise):  composition / layout
  - mid stage:                  subject + lighting
  - late stage (low noise):     fine textures + sharpness

Output is a JSON in the same format as the Qwen rewrites cache that the
SD3.5 / FLUX samplers consume via --rewrites_file. The sampler then exposes
each stage suffix as a distinct prompt variant_idx, which MCTS can branch on
per key-step.

Usage:
    python make_stage_rewrites.py \
        --prompt_file prompts.txt \
        --out_file stage_rewrites.json \
        --mode 3level

Modes:
    none       : 1 variant (original prompt only)
    2level     : 2 variants (early/composition, late/details)
    3level     : 3 variants (composition, subject, details)
    4level     : 4 variants (composition, subject, lighting, details)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# Suffix templates per stage (appended to the original prompt).
# Empty string = identity (use original prompt as-is).
SUFFIX_PRESETS = {
    "none": [""],
    "2level": [
        ", clear composition, balanced lighting",
        ", sharp details, fine textures, high resolution",
    ],
    "3level": [
        ", clear composition, wide framing, establishing shot",
        ", focused subject, balanced lighting, detailed scene",
        ", sharp fine textures, crisp details, photographic quality",
    ],
    "4level": [
        ", wide composition, establishing shot",
        ", clear subject, focused framing",
        ", dramatic lighting, balanced exposure",
        ", sharp details, fine textures, high resolution",
    ],
    # Style-axis variants (orthogonal axis you can mix in if you want).
    "stylistic": [
        ", photorealistic, cinematic lighting",
        ", artistic, painterly, soft focus",
        ", ultra-detailed, 4k, professional photography",
    ],
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt_file", required=True,
                   help="One prompt per line.")
    p.add_argument("--out_file", required=True,
                   help="Output JSON. Format: {prompt: [variant_0, variant_1, ...]}.")
    p.add_argument("--mode", default="3level",
                   choices=list(SUFFIX_PRESETS.keys()))
    p.add_argument("--include_original_first", action="store_true",
                   help="Prepend the unmodified prompt as variant_0 (shifts stage variants to 1..N).")
    args = p.parse_args()

    suffixes = SUFFIX_PRESETS[args.mode]
    if args.include_original_first:
        suffixes = [""] + suffixes

    prompts = []
    with open(args.prompt_file) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)

    out: dict[str, list[str]] = {}
    for prompt in prompts:
        variants = [(prompt + s).strip() for s in suffixes]
        # De-dup but keep order (the empty suffix variant is identical to prompt).
        seen = set()
        deduped = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                deduped.append(v)
        out[prompt] = deduped

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[stage-rewrites] wrote {len(out)} prompts × {len(suffixes)} variants → {args.out_file}")
    print(f"[stage-rewrites] mode={args.mode}  suffixes:")
    for i, s in enumerate(suffixes):
        preview = s.strip(", ") or "<original>"
        print(f"  stage {i}: {preview}")


if __name__ == "__main__":
    main()
