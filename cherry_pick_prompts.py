#!/usr/bin/env python3
"""Download HPSv2 + DrawBench, sample N unique prompts per backend.

Output: <out_dir>/backend_<name>.txt — one prompt per line.
Different sampling seed per backend ensures no prompt overlap.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

BACKENDS = ["sid", "senseflow_large", "sd35_base", "flux_schnell"]


def _load_hpsv2_prompts() -> list[str]:
    """Pull HPSv2 prompts from HuggingFace (xswu/HPSv2 — all 4 splits)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("pip install datasets") from e
    prompts: list[str] = []
    # Splits: photo, anime, concept-art, paintings.
    for split in ["test"]:
        try:
            ds = load_dataset("xswu/HPSv2", split=split)
            for row in ds:
                p = row.get("prompt") or row.get("text") or ""
                if isinstance(p, str) and p.strip():
                    prompts.append(p.strip())
        except Exception as exc:
            print(f"[cherry] warn: failed to load HPSv2 split={split}: {exc}")
    return prompts


def _load_drawbench_prompts() -> list[str]:
    """Pull DrawBench prompts (sayakpaul/drawbench)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("pip install datasets") from e
    prompts: list[str] = []
    try:
        ds = load_dataset("sayakpaul/drawbench", split="train")
        for row in ds:
            p = row.get("Prompts") or row.get("prompt") or row.get("text") or ""
            if isinstance(p, str) and p.strip():
                prompts.append(p.strip())
    except Exception as exc:
        print(f"[cherry] warn: failed to load DrawBench: {exc}")
    return prompts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_prompts", type=int, default=100)
    p.add_argument("--out_dir", default="./cherry_pick_prompts")
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--backends", nargs="+", default=BACKENDS)
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[cherry] loading HPSv2 prompts ...")
    hpsv2 = _load_hpsv2_prompts()
    print(f"[cherry]   {len(hpsv2)} HPSv2 prompts")
    print("[cherry] loading DrawBench prompts ...")
    drawbench = _load_drawbench_prompts()
    print(f"[cherry]   {len(drawbench)} DrawBench prompts")

    union = list({*hpsv2, *drawbench})  # de-dupe across sources
    union.sort()  # deterministic order before sampling
    if len(union) < args.n_prompts:
        raise RuntimeError(f"Union has only {len(union)} prompts; need {args.n_prompts}")
    print(f"[cherry] union (deduped): {len(union)} prompts")

    for i, backend in enumerate(args.backends):
        rng = random.Random(args.base_seed + 7 * i + 1)
        sample = rng.sample(union, args.n_prompts)
        out_path = out_dir / f"backend_{backend}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            for s in sample:
                # CSV-safe: strip newlines.
                f.write(s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ") + "\n")
        print(f"[cherry] wrote {len(sample)} prompts to {out_path}")


if __name__ == "__main__":
    main()
