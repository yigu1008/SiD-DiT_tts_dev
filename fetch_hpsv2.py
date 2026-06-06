#!/usr/bin/env python3
"""Fetch HPSv2 + DrawBench prompts and write a plain `one-prompt-per-line` text
file the rest of the pipeline (hpsv2_*_ddp_suite.sh, run_my_actdiff_grid.sh,
etc.) can consume directly.

Mirrors `cherry_pick_prompts.py` exactly:
  - HPSv2 prompts come from `Lakonik/t2i-prompts-hpsv2:test` (3200 prompts)
  - DrawBench from `sayakpaul/drawbench:train` (200 prompts)
  - De-duped union -> ~3372 unique prompts
  - First N (or random N with --shuffle) written to --out_file

Usage:
    python fetch_hpsv2.py --out_file prompts.txt
    python fetch_hpsv2.py --out_file prompts.txt --n_prompts 200 --shuffle
    python fetch_hpsv2.py --out_file prompts.txt --n_prompts 200 --hpsv2_only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


_HPSV2_CANDIDATES = [
    ("Lakonik/t2i-prompts-hpsv2", "test"),
]
_DRAWBENCH_CANDIDATES = [
    ("sayakpaul/drawbench", "train"),
    ("shunk031/DrawBench", "train"),
]


def _load_hf_prompts(candidates: list[tuple[str, str]], label: str) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("pip install datasets") from e
    for repo, split in candidates:
        try:
            ds = load_dataset(repo, split=split)
        except Exception as exc:
            print(f"[hpsv2] {label}: {repo}:{split} unavailable ({type(exc).__name__})")
            continue
        prompts: list[str] = []
        for row in ds:
            p = (
                row.get("prompt") or row.get("Prompts") or row.get("text") or
                row.get("caption") or ""
            )
            if isinstance(p, str) and p.strip():
                prompts.append(p.strip())
        if prompts:
            print(f"[hpsv2] {label}: loaded {len(prompts)} from {repo}:{split}")
            return prompts
    return []


def _load_local_fallback(repo_root: Path) -> list[str]:
    for path in (repo_root / "hpsv2_subset.txt", repo_root.parent / "hpsv2_subset.txt"):
        if path.exists():
            with path.open(encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
            if prompts:
                print(f"[hpsv2] local fallback: loaded {len(prompts)} from {path}")
                return prompts
    return []


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out_file", default="prompts.txt", type=Path,
                   help="Output text file (one prompt per line).")
    p.add_argument("--n_prompts", type=int, default=-1,
                   help="Limit to first N (-1 = all).")
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle before truncation (use --seed for reproducibility).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hpsv2_only", action="store_true",
                   help="Skip DrawBench, write only HPSv2 prompts.")
    p.add_argument("--drawbench_only", action="store_true",
                   help="Skip HPSv2, write only DrawBench prompts.")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent

    prompts: list[str] = []
    if not args.drawbench_only:
        print("[hpsv2] loading HPSv2 prompts ...")
        h = _load_hf_prompts(_HPSV2_CANDIDATES, "HPSv2")
        if not h:
            h = _load_local_fallback(repo_root)
        prompts.extend(h)
        print(f"[hpsv2]   {len(h)} HPSv2 prompts")

    if not args.hpsv2_only:
        print("[hpsv2] loading DrawBench prompts ...")
        d = _load_hf_prompts(_DRAWBENCH_CANDIDATES, "DrawBench")
        prompts.extend(d)
        print(f"[hpsv2]   {len(d)} DrawBench prompts")

    if not prompts:
        print("[FATAL] no prompts loaded from any source.", file=sys.stderr)
        sys.exit(1)

    # Dedupe (preserve order).
    seen: set = set()
    deduped: list[str] = []
    for s in prompts:
        if s and s not in seen:
            seen.add(s)
            deduped.append(s)
    prompts = deduped
    print(f"[hpsv2] union (deduped): {len(prompts)} prompts")

    if args.shuffle:
        import random
        rng = random.Random(args.seed)
        rng.shuffle(prompts)

    if args.n_prompts > 0 and len(prompts) > args.n_prompts:
        prompts = prompts[: args.n_prompts]

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        for line in prompts:
            f.write(line.replace("\n", " ").strip() + "\n")
    print(f"[hpsv2] wrote {len(prompts)} prompts -> {args.out_file}")
    print(f"[hpsv2] preview:")
    for line in prompts[:3]:
        print(f"  {line[:120]}")


if __name__ == "__main__":
    main()
