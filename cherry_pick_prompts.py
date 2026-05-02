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


# HPSv2 prompts: the official author repo `xswu/HPSv2` is a *code* repo, not a
# dataset. The HPSv2 prompts ship inside the pi-Flow paper's mirror at
# `Lakonik/t2i-prompts-hpsv2` (test split, 3200 prompts, column `prompt`).
_HPSV2_CANDIDATES = [
    ("Lakonik/t2i-prompts-hpsv2", "test"),
]
# DrawBench: `sayakpaul/drawbench` (CSV, train split, column `Prompts`) is the
# canonical mirror; `shunk031/DrawBench` is an alternative.
_DRAWBENCH_CANDIDATES = [
    ("sayakpaul/drawbench", "train"),
    ("shunk031/DrawBench", "train"),
]


def _load_hf_prompts(candidates: list[tuple[str, str]], label: str) -> list[str]:
    """Try multiple HF dataset names; return prompts from the first that works."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("pip install datasets") from e
    for repo, split in candidates:
        try:
            ds = load_dataset(repo, split=split)
        except Exception as exc:
            print(f"[cherry] {label}: {repo}:{split} unavailable ({type(exc).__name__})")
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
            print(f"[cherry] {label}: loaded {len(prompts)} from {repo}:{split}")
            return prompts
    return []


def _load_local_fallback(repo_root: Path) -> list[str]:
    """If HF is unreachable, fall back to local hpsv2_subset.txt in the repo."""
    candidates = [
        repo_root / "hpsv2_subset.txt",
        repo_root.parent / "hpsv2_subset.txt",
    ]
    for path in candidates:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
            if prompts:
                print(f"[cherry] local fallback: loaded {len(prompts)} prompts from {path}")
                return prompts
    return []


def _load_hpsv2_prompts() -> list[str]:
    return _load_hf_prompts(_HPSV2_CANDIDATES, "HPSv2")


def _load_drawbench_prompts() -> list[str]:
    return _load_hf_prompts(_DRAWBENCH_CANDIDATES, "DrawBench")


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
    if len(union) < args.n_prompts:
        # HF unreachable or returned too few — fall back to local subset.
        repo_root = Path(__file__).resolve().parent
        local = _load_local_fallback(repo_root)
        union = list({*union, *local})
    union.sort()  # deterministic order before sampling
    if len(union) < args.n_prompts:
        raise RuntimeError(
            f"Union has only {len(union)} prompts; need {args.n_prompts}. "
            f"HF datasets unreachable AND local hpsv2_subset.txt missing/short."
        )
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
