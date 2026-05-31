#!/usr/bin/env python3
"""Fetch DPG-Bench prompts and write a plain `one-prompt-per-line` text file
the rest of the pipeline (cherry_pick_select.py, hpsv2_*_ddp_suite.sh, ...)
can consume directly.

DPG-Bench is from the ELLA paper -- 1065 dense, compositional prompts that
stress long prompt-following.  We try multiple HuggingFace datasets in order
because hosting moves around, then fall back to the GitHub raw JSON.

Usage:
    python fetch_dpg_bench.py --out_file dpg_bench_prompts.txt
    python fetch_dpg_bench.py --out_file dpg_bench_prompts.txt --n_prompts 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

# HF dataset candidates (try in order).  None of these are official ELLA — the
# DPG-Bench data has been redistributed by the community.
_HF_CANDIDATES = [
    ("Geonmo/DPG-Bench", None),
    ("xhinker/DPG-Bench", None),
    ("alfredplpl/DPG-bench", None),
    ("openMUSE/dpg-bench", None),
]

# Direct raw URLs (GitHub mirrors of the ELLA repo).
_RAW_URL_CANDIDATES = [
    "https://raw.githubusercontent.com/TencentQQGYLab/ELLA/main/dpg_bench/prompts.csv",
    "https://raw.githubusercontent.com/TencentQQGYLab/ELLA/main/dpg_bench/dpg_bench.json",
]


def _try_hf() -> list[str]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        print(f"  [hf] datasets package unavailable: {exc}")
        return []
    for repo, split in _HF_CANDIDATES:
        try:
            print(f"  [hf] trying {repo}{':' + split if split else ''} ...")
            ds = load_dataset(repo, split=split or "train", trust_remote_code=False)
        except Exception as exc:
            print(f"  [hf]   skip ({type(exc).__name__}: {exc})")
            continue
        # Find the prompt column heuristically
        cols = list(ds.column_names) if hasattr(ds, "column_names") else list(ds.features.keys())
        prompt_col = None
        for c in ("prompt", "text", "caption", "Prompt", "input"):
            if c in cols:
                prompt_col = c
                break
        if prompt_col is None:
            print(f"  [hf]   no prompt column in {cols}; skip")
            continue
        prompts = [str(x).strip() for x in ds[prompt_col] if x]
        print(f"  [hf] loaded {len(prompts)} prompts from {repo} (col={prompt_col})")
        return prompts
    return []


def _try_github_raw() -> list[str]:
    for url in _RAW_URL_CANDIDATES:
        try:
            print(f"  [raw] trying {url} ...")
            with urllib.request.urlopen(url, timeout=20) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
        except Exception as exc:
            print(f"  [raw]   skip ({type(exc).__name__}: {exc})")
            continue
        prompts: list[str] = []
        if url.endswith(".json"):
            try:
                obj = json.loads(body)
                if isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, str):
                            prompts.append(item.strip())
                        elif isinstance(item, dict):
                            for key in ("prompt", "text", "caption"):
                                if key in item and isinstance(item[key], str):
                                    prompts.append(item[key].strip())
                                    break
                elif isinstance(obj, dict):
                    for v in obj.values():
                        if isinstance(v, str):
                            prompts.append(v.strip())
                        elif isinstance(v, dict):
                            for key in ("prompt", "text"):
                                if key in v:
                                    prompts.append(str(v[key]).strip())
                                    break
            except Exception as exc:
                print(f"  [raw]   JSON parse failed: {exc}")
                continue
        else:
            # CSV fallback: first column or column named "prompt"
            import csv
            from io import StringIO
            try:
                reader = csv.DictReader(StringIO(body))
                if reader.fieldnames:
                    col = next((c for c in ("prompt", "Prompt", "text", "caption")
                                if c in reader.fieldnames), reader.fieldnames[0])
                    for row in reader:
                        v = row.get(col)
                        if v:
                            prompts.append(str(v).strip())
            except Exception as exc:
                print(f"  [raw]   CSV parse failed: {exc}")
                continue
        prompts = [p for p in prompts if p]
        if prompts:
            print(f"  [raw] loaded {len(prompts)} prompts from {url}")
            return prompts
    return []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_file", default="dpg_bench_prompts.txt", type=Path,
                   help="Output text file (one prompt per line).")
    p.add_argument("--n_prompts", type=int, default=-1,
                   help="Limit to first N prompts (-1 = all).")
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle before truncation.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    prompts = _try_hf()
    if not prompts:
        prompts = _try_github_raw()
    if not prompts:
        print("[FATAL] could not fetch DPG-Bench from any known source.")
        print("        Try manually: pip install datasets; then huggingface-cli login")
        sys.exit(1)

    # De-dup while keeping order.
    seen: set = set()
    deduped: list[str] = []
    for p_ in prompts:
        if p_ and p_ not in seen:
            seen.add(p_)
            deduped.append(p_)
    prompts = deduped
    print(f"  unique prompts: {len(prompts)}")

    if args.shuffle:
        import random
        rng = random.Random(args.seed)
        rng.shuffle(prompts)

    if args.n_prompts > 0 and len(prompts) > args.n_prompts:
        prompts = prompts[: args.n_prompts]

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        for line in prompts:
            f.write(line.replace("\n", " ").strip() + "\n")
    print(f"  wrote {len(prompts)} prompts -> {args.out_file}")
    print(f"  preview (first 3):")
    for line in prompts[:3]:
        print(f"    {line[:120]}")


if __name__ == "__main__":
    main()
