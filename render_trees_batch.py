#!/usr/bin/env python3
"""Render a batch of ActDiff tree visualizations from an existing run's outputs.

No re-run needed — reads `rank_*.jsonl` per-prompt diagnostics that bon_mcts
runs already wrote to disk.

Usage:
    # Render trees for prompts 0..49 from a FLUX run:
    python render_trees_batch.py \
        --run_root /mnt/data/v-yigu/all_in_one/flux-newcfg/composite/flux_schnell/seed42 \
        --method bon_mcts \
        --prompt_range 0:50 \
        --out_dir figures/raw/flux_trees

    # Or just a few specific prompts:
    python render_trees_batch.py \
        --run_root /mnt/data/v-yigu/all_method_ablations/step-reward-test \
        --method bon_mcts \
        --prompts 0 5 10 17 42

Output: PNG + JSON pair per prompt in --out_dir.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_prompt_list(args) -> list[int]:
    out: list[int] = []
    if args.prompt_range:
        start, end = args.prompt_range.split(":")
        out.extend(range(int(start), int(end)))
    if args.prompts:
        out.extend(int(p) for p in args.prompts)
    seen = set()
    deduped = []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_root", required=True, type=Path,
                   help="Run root containing <method>/run_*/rank_*.jsonl.")
    p.add_argument("--method", default="bon_mcts",
                   help="Which method subdir to read.")
    p.add_argument("--prompt_range", default=None,
                   help="Inclusive-exclusive range, e.g. 0:50 -> prompts 0..49.")
    p.add_argument("--prompts", nargs="+", default=None,
                   help="Explicit list of prompt indices.")
    p.add_argument("--out_dir", default=Path("figures/raw/trees_batch"), type=Path)
    p.add_argument("--title_prefix", default="ActDiff trace",
                   help="Prepended to per-prompt titles.")
    p.add_argument("--script", default="plot_actdiff_tree.py",
                   help="Path to the per-prompt renderer.")
    args = p.parse_args()

    if not args.prompt_range and not args.prompts:
        args.prompt_range = "0:10"  # safe default

    prompts = _parse_prompt_list(args)
    if not prompts:
        raise SystemExit("No prompts specified.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"# rendering {len(prompts)} trees from {args.run_root}")
    print(f"# method = {args.method}")
    print(f"# out_dir = {args.out_dir}\n")

    n_ok = 0
    n_skip = 0
    for pi in prompts:
        title = f"{args.title_prefix} -- {args.method} -- prompt #{pi}"
        cmd = [
            sys.executable, args.script,
            "--mode", "real",
            "--run_root", str(args.run_root),
            "--method", args.method,
            "--prompt_index", str(pi),
            "--out_dir", str(args.out_dir),
            "--title", title,
        ]
        try:
            res = subprocess.run(cmd, check=False, capture_output=True, text=True)
        except Exception as exc:
            print(f"  prompt {pi:4d}  [error invoking]: {exc}")
            n_skip += 1
            continue
        if res.returncode != 0:
            print(f"  prompt {pi:4d}  [rc={res.returncode}]  {res.stderr.strip()[:100]}")
            n_skip += 1
            continue
        # Heuristic: detect fallback-to-schematic from stdout.
        if "falling back to schematic" in (res.stdout + res.stderr):
            print(f"  prompt {pi:4d}  [no diagnostics — schematic fallback]")
            n_skip += 1
        else:
            n_ok += 1
            if n_ok % 5 == 0 or n_ok == 1:
                print(f"  prompt {pi:4d}  OK")
    print(f"\n# rendered {n_ok} / {len(prompts)} ({n_skip} skipped/fallback)")
    print(f"# browse: {args.out_dir}")


if __name__ == "__main__":
    main()
