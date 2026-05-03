#!/usr/bin/env python3
"""Compare MCTS-hyperparam ablation cells side-by-side.

Reads each cell's `bon_mcts/aggregate_ddp.json` + `best_images_multi_reward_aggregate.json`
and prints a table per backend of (cell, mean_search, eval_imagereward, eval_hpsv3,
mean_delta_vs_baseline). Also computes the "winner" cell per backend.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

CELLS = [
    "default", "n_sims_15", "n_sims_60", "n_sims_120",
    "topk_1", "topk_4", "ucb_c_0.5", "ucb_c_2.0",
]
BACKENDS = ["sid", "senseflow_large", "sd35_base"]


def _aggregate_cell(cell_root: Path) -> dict | None:
    """Average mean_search etc. across all seed*/ inside this cell."""
    rows = []
    for seed_dir in sorted(cell_root.glob("seed*/run_*/bon_mcts")):
        agg = seed_dir / "aggregate_ddp.json"
        if not agg.exists():
            continue
        try:
            payload = json.loads(agg.read_text())
        except json.JSONDecodeError:
            continue

        eval_path = seed_dir / "best_images_multi_reward_aggregate.json"
        eval_means = {}
        if eval_path.exists():
            try:
                stats = json.loads(eval_path.read_text()).get("backend_stats", {}) or {}
                for k, v in stats.items():
                    if isinstance(v, dict) and v.get("mean") is not None:
                        eval_means[k] = float(v["mean"])
            except json.JSONDecodeError:
                pass

        rows.append({
            "mean_search": float(payload.get("mean_search_score") or 0.0),
            "mean_baseline": float(payload.get("mean_baseline_score") or 0.0),
            "mean_delta": float(payload.get("mean_delta_score") or 0.0),
            "n": int(payload.get("num_samples") or 0),
            **{f"eval_{k}": v for k, v in eval_means.items()},
        })
    if not rows:
        return None

    avg = {}
    for k in rows[0]:
        vals = [r[k] for r in rows if k in r]
        if not vals:
            continue
        if k == "n":
            avg[k] = sum(vals)
        else:
            avg[k] = sum(vals) / len(vals)
    avg["n_seeds"] = len(rows)
    return avg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="e.g. /mnt/data/v-yigu/mcts_param/<run_tag>")
    p.add_argument("--cells", nargs="+", default=CELLS)
    p.add_argument("--backends", nargs="+", default=BACKENDS)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()

    print(f"{'backend':<18} {'cell':<14} {'n_seeds':>7} {'n_samples':>9} "
          f"{'mean_search':>12} {'eval_IR':>8} {'eval_HPSv3':>10} "
          f"{'Δ_vs_base':>10}")
    print("─" * 92)

    by_backend: dict[str, list[dict]] = defaultdict(list)

    for backend in args.backends:
        backend_root = root / backend
        if not backend_root.exists():
            print(f"{backend:<18} (missing dir)")
            continue

        for cell in args.cells:
            cell_root = backend_root / cell
            if not cell_root.exists():
                print(f"{backend:<18} {cell:<14} (missing)")
                continue
            agg = _aggregate_cell(cell_root)
            if agg is None:
                print(f"{backend:<18} {cell:<14} (no aggregate_ddp.json found)")
                continue
            row = {"backend": backend, "cell": cell, **agg}
            by_backend[backend].append(row)
            print(
                f"{backend:<18} {cell:<14} {agg['n_seeds']:>7} {agg.get('n', 0):>9} "
                f"{agg['mean_search']:>12.4f} "
                f"{agg.get('eval_imagereward', 0.0):>8.4f} "
                f"{agg.get('eval_hpsv3', 0.0):>10.4f} "
                f"{agg['mean_delta']:>+10.4f}"
            )
        print()

    print("Winners per backend (highest mean_search):")
    for backend, rows in by_backend.items():
        if not rows:
            continue
        best = max(rows, key=lambda r: r["mean_search"])
        worst = min(rows, key=lambda r: r["mean_search"])
        spread = best["mean_search"] - worst["mean_search"]
        print(f"  {backend:<18} best={best['cell']:<14} ({best['mean_search']:.4f})  "
              f"worst={worst['cell']:<14} ({worst['mean_search']:.4f})  "
              f"spread={spread:.4f}")


if __name__ == "__main__":
    main()
