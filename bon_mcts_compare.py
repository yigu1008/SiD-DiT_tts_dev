#!/usr/bin/env python3
"""Compare bon_mcts config variants from the prescreen-ablation AMLT run.

Reads each config's `bon_mcts/aggregate_ddp.json` and emits a side-by-side
table. Also computes per-config:
  - prescreen-rank-1 vs winner_seed agreement rate (from per-prompt diagnostics)
  - mean prescreen score vs mean refine score

Usage:
    python bon_mcts_compare.py --root /mnt/data/v-yigu/bon_mcts_ablation/bon_mcts_ablation
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


CONFIG_VARIANTS = ["default", "prescreen_high_cfg", "wide_topk", "large_pool", "vanilla_refine"]
BACKENDS = ["sid", "senseflow_large"]


def _find_aggregate(variant_root: Path) -> Path | None:
    """Walk variant_root looking for bon_mcts/aggregate_ddp.json."""
    for p in variant_root.rglob("bon_mcts/aggregate_ddp.json"):
        return p
    return None


def _find_baseline_aggregate(variant_root: Path) -> Path | None:
    for p in variant_root.rglob("base/aggregate_ddp.json"):
        return p
    return None


def _per_prompt_diagnostics(variant_root: Path) -> tuple[int, int]:
    """Return (n_with_diagnostics, n_prescreen_top1_matches_winner)."""
    n_total = 0
    n_match = 0
    for log_path in variant_root.rglob("bon_mcts/logs/rank_*.jsonl"):
        with log_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                diag = (row.get("search_diagnostics") or {}).get("bon_mcts")
                if not diag:
                    continue
                ranked = diag.get("prescreen_ranked") or []
                if not ranked:
                    continue
                top_seed = int(ranked[0].get("seed", -1))
                winner = int(diag.get("winner_seed", -2))
                n_total += 1
                if top_seed == winner:
                    n_match += 1
    return n_total, n_match


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True,
                   help="e.g. /mnt/data/v-yigu/bon_mcts_ablation/bon_mcts_ablation")
    p.add_argument("--variants", nargs="+", default=CONFIG_VARIANTS)
    p.add_argument("--backends", nargs="+", default=BACKENDS)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()

    rows: list[dict] = []
    for backend in args.backends:
        baseline_score = None
        for variant in args.variants:
            variant_root = root / backend / variant
            if not variant_root.exists():
                rows.append({"backend": backend, "variant": variant, "status": "missing-dir"})
                continue
            agg = _find_aggregate(variant_root)
            if agg is None:
                rows.append({"backend": backend, "variant": variant, "status": "no-aggregate"})
                continue
            payload = json.loads(agg.read_text())
            mean_search = float(payload.get("mean_search_score") or 0.0)
            mean_baseline = float(payload.get("mean_baseline_score") or 0.0)
            mean_delta = float(payload.get("mean_delta_score") or 0.0)
            num_samples = int(payload.get("num_samples") or 0)
            if baseline_score is None:
                baseline_score = mean_baseline

            n_total, n_match = _per_prompt_diagnostics(variant_root)
            agreement = (n_match / n_total) if n_total > 0 else None

            rows.append({
                "backend": backend,
                "variant": variant,
                "n": num_samples,
                "mean_baseline": mean_baseline,
                "mean_search": mean_search,
                "mean_delta": mean_delta,
                "agreement_rate": agreement,
                "n_diag": n_total,
                "status": "ok",
            })

    # Print compact comparison table.
    print(f"{'backend':<18} {'variant':<22} {'n':>3} {'baseline':>9} {'search':>9} {'Δ':>8} {'agreement':>10}")
    print("─" * 84)
    for r in rows:
        if r.get("status") != "ok":
            print(f"{r['backend']:<18} {r['variant']:<22} {'':>3} {'':>9} {'':>9} {'':>8} {'':>10}  [{r['status']}]")
            continue
        agr = r["agreement_rate"]
        agr_str = f"{agr:.2%} ({r['n_diag']})" if agr is not None else "—"
        print(
            f"{r['backend']:<18} {r['variant']:<22} {r['n']:>3} "
            f"{r['mean_baseline']:>9.4f} {r['mean_search']:>9.4f} {r['mean_delta']:>+8.4f} "
            f"{agr_str:>10}"
        )

    # Best variant per backend.
    print("\nWinners (highest mean_search per backend):")
    by_backend: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("status") == "ok":
            by_backend[r["backend"]].append(r)
    for backend, rs in by_backend.items():
        if not rs:
            continue
        best = max(rs, key=lambda x: x["mean_search"])
        worst = min(rs, key=lambda x: x["mean_search"])
        print(f"  {backend}: best={best['variant']} ({best['mean_search']:.4f})  "
              f"worst={worst['variant']} ({worst['mean_search']:.4f})  "
              f"spread={best['mean_search']-worst['mean_search']:.4f}")


if __name__ == "__main__":
    main()
