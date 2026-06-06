#!/usr/bin/env python3
"""Plot from a wide-format CSV: first col = x (NFE), other cols = methods.

Drops methods listed in --drop, bolds the column named in --ours_label.
y-axis label is auto-inferred from the filename (between "<backend>_" and "_vs_").

Usage:
  python plot_nfe_wide.py --csv flux_schnell_mean_search_vs_nfe_transformer.csv \
      --out_dir plots --drop baseline dynamic_cfg_x0 greedy ga beam dts \
      --ours_label ours
"""
from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_OURS_COLOR = "#d62728"
_OTHER_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2",
                 "#17becf", "#bcbd22", "#7f7f7f", "#ff7f0e", "#1abc9c"]
_OTHER_MARKERS = ["s", "D", "^", "v", "P", "X", "p", "h", "o", "*"]


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--drop", nargs="*",
                   default=["baseline", "dynamic_cfg_x0", "greedy", "ga", "beam", "dts"])
    p.add_argument("--ours_label", default="ours",
                   help="Column name to bold (case-insensitive match).")
    p.add_argument("--ours_display", default=None,
                   help="Legend label for the bolded ours curve (default: '<col> (ours)').")
    # Paper-style default rename map (override with --rename).
    _PAPER_RENAME = {
        "bon": "BoN", "sop": "SoP", "smc": "SMC/DAS",
        "bon_actdiff_cfg": "BoN+ActDiff (CFG)",
        "bon_actdiff_full": "BoN+ActDiff (CFG+Prompt)",
        "sop_actdiff_cfg": "SoP+ActDiff (CFG)",
        "sop_actdiff_full": "SoP+ActDiff (CFG+Prompt)",
        "smc_actdiff_cfg": "SMC+ActDiff (CFG)",
        "smc_actdiff_full": "SMC+ActDiff (CFG+Prompt)",
        "fksteering": "FK-Steering", "greedy_prompt": "Greedy (Prompt)",
        "bon_mcts": "ours", "bon_mcts_full": "MCTS (full)",
        "bon_mcts_static_cfg": "MCTS (static CFG)",
        "bon_mcts_adaptive_cfg": "MCTS (adaptive CFG)",
        "bon_mcts_rewrite_only": "MCTS (rewrite only)",
        "bon_mcts_neg": "MCTS+NegBank",
        "bon_mcts_sigma": "MCTS+Sigma",
        "bon_mcts_axes": "MCTS+Axes",
    }
    p.add_argument("--rename", nargs="*",
                   default=[f"{k}:{v}" for k, v in _PAPER_RENAME.items()],
                   help="Rename pairs old:new for legend labels (e.g. smc:DAS).")
    p.add_argument("--no_log2_x", action="store_true")
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--title", default=None, help="Override plot title.")
    p.add_argument("--xlabel", default=None, help="Override x-axis label.")
    p.add_argument("--ylabel", default=None, help="Override y-axis label.")
    args = p.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse "<backend>_<ycol>_vs_<xcol>.csv"
    stem = csv_path.stem
    m = re.match(r"(.+?)_(.+?)_vs_(.+)$", stem)
    if m:
        backend, y_label, x_label = m.groups()
    else:
        backend, y_label, x_label = stem, "value", "nfe"

    with csv_path.open() as f:
        rows = list(csv.reader(f))
    header = rows[0]
    x_col = header[0]
    method_cols = header[1:]

    drop_lc = {d.lower() for d in args.drop}
    ours_lc = args.ours_label.lower()
    rename_map = {}
    for spec in args.rename:
        if ":" in spec:
            old, new = spec.split(":", 1)
            rename_map[old.strip().lower()] = new.strip()

    # Column index → (method_label, is_ours)
    columns = []
    for i, name in enumerate(method_cols, start=1):
        nm_lc = name.strip().lower()
        if nm_lc in drop_lc:
            continue
        is_ours = (nm_lc == ours_lc)
        display_name = rename_map.get(nm_lc, name.strip())
        columns.append((i, display_name, is_ours))

    # Sort: non-ours first (for color slot stability), ours last (drawn on top).
    columns.sort(key=lambda c: (c[2], c[1].lower()))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    non_ours_idx = 0
    for col_idx, name, is_ours in columns:
        xs, ys = [], []
        for r in rows[1:]:
            if col_idx >= len(r):
                continue
            x = _to_float(r[0])
            y = _to_float(r[col_idx])
            if x is None or y is None or x <= 0:
                continue
            xs.append(x); ys.append(y)
        if not xs:
            continue
        order = sorted(range(len(xs)), key=lambda k: xs[k])
        xs = [xs[k] for k in order]
        ys = [ys[k] for k in order]
        if is_ours:
            ours_legend = args.ours_display or f"{name} (ours)"
            ax.plot(xs, ys, marker="o", color=_OURS_COLOR, linewidth=3.5,
                    markersize=9, zorder=10, label=ours_legend)
        else:
            color = _OTHER_COLORS[non_ours_idx % len(_OTHER_COLORS)]
            marker = _OTHER_MARKERS[non_ours_idx % len(_OTHER_MARKERS)]
            non_ours_idx += 1
            ax.plot(xs, ys, marker=marker, color=color, linewidth=1.4,
                    markersize=6, alpha=0.85, zorder=2, label=name)

    if not args.no_log2_x:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(args.xlabel or x_col)
    ax.set_ylabel(args.ylabel or y_label)
    ax.set_title(args.title or f"{backend}: {y_label} vs {x_col}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.92)

    stem = out_dir / f"{backend}_{y_label}_vs_{x_col}"
    fig.tight_layout()
    # PNG (raster preview) + PDF (vector, selectable text for LaTeX)
    fig.savefig(f"{stem}.png", dpi=args.dpi)
    fig.savefig(f"{stem}.pdf", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_nfe_wide] wrote {stem}.png and {stem}.pdf")


if __name__ == "__main__":
    main()
