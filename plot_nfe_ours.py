#!/usr/bin/env python3
"""Plot eval reward vs NFE per backend, dropping unscalable methods and
bolding our method.

Defaults:
  - drop: baseline, dynamic_cfg_x0           (single low-NFE points)
  - rename: bon_mcts → ours_mcts              (display label)
  - bold any method whose final label starts with "ours"  (linewidth 3.5,
    z-order on top, distinct marker, color from a high-contrast slot)

Usage:
  python plot_nfe_ours.py \
      --csv /Users/guyi/Downloads/nfes/plots_sid/combined.csv \
      --out_dir /Users/guyi/Downloads/nfes/plots_sid_ours \
      --y_col eval_imagereward
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DROP = ["baseline", "dynamic_cfg_x0"]
# Paper-style display names.  Override on the CLI with --rename old:new ....
# "ours_mcts" stays magical: it triggers the bold/red highlight in this script.
DEFAULT_RENAME = {
    "bon":                    "BoN",
    "bon_actdiff_cfg":        "BoN+ActDiff (CFG)",
    "bon_actdiff_full":       "BoN+ActDiff (CFG+Prompt)",
    "sop":                    "SoP",
    "sop_actdiff_cfg":        "SoP+ActDiff (CFG)",
    "sop_actdiff_full":       "SoP+ActDiff (CFG+Prompt)",
    "smc":                    "SMC/DAS",
    "smc_actdiff_cfg":        "SMC+ActDiff (CFG)",
    "smc_actdiff_full":       "SMC+ActDiff (CFG+Prompt)",
    "fksteering":             "FK-Steering",
    "greedy":                 "Greedy",
    "greedy_prompt":          "Greedy (Prompt)",
    "ga":                     "GA",
    "beam":                   "Beam",
    "noise":                  "NoiseInject",
    "bon_mcts":               "ours_mcts",   # MUST stay as ours_mcts -> gets bolded
    "bon_mcts_static_cfg":    "MCTS (static CFG)",
    "bon_mcts_adaptive_cfg":  "MCTS (adaptive CFG)",
    "bon_mcts_rewrite_only":  "MCTS (rewrite only)",
    "bon_mcts_full":          "MCTS (full)",
    "bon_mcts_neg":           "MCTS+NegBank",
    "bon_mcts_sigma":         "MCTS+Sigma",
    "bon_mcts_axes":          "MCTS+Axes",
}

# Color palette — first slot reserved for ours_mcts.
_OURS_COLOR = "#d62728"  # vivid red
_OTHER_COLORS = [
    "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2",
    "#17becf", "#bcbd22", "#7f7f7f", "#ff7f0e", "#1abc9c",
]
_OTHER_MARKERS = ["s", "D", "^", "v", "P", "X", "p", "h", "o", "*"]


def _load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="combined.csv from nfe_sweep_combine.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--y_col", default="eval_imagereward",
                   help="Y-axis column (default eval_imagereward).")
    p.add_argument("--x_col", default="target_nfe",
                   help="X-axis column (default target_nfe — transformer NFE).")
    p.add_argument("--drop", nargs="*", default=DEFAULT_DROP,
                   help="Methods to drop (default: baseline dynamic_cfg_x0).")
    p.add_argument("--rename", nargs="*", default=[f"{k}:{v}" for k, v in DEFAULT_RENAME.items()],
                   help="Rename pairs old:new (default: bon_mcts:ours_mcts).")
    p.add_argument("--ours_label", default="ours_mcts",
                   help="Renamed label to bold/highlight (default ours_mcts).")
    p.add_argument("--ours_display", default=None,
                   help="Legend label for the bolded ours curve (default: '<label> (ours)').")
    p.add_argument("--backends", nargs="*", default=None,
                   help="Filter to subset of backends; default = all in csv.")
    p.add_argument("--status", default="ok")
    p.add_argument("--no_log2_x", action="store_true")
    p.add_argument("--dpi", type=int, default=180)
    args = p.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(csv_path)
    drop = set(args.drop)
    rename = {}
    for spec in args.rename:
        if ":" in spec:
            old, new = spec.split(":", 1)
            rename[old.strip()] = new.strip()

    # Group: backend → method_label → list[(x, y)]
    by_backend: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if args.status != "all" and row.get("status", "ok") != args.status:
            continue
        method = row.get("method", "").strip()
        if method in drop:
            continue
        method = rename.get(method, method)
        backend = row.get("backend", "").strip()
        if args.backends and backend not in args.backends:
            continue
        x = _to_int(row.get(args.x_col))
        y = _to_float(row.get(args.y_col))
        if x is None or y is None or x <= 0:
            continue
        by_backend[backend][method].append((x, y))

    if not by_backend:
        print("[plot_nfe_ours] no data to plot after filtering")
        return

    for backend, by_method in by_backend.items():
        fig, ax = plt.subplots(figsize=(8, 5.5))
        # Sort method order: ours_label last so it draws on top.
        method_order = sorted(by_method.keys(), key=lambda m: (m == args.ours_label, m))
        non_ours_idx = 0
        for method in method_order:
            pts = sorted(by_method[method])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            is_ours = (method == args.ours_label)
            if is_ours:
                color = _OURS_COLOR
                marker = "o"
                lw = 3.5
                ms = 9
                alpha = 1.0
                zorder = 10
                label = args.ours_display or f"{method} (ours)"
            else:
                color = _OTHER_COLORS[non_ours_idx % len(_OTHER_COLORS)]
                marker = _OTHER_MARKERS[non_ours_idx % len(_OTHER_MARKERS)]
                non_ours_idx += 1
                lw = 1.4
                ms = 6
                alpha = 0.85
                zorder = 2
                label = method
            ax.plot(xs, ys, marker=marker, color=color, linewidth=lw, markersize=ms,
                    alpha=alpha, label=label, zorder=zorder)

        if not args.no_log2_x:
            ax.set_xscale("log", base=2)
        ax.set_xlabel(args.x_col)
        ax.set_ylabel(args.y_col)
        ax.set_title(f"{backend}: {args.y_col} vs {args.x_col}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9, framealpha=0.92)
        stem = out_dir / f"{backend}_{args.y_col}_vs_{args.x_col}"
        fig.tight_layout()
        # PNG (raster, for slides/preview)
        fig.savefig(f"{stem}.png", dpi=args.dpi)
        # PDF (vector, with selectable text -- for paper LaTeX includegraphics)
        fig.savefig(f"{stem}.pdf", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_nfe_ours] wrote {stem}.png and {stem}.pdf")


if __name__ == "__main__":
    main()
