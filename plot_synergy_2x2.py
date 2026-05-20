#!/usr/bin/env python3
"""2x2 synergy plot for dynamic-CFG × dynamic-prompt.

Reads a summary.tsv (produced by run_synergy_ablation.sh) with at least the
four cells:
    bon_mcts_static_cfg     (static CFG, static prompt)   = base
    bon_mcts_adaptive_cfg   (dynamic CFG, static prompt)  = +CFG
    bon_mcts_rewrite_only   (static CFG, dynamic prompt)  = +prompt
    bon_mcts_full           (dynamic CFG, dynamic prompt) = both

Computes and reports:
    ME_cfg     = ((+CFG) + (both)) / 2 - (base + (+prompt)) / 2
    ME_prompt  = ((+prompt) + (both)) / 2 - (base + (+CFG)) / 2
    interaction = both - ((+CFG) + (+prompt) - base)   # > 0 ⇒ super-additive (1+1>2)

Emits a 2x2 grid bar chart with interaction annotated.

Usage:
    python plot_synergy_2x2.py --summary <RUN_ROOT>/summary.tsv \
        [--metric eval_ir|mean_search|eval_hpsv3] [--out_png synergy.png]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CELLS = {
    "bon_mcts_static_cfg":   ("base",       "static",  "static"),
    "bon_mcts_adaptive_cfg": ("+CFG",       "dynamic", "static"),
    "bon_mcts_rewrite_only": ("+prompt",    "static",  "dynamic"),
    "bon_mcts_full":         ("both",       "dynamic", "dynamic"),
}


def _read(summary: Path, metric: str) -> dict[str, float]:
    rows: dict[str, float] = {}
    with open(summary) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            m = row.get("method", "").strip()
            v = row.get(metric, "").strip()
            if m in CELLS and v not in ("", "-"):
                try:
                    rows[m] = float(v)
                except ValueError:
                    pass
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", required=True, type=Path)
    p.add_argument("--metric", default="eval_ir", choices=["eval_ir", "mean_search", "eval_hpsv3"])
    p.add_argument("--out_png", default=None, help="Default: <summary_dir>/synergy_<metric>.png")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    rows = _read(args.summary, args.metric)
    missing = [c for c in CELLS if c not in rows]
    if missing:
        raise SystemExit(f"summary.tsv missing cells: {missing}\nHave: {list(rows)}")

    base   = rows["bon_mcts_static_cfg"]
    plus_c = rows["bon_mcts_adaptive_cfg"]
    plus_p = rows["bon_mcts_rewrite_only"]
    both   = rows["bon_mcts_full"]

    me_cfg      = ((plus_c + both) - (base + plus_p)) / 2.0
    me_prompt   = ((plus_p + both) - (base + plus_c)) / 2.0
    additive    = plus_c + plus_p - base
    interaction = both - additive

    print(f"# metric: {args.metric}")
    print(f"  base                                = {base:.4f}")
    print(f"  +CFG       (dynamic cfg only)       = {plus_c:.4f}   Δ = {plus_c - base:+.4f}")
    print(f"  +prompt    (dynamic prompt only)    = {plus_p:.4f}   Δ = {plus_p - base:+.4f}")
    print(f"  both       (cfg + prompt)           = {both:.4f}   Δ = {both - base:+.4f}")
    print()
    print(f"  ME_cfg     (main effect, CFG)       = {me_cfg:+.4f}")
    print(f"  ME_prompt  (main effect, prompt)    = {me_prompt:+.4f}")
    print(f"  additive prediction (no synergy)    = {additive:.4f}")
    print(f"  interaction = both - additive       = {interaction:+.4f}   "
          f"({'SUPER-ADDITIVE (1+1>2)' if interaction > 0 else 'SUB-ADDITIVE'})")

    # Plot: 2x2 grid (CFG axis × prompt axis).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left panel: 4 bars side-by-side.
    labels = ["base\n(no/no)", "+CFG\n(yes/no)", "+prompt\n(no/yes)", "both\n(yes/yes)"]
    vals = [base, plus_c, plus_p, both]
    colors = ["#888888", "#3477B3", "#D87F4A", "#C0392B"]
    bars = axes[0].bar(labels, vals, color=colors)
    # Annotate additive prediction
    axes[0].axhline(additive, ls="--", lw=1.2, color="#444",
                    label=f"additive prediction = {additive:.3f}")
    for b, v in zip(bars, vals):
        axes[0].annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                         ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel(args.metric)
    axes[0].set_title(args.title or f"2×2 factorial — {args.metric}")
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    ymin = min(vals + [additive])
    ymax = max(vals + [additive])
    pad = (ymax - ymin) * 0.15
    axes[0].set_ylim(ymin - pad, ymax + pad)

    # Right panel: interaction plot — lines connecting (CFG-off, CFG-on) at each prompt setting.
    x = [0, 1]
    static_line  = [base,   plus_c]   # prompt static
    dynamic_line = [plus_p, both]     # prompt dynamic
    axes[1].plot(x, static_line,  "o-", color="#3477B3", lw=2.0, label="prompt: static",  markersize=7)
    axes[1].plot(x, dynamic_line, "s-", color="#C0392B", lw=2.0, label="prompt: dynamic", markersize=7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["CFG: static", "CFG: dynamic"])
    axes[1].set_ylabel(args.metric)
    axes[1].set_title(f"interaction = {interaction:+.3f}\n"
                      f"({'SUPER-ADDITIVE' if interaction > 0 else 'sub-additive'})")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.3)
    # Annotate each point
    for xi, (a, b) in zip(x, [(static_line[0], dynamic_line[0]), (static_line[1], dynamic_line[1])]):
        axes[1].annotate(f"{a:.3f}", (xi, a), textcoords="offset points", xytext=(8, -2),  fontsize=8)
        axes[1].annotate(f"{b:.3f}", (xi, b), textcoords="offset points", xytext=(8,  2),  fontsize=8)

    fig.tight_layout()
    out = Path(args.out_png) if args.out_png else (args.summary.parent / f"synergy_{args.metric}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  saved: {out}")


if __name__ == "__main__":
    main()
