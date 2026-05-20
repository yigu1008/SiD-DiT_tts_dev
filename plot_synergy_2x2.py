#!/usr/bin/env python3
"""Paper-grade 2x2 synergy plot for dynamic-CFG × dynamic-prompt.

Reads a summary.tsv (run_synergy_ablation.sh or run_all_method_ablations.sh)
with at least the four corner cells:
    bon_mcts_static_cfg     (static CFG, static prompt)   = base
    bon_mcts_adaptive_cfg   (dynamic CFG, static prompt)  = +CFG
    bon_mcts_rewrite_only   (static CFG, dynamic prompt)  = +prompt
    bon_mcts_full           (dynamic CFG, dynamic prompt) = both

Renders ONE of two layouts:
  - default (numbers only): bars + interaction lines
  - with --example_strip:   bars + image strip showing 4 corner cells for a
                            cherry-picked prompt index, visual proof of 1+1>2.

Numerical output (console + image annotation):
    ME_cfg     = ((+CFG) + (both)) / 2 - (base + (+prompt)) / 2
    ME_prompt  = ((+prompt) + (both)) / 2 - (base + (+CFG)) / 2
    interaction = both - ((+CFG) + (+prompt) - base)
                  > 0 ⇒ SUPER-additive (1+1>2)

Usage:
    python plot_synergy_2x2.py --summary <RUN_ROOT>/summary.tsv
    python plot_synergy_2x2.py --summary <RUN_ROOT>/summary.tsv \
        --metric eval_ir --example_strip 7 --run_root <RUN_ROOT>
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


# Method → (display_label, cfg_dynamic, prompt_dynamic)
CELLS = {
    "bon_mcts_static_cfg":   ("base",    False, False),
    "bon_mcts_adaptive_cfg": ("+CFG",    True,  False),
    "bon_mcts_rewrite_only": ("+prompt", False, True),
    "bon_mcts_full":         ("both",    True,  True),
}

# Brand colors — distinct enough to read at print resolution.
COLORS = {
    "base":    "#7E8489",   # neutral grey
    "+CFG":    "#3D77BE",   # blue
    "+prompt": "#E08540",   # orange
    "both":    "#B73B3B",   # red (the "win" cell)
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


def _find_best_image(run_root: Path, method: str, prompt_index: int) -> Path | None:
    """Locate the chosen best image for `method` at `prompt_index`.

    Image naming differs slightly across runners; we accept the first match
    under any subdir containing both `method/` and a filename that includes
    the zero-padded prompt index.
    """
    pad_re = re.compile(rf"(?:^|[^0-9])0*{prompt_index}([^0-9]|$)")
    candidates: list[Path] = []
    # Common locations: <run>/<method>/best_images/, <run>/<method>/images/best/, etc.
    for pat in (
        f"**/{method}/**/best_images/*.png",
        f"**/{method}/**/best_images/*.jpg",
        f"**/{method}/**/images/*.png",
        f"**/{method}/**/*.png",
    ):
        for path in sorted(run_root.glob(pat)):
            if pad_re.search(path.name):
                candidates.append(path)
                break
        if candidates:
            break
    return candidates[0] if candidates else None


def _set_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })


def _bar_panel(ax: plt.Axes, vals: dict[str, float], metric: str) -> None:
    labels = ["base\n(CFG: static\nprompt: static)",
              "+CFG\n(CFG: dynamic\nprompt: static)",
              "+prompt\n(CFG: static\nprompt: dynamic)",
              "both\n(CFG: dynamic\nprompt: dynamic)"]
    keys = ["base", "+CFG", "+prompt", "both"]
    data = [vals["base"], vals["+CFG"], vals["+prompt"], vals["both"]]
    colors = [COLORS[k] for k in keys]

    bars = ax.bar(labels, data, color=colors, edgecolor="black", linewidth=0.6, width=0.62)

    # Additive prediction reference
    additive = vals["+CFG"] + vals["+prompt"] - vals["base"]
    ax.axhline(additive, ls="--", lw=1.5, color="#222",
               label=f"additive prediction = {additive:.3f}")

    # Annotate each bar with its value
    for b, v in zip(bars, data):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Annotate "both" bar with the synergy arrow
    interaction = vals["both"] - additive
    if interaction > 0:
        ax.annotate(
            f"synergy\n+{interaction:.3f}",
            xy=(3, vals["both"]),
            xytext=(3, vals["both"] + (max(data) - min(data)) * 0.18),
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=COLORS["both"],
            arrowprops=dict(arrowstyle="->", color=COLORS["both"], lw=1.6),
        )

    ax.set_ylabel(metric)
    ax.set_title("2×2 factorial — bars")
    ax.legend(loc="lower left", frameon=False)
    lo = min(data + [additive])
    hi = max(data + [additive])
    pad = (hi - lo) * 0.35 + 1e-6
    ax.set_ylim(lo - pad * 0.4, hi + pad)


def _interaction_panel(ax: plt.Axes, vals: dict[str, float], metric: str) -> None:
    x = [0, 1]
    static_line  = [vals["base"],    vals["+CFG"]]    # prompt: static
    dynamic_line = [vals["+prompt"], vals["both"]]    # prompt: dynamic
    ax.plot(x, static_line,  "o-", color=COLORS["base"],    lw=2.4, markersize=10,
            label="prompt: static")
    ax.plot(x, dynamic_line, "s-", color=COLORS["both"],    lw=2.4, markersize=10,
            label="prompt: dynamic")
    ax.set_xticks(x)
    ax.set_xticklabels(["CFG: static", "CFG: dynamic"])
    ax.set_ylabel(metric)
    additive = vals["+CFG"] + vals["+prompt"] - vals["base"]
    interaction = vals["both"] - additive
    verdict = "SUPER-additive (1 + 1 > 2)" if interaction > 0 else "sub-additive"
    ax.set_title(f"interaction lines    {verdict}: Δ = {interaction:+.3f}")
    ax.legend(frameon=False, loc="lower right")

    # Point labels
    for xi, a, b in zip(x, static_line, dynamic_line):
        ax.annotate(f"{a:.3f}", (xi, a), textcoords="offset points",
                    xytext=(10, -4), fontsize=10)
        ax.annotate(f"{b:.3f}", (xi, b), textcoords="offset points",
                    xytext=(10,  6), fontsize=10)


def _image_strip_panel(ax: plt.Axes, run_root: Path, prompt_index: int) -> None:
    from PIL import Image
    methods_in_order = ["bon_mcts_static_cfg", "bon_mcts_adaptive_cfg",
                        "bon_mcts_rewrite_only", "bon_mcts_full"]
    labels = ["base", "+CFG", "+prompt", "both"]
    n = len(methods_in_order)
    ax.axis("off")
    sub = ax.figure.subplots(1, n, gridspec_kw={"wspace": 0.04},
                              subplot_kw={"frame_on": False},
                              squeeze=False)[0]
    # Hack: matplotlib doesn't let us cleanly carve subplots inside ax, so caller
    # must pass a pre-cleared row of axes; here we re-purpose the parent ax bbox.
    for axi, m, lab in zip(sub, methods_in_order, labels):
        img_path = _find_best_image(run_root, m, prompt_index)
        if img_path is None:
            axi.text(0.5, 0.5, f"(no image\nfor #{prompt_index})", ha="center", va="center")
            axi.set_xticks([]); axi.set_yticks([])
            for s in axi.spines.values(): s.set_visible(False)
        else:
            img = Image.open(img_path).convert("RGB")
            axi.imshow(img)
            axi.set_xticks([]); axi.set_yticks([])
            for s in axi.spines.values():
                s.set_visible(True); s.set_linewidth(2.4); s.set_edgecolor(COLORS[lab])
        axi.set_title(lab, color=COLORS[lab], fontsize=11, fontweight="bold")
    sub[0].figure.suptitle("")  # avoid stale suptitle


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", required=True, type=Path)
    p.add_argument("--metric", default="eval_ir",
                   choices=["eval_ir", "mean_search", "eval_hpsv3"])
    p.add_argument("--out_png", default=None)
    p.add_argument("--example_strip", type=int, default=None,
                   help="Prompt index to render as a 4-cell image strip beside the bars.")
    p.add_argument("--run_root", type=Path, default=None,
                   help="Required with --example_strip; root containing the per-method dirs.")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    rows = _read(args.summary, args.metric)
    missing = [c for c in CELLS if c not in rows]
    if missing:
        raise SystemExit(f"summary.tsv missing cells: {missing}\nHave: {list(rows)}")

    vals = {
        "base":    rows["bon_mcts_static_cfg"],
        "+CFG":    rows["bon_mcts_adaptive_cfg"],
        "+prompt": rows["bon_mcts_rewrite_only"],
        "both":    rows["bon_mcts_full"],
    }
    additive    = vals["+CFG"] + vals["+prompt"] - vals["base"]
    me_cfg      = ((vals["+CFG"] + vals["both"]) - (vals["base"] + vals["+prompt"])) / 2.0
    me_prompt   = ((vals["+prompt"] + vals["both"]) - (vals["base"] + vals["+CFG"])) / 2.0
    interaction = vals["both"] - additive

    print(f"# metric: {args.metric}")
    for k, v in vals.items():
        print(f"  {k:8s} = {v:.4f}")
    print(f"  additive prediction         = {additive:.4f}")
    print(f"  ME_cfg                      = {me_cfg:+.4f}")
    print(f"  ME_prompt                   = {me_prompt:+.4f}")
    print(f"  interaction (both-additive) = {interaction:+.4f}    "
          f"({'SUPER-ADDITIVE (1+1>2)' if interaction > 0 else 'sub-additive'})")

    _set_paper_style()

    if args.example_strip is not None:
        if args.run_root is None:
            raise SystemExit("--example_strip requires --run_root")
        fig = plt.figure(figsize=(15, 5.5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1, 0.9],
                              wspace=0.22, hspace=0.45)
        ax_bars = fig.add_subplot(gs[0, 0])
        ax_inter = fig.add_subplot(gs[1, 0])
        ax_strip = fig.add_subplot(gs[:, 1])
        _bar_panel(ax_bars, vals, args.metric)
        _interaction_panel(ax_inter, vals, args.metric)
        _image_strip_panel(ax_strip, args.run_root, args.example_strip)
        ax_strip.set_title(f"example: prompt #{args.example_strip}",
                            fontsize=12, fontweight="bold", pad=6)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        _bar_panel(axes[0], vals, args.metric)
        _interaction_panel(axes[1], vals, args.metric)

    if args.title:
        fig.suptitle(args.title, fontsize=14, fontweight="bold", y=1.02)

    out = Path(args.out_png) if args.out_png else (args.summary.parent / f"synergy_{args.metric}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"\n  saved: {out}")


if __name__ == "__main__":
    main()
