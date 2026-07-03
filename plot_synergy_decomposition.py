#!/usr/bin/env python3
"""Concrete "why prompt + cfg is 1+1>2" figure: images + reward waterfall.

For one prompt, decompose the reward into its additive parts and show that doing
BOTH exceeds the additive prediction -- which is exactly what 1+1>2 means:

    gain(both)  >  gain(cfg) + gain(prompt)

Top row: the four images (baseline / +cfg / +prompt / +both), best cfg picked per
axis. Bottom: a waterfall bar chart where the +both bar stacks
    base + Δcfg + Δprompt + SYNERGY
so the green synergy segment sitting ABOVE the additive-prediction line is the
visual ">2". By default the prompt with the largest positive synergy is chosen.

Usage:
  python plot_synergy_decomposition.py --run_root <grid_run> --seed 42
  python plot_synergy_decomposition.py --run_root <grid_run> --prompt_index 3
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image


def load(run_root: str, seed: str):
    csv_path = os.path.join(run_root, "cfg_prompt_grid.csv")
    rew: dict[tuple[int, int, float], float] = {}
    text: dict[int, dict[int, str]] = {}
    cfgs, variants = set(), set()
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        if str(int(r["seed"])) != str(seed):
            continue
        pi, vi, c = int(r["prompt_index"]), int(r["variant"]), float(r["cfg"])
        rew[(pi, vi, c)] = float(r["reward"])
        text.setdefault(pi, {})[vi] = r.get("variant_text", "")
        cfgs.add(c); variants.add(vi)
    return rew, text, sorted(cfgs), sorted(variants)


def corners(rew, pi, cfg_lo, cfgs, rewrites):
    base = rew.get((pi, 0, cfg_lo))
    if base is None:
        return None
    cfg_val, c_cfg = max((rew.get((pi, 0, c), -1e9), c) for c in cfgs)
    prompt_val, v_p = max((rew.get((pi, v, cfg_lo), -1e9), v) for v in rewrites)
    both_val, v_b, c_b = max((rew.get((pi, v, c), -1e9), v, c) for v in rewrites for c in cfgs)
    if min(cfg_val, prompt_val, both_val) < -1e8:
        return None
    syn = both_val - (cfg_val + prompt_val - base)
    return {"base": (0, cfg_lo, base), "cfg": (0, c_cfg, cfg_val),
            "prompt": (v_p, cfg_lo, prompt_val), "both": (v_b, c_b, both_val), "syn": syn}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_root", required=True)
    p.add_argument("--seed", default="42")
    p.add_argument("--prompt_index", type=int, default=None, help="default: max-synergy prompt")
    p.add_argument("--cfg_lo", type=float, default=None, help="baseline cfg (default: min in CSV)")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    rew, text, cfgs, variants = load(a.run_root, a.seed)
    if not rew:
        raise SystemExit(f"no rows for seed {a.seed}")
    cfg_lo = a.cfg_lo if a.cfg_lo is not None else cfgs[0]
    rewrites = [v for v in variants if v != 0] or [0]
    pis = sorted({pi for (pi, _v, _c) in rew})
    cand = {pi: c for pi in pis if (c := corners(rew, pi, cfg_lo, cfgs, rewrites))}
    if not cand:
        raise SystemExit("no prompt had all four corners")
    pi = a.prompt_index if a.prompt_index is not None else max(cand, key=lambda i: cand[i]["syn"])
    c = cand[pi]

    base = c["base"][2]; d_cfg = c["cfg"][2] - base; d_pr = c["prompt"][2] - base
    both = c["both"][2]; add_pred = base + d_cfg + d_pr; syn = c["syn"]
    img_dir = os.path.join(a.run_root, "images")

    def thumb(cell):
        v, cf, _s = cell
        pth = os.path.join(img_dir, f"p{pi:05d}_s{a.seed}_v{v}_cfg{cf:.2f}.png")
        return Image.open(pth).convert("RGB") if os.path.exists(pth) else None

    cells = [("baseline", c["base"], (150, 150, 150)),
             ("+cfg", c["cfg"], (70, 130, 210)),
             ("+prompt", c["prompt"], (230, 150, 60)),
             ("+both", c["both"], (90, 200, 120))]

    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[3.2, 2.6], hspace=0.16, wspace=0.05)
    for ci, (lab, cell, _col) in enumerate(cells):
        ax = fig.add_subplot(gs[0, ci])
        im = thumb(cell)
        if im is not None:
            ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        v, cf, s = cell
        edge = "#5ac878" if lab == "+both" else "#888"
        for sp in ax.spines.values():
            sp.set_color(edge); sp.set_linewidth(3.5 if lab == "+both" else 1.5)
        tag = f"cfg {cf:g}" + ("" if ci in (0, 1) else f", v{v}")
        ax.set_title(f"{lab}\n{tag}   reward {s:.3f}", fontsize=12,
                     color=("#2e7d32" if lab == "+both" else "black"),
                     fontweight=("bold" if lab == "+both" else "normal"))

    axb = fig.add_subplot(gs[1, :])
    xs = [0, 1, 2, 3]
    grey, blue, orange, green = "#b0b0b0", "#4682d2", "#e6963c", "#5ac878"
    # baseline / +cfg / +prompt : base + single increment
    axb.bar(0, base, color=grey, width=0.62)
    axb.bar(1, base, color=grey, width=0.62); axb.bar(1, d_cfg, bottom=base, color=blue, width=0.62)
    axb.bar(2, base, color=grey, width=0.62); axb.bar(2, d_pr, bottom=base, color=orange, width=0.62)
    # +both waterfall: base + Δcfg + Δprompt + synergy
    axb.bar(3, base, color=grey, width=0.62)
    axb.bar(3, d_cfg, bottom=base, color=blue, width=0.62)
    axb.bar(3, d_pr, bottom=base + d_cfg, color=orange, width=0.62)
    syn_col = green if syn >= 0 else "#d3564b"
    axb.bar(3, syn, bottom=add_pred, color=syn_col, width=0.62,
            hatch="//", edgecolor="white", linewidth=0)
    # additive-prediction reference line
    axb.hlines(add_pred, 2.62, 3.38, color="black", linestyle="--", linewidth=1.6)
    axb.annotate("additive prediction\n(base+Δcfg+Δprompt)", xy=(3.38, add_pred),
                 xytext=(3.5, add_pred), fontsize=10, va="center", color="black")
    axb.annotate(f"synergy {syn:+.3f}", xy=(3, add_pred + syn / 2), xytext=(3, add_pred + syn / 2),
                 ha="center", va="center", fontsize=11, fontweight="bold",
                 color=("#2e7d32" if syn >= 0 else "#b3261e"))
    for xi, val in zip(xs, [base, c["cfg"][2], c["prompt"][2], both]):
        axb.text(xi, val + 0.008, f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    axb.set_xticks(xs); axb.set_xticklabels(["baseline", "+cfg", "+prompt", "+both"], fontsize=12)
    lo = min(0.0, base) if base < 0 else max(0.0, base - 0.12)
    axb.set_ylim(lo, max(both, add_pred) + 0.06)
    axb.set_ylabel("composite reward", fontsize=12)
    axb.legend(handles=[Patch(color=grey, label="baseline"), Patch(color=blue, label="Δ cfg"),
                        Patch(color=orange, label="Δ prompt"),
                        Patch(facecolor=syn_col, hatch="//", label="synergy (1+1>2)")],
               loc="upper left", fontsize=10, framealpha=0.9)

    verdict = ("1+1 > 2" if syn > 0 else ("1+1 ≈ 2" if abs(syn) < 1e-3 else "1+1 < 2 (sub-additive)"))
    fig.suptitle(f"p{pi:05d}  “{text.get(pi, {}).get(0, '')[:60]}”   —   "
                 f"gain(both) {both - base:+.3f}  vs  gain(cfg)+gain(prompt) {d_cfg + d_pr:+.3f}   ⇒  {verdict}",
                 fontsize=14, y=0.98)
    out = a.out or os.path.join(a.run_root, f"synergy_decomp_p{pi:05d}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[decomp] wrote {out}  p{pi:05d} synergy {syn:+.4f} ({verdict})")


if __name__ == "__main__":
    main()
