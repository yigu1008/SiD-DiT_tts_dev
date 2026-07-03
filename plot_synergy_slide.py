#!/usr/bin/env python3
"""Slide-ready 2x2 'prompt x CFG' figure for one prompt (academic presentation).

Reframes the 4-image strip as a 2x2 matrix so the two orthogonal knobs are
explicit:
    rows    = prompt   (original  ->  LLM-enhanced)
    columns = CFG scale (original  ->  scaled)
    corners = baseline / +CFG / +prompt / +both
The bottom-right (+both) cell is highlighted; axis arrows show reward rising
along each knob, and a diagonal annotation gives the total gain. Light
background, large type -- drops straight onto a slide.

Usage:
  python plot_synergy_slide.py --run_root <grid_run> --prompt_index 1 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def load(run_root, seed):
    rew, text, cfgs, variants = {}, {}, set(), set()
    for r in csv.DictReader(open(os.path.join(run_root, "cfg_prompt_grid.csv"), encoding="utf-8")):
        if str(int(r["seed"])) != str(seed):
            continue
        pi, vi, c = int(r["prompt_index"]), int(r["variant"]), float(r["cfg"])
        rew[(pi, vi, c)] = float(r["reward"])
        text.setdefault(pi, {})[vi] = r.get("variant_text", "")
        cfgs.add(c); variants.add(vi)
    return rew, text, sorted(cfgs), sorted(variants)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_root", required=True)
    p.add_argument("--prompt_index", type=int, default=1)
    p.add_argument("--seed", default="42")
    p.add_argument("--cfg_lo", type=float, default=None, help="baseline cfg (default: min in CSV)")
    p.add_argument("--row_labels", default="v1|v2")
    p.add_argument("--col_labels", default="Original CFG|Scaled CFG")
    p.add_argument("--title", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    rew, text, cfgs, variants = load(a.run_root, a.seed)
    pi, cfg_lo = a.prompt_index, (a.cfg_lo if a.cfg_lo is not None else cfgs[0])
    rewrites = [v for v in variants if v != 0] or [0]
    base = rew.get((pi, 0, cfg_lo))
    if base is None:
        raise SystemExit(f"no baseline for p{pi:05d} seed {a.seed} at cfg {cfg_lo}")
    cfg_val, c_cfg = max((rew.get((pi, 0, c), -1e9), c) for c in cfgs)
    prompt_val, v_p = max((rew.get((pi, v, cfg_lo), -1e9), v) for v in rewrites)
    both_val, v_b, c_b = max((rew.get((pi, v, c), -1e9), v, c) for v in rewrites for c in cfgs)

    # 2x2: [row=prompt][col=cfg]; use the +both rewrite for the whole bottom row,
    # and the +both cfg for the whole right column, so the matrix is a clean grid.
    cells = {
        (0, 0): (0,   cfg_lo, base,       "baseline"),
        (0, 1): (0,   c_b,    rew.get((pi, 0, c_b), cfg_val),  "+CFG"),
        (1, 0): (v_b, cfg_lo, rew.get((pi, v_b, cfg_lo), prompt_val), "+prompt"),
        (1, 1): (v_b, c_b,    both_val,    "+both"),
    }
    img_dir = os.path.join(a.run_root, "images")

    def thumb(v, c):
        pth = os.path.join(img_dir, f"p{pi:05d}_s{a.seed}_v{v}_cfg{c:.2f}.png")
        return Image.open(pth).convert("RGB") if os.path.exists(pth) else None

    rlab = a.row_labels.split("|"); clab = a.col_labels.split("|")
    fig = plt.figure(figsize=(9.6, 10.2))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, left=0.14, right=0.98, top=0.90, bottom=0.06,
                          hspace=0.06, wspace=0.04)
    for (ri, ci), (v, c, s, tag) in cells.items():
        ax = fig.add_subplot(gs[ri, ci])
        im = thumb(v, c)
        if im is not None:
            ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        best = (ri, ci) == (1, 1)
        for sp in ax.spines.values():
            sp.set_color("#2e9e57" if best else "#bbbbbb")
            sp.set_linewidth(5 if best else 1.5)
        # reward badge (top-left of each image)
        ax.text(0.03, 0.96, f"{s:.3f}", transform=ax.transAxes, fontsize=17, fontweight="bold",
                va="top", ha="left", color="white",
                bbox=dict(boxstyle="round,pad=0.28", fc=("#2e9e57" if best else "#333333"), ec="none"))
        ax.text(0.97, 0.04, tag + (" ★" if best else ""), transform=ax.transAxes, fontsize=14,
                va="bottom", ha="right", color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc=("#2e9e57" if best else "#00000088"), ec="none"))
        if ri == 0:
            ax.set_title(clab[ci], fontsize=17, fontweight="bold", pad=8)
        if ci == 0:
            ax.set_ylabel(rlab[ri], fontsize=17, fontweight="bold", labelpad=10)

    # axis arrows: CFG (top, →) and prompt (left, ↓), both "reward increases"
    fig.text(0.56, 0.935, "CFG scale  →", ha="center", fontsize=15, color="#c0392b", fontweight="bold")
    fig.text(0.055, 0.48, "prompt quality  →", rotation=90, va="center", fontsize=15,
             color="#c0392b", fontweight="bold")
    # bottom callout: both beats EITHER single axis (the defensible claim)
    gain = both_val - base
    single_cfg = rew.get((pi, 0, c_b), cfg_val)
    single_prompt = rew.get((pi, v_b, cfg_lo), prompt_val)
    fig.text(0.56, 0.015,
             f"both together {both_val:.3f}  >  +CFG alone {single_cfg:.3f}   and   "
             f">  +prompt alone {single_prompt:.3f}    (baseline {base:.3f}, +{gain:.3f})",
             ha="center", fontsize=13.5, color="#2e7d32", fontweight="bold")

    title = a.title or f"CFG scale × prompt version — “{text.get(pi, {}).get(0, '')[:48]}”"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.975)
    out = a.out or os.path.join(a.run_root, f"synergy_slide_p{pi:05d}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    v1_text = text.get(pi, {}).get(0, "")
    v2_text = text.get(pi, {}).get(v_b, "")
    print(f"[slide] wrote {out}  base {base:.3f} -> both {both_val:.3f}  (+{gain:.3f})")
    print(f"[slide] v1 (original prompt): {v1_text}")
    print(f"[slide] v2 (rewrite, variant {v_b}): {v2_text}")


if __name__ == "__main__":
    main()
