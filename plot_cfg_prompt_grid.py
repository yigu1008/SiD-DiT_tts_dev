#!/usr/bin/env python3
"""Visualize the (cfg x prompt-variant) reward rectangle from run_cfg_prompt_grid.py.

Per prompt: a heatmap of reward over cfg (x) x variant (y), with the synergy read
straight off the rectangle:

  baseline        = R[variant 0, cfg=baseline_cfg]     (original prompt, no guidance)
  best cfg-only   = max over cfg at variant 0
  best prompt-only= max over variant at cfg=baseline_cfg
  best both       = max over the whole rectangle
  synergy         = best_both - (best_cfg_only + best_prompt_only - baseline)   (>0 => 1+1>2)

Aggregates over seeds by max (best-of-seed) unless --seed_agg mean.

Usage:
  python plot_cfg_prompt_grid.py --grid_csv /data/ygu/cfg_prompt_grid/exp1/cfg_prompt_grid.csv \
    --baseline_cfg 1.0 --rank_by_synergy --top 12 --out cfg_prompt_grid.png
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path: str):
    # {prompt: {(variant, cfg): [rewards]}}
    data: dict[int, dict[tuple[int, float], list[float]]] = defaultdict(lambda: defaultdict(list))
    cfgs, variants = set(), set()
    for row in csv.DictReader(open(path, encoding="utf-8")):
        pi = int(row["prompt_index"]); v = int(row["variant"]); c = float(row["cfg"])
        data[pi][(v, c)].append(float(row["reward"]))
        cfgs.add(c); variants.add(v)
    return data, sorted(cfgs), sorted(variants)


def _matrix(cell, cfgs, variants, agg):
    M = np.full((len(variants), len(cfgs)), np.nan)
    for i, v in enumerate(variants):
        for j, c in enumerate(cfgs):
            vals = cell.get((v, c))
            if vals:
                M[i, j] = (max(vals) if agg == "max" else float(np.mean(vals)))
    return M


def _synergy(M, cfgs, baseline_cfg):
    jb = int(np.argmin([abs(c - baseline_cfg) for c in cfgs]))
    base = M[0, jb]
    cfg_only = np.nanmax(M[0, :])          # variant 0, search cfg
    prompt_only = np.nanmax(M[:, jb])      # cfg baseline, search variant
    both = np.nanmax(M)                    # search both
    return both - (cfg_only + prompt_only - base), base, cfg_only, prompt_only, both


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid_csv", required=True)
    p.add_argument("--baseline_cfg", type=float, default=1.0)
    p.add_argument("--seed_agg", choices=["max", "mean"], default="max")
    p.add_argument("--prompts", default=None, help="subset '0-11'/'0,3,7' (default: all)")
    p.add_argument("--rank_by_synergy", action="store_true")
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--out", default="cfg_prompt_grid.png")
    a = p.parse_args()

    data, cfgs, variants = _load(a.grid_csv)
    pis = sorted(data)
    if a.prompts:
        want = set()
        for tok in a.prompts.replace(",", " ").split():
            if "-" in tok:
                x, y = tok.split("-"); want.update(range(int(x), int(y) + 1))
            elif tok.isdigit():
                want.add(int(tok))
        pis = [i for i in pis if i in want]

    syn = {}
    for pi in pis:
        M = _matrix(data[pi], cfgs, variants, a.seed_agg)
        syn[pi] = _synergy(M, cfgs, a.baseline_cfg)
    if a.rank_by_synergy:
        pis.sort(key=lambda i: syn[i][0], reverse=True)
    pis = pis[: a.top]
    if not pis:
        raise SystemExit("no prompts to plot")

    ncol = min(a.cols, len(pis)); nrow = (len(pis) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False)
    vmin = min(np.nanmin(_matrix(data[i], cfgs, variants, a.seed_agg)) for i in pis)
    vmax = max(np.nanmax(_matrix(data[i], cfgs, variants, a.seed_agg)) for i in pis)
    for k, pi in enumerate(pis):
        ax = axes[k // ncol][k % ncol]
        M = _matrix(data[pi], cfgs, variants, a.seed_agg)
        s, base, co, po, both = syn[pi]
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_xticks(range(len(cfgs))); ax.set_xticklabels([f"{c:g}" for c in cfgs], fontsize=7)
        ax.set_yticks(range(len(variants))); ax.set_yticklabels([f"v{v}" for v in variants], fontsize=7)
        ax.set_xlabel("cfg", fontsize=8); ax.set_ylabel("prompt", fontsize=8)
        # mark the argmax cell (best joint)
        bi, bj = np.unravel_index(np.nanargmax(M), M.shape)
        ax.plot(bj, bi, "r*", ms=10)
        col = "tab:green" if s > 0 else "tab:red"
        ax.set_title(f"p{pi:05d}  synergy {s:+.3f} {'(1+1>2)' if s>0 else ''}\n"
                     f"base {base:.2f}  +cfg {co:.2f}  +prm {po:.2f}  both {both:.2f}",
                     fontsize=7.5, color=col)
    for k in range(len(pis), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label="reward")
    fig.suptitle(f"cfg x prompt reward rectangle  (agg={a.seed_agg}, baseline_cfg={a.baseline_cfg})", fontsize=11)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    n_pos = sum(1 for pi in pis if syn[pi][0] > 0)
    print(f"[grid-plot] wrote {a.out}  ({len(pis)} prompts, {n_pos} with synergy>0, "
          f"mean synergy {np.mean([syn[i][0] for i in pis]):+.4f})")


if __name__ == "__main__":
    main()
