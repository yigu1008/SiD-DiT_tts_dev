#!/usr/bin/env python3
"""Visualize the per-step CFG intervention grid from run_cfg_step_grid.py.

Per prompt: a heatmap of reward GAIN vs the all-baseline (constant cfg) over
step (y) x cfg (x). Warm (>0) cells = bumping cfg at THAT step beats the
constant-cfg baseline -- the per-step guidance-schedule effect. The best
(step, cfg) cell is starred; panels are ranked by max gain so the prompts that
benefit from a per-step cfg bump surface first.

Usage:
  python plot_cfg_step_grid.py --grid_csv .../cfg_step_grid.csv \
    --rank_by_gain --top 12 --out cfg_step_rectangles.png
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
    rew = defaultdict(lambda: defaultdict(list))   # prompt -> (step,cfg) -> [reward]
    base = defaultdict(list)                        # prompt -> [baseline reward]
    cfgs, steps = set(), set()
    for r in csv.DictReader(open(path, encoding="utf-8")):
        pi = int(r["prompt_index"]); rw = float(r["reward"])
        if int(r["is_baseline"]) == 1 or int(r["step"]) < 0:
            base[pi].append(rw); continue
        s = int(r["step"]); c = float(r["cfg"])
        rew[pi][(s, c)].append(rw); steps.add(s); cfgs.add(c)
    return rew, base, sorted(cfgs), sorted(steps)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid_csv", required=True)
    p.add_argument("--seed_agg", choices=["max", "mean"], default="max")
    p.add_argument("--rank_by_gain", action="store_true")
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--aggregate", action="store_true",
                   help="One summary heatmap: mean gain per (step x cfg) over all prompts (easiest read).")
    p.add_argument("--out", default="cfg_step_rectangles.png")
    a = p.parse_args()

    rew, base, cfgs, steps = _load(a.grid_csv)
    agg = (lambda xs: max(xs)) if a.seed_agg == "max" else (lambda xs: float(np.mean(xs)))

    def delta_matrix(pi):
        b = agg(base[pi]) if base[pi] else 0.0
        M = np.full((len(steps), len(cfgs)), np.nan)
        for i, s in enumerate(steps):
            for j, c in enumerate(cfgs):
                vals = rew[pi].get((s, c))
                if vals:
                    M[i, j] = agg(vals) - b
        return M, b

    pis = sorted(rew)

    if a.aggregate:
        # mean gain per (step, cfg) across all prompts -> one summary heatmap
        stacks = np.stack([delta_matrix(pi)[0] for pi in pis])
        Mbar = np.nanmean(stacks, axis=0)
        lim = float(np.nanmax(np.abs(Mbar))) or 1e-6
        fig, ax = plt.subplots(figsize=(1.1 * len(cfgs) + 2, 0.6 * len(steps) + 2))
        im = ax.imshow(Mbar, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
        ax.set_xticks(range(len(cfgs))); ax.set_xticklabels([f"{c:g}" for c in cfgs])
        ax.set_yticks(range(len(steps))); ax.set_yticklabels([f"step {s}" for s in steps])
        ax.set_xlabel("cfg bumped at step"); ax.set_ylabel("step")
        for i in range(len(steps)):
            for j in range(len(cfgs)):
                ax.text(j, i, f"{Mbar[i, j]:+.3f}", ha="center", va="center", fontsize=8,
                        color=("white" if abs(Mbar[i, j]) > 0.6 * lim else "black"))
        bi, bj = np.unravel_index(np.nanargmax(Mbar), Mbar.shape)
        ax.plot(bj, bi, "k*", ms=14)
        fig.colorbar(im, ax=ax, label="mean reward gain vs constant-cfg baseline")
        ax.set_title(f"per-step CFG effect (mean over {len(pis)} prompts)\n"
                     f"best: step {steps[bi]} @ cfg {cfgs[bj]:g}  ({Mbar[bi, bj]:+.3f})")
        fig.savefig(a.out, dpi=130, bbox_inches="tight")
        print(f"[cfg-step-plot] aggregate -> {a.out}  best step {steps[bi]} cfg {cfgs[bj]:g} mean-gain {Mbar[bi, bj]:+.4f}")
        return

    gain = {pi: (np.nanmax(delta_matrix(pi)[0]) if np.isfinite(delta_matrix(pi)[0]).any() else -1e9) for pi in pis}
    if a.rank_by_gain:
        pis.sort(key=lambda i: gain[i], reverse=True)
    pis = pis[: a.top]
    if not pis:
        raise SystemExit("no prompts to plot")

    lim = max(abs(np.nanmin([delta_matrix(i)[0] for i in pis])), abs(np.nanmax([delta_matrix(i)[0] for i in pis])))
    ncol = min(a.cols, len(pis)); nrow = (len(pis) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False)
    for k, pi in enumerate(pis):
        ax = axes[k // ncol][k % ncol]
        M, b = delta_matrix(pi)
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
        ax.set_xticks(range(len(cfgs))); ax.set_xticklabels([f"{c:g}" for c in cfgs], fontsize=7)
        ax.set_yticks(range(len(steps))); ax.set_yticklabels([f"s{s}" for s in steps], fontsize=7)
        ax.set_xlabel("cfg@step", fontsize=8); ax.set_ylabel("step", fontsize=8)
        if np.isfinite(M).any():
            bi, bj = np.unravel_index(np.nanargmax(M), M.shape)
            ax.plot(bj, bi, "k*", ms=11)
            g = float(M[bi, bj])
        else:
            g = float("nan")
        col = "tab:green" if g > 0 else "tab:red"
        ax.set_title(f"p{pi:05d}  base {b:.2f}  best-step-gain {g:+.3f}\n"
                     f"{'helps at step '+str(steps[bi])+' cfg '+format(cfgs[bj],'g') if g>0 else 'no per-step cfg gain'}",
                     fontsize=7.5, color=col)
    for k in range(len(pis), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label="reward gain vs constant-cfg baseline")
    fig.suptitle(f"per-step CFG intervention  (agg={a.seed_agg})  warm = bumping cfg at that step helps", fontsize=11)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    n_pos = sum(1 for pi in pis if gain[pi] > 0)
    print(f"[cfg-step-plot] wrote {a.out}  ({len(pis)} prompts, {n_pos} with a per-step cfg gain>0)")


if __name__ == "__main__":
    main()
