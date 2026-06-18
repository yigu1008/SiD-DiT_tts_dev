#!/usr/bin/env python3
"""Qualitative test-time scaling figure: reward vs. search compute.

Illustrative (NOT measured) curves showing how reward scales with test-time
compute as the search method grows from a single MCTS tree into BoN-MCTS
(best-of-N over independent MCTS trees). Rendered for FLUX.1-schnell and
SiD (SD3.5) only.

    python plot_test_time_scaling.py
    python plot_test_time_scaling.py --out figures/raw/tts_scaling.png

Companion JSON records the synthetic curve params so the figure is
re-renderable. The numbers are qualitative placeholders, not benchmark data.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── Colors (shared with plot_actdiff_tree.py) ───────────────────────────────
C_NOISE       = "#888888"   # best-of-N sampling baseline
C_MCTS        = "#E08540"   # single-tree MCTS regime
C_BONMCTS     = "#B73B3B"   # BoN-MCTS regime
C_MCTS_BAND   = "#F2D6BE"
C_BON_BAND    = "#E6BFBF"

# Qualitative per-model curve params. budget = number of reward evaluations.
#   r(u) = r0 + gain * (1 - exp(-k * (u - u0))),  u = log2(budget)
# transition = budget at which one MCTS tree's budget is spent and the method
# starts adding parallel trees (-> BoN-MCTS).
MODELS = {
    "flux": {
        "title": "FLUX.1-schnell",
        "budget_min": 8,
        "budget_max": 8192,
        "transition": 256,
        "method": {"r0": 0.20, "gain": 0.62, "k": 0.32},
        "baseline": {"r0": 0.20, "gain": 0.30, "k": 0.22},
    },
    "sid35": {
        "title": "SiD (SD3.5)",
        "budget_min": 8,
        "budget_max": 8192,
        "transition": 256,
        "method": {"r0": 0.24, "gain": 0.66, "k": 0.30},
        "baseline": {"r0": 0.24, "gain": 0.33, "k": 0.20},
    },
}


def _set_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _saturating(u: np.ndarray, r0: float, gain: float, k: float, u0: float) -> np.ndarray:
    """Concave, monotonically increasing reward in log-budget u."""
    return r0 + gain * (1.0 - np.exp(-k * np.maximum(0.0, u - u0)))


def _draw_panel(ax, cfg: dict, rng: np.random.Generator) -> None:
    budgets = np.logspace(
        np.log2(cfg["budget_min"]), np.log2(cfg["budget_max"]), 200, base=2.0
    )
    u = np.log2(budgets)
    u0 = float(u[0])

    base = _saturating(u, u0=u0, **cfg["baseline"])
    method = _saturating(u, u0=u0, **cfg["method"])

    # Light qualitative wiggle + band so it reads as empirical, not a formula.
    jitter = 0.012 * rng.standard_normal(method.shape)
    method_w = method + np.cumsum(jitter) * 0.15
    band = 0.025 + 0.01 * (u - u0) / (u[-1] - u0)

    # Split the method curve at the MCTS -> BoN-MCTS transition.
    t = float(cfg["transition"])
    mcts_mask = budgets <= t
    bon_mask = budgets >= t

    # Regime shading.
    ax.axvspan(cfg["budget_min"], t, color=C_MCTS_BAND, alpha=0.35, lw=0)
    ax.axvspan(t, cfg["budget_max"], color=C_BON_BAND, alpha=0.30, lw=0)
    ax.axvline(t, color="#666666", ls="--", lw=1.0, alpha=0.8)

    # Baseline (best-of-N sampling).
    ax.plot(budgets, base, color=C_NOISE, lw=2.0, ls=(0, (4, 2)), zorder=3)

    # Method curve: orange in the MCTS regime, red in the BoN-MCTS regime.
    ax.fill_between(budgets, method_w - band, method_w + band, color=C_BONMCTS, alpha=0.12, lw=0, zorder=2)
    ax.plot(budgets[mcts_mask], method_w[mcts_mask], color=C_MCTS, lw=2.6, zorder=4)
    ax.plot(budgets[bon_mask], method_w[bon_mask], color=C_BONMCTS, lw=2.6, zorder=4)

    # A few markers to suggest discrete configs (qualitative).
    mk = np.array([cfg["budget_min"], t / 4, t, t * 4, cfg["budget_max"]])
    mk = mk[(mk >= cfg["budget_min"]) & (mk <= cfg["budget_max"])]
    mk_u = np.log2(mk)
    mk_r = _saturating(mk_u, u0=u0, **cfg["method"])
    mk_color = [C_MCTS if b <= t else C_BONMCTS for b in mk]
    ax.scatter(mk, mk_r, c=mk_color, s=42, zorder=5, edgecolors="white", linewidths=0.8)

    # Regime labels.
    ymin, ymax = base.min() - 0.04, method_w.max() + 0.06
    ax.text(np.sqrt(cfg["budget_min"] * t), ymax - 0.02, "MCTS\n(single tree)",
            ha="center", va="top", fontsize=9.5, color=C_MCTS, fontweight="bold")
    ax.text(np.sqrt(t * cfg["budget_max"]), ymax - 0.02, "BoN-MCTS\n(best-of-N trees)",
            ha="center", va="top", fontsize=9.5, color=C_BONMCTS, fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.set_xlim(cfg["budget_min"], cfg["budget_max"])
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("test-time compute  (reward evaluations)")
    ax.set_title(cfg["title"])
    ax.grid(True, which="major", axis="y", alpha=0.25, lw=0.6)


def render(models: list[str], seed: int, title: str):
    _set_paper_style()
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.4), sharey=True)
    if n == 1:
        axes = [axes]
    rng = np.random.default_rng(seed)
    for ax, key in zip(axes, models):
        _draw_panel(ax, MODELS[key], rng)
    axes[0].set_ylabel("reward  (qualitative)")

    legend = [
        Line2D([0], [0], color=C_MCTS, lw=2.6, label="MCTS (single tree)"),
        Line2D([0], [0], color=C_BONMCTS, lw=2.6, label="BoN-MCTS (best-of-N trees)"),
        Line2D([0], [0], color=C_NOISE, lw=2.0, ls=(0, (4, 2)), label="best-of-N sampling"),
        Patch(facecolor=C_MCTS_BAND, alpha=0.5, label="MCTS regime"),
        Patch(facecolor=C_BON_BAND, alpha=0.5, label="BoN-MCTS regime"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=9.5)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.20, wspace=0.08)
    return fig


def _draw_real_panel(ax, cfg: dict, transition: float) -> None:
    """Measured HPSv3-vs-budget panel: points + line + SEM band, MCTS->BoN split."""
    budget = np.asarray(cfg["budget"], dtype=float)
    order = np.argsort(budget)
    budget = budget[order]
    mean = np.asarray(cfg["reward_mean"], dtype=float)[order]
    sem_raw = cfg.get("reward_sem")
    sem = np.zeros_like(mean) if sem_raw is None else np.asarray(sem_raw, dtype=float)[order]

    bmin, bmax, t = float(budget.min()), float(budget.max()), float(transition)

    ax.axvspan(bmin, t, color=C_MCTS_BAND, alpha=0.35, lw=0)
    ax.axvspan(t, bmax, color=C_BON_BAND, alpha=0.30, lw=0)
    ax.axvline(t, color="#666666", ls="--", lw=1.0, alpha=0.8)

    ax.fill_between(budget, mean - sem, mean + sem, color=C_BONMCTS, alpha=0.12, lw=0, zorder=2)
    mcts_mask = budget <= t
    bon_mask = budget >= t
    ax.plot(budget[mcts_mask], mean[mcts_mask], color=C_MCTS, lw=2.6, zorder=4)
    ax.plot(budget[bon_mask], mean[bon_mask], color=C_BONMCTS, lw=2.6, zorder=4)

    ax.errorbar(budget, mean, yerr=sem, fmt="none", ecolor="#999999", elinewidth=1.0, capsize=3, zorder=4)
    mk_color = [C_MCTS if b <= t else C_BONMCTS for b in budget]
    ax.scatter(budget, mean, c=mk_color, s=42, zorder=5, edgecolors="white", linewidths=0.8)

    ymin = float((mean - sem).min()) - 0.04
    ymax = float((mean + sem).max()) + 0.06
    ax.text(np.sqrt(bmin * t), ymax - 0.02, "MCTS\n(single tree)",
            ha="center", va="top", fontsize=9.5, color=C_MCTS, fontweight="bold")
    ax.text(np.sqrt(t * bmax), ymax - 0.02, "BoN-MCTS\n(best-of-N trees)",
            ha="center", va="top", fontsize=9.5, color=C_BONMCTS, fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.set_xlim(bmin, bmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("test-time compute  (HPSv3 reward evaluations)")
    ax.set_title(cfg["title"])
    ax.grid(True, which="major", axis="y", alpha=0.25, lw=0.6)


def render_real(data: dict, title: str):
    """Render measured curves from a --data_json payload.

    Schema:
      {"transition": 64,
       "models": {"sid35": {"title": "SiD (SD3.5)",
                            "budget": [...], "reward_mean": [...], "reward_sem": [...]},
                  "flux":  {...}}}
    """
    _set_paper_style()
    models = data["models"]
    keys = list(models.keys())
    transition = float(data.get("transition", 64))
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        _draw_real_panel(ax, models[key], transition)
    axes[0].set_ylabel("HPSv3 reward  (mean ± SEM)")

    legend = [
        Line2D([0], [0], color=C_MCTS, lw=2.6, label="MCTS (single tree)"),
        Line2D([0], [0], color=C_BONMCTS, lw=2.6, label="BoN-MCTS (best-of-N trees)"),
        Patch(facecolor=C_MCTS_BAND, alpha=0.5, label="MCTS regime"),
        Patch(facecolor=C_BON_BAND, alpha=0.5, label="BoN-MCTS regime"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=9.5)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.20, wspace=0.12)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description="Test-time scaling plot (MCTS -> BoN-MCTS); qualitative or measured.")
    p.add_argument("--out", default=None, type=Path,
                   help="Output PNG. When omitted, writes <out_dir>/tts_scaling_<timestamp>.png.")
    p.add_argument("--out_dir", default=Path("figures/raw"), type=Path)
    p.add_argument("--save_json", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--models", nargs="+", default=["flux", "sid35"], choices=list(MODELS.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--title", default="Test-time scaling: MCTS → BoN-MCTS")
    p.add_argument("--data_json", default=None, type=Path,
                   help="Measured-data JSON (HPSv3 vs budget per model). When set, "
                        "overrides the synthetic curves and plots real points + SEM band.")
    args = p.parse_args()

    if args.data_json is not None:
        data = json.loads(Path(args.data_json).read_text())
        fig = render_real(data, args.title)
    else:
        fig = render(args.models, args.seed, args.title)

    is_real = args.data_json is not None
    prefix = "tts_scaling_real" if is_real else "tts_scaling"
    if args.out is not None:
        out_png = args.out
        out_png.parent.mkdir(parents=True, exist_ok=True)
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_png = args.out_dir / f"{prefix}_{ts}.png"

    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    print(f"[tts] wrote {out_png}")

    if args.save_json:
        out_json = out_png.with_suffix(".json")
        if is_real:
            payload = {
                "kind": "measured_test_time_scaling",
                "note": "HPSv3 reward vs. test-time compute (measured).",
                "title": args.title,
                "source_data_json": str(args.data_json),
                **data,
            }
        else:
            payload = {
                "kind": "qualitative_test_time_scaling",
                "note": "Illustrative curves, not measured benchmark data.",
                "title": args.title,
                "seed": args.seed,
                "models": {k: MODELS[k] for k in args.models},
            }
        out_json.write_text(json.dumps(payload, indent=2))
        print(f"[tts] wrote {out_json}")


if __name__ == "__main__":
    main()
