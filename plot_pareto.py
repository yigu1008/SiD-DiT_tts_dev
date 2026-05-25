#!/usr/bin/env python3
"""Pareto-frontier analysis across (imagereward, hpsv3, hpsv2, pickscore).

Reads per-image scores from the eval JSONs each method emits (one per
prompt, with all 4 rewards present), aggregates per-method, and renders:

  - A 2x2 grid of pairwise reward-vs-reward scatter plots, with each
    method's Pareto frontier traced in its own color.  The "both"/"ours"
    method is bolded.
  - A table (printed + CSV) of per-method per-reward means, plus a
    4-D hypervolume estimate (relative to a fixed reference point).

Inputs: a run root with subdirs `<run_root>/<method>/...` containing
        `best_images_multi_reward.json` (per-image) or
        `best_images_multi_reward_aggregate.json` (per-method summary).

Usage:
    python plot_pareto.py --run_root <RUN_ROOT> \
        [--methods baseline bon_mcts bon_mcts_full ...] \
        [--out_png <RUN_ROOT>/pareto.png] \
        [--ours bon_mcts_full]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REWARDS = ("imagereward", "hpsv3", "hpsv2", "pickscore")

_COLOR_CYCLE = [
    "#7E8489", "#3D77BE", "#E08540", "#B73B3B",
    "#6A5ACD", "#2BA065", "#A3B72E", "#7F3A8D",
    "#1F77B4", "#FF7F0E",
]


def _as_scalar(x) -> float | None:
    """Best-effort scalar extraction from int / float / dict / list / nested.

    Many eval JSONs store per-image scores as `{reward: {"score": 0.42}}` or
    `{reward: {"value": 0.42, "raw": ...}}` rather than plain floats. We try
    common scalar keys in order before giving up.
    """
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        for k in ("score", "value", "mean", "reward", "ir", "hps"):
            if k in x:
                return _as_scalar(x[k])
        return None
    if isinstance(x, list) and len(x) > 0:
        return _as_scalar(x[0])
    return None


def _per_image_jsons(run_root: Path, method: str) -> list[Path]:
    out: list[Path] = []
    for pat in (f"**/{method}/**/best_images_multi_reward.json",
                f"**/{method}/**/best_images_multi_reward_aggregate.json"):
        out.extend(sorted(run_root.glob(pat)))
    return out


def _load_method_scores(run_root: Path, method: str) -> list[dict[str, float]]:
    """Return list of {reward: value} dicts, one per image."""
    rows: list[dict[str, float]] = []
    seen_paths: set[str] = set()
    for jp in _per_image_jsons(run_root, method):
        # Prefer the per-image file over its aggregate sibling — if both
        # exist in the same dir we don't want to double-count the aggregate's
        # backend_stats means as "extra" data.
        agg_sibling = jp.parent / "best_images_multi_reward_aggregate.json"
        if jp == agg_sibling and str(jp.parent) in seen_paths:
            continue
        if jp.name == "best_images_multi_reward.json":
            seen_paths.add(str(jp.parent))
        try:
            data = json.loads(jp.read_text())
        except Exception as exc:
            print(f"    skip {jp}: {type(exc).__name__}: {exc}")
            continue
        # Case A: top-level per-image array.
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                d: dict[str, float] = {}
                for r in REWARDS:
                    v = item.get(r)
                    if v is None and isinstance(item.get("scores"), dict):
                        v = item["scores"].get(r)
                    s = _as_scalar(v)
                    if s is not None:
                        d[r] = s
                if len(d) >= 2:
                    rows.append(d)
            continue
        if not isinstance(data, dict):
            continue
        # Case B0 (preferred): eval-script layout {"aggregate":…, "rows":[…]}.
        if isinstance(data.get("rows"), list):
            for item in data["rows"]:
                if not isinstance(item, dict):
                    continue
                scores = item.get("scores") if isinstance(item.get("scores"), dict) else item
                d = {}
                for r in REWARDS:
                    v = scores.get(r) if isinstance(scores, dict) else None
                    s = _as_scalar(v)
                    if s is not None:
                        d[r] = s
                if len(d) >= 2:
                    rows.append(d)
            continue
        # Case B1: aggregate-only file — {"backend_stats": {reward: {mean,…}}}.
        # No per-image values exist; fall back to 1 point per method using means.
        if isinstance(data.get("backend_stats"), dict):
            bs = data["backend_stats"]
            d = {}
            for r in REWARDS:
                rec = bs.get(r)
                s = _as_scalar(rec)  # picks up .mean via _as_scalar
                if s is not None:
                    d[r] = s
            if len(d) >= 2:
                rows.append(d)
            continue
        # Case B2: legacy {reward: {"mean": X, "values": [...]}} layout.
        values_lists: dict[str, list[float]] = {}
        for r in REWARDS:
            rec = data.get(r)
            arr = None
            if isinstance(rec, dict) and isinstance(rec.get("values"), list):
                arr = rec["values"]
            elif isinstance(rec, list):
                arr = rec
            if arr is None:
                continue
            keep = []
            for v in arr:
                s = _as_scalar(v)
                if s is not None:
                    keep.append(s)
            if keep:
                values_lists[r] = keep
        if values_lists:
            n = min(len(v) for v in values_lists.values())
            for i in range(n):
                d = {r: values_lists[r][i] for r in values_lists}
                rows.append(d)
            continue
        # Case C (last resort): per-image dict — {image_name: {reward: …}}.
        for v in data.values():
            if not isinstance(v, dict):
                continue
            d = {}
            for r in REWARDS:
                if r not in v:
                    continue
                s = _as_scalar(v[r])
                if s is not None:
                    d[r] = s
            if len(d) >= 2:
                rows.append(d)
    return rows


def _pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the upper-right Pareto frontier (max-max objective)."""
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda p: (-p[0], -p[1]))
    front: list[tuple[float, float]] = []
    best_y = -float("inf")
    for x, y in sorted_pts:
        if y > best_y:
            front.append((x, y))
            best_y = y
    return sorted(front, key=lambda p: p[0])


def _hypervolume_4d(points: list[dict[str, float]], ref: dict[str, float]) -> float:
    """Cheap upper bound on 4D hypervolume — sum of per-point boxes minus
    intersections is exact only for axis-aligned-non-dominated points.
    Here we use sum of (p - ref) products over a Pareto-filtered set.
    """
    if not points:
        return 0.0
    # Filter to 4D Pareto front (max each dim).
    pts = [tuple(float(p.get(r, ref[r])) for r in REWARDS) for p in points]
    n = len(pts)
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if all(pts[j][k] >= pts[i][k] for k in range(4)) and any(pts[j][k] > pts[i][k] for k in range(4)):
                keep[i] = False
                break
    front = [pts[i] for i in range(n) if keep[i]]
    # Approximation: inclusion-exclusion is exponential; instead report the
    # sum-of-boxes (loose upper bound).  This gives a comparable scalar across
    # methods.  For exact HV, swap in pygmo / pymoo if installed.
    total = 0.0
    ref_t = tuple(ref[r] for r in REWARDS)
    for p in front:
        prod = 1.0
        for k in range(4):
            prod *= max(0.0, p[k] - ref_t[k])
        total += prod
    return total


def _set_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_root", required=True, type=Path)
    p.add_argument("--methods", nargs="+", default=None,
                   help="Methods to include.  Defaults to all subdirs with scores.")
    p.add_argument("--ours", default="bon_mcts_full",
                   help="Method to bold/highlight in legend.")
    p.add_argument("--out_png", default=None)
    p.add_argument("--out_csv", default=None)
    args = p.parse_args()

    # Auto-discover methods if not given.
    if args.methods is None:
        methods = sorted({pp.parts[-2] for pat in ("best_images_multi_reward*.json",)
                          for pp in args.run_root.glob(f"**/{pat}")})
        methods = [m for m in methods if not m.startswith("_") and m != "_prompts"]
    else:
        methods = list(args.methods)
    print(f"# methods: {methods}")

    method_data: dict[str, list[dict[str, float]]] = {}
    for m in methods:
        rows = _load_method_scores(args.run_root, m)
        if rows:
            method_data[m] = rows
            print(f"  {m:30s}  n_images={len(rows)}")
    if not method_data:
        raise SystemExit("No method scores found.  Check --run_root and method names.")

    # Per-method means table
    means: dict[str, dict[str, float]] = {}
    for m, rows in method_data.items():
        avg = {}
        for r in REWARDS:
            vals = [row[r] for row in rows if r in row and row[r] is not None]
            avg[r] = sum(vals) / len(vals) if vals else float("nan")
        means[m] = avg

    print("\n# per-method means")
    print(f"  {'method':30s}  " + "  ".join(f"{r:>11s}" for r in REWARDS) + "  hv4d")
    # Reference point for HV: floor across all methods + a small margin
    ref = {r: min(means[m][r] for m in means) - 0.01 for r in REWARDS}
    for m in methods:
        if m not in means:
            continue
        hv = _hypervolume_4d(method_data[m], ref)
        row = f"  {m:30s}  " + "  ".join(f"{means[m][r]:>11.4f}" for r in REWARDS) + f"  {hv:.4f}"
        if m == args.ours:
            row = row + "  ← ours"
        print(row)

    # CSV out
    out_csv = Path(args.out_csv) if args.out_csv else args.run_root / "pareto_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + list(REWARDS) + ["hv4d", "n_images"])
        for m in methods:
            if m not in means:
                continue
            hv = _hypervolume_4d(method_data[m], ref)
            w.writerow([m] + [means[m][r] for r in REWARDS] + [hv, len(method_data[m])])
    print(f"\n  csv: {out_csv}")

    # Pairwise Pareto plot
    _set_paper_style()
    pairs = [("imagereward", "hpsv3"), ("imagereward", "hpsv2"),
             ("imagereward", "pickscore"), ("hpsv3", "hpsv2"),
             ("hpsv3", "pickscore"), ("hpsv2", "pickscore")]
    # Pre-flight: which rewards have ANY data across methods?
    reward_has_data = {r: False for r in REWARDS}
    for m in methods:
        if m not in method_data:
            continue
        for row in method_data[m]:
            for r in REWARDS:
                if r in row and row[r] is not None:
                    reward_has_data[r] = True
    missing_rewards = [r for r, has in reward_has_data.items() if not has]
    if missing_rewards:
        print(f"\n  ⚠ no data for rewards: {missing_rewards} — panels involving them will be annotated.")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (ra, rb) in zip(axes.flat, pairs):
        if not reward_has_data[ra] or not reward_has_data[rb]:
            ax.text(0.5, 0.5,
                    f"no data\n({ra if not reward_has_data[ra] else rb} missing)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#888888", style="italic")
            ax.set_xlabel(ra); ax.set_ylabel(rb)
            ax.set_title(f"{ra} × {rb}")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        for i, m in enumerate(methods):
            if m not in method_data:
                continue
            color = _COLOR_CYCLE[i % len(_COLOR_CYCLE)]
            is_ours = (m == args.ours)
            xs = [row[ra] for row in method_data[m] if ra in row and rb in row]
            ys = [row[rb] for row in method_data[m] if ra in row and rb in row]
            ax.scatter(xs, ys,
                       s=22 if is_ours else 12,
                       alpha=0.7,
                       color=color,
                       edgecolor="black" if is_ours else "none",
                       linewidth=0.5 if is_ours else 0,
                       label=(f"{m} (ours)" if is_ours else m))
            front = _pareto_frontier(list(zip(xs, ys)))
            if front:
                fx = [p[0] for p in front]
                fy = [p[1] for p in front]
                ax.plot(fx, fy,
                        "-",
                        color=color,
                        lw=2.4 if is_ours else 1.4,
                        alpha=0.9)
        ax.set_xlabel(ra); ax.set_ylabel(rb)
        ax.set_title(f"{ra} × {rb}")
    axes[0, 0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Pareto frontiers across 4 rewards", fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    out_png = Path(args.out_png) if args.out_png else args.run_root / "pareto.png"
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    print(f"  png: {out_png}")


if __name__ == "__main__":
    main()
