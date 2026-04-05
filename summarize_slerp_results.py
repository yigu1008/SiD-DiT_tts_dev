"""
Summarize results from a sandbox_slerp_nlerp_unified_sana output directory.

Reads result.json from each p*/  sub-directory.  When design-ablation files
are present (mcts_variant4_action_search.json, mcts_globalblend_*_action_search.json)
they are loaded automatically — no extra flags needed.

Usage:
    python summarize_slerp_results.py --out_dir ./sandbox_slerp_nlerp_unified_sana_out
    python summarize_slerp_results.py --out_dir ./out --csv results.csv
    python summarize_slerp_results.py --out_dir ./out --top 10 --sort_by mcts
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


# ── helpers ────────────────────────────────────────────────────────────────────

def _f(v: Any, fmt: str = ".4f") -> str:
    if v is None:
        return "n/a"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [warn] {path.name}: {e}", file=sys.stderr)
        return None


def _best_score(d: dict | None) -> float | None:
    """Extract best/final score from a search result dict."""
    if not isinstance(d, dict):
        return None
    for k in ("final_score", "best_score"):
        if k in d:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                pass
    return None


def _best_sweep_score(sweep: list) -> float | None:
    scores = [float(r["score"]) for r in (sweep or []) if isinstance(r.get("score"), (int, float))]
    return max(scores) if scores else None


def _best_sweep_entry(sweep: list) -> dict | None:
    if not sweep:
        return None
    return max(sweep, key=lambda r: float(r.get("score", -1e9)))


# ── per-prompt loading ─────────────────────────────────────────────────────────

def load_prompt_row(prompt_dir: Path) -> dict | None:
    data = _load_json(prompt_dir / "result.json")
    if data is None:
        return None

    slug = str(data.get("slug", prompt_dir.name))
    prompt = str(data.get("prompt", ""))
    baseline = data.get("baseline_score")
    baseline = float(baseline) if baseline is not None else None

    best_sweep = _best_sweep_score(data.get("sweep") or [])
    best_entry = _best_sweep_entry(data.get("sweep") or [])

    # GA
    ga_score = _best_score(data.get("ga"))
    if ga_score is None:
        ga_score = _best_score(_load_json(prompt_dir / "ga_action_search.json"))

    # Main MCTS
    mcts_score = _best_score(data.get("mcts"))
    if mcts_score is None:
        mcts_score = _best_score(_load_json(prompt_dir / "mcts_action_search.json"))

    # Weight ablation (mcts_fixed / mcts_spsa)
    abl = data.get("mcts_ablation") or {}
    mcts_fixed = _best_score(abl.get("fixed"))
    mcts_spsa  = _best_score(abl.get("spsa"))

    # Design ablation — prefer nested result.json keys, fall back to side-car files
    da = data.get("mcts_design_ablation") or {}
    mcts_variant4 = (
        _best_score(da.get("variant4"))
        or _best_score(_load_json(prompt_dir / "mcts_variant4_action_search.json"))
    )
    mcts_gb_fixed = (
        _best_score(da.get("global_blend_fixed"))
        or _best_score(_load_json(prompt_dir / "mcts_globalblend_fixed_action_search.json"))
    )
    mcts_gb_spsa = (
        _best_score(da.get("global_blend_spsa"))
        or _best_score(_load_json(prompt_dir / "mcts_globalblend_spsa_action_search.json"))
    )
    mcts_gb_nlerp = (
        _best_score(da.get("global_blend_nlerp_spsa"))
        or _best_score(_load_json(prompt_dir / "mcts_globalblend_nlerp_spsa_action_search.json"))
    )
    mcts_gb_slerp = (
        _best_score(da.get("global_blend_slerp_spsa"))
        or _best_score(_load_json(prompt_dir / "mcts_globalblend_slerp_spsa_action_search.json"))
    )

    candidates = [x for x in [
        baseline, best_sweep, ga_score, mcts_score,
        mcts_fixed, mcts_spsa, mcts_variant4,
        mcts_gb_fixed, mcts_gb_spsa, mcts_gb_nlerp, mcts_gb_slerp,
    ] if x is not None]
    overall_best = max(candidates) if candidates else None
    delta_best   = (overall_best - baseline) if (overall_best is not None and baseline is not None) else None

    return {
        "slug": slug,
        "prompt": prompt[:80],
        "baseline": baseline,
        "best_sweep": best_sweep,
        "best_sweep_family": best_entry.get("family") if best_entry else None,
        "best_sweep_profile": best_entry.get("profile") if best_entry else None,
        "best_sweep_cfg": best_entry.get("cfg") if best_entry else None,
        "ga": ga_score,
        "mcts": mcts_score,
        "mcts_fixed": mcts_fixed,
        "mcts_spsa": mcts_spsa,
        "mcts_variant4": mcts_variant4,
        "mcts_gb_fixed": mcts_gb_fixed,
        "mcts_gb_spsa": mcts_gb_spsa,
        "mcts_gb_nlerp_spsa": mcts_gb_nlerp,
        "mcts_gb_slerp_spsa": mcts_gb_slerp,
        "overall_best": overall_best,
        "delta_vs_baseline": delta_best,
    }


# ── stats ──────────────────────────────────────────────────────────────────────

def _mean(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def _pct_positive(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return 100.0 * sum(1 for x in v if x > 0) / len(v) if v else None


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize slerp/nlerp sandbox results.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--csv", default=None, help="Write per-prompt CSV to this path.")
    p.add_argument("--top", type=int, default=0, help="Print top-N rows (0 = all).")
    p.add_argument("--sort_by", choices=["baseline", "ga", "mcts", "sweep", "best"], default="best")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        sys.exit(f"Error: {out_dir} is not a directory.")

    prompt_dirs = sorted(d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith("p"))
    if not prompt_dirs:
        sys.exit(f"No p* directories found under {out_dir}.")

    rows: list[dict] = []
    for pd in prompt_dirs:
        row = load_prompt_row(pd)
        if row is not None:
            rows.append(row)

    if not rows:
        sys.exit("No result.json files found.")

    # Auto-detect which columns have any data
    has_ga          = any(r["ga"] is not None for r in rows)
    has_mcts        = any(r["mcts"] is not None for r in rows)
    has_sweep       = any(r["best_sweep"] is not None for r in rows)
    has_mcts_fixed  = any(r["mcts_fixed"] is not None for r in rows)
    has_mcts_spsa   = any(r["mcts_spsa"] is not None for r in rows)
    has_variant4    = any(r["mcts_variant4"] is not None for r in rows)
    has_gb_fixed    = any(r["mcts_gb_fixed"] is not None for r in rows)
    has_gb_spsa     = any(r["mcts_gb_spsa"] is not None for r in rows)
    has_gb_nlerp    = any(r["mcts_gb_nlerp_spsa"] is not None for r in rows)
    has_gb_slerp    = any(r["mcts_gb_slerp_spsa"] is not None for r in rows)
    has_design_abl  = has_variant4 or has_gb_fixed or has_gb_spsa

    # ── sort ─────────────────────────────────────────────────────────────────
    sort_key = {
        "baseline": lambda r: r["baseline"] or -1e9,
        "ga":       lambda r: r["ga"] or -1e9,
        "mcts":     lambda r: r["mcts"] or -1e9,
        "sweep":    lambda r: r["best_sweep"] or -1e9,
        "best":     lambda r: r["overall_best"] or -1e9,
    }[args.sort_by]
    rows.sort(key=sort_key, reverse=True)
    display = rows[: args.top] if args.top > 0 else rows

    # ── build column spec ────────────────────────────────────────────────────
    # (label, row_key)
    cols = [("baseline", "baseline")]
    if has_sweep:       cols.append(("sweep_best", "best_sweep"))
    if has_ga:          cols.append(("ga", "ga"))
    if has_mcts:        cols.append(("mcts", "mcts"))
    if has_mcts_fixed:  cols.append(("mcts_fixed", "mcts_fixed"))
    if has_mcts_spsa:   cols.append(("mcts_spsa", "mcts_spsa"))
    if has_variant4:    cols.append(("variant4", "mcts_variant4"))
    if has_gb_fixed:    cols.append(("gb_fixed", "mcts_gb_fixed"))
    if has_gb_spsa:     cols.append(("gb_spsa", "mcts_gb_spsa"))
    if has_gb_nlerp:    cols.append(("gb_nlerp", "mcts_gb_nlerp_spsa"))
    if has_gb_slerp:    cols.append(("gb_slerp", "mcts_gb_slerp_spsa"))
    cols.append(("best", "overall_best"))
    cols.append(("Δbest", "delta_vs_baseline"))

    W = 9
    header = f"{'slug':<8}  " + "  ".join(f"{lbl:>{W}}" for lbl, _ in cols)
    sep = "-" * len(header)

    print(f"\nResults : {out_dir}")
    print(f"Prompts : {len(rows)} found, {len(display)} displayed  |  sort={args.sort_by}")
    if has_design_abl:
        active = [l for l, _ in cols if l.startswith(("variant", "gb_"))]
        print(f"Design ablation columns detected: {' '.join(active)}")
    print(sep)
    print(header)
    print(sep)
    for r in display:
        vals = []
        for lbl, key in cols:
            fmt = "+.4f" if key == "delta_vs_baseline" else ".4f"
            vals.append(f"{_f(r[key], fmt):>{W}}")
        print(f"{r['slug']:<8}  " + "  ".join(vals))
    print(sep)

    # ── aggregate ────────────────────────────────────────────────────────────
    stat_rows = [
        ("baseline",      [r["baseline"] for r in rows],           False),
        ("sweep_best",    [r["best_sweep"] for r in rows],         False),
        ("ga",            [r["ga"] for r in rows],                 False),
        ("mcts",          [r["mcts"] for r in rows],               False),
        ("mcts_fixed",    [r["mcts_fixed"] for r in rows],         False),
        ("mcts_spsa",     [r["mcts_spsa"] for r in rows],          False),
        ("variant4",      [r["mcts_variant4"] for r in rows],      False),
        ("gb_fixed",      [r["mcts_gb_fixed"] for r in rows],      False),
        ("gb_spsa",       [r["mcts_gb_spsa"] for r in rows],       False),
        ("gb_nlerp_spsa", [r["mcts_gb_nlerp_spsa"] for r in rows], False),
        ("gb_slerp_spsa", [r["mcts_gb_slerp_spsa"] for r in rows], False),
        ("overall_best",  [r["overall_best"] for r in rows],       False),
        ("delta_vs_base", [r["delta_vs_baseline"] for r in rows],  True),
    ]
    print("\nAggregate (all prompts):")
    print(f"  {'method':<20}  {'mean':>9}  {'n':>5}  {'%>0':>6}")
    print(f"  {'-'*20}  {'-'*9}  {'-'*5}  {'-'*6}")
    for label, vals, show_pct in stat_rows:
        n = sum(1 for x in vals if x is not None)
        if n == 0:
            continue
        m = _mean(vals)
        pct_str = f"{_pct_positive(vals):>5.1f}%" if show_pct else "      "
        print(f"  {label:<20}  {_f(m):>9}  {n:>5}  {pct_str}")

    # Best sweep breakdown by family
    fam_scores: dict[str, list[float]] = {}
    for r in rows:
        if r["best_sweep_family"] and r["best_sweep"] is not None:
            fam_scores.setdefault(r["best_sweep_family"], []).append(float(r["best_sweep"]))
    if fam_scores:
        print("\nBest-sweep by interp family:")
        for fam, scores in sorted(fam_scores.items()):
            print(f"  {fam:<12}  mean={_f(_mean(scores))}  n={len(scores)}")

    # ── CSV ──────────────────────────────────────────────────────────────────
    if args.csv:
        csv_path = Path(args.csv)
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written: {csv_path}")


if __name__ == "__main__":
    main()
