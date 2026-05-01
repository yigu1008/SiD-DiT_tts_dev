#!/usr/bin/env python3
"""Crawl SD3.5 + FLUX sweep outputs and emit one canonical NFE-vs-reward CSV.

Canonical schema (frozen for the all-models NFE-vs-reward plot):

    backend, method, search_reward,
    nfe_transformer,    # mean transformer forward count per prompt
    nfe_reward_eval,    # analytical reward-eval count per prompt (see methods)
    uses_reward_diff,   # 1 if method backprops through reward, else 0
    mean_search,        # search-time reward (y-axis)
    mean_baseline,
    mean_delta,
    n_prompts,
    knob,               # the swept-over value (target_nfe / score_every / K:M)
    status,
    run_path

Inputs are sweep_root directories produced by nfe_sweep_sd35.sh / nfe_sweep_flux.sh.
The crawler walks `<sweep_root>/<method>_nfe<X>/run_*/<method>/` looking for
aggregate_ddp.json + best_images_multi_reward_aggregate.json.

`search_reward` is read from a sibling `meta.json` if present, else inferred
from the directory path; if both fail it is left blank for the user to fill.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

REWARD_DIFF_METHODS = {"smc", "fksteering"}


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // max(1, int(b)))


def _nfe_for(method: str, knob: str, steps: int, baseline_cfg: float, beam_cfg_n: int = 2) -> tuple[int, int]:
    """Return (nfe_transformer, nfe_reward_eval) per prompt for this (method, knob, steps).

    Analytical: matches the formulas documented in nfe_sweep_sd35.sh.
    For methods with cfg_scale != 0/1, every transformer call is doubled
    (uncond + cond) — we don't double here; sweep budgets are already in
    "transformer-call" units.
    """
    s = int(steps)
    try:
        target = int(knob)
    except (TypeError, ValueError):
        target = 0  # used for sop K:M parsing below

    if method == "baseline":
        return s, 1
    if method == "bon":
        n = max(1, _ceil_div(target, s))
        return n * s, n
    if method == "beam":
        denom = s * max(1, beam_cfg_n)
        w = max(1, _ceil_div(target, denom))
        return w * beam_cfg_n * s, w * beam_cfg_n * s
    if method in {"smc", "fksteering"}:
        k = max(2, _ceil_div(target, s))
        return k * s, k * s
    if method == "greedy":
        n = max(1, _ceil_div(target, s))
        return n * s, n * s
    if method == "ga":
        gens = 8  # matches GA_GENERATIONS_SWEEP default
        denom = gens * s
        p = max(2, _ceil_div(target, denom))
        return p * gens * s, p * gens
    if method in {"dts", "dts_star"}:
        m = max(1, _ceil_div(target, s))
        return m * s, m * s
    if method == "dynamic_cfg_x0":
        # knob is score_every. Grid size + start_frac assumed already configured;
        # we report transformer = STEPS (cfg-split shared), reward_eval analytical
        # using grid size 4 (per the SD3.5/SID/SCHNELL banks) and a default
        # 0.5 start_frac (sid/senseflow/schnell) or 0.25 (sd35_base).
        every = max(1, target)
        scored_steps = max(1, _ceil_div(int(0.75 * s if s >= 8 else 0.5 * s), every))
        grid_size = 4
        return s, scored_steps * grid_size
    if method == "sop":
        # knob is "K:M".
        m = re.match(r"(\d+):(\d+)", str(knob))
        if not m:
            return 0, 0
        k_val, m_val = int(m.group(1)), int(m.group(2))
        # start_frac 0.5 for 4-step, 0.25 for 28-step.
        pre = int(0.5 * s) if s <= 8 else int(0.25 * s)
        branch = s - pre
        return k_val * pre + k_val * m_val * branch, k_val * m_val * branch + k_val
    return 0, 0


def _infer_search_reward(run_path: Path) -> str:
    """Infer search_reward from path tokens; fall back to empty string."""
    parts = run_path.parts
    for token in ("hpsv3", "imagereward", "hpsv2", "pickscore"):
        if token in parts:
            return token
    return ""


def _read_meta(method_dir: Path) -> dict:
    for candidate in (method_dir, method_dir.parent, method_dir.parent.parent):
        m = candidate / "meta.json"
        if m.exists():
            try:
                return json.loads(m.read_text())
            except Exception:
                pass
    return {}


def _crawl_sweep_root(sweep_root: Path, default_backend: str) -> list[dict]:
    rows: list[dict] = []
    for cfg_dir in sorted(sweep_root.glob("*_nfe*")):
        label = cfg_dir.name
        m = re.match(r"(.+)_nfe(.+)", label)
        if not m:
            continue
        method, knob = m.group(1), m.group(2)
        if method == "noise_inject" or method == "mcts" or method == "bon_mcts":
            continue  # excluded per spec
        # Locate the actual method dir (one level under run_*/<method>/).
        method_dir = None
        for run_dir in sorted(cfg_dir.glob("run_*")):
            for cand in run_dir.iterdir():
                if not cand.is_dir():
                    continue
                if (cand / "aggregate_ddp.json").exists():
                    method_dir = cand
                    break
            if method_dir:
                break
        if not method_dir:
            rows.append({
                "backend": default_backend, "method": method, "knob": knob,
                "status": "missing",
            })
            continue

        agg = json.loads((method_dir / "aggregate_ddp.json").read_text())
        eval_means: dict[str, float] = {}
        eval_path = method_dir / "best_images_multi_reward_aggregate.json"
        if eval_path.exists():
            stats = json.loads(eval_path.read_text()).get("backend_stats", {}) or {}
            for b, s in stats.items():
                if isinstance(s, dict) and s.get("mean") is not None:
                    eval_means[b] = float(s["mean"])

        meta = _read_meta(method_dir)
        search_reward = (
            meta.get("search_reward")
            or _infer_search_reward(method_dir)
        )

        # Restore "fksteering" label even though runner method is "smc".
        canonical_method = method
        if method == "fksteering":
            canonical_method = "fksteering"

        # Lookup default STEPS/baseline for NFE reconstruction.
        steps_lookup = {
            "sid": 4, "senseflow_large": 4, "sd35_base": 28,
            "flux_schnell": 4, "flux": 28, "tdd_flux": 8,
        }
        steps = steps_lookup.get(default_backend, 4)
        nfe_xfm, nfe_rev = _nfe_for(canonical_method, knob, steps, 0.0)

        row = {
            "backend": default_backend,
            "method": canonical_method,
            "search_reward": search_reward,
            "nfe_transformer": nfe_xfm,
            "nfe_reward_eval": nfe_rev,
            "uses_reward_diff": 1 if canonical_method in REWARD_DIFF_METHODS else 0,
            "mean_search": agg.get("mean_search_score"),
            "mean_baseline": agg.get("mean_baseline_score"),
            "mean_delta": agg.get("mean_delta_score"),
            "n_prompts": agg.get("num_samples"),
            "knob": knob,
            "status": "ok",
            "run_path": str(method_dir),
        }
        for ev_b, ev_v in eval_means.items():
            row[f"eval_{ev_b}"] = ev_v
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root", action="append", default=[],
        help="Sweep root dir(s). Each is <OUT_ROOT_BASE>/<backend>/sweep_<TS>. "
             "Pass multiple --root flags or use --auto.",
    )
    p.add_argument(
        "--auto", default=None,
        help="Glob root containing per-backend sweep_<TS> dirs (e.g. /tmp/sd35_nfe_sweep).",
    )
    p.add_argument("--out", default="combined.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    roots: list[tuple[Path, str]] = []
    for r in args.root:
        rp = Path(r)
        backend = rp.parent.name if rp.parent.name not in {"", "."} else "unknown"
        roots.append((rp, backend))
    if args.auto:
        for sweep in sorted(Path(args.auto).glob("*/sweep_*")):
            roots.append((sweep, sweep.parent.name))

    if not roots:
        print("No --root or --auto inputs given", file=sys.stderr)
        sys.exit(1)

    all_rows: list[dict] = []
    eval_cols: set[str] = set()
    for sweep_root, backend in roots:
        if not sweep_root.exists():
            print(f"[crawl] WARN missing: {sweep_root}", file=sys.stderr)
            continue
        rows = _crawl_sweep_root(sweep_root, backend)
        for r in rows:
            for k in r:
                if k.startswith("eval_"):
                    eval_cols.add(k)
        all_rows.extend(rows)
        print(f"[crawl] {sweep_root} ({backend}): {len(rows)} rows")

    base_cols = [
        "backend", "method", "search_reward",
        "nfe_transformer", "nfe_reward_eval", "uses_reward_diff",
        "mean_search", "mean_baseline", "mean_delta",
        "n_prompts", "knob", "status", "run_path",
    ]
    cols = base_cols + sorted(eval_cols)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({c: r.get(c, "") for c in cols})
    print(f"[crawl] wrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
