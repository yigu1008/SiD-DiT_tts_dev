#!/usr/bin/env python3
"""Aggregate per-seed method runs into across-run error bars.

Reads the per-seed RUN_ROOTs produced by run_all_method_ablations.sh
(parent/seed_<S>/.../<method>/.../best_images_multi_reward_aggregate.json)
and reports, per (method, backend), the mean of the per-run means together
with the across-run sample std and standard error (SEM = std / sqrt(n_runs)).

The eval JSON already carries a within-run mean/std over prompts; this script
adds the across-run uncertainty that you actually want as an error bar when
comparing methods, by treating each seed as one independent replicate.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics


def _find_eval_json(seed_dir: str, method: str) -> str | None:
    pat = os.path.join(seed_dir, "**", method, "**", "best_images_multi_reward_aggregate.json")
    hits = sorted(glob.glob(pat, recursive=True))
    return hits[0] if hits else None


def _find_search_json(seed_dir: str, method: str) -> str | None:
    pat = os.path.join(seed_dir, "**", method, "**", "aggregate_ddp.json")
    hits = sorted(glob.glob(pat, recursive=True))
    return hits[0] if hits else None


def _across_run_stats(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n_runs": 0, "mean": None, "std": None, "sem": None, "per_run": []}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0  # sample std (ddof=1)
    sem = (std / math.sqrt(n)) if n > 1 else 0.0
    return {
        "n_runs": n,
        "mean": float(mean),
        "std": float(std),
        "sem": float(sem),
        "per_run": [float(v) for v in values],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", required=True, help="Dir containing seed_<S> subdirs.")
    ap.add_argument("--methods", required=True, help="Space-separated method names.")
    ap.add_argument("--backends", default="imagereward hpsv3", help="Space-separated eval backends.")
    ap.add_argument("--out_json", default=None, help="Defaults to <parent>/errorbars.json")
    ap.add_argument("--out_csv", default=None, help="Defaults to <parent>/errorbars.csv")
    args = ap.parse_args()

    methods = args.methods.split()
    backends = args.backends.split()
    out_json = args.out_json or os.path.join(args.parent, "errorbars.json")
    out_csv = args.out_csv or os.path.join(args.parent, "errorbars.csv")

    seed_dirs = sorted(glob.glob(os.path.join(args.parent, "seed_*")))
    if not seed_dirs:
        raise SystemExit(f"[errorbars] no seed_* dirs under {args.parent}")
    print(f"[errorbars] {len(seed_dirs)} seed dirs: {[os.path.basename(d) for d in seed_dirs]}")

    report: dict = {"parent": args.parent, "seed_dirs": seed_dirs, "methods": {}}
    rows: list[dict] = []

    for method in methods:
        report["methods"][method] = {}
        # Collect per-run search reward + per-backend eval means.
        search_vals: list[float] = []
        backend_vals: dict[str, list[float]] = {b: [] for b in backends}
        for sd in seed_dirs:
            sj = _find_search_json(sd, method)
            if sj and os.path.isfile(sj):
                try:
                    sd_json = json.load(open(sj))
                    # aggregate_ddp.json stores the key as "mean_search_score";
                    # fall back to "mean_search" for any older runs.
                    ms = sd_json.get("mean_search_score", sd_json.get("mean_search"))
                    if ms is not None:
                        search_vals.append(float(ms))
                except (ValueError, OSError, KeyError):
                    pass
            ej = _find_eval_json(sd, method)
            if not (ej and os.path.isfile(ej)):
                print(f"[errorbars]   WARN missing eval json for method={method} in {os.path.basename(sd)}")
                continue
            try:
                d = json.load(open(ej))
            except (ValueError, OSError):
                print(f"[errorbars]   WARN unreadable eval json {ej}")
                continue
            # best_images_multi_reward_aggregate.json nests per-backend means
            # under "backend_stats"; older flat-keyed runs fall back to top level.
            stats = d.get("backend_stats", d)
            for b in backends:
                v = stats.get(b, {}).get("mean")
                if v is not None:
                    backend_vals[b].append(float(v))

        report["methods"][method]["search_reward"] = _across_run_stats(search_vals)
        report["methods"][method]["eval"] = {}
        for b in backends:
            st = _across_run_stats(backend_vals[b])
            report["methods"][method]["eval"][b] = st
            rows.append({
                "method": method,
                "metric": f"eval_{b}",
                "n_runs": st["n_runs"],
                "mean": st["mean"],
                "std": st["std"],
                "sem": st["sem"],
                "per_run": " ".join(f"{x:.5f}" for x in st["per_run"]),
            })
        st = report["methods"][method]["search_reward"]
        rows.append({
            "method": method,
            "metric": "search_reward",
            "n_runs": st["n_runs"],
            "mean": st["mean"],
            "std": st["std"],
            "sem": st["sem"],
            "per_run": " ".join(f"{x:.5f}" for x in st["per_run"]),
        })

    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "metric", "n_runs", "mean", "std", "sem", "per_run"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[errorbars] wrote {out_json}")
    print(f"[errorbars] wrote {out_csv}")
    print(f"{'method':<14} {'metric':<16} {'n':>3} {'mean':>10} {'std':>10} {'sem':>10}")
    for r in rows:
        if r["mean"] is None:
            print(f"{r['method']:<14} {r['metric']:<16} {r['n_runs']:>3} {'--':>10}")
            continue
        print(f"{r['method']:<14} {r['metric']:<16} {r['n_runs']:>3} "
              f"{r['mean']:>10.5f} {r['std']:>10.5f} {r['sem']:>10.5f}")


if __name__ == "__main__":
    main()
