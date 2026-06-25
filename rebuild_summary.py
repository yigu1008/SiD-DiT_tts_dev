#!/usr/bin/env python3
"""Rebuild summary.tsv from a crashed run's per-method output dirs.

When the suite crashes mid-run, the driver bash dies before writing
summary.tsv, but completed methods still left their `aggregate_ddp.json` +
`best_images_multi_reward_aggregate.json` files on disk.  This script walks
the run root, finds whichever methods finished, and emits a fresh summary.tsv
the plot scripts can consume.

Usage:
    python rebuild_summary.py --run_root /mnt/data/v-yigu/synergy/synergy
    python rebuild_summary.py --run_root <RUN_ROOT> --layout synergy
        # then: python plot_synergy_2x2.py --summary <RUN_ROOT>/summary.tsv

Layouts (all carry a search_reward column so rows from different SEARCH_REWARD
runs under one run_root never collapse onto one method name):
    synergy     : columns method, search_reward, cfg_dynamic, prompt_dynamic, mean_search, eval_ir, eval_hpsv3
    all_method  : columns method, search_reward, cfg_dynamic, prompt_dynamic, neg_bank, sigma_bank, mean_search, eval_ir, eval_hpsv3
    pareto      : columns method, search_reward, mean_imagereward, mean_hpsv3, mean_hpsv2, mean_pickscore
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Per-method axis labels (mirror the driver bash case blocks).
_CFG_DYN = {
    "baseline": "-",
    "greedy_prompt": "static",
    "bon": "-",
    "bon_mcts": "dynamic",
    "bon_mcts_static_cfg": "static",
    "bon_mcts_adaptive_cfg": "dynamic",
    "bon_mcts_rewrite_only": "static",
    "bon_mcts_full": "dynamic",
    "bon_mcts_neg": "dynamic",
    "bon_mcts_sigma": "dynamic",
    "bon_mcts_axes": "dynamic",
    "bon_mcts_step_reward": "dynamic",
    "bon_mcts_multiseed": "dynamic",
    "bon_mcts_singleseed": "dynamic",
}
_PRM_DYN = {
    "baseline": "-",
    "greedy_prompt": "greedy",
    "bon": "-",
    "bon_mcts": "static",
    "bon_mcts_static_cfg": "static",
    "bon_mcts_adaptive_cfg": "static",
    "bon_mcts_rewrite_only": "dynamic",
    "bon_mcts_full": "dynamic",
    "bon_mcts_neg": "static",
    "bon_mcts_sigma": "static",
    "bon_mcts_axes": "static",
    "bon_mcts_step_reward": "static",
    "bon_mcts_multiseed": "static",
    "bon_mcts_singleseed": "static",
}


def _agg_search_reward(agg: Path) -> str:
    """Read the search_reward provenance tag written into aggregate_ddp.json.
    Older aggregates predate the tag -> '' (grouped as 'unknown')."""
    try:
        return str(json.loads(agg.read_text()).get("search_reward", "") or "")
    except Exception:
        return ""


def _find_method_aggs(run_root: Path) -> dict[tuple[str, str], tuple[Path | None, Path | None]]:
    """Return {(search_reward, method): (aggregate_ddp_path, best_images_aggregate_path)}.

    Keying by (search_reward, method) -- not method alone -- so that a run_root
    spanning multiple SEARCH_REWARD subtrees (e.g. a composite_hpsv3_ir run and a
    raw hpsv3 run) does NOT collapse same-named methods onto one row. Mixing those
    would put a ~0.8 normalized-composite mean_search next to a ~14 raw-hpsv3 one
    in the same table."""
    out: dict[tuple[str, str], tuple[Path | None, Path | None]] = {}
    for agg in sorted(run_root.glob("**/aggregate_ddp.json")):
        # Path layout: <root>/[run_*/]<method>/aggregate_ddp.json
        parts = agg.relative_to(run_root).parts
        if len(parts) < 2:
            continue
        method = parts[-2]
        if method.startswith("_") or method in {"prompts", "trees"}:
            continue
        reward = _agg_search_reward(agg) or "unknown"
        sibling = agg.parent / "best_images_multi_reward_aggregate.json"
        out[(reward, method)] = (agg, sibling if sibling.exists() else None)
    return out


def _extract_scalar(p: Path, *keys) -> str:
    try:
        data = json.loads(p.read_text())
    except Exception:
        return ""
    cur = data
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return ""
    if isinstance(cur, (int, float)):
        return f"{float(cur):.6f}"
    return str(cur) if cur is not None else ""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_root", required=True, type=Path)
    p.add_argument("--layout", default="synergy",
                   choices=["synergy", "all_method", "pareto"])
    p.add_argument("--out", default=None, type=Path,
                   help="Default: <run_root>/summary.tsv")
    args = p.parse_args()

    methods = _find_method_aggs(args.run_root)
    if not methods:
        raise SystemExit(f"No aggregate_ddp.json files found under {args.run_root}")

    out = args.out or (args.run_root / "summary.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if args.layout == "synergy":
            w.writerow(["method", "search_reward", "cfg_dynamic", "prompt_dynamic", "mean_search", "eval_ir", "eval_hpsv3"])
        elif args.layout == "all_method":
            w.writerow(["method", "search_reward", "cfg_dynamic", "prompt_dynamic", "neg_bank", "sigma_bank",
                        "mean_search", "eval_ir", "eval_hpsv3"])
        else:  # pareto
            w.writerow(["method", "search_reward",
                        "mean_imagereward", "mean_hpsv3", "mean_hpsv2", "mean_pickscore"])
        for (reward, method), (agg, ir_eval) in sorted(methods.items()):
            # aggregate_ddp.json stores "mean_search_score"; fall back to the
            # legacy "mean_search" key for older runs.
            ms = ((_extract_scalar(agg, "mean_search_score") or _extract_scalar(agg, "mean_search"))
                  if agg else "")
            eir = _extract_scalar(ir_eval, "imagereward", "mean") if ir_eval else ""
            eh = _extract_scalar(ir_eval, "hpsv3", "mean") if ir_eval else ""
            eh2 = _extract_scalar(ir_eval, "hpsv2", "mean") if ir_eval else ""
            eps = _extract_scalar(ir_eval, "pickscore", "mean") if ir_eval else ""
            cfg_d = _CFG_DYN.get(method, "?")
            prm_d = _PRM_DYN.get(method, "?")
            if args.layout == "synergy":
                w.writerow([method, reward, cfg_d, prm_d, ms, eir, eh])
            elif args.layout == "all_method":
                w.writerow([method, reward, cfg_d, prm_d, "-", "-", ms, eir, eh])
            else:  # pareto
                w.writerow([method, reward, eir, eh, eh2, eps])
            print(f"  [{reward}] {method}: mean_search={ms or '-'}  eval_ir={eir or '-'}  eval_hpsv3={eh or '-'}")
    print(f"\n  rebuilt summary -> {out}  ({len(methods)} methods)")


if __name__ == "__main__":
    main()
