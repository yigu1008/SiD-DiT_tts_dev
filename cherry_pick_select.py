#!/usr/bin/env python3
"""Post-process suite outputs to find bon_mcts winners and pick top-N examples.

Reads each method's `best_images_multi_reward.json` (one per method dir under
RUN_ROOT), computes hpsv3 + imagereward sum per (prompt, seed) per method,
ranks methods, and selects (prompt, seed) examples where bon_mcts is rank-1.

Output:
  <out_dir>/winners.json     — manifest with full per-(prompt, seed) detail
  <out_dir>/winners/         — copies of all 5-method images for each winner,
                               named {prompt_idx}_seed{seed}_{method}.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

METHODS = ["bon", "smc", "fksteering", "dts_star", "bon_mcts"]


def _find_method_dirs(run_root: Path) -> dict[str, Path]:
    """Return method_name → first-found method dir under run_root.

    The suite may nest under run_root/<seed_or_run_dir>/<method>/. We crawl
    one level deep first, then deeper if needed.
    """
    found: dict[str, Path] = {}
    for method in METHODS:
        # Try direct: run_root/<method>/best_images_multi_reward.json
        direct = run_root / method
        if (direct / "best_images_multi_reward.json").exists():
            found[method] = direct
            continue
        # Else search.
        for cand in run_root.glob(f"**/{method}/best_images_multi_reward.json"):
            found[method] = cand.parent
            break
    return found


def _load_per_prompt_scores(method_dir: Path) -> dict[tuple[int, int], dict]:
    """Return {(prompt_index, seed): {hpsv3, imagereward, image_path}} for one method."""
    eval_path = method_dir / "best_images_multi_reward.json"
    if not eval_path.exists():
        return {}
    payload = json.loads(eval_path.read_text())
    out: dict[tuple[int, int], dict] = {}
    rows = payload.get("rows") or payload.get("entries") or []
    for row in rows:
        try:
            p_idx = int(row.get("prompt_index", -1))
            seed = int(row.get("seed", -1))
        except (ValueError, TypeError):
            continue
        if p_idx < 0 or seed < 0:
            continue
        per_backend = row.get("per_backend") or row.get("rewards") or {}
        hpsv3 = float(per_backend.get("hpsv3", float("nan"))) if isinstance(per_backend, dict) else float("nan")
        ir = float(per_backend.get("imagereward", float("nan"))) if isinstance(per_backend, dict) else float("nan")
        out[(p_idx, seed)] = {
            "hpsv3": hpsv3,
            "imagereward": ir,
            "image_path": row.get("image_path"),
            "prompt": row.get("prompt"),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_root", required=True,
                   help="Dir containing subdirs for each method (or one level deeper).")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_winners", type=int, default=8)
    p.add_argument("--rank_metric", choices=["sum", "mean"], default="sum",
                   help="Method score = sum(hpsv3, imagereward) [default] or mean.")
    args = p.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    winners_dir = out_dir / "winners"
    winners_dir.mkdir(parents=True, exist_ok=True)

    method_dirs = _find_method_dirs(run_root)
    missing = [m for m in METHODS if m not in method_dirs]
    if missing:
        print(f"[select] WARN missing method dirs: {missing}")

    # Collect per-method scores indexed by (prompt_index, seed).
    per_method: dict[str, dict[tuple[int, int], dict]] = {}
    for m, d in method_dirs.items():
        per_method[m] = _load_per_prompt_scores(d)
        print(f"[select] {m}: {len(per_method[m])} (prompt, seed) entries from {d}")

    if "bon_mcts" not in per_method:
        raise RuntimeError("No bon_mcts results found — cannot pick winners.")

    # Find common keys across all 5 methods.
    common_keys = set(per_method["bon_mcts"].keys())
    for m in METHODS:
        if m in per_method:
            common_keys &= set(per_method[m].keys())
    print(f"[select] common (prompt, seed) keys across {len(per_method)} methods: {len(common_keys)}")

    # Rank methods per (prompt, seed); collect bon_mcts winners with margin.
    candidates: list[dict] = []
    for key in sorted(common_keys):
        scores = {}
        for m in METHODS:
            if m not in per_method:
                continue
            entry = per_method[m].get(key)
            if not entry:
                continue
            h = entry["hpsv3"]; r = entry["imagereward"]
            if h != h or r != r:  # NaN check
                continue
            score = (h + r) if args.rank_metric == "sum" else 0.5 * (h + r)
            scores[m] = score
        if "bon_mcts" not in scores or len(scores) < 5:
            continue
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top_method, top_score = ranked[0]
        if top_method != "bon_mcts":
            continue
        runner_up_score = ranked[1][1]
        margin = top_score - runner_up_score
        candidates.append({
            "prompt_index": key[0],
            "seed": key[1],
            "prompt": per_method["bon_mcts"][key].get("prompt"),
            "scores": scores,
            "margin_over_runner_up": margin,
            "image_paths": {
                m: per_method.get(m, {}).get(key, {}).get("image_path")
                for m in METHODS if m in per_method
            },
        })

    # Sort candidates by margin (largest first) and take top N.
    candidates.sort(key=lambda r: -float(r.get("margin_over_runner_up", 0.0)))
    winners = candidates[: int(args.n_winners)]

    # Copy raw images into winners/ with tagged filenames.
    for rec in winners:
        p_idx = int(rec["prompt_index"]); seed = int(rec["seed"])
        for m, src_path in rec["image_paths"].items():
            if not src_path:
                continue
            src = Path(src_path)
            if not src.exists():
                # Best-image multi-reward JSON uses paths relative to run_root.
                src = run_root / src_path
            if not src.exists():
                print(f"[select] WARN missing source image: {src_path}")
                continue
            tag = f"p{p_idx:04d}_seed{seed:05d}_{m}{src.suffix}"
            dst = winners_dir / tag
            shutil.copy2(src, dst)
            rec.setdefault("copied_paths", {})[m] = str(dst)

    manifest = {
        "n_candidates": len(candidates),
        "n_winners_requested": int(args.n_winners),
        "n_winners_returned": len(winners),
        "rank_metric": args.rank_metric,
        "methods": METHODS,
        "winners": winners,
    }
    (out_dir / "winners.json").write_text(json.dumps(manifest, indent=2))
    print(f"[select] wrote {len(winners)} winners to {out_dir}/winners/ (manifest: winners.json)")
    print(f"[select] {len(candidates)} bon_mcts-rank-1 candidates total; took top {len(winners)} by margin.")


if __name__ == "__main__":
    main()
