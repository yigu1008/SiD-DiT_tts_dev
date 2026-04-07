#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

# ImageReward inference does not need wandb logging and wandb can be broken on clusters.
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("SID_FORCE_WANDB_STUB", "1")

from reward_unified import UnifiedRewardScorer


@dataclass
class ImageRecord:
    prompt_index: int
    slug: str
    sample_index: int
    prompt: str
    image_path: str
    objective_score: float | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-evaluate generated best images with multiple reward backends.")
    p.add_argument("--layout", choices=["sd35", "sana", "flux"], required=True)
    p.add_argument("--method_out", required=True, help="Method output directory (e.g., .../run_xxx/ga)")
    p.add_argument("--method", required=True, help="Method name: baseline|greedy|mcts|ga|smc")
    p.add_argument("--backends", nargs="+", default=["imagereward", "hpsv2", "pickscore"])
    p.add_argument("--reward_device", default="cpu")
    p.add_argument("--image_reward_model", default="ImageReward-v1.0")
    p.add_argument("--pickscore_model", default="yuvalkirstain/PickScore_v1")
    p.add_argument("--unifiedreward_model", default="CodeGoat24/UnifiedReward-qwen-7b")
    p.add_argument("--reward_api_base", default=None)
    p.add_argument("--reward_api_key", default="unifiedreward")
    p.add_argument("--reward_api_model", default="UnifiedReward-7b-v1.5")
    p.add_argument("--reward_max_new_tokens", type=int, default=512)
    p.add_argument("--reward_prompt_mode", choices=["standard", "strict"], default="standard")
    p.add_argument("--allow_missing_backends", action="store_true")
    p.add_argument("--out_json", default=None)
    p.add_argument("--out_aggregate", default=None)
    return p.parse_args()


def _slug_to_index(slug: str) -> int:
    m = re.search(r"(\d+)", str(slug))
    if not m:
        return -1
    return int(m.group(1))


def _sd35_mode_key_and_suffix(method: str) -> tuple[str, str]:
    m = str(method).strip().lower()
    if m == "baseline":
        return "base", "base"
    if m == "base":
        return "base", "base"
    if m.startswith("mcts"):
        # SD3.5 DDP writes mode/image suffix as "mcts" even for method aliases
        # like mcts_lookahead_dynamiccfg, mcts_dynamiccfg_only, etc.
        return "mcts", "mcts"
    if m in {"greedy", "ga", "smc", "bon", "beam"}:
        return m, m
    return m, m


def _collect_sd35_records(method_out: str, method: str) -> tuple[list[ImageRecord], list[str]]:
    records: list[ImageRecord] = []
    missing: list[str] = []
    logs_dir = os.path.join(method_out, "logs")
    image_dir = os.path.join(method_out, "images")

    mode_key, suffix = _sd35_mode_key_and_suffix(method)

    for log_path in sorted(glob.glob(os.path.join(logs_dir, "rank_*.jsonl"))):
        if os.path.basename(log_path).endswith("_rewrite_examples.jsonl"):
            continue
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if str(row.get("mode")) != mode_key:
                    continue
                prompt_index = int(row.get("prompt_index", -1))
                slug = f"p{prompt_index:05d}" if prompt_index >= 0 else "p?????"
                image_path = os.path.join(image_dir, f"{slug}_{suffix}.png")
                rec = ImageRecord(
                    prompt_index=prompt_index,
                    slug=slug,
                    sample_index=0,
                    prompt=str(row.get("prompt", "")),
                    image_path=image_path,
                    objective_score=float(row.get("score")) if row.get("score") is not None else None,
                )
                if os.path.exists(image_path):
                    records.append(rec)
                else:
                    missing.append(image_path)
    records.sort(key=lambda r: (r.prompt_index, r.sample_index))
    return records, missing


def _collect_sana_flux_records(method_out: str, method: str) -> tuple[list[ImageRecord], list[str]]:
    records: list[ImageRecord] = []
    missing: list[str] = []
    for summary_path in sorted(glob.glob(os.path.join(method_out, "rank_*", "summary.json"))):
        rank_dir = os.path.dirname(summary_path)
        with open(summary_path, encoding="utf-8") as f:
            payload = json.load(f)
        for row in payload:
            slug = str(row.get("slug", ""))
            prompt = str(row.get("prompt", ""))
            prompt_index = _slug_to_index(slug)
            samples = row.get("samples", [])
            for sample_idx, sample in enumerate(samples):
                if method == "baseline":
                    img_name = f"{slug}_s{sample_idx}_baseline.png"
                    objective_score = sample.get("baseline_score")
                else:
                    img_name = f"{slug}_s{sample_idx}_{method}.png"
                    objective_score = sample.get("search_score")
                image_path = os.path.join(rank_dir, img_name)
                rec = ImageRecord(
                    prompt_index=prompt_index,
                    slug=slug,
                    sample_index=int(sample_idx),
                    prompt=prompt,
                    image_path=image_path,
                    objective_score=float(objective_score) if objective_score is not None else None,
                )
                if os.path.exists(image_path):
                    records.append(rec)
                else:
                    missing.append(image_path)
    records.sort(key=lambda r: (r.prompt_index, r.sample_index))
    return records, missing


def collect_records(layout: str, method_out: str, method: str) -> tuple[list[ImageRecord], list[str]]:
    if layout == "sd35":
        return _collect_sd35_records(method_out, method)
    return _collect_sana_flux_records(method_out, method)


def _build_scorer(args: argparse.Namespace, backend: str) -> UnifiedRewardScorer:
    return UnifiedRewardScorer(
        device=str(args.reward_device),
        backend=str(backend),
        image_reward_model=str(args.image_reward_model),
        pickscore_model=str(args.pickscore_model),
        unifiedreward_model=str(args.unifiedreward_model),
        unifiedreward_api_base=args.reward_api_base,
        unifiedreward_api_key=str(args.reward_api_key),
        unifiedreward_api_model=str(args.reward_api_model),
        max_new_tokens=int(args.reward_max_new_tokens),
        unifiedreward_prompt_mode=str(args.reward_prompt_mode),
    )


def score_records(
    args: argparse.Namespace,
    records: list[ImageRecord],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, dict[str, Any]]]:
    scorers: dict[str, UnifiedRewardScorer] = {}
    init_errors: dict[str, str] = {}
    for backend in args.backends:
        try:
            scorers[backend] = _build_scorer(args, backend)
        except Exception as exc:  # pragma: no cover
            init_errors[backend] = f"{type(exc).__name__}: {exc}"

    if init_errors and not args.allow_missing_backends:
        bad = "; ".join(f"{k}: {v}" for k, v in init_errors.items())
        raise RuntimeError(f"Failed to initialize requested backends: {bad}")

    out_rows: list[dict[str, Any]] = []
    eval_errors: dict[str, dict[str, Any]] = {}
    for rec in records:
        row = {
            "prompt_index": int(rec.prompt_index),
            "slug": rec.slug,
            "sample_index": int(rec.sample_index),
            "prompt": rec.prompt,
            "image_path": rec.image_path,
            "method": args.method,
            "objective_score": rec.objective_score,
            "scores": {},
        }
        with Image.open(rec.image_path) as img:
            img = img.convert("RGB")
            for backend, scorer in scorers.items():
                try:
                    row["scores"][backend] = float(scorer.score(rec.prompt, img))
                except Exception as exc:  # pragma: no cover
                    row["scores"][backend] = None
                    eval_errors.setdefault(backend, {"count": 0, "examples": []})
                    eval_errors[backend]["count"] += 1
                    if len(eval_errors[backend]["examples"]) < 3:
                        eval_errors[backend]["examples"].append(
                            {
                                "image_path": rec.image_path,
                                "prompt_index": int(rec.prompt_index),
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    if not args.allow_missing_backends:
                        raise
        out_rows.append(row)

    return out_rows, init_errors, eval_errors


def summarize(
    args: argparse.Namespace,
    records: list[ImageRecord],
    scored_rows: list[dict[str, Any]],
    missing_paths: list[str],
    init_errors: dict[str, str],
    eval_errors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    backend_stats: dict[str, Any] = {}
    for backend in args.backends:
        vals: list[float] = []
        for row in scored_rows:
            val = row.get("scores", {}).get(backend)
            if isinstance(val, (float, int)):
                vals.append(float(val))
        if vals:
            backend_stats[backend] = {
                "count": len(vals),
                "mean": float(statistics.fmean(vals)),
                "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
                "min": float(min(vals)),
                "max": float(max(vals)),
            }
        else:
            backend_stats[backend] = {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }

    return {
        "layout": args.layout,
        "method": args.method,
        "method_out": str(Path(args.method_out).resolve()),
        "backends_requested": list(args.backends),
        "reward_device": str(args.reward_device),
        "num_candidate_records": len(records) + len(missing_paths),
        "num_images_found": len(records),
        "num_images_scored": len(scored_rows),
        "num_missing_images": len(missing_paths),
        "missing_image_examples": missing_paths[:20],
        "backend_init_errors": init_errors,
        "backend_eval_errors": eval_errors,
        "backend_stats": backend_stats,
    }


def main() -> None:
    args = parse_args()
    args.method_out = str(Path(args.method_out).expanduser().resolve())
    if args.out_json is None:
        args.out_json = os.path.join(args.method_out, "best_images_multi_reward.json")
    if args.out_aggregate is None:
        args.out_aggregate = os.path.join(args.method_out, "best_images_multi_reward_aggregate.json")

    records, missing = collect_records(args.layout, args.method_out, args.method)
    if not records:
        raise RuntimeError(
            f"No saved best-image records found. layout={args.layout} method={args.method} method_out={args.method_out}"
        )

    scored_rows, init_errors, eval_errors = score_records(args, records)
    aggregate = summarize(args, records, scored_rows, missing, init_errors, eval_errors)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "aggregate": aggregate,
                "rows": scored_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(args.out_aggregate, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    print(
        f"[eval] method={args.method} layout={args.layout} "
        f"found={aggregate['num_images_found']} scored={aggregate['num_images_scored']} "
        f"missing={aggregate['num_missing_images']}"
    )
    for backend in args.backends:
        stats = aggregate["backend_stats"].get(backend, {})
        mean = stats.get("mean")
        count = stats.get("count")
        if mean is None:
            print(f"[eval] {backend}: count={count} mean=n/a")
        else:
            print(f"[eval] {backend}: count={count} mean={float(mean):.6f}")


if __name__ == "__main__":
    main()
