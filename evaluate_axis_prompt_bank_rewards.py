#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

# Keep ImageReward independent of cluster wandb/protobuf drift.
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("SID_FORCE_WANDB_STUB", "1")

from reward_unified import UnifiedRewardScorer


@dataclass
class EvalRecord:
    prompt_index: int
    prompt: str
    mode: str
    axis: str | None
    seed: int
    image_path: str
    axis_schedule: list[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate axis-prompt-bank outputs with imagereward/hpsv2 (and optional others).")
    p.add_argument("--run_dir", required=True, help="Output dir from sd35_axis_prompt_bank_pipeline.py")
    p.add_argument("--backends", nargs="+", default=["imagereward", "hpsv2"])
    p.add_argument("--reward_device", default="cpu")
    p.add_argument("--allow_missing_backends", action="store_true")
    p.add_argument("--include_modes", nargs="+", default=["fixed", "stepaware"], choices=["fixed", "stepaware"])
    p.add_argument("--image_reward_model", default="ImageReward-v1.0")
    p.add_argument("--pickscore_model", default="yuvalkirstain/PickScore_v1")
    p.add_argument("--unifiedreward_model", default="CodeGoat24/UnifiedReward-qwen-7b")
    p.add_argument("--reward_api_base", default=None)
    p.add_argument("--reward_api_key", default="unifiedreward")
    p.add_argument("--reward_api_model", default="UnifiedReward-7b-v1.5")
    p.add_argument("--reward_max_new_tokens", type=int, default=512)
    p.add_argument("--reward_prompt_mode", choices=["standard", "strict"], default="standard")
    p.add_argument("--out_json", default=None)
    p.add_argument("--out_tsv", default=None)
    p.add_argument("--out_final_rewards_txt", default=None, help="Simple per-image final reward output.")
    p.add_argument("--out_simple_summary_txt", default=None, help="Simple aggregate reward summary.")
    return p.parse_args()


def _load_prompt_bank_records(run_dir: str, include_modes: set[str]) -> tuple[list[EvalRecord], list[str]]:
    root = Path(run_dir).expanduser().resolve()
    missing: list[str] = []
    rows: list[EvalRecord] = []
    for pb_path in sorted(root.glob("p*/prompt_bank.json")):
        payload = json.loads(pb_path.read_text(encoding="utf-8"))
        p_idx = int(payload.get("prompt_index", -1))
        prompt = str(payload.get("prompt", ""))
        sampling = payload.get("sampling", {})
        if "fixed" in include_modes:
            for item in sampling.get("fixed_manifest", []):
                image_path = str(item.get("path", ""))
                rec = EvalRecord(
                    prompt_index=p_idx,
                    prompt=prompt,
                    mode="fixed",
                    axis=str(item.get("axis")) if item.get("axis") is not None else None,
                    seed=int(item.get("seed", -1)),
                    image_path=image_path,
                    axis_schedule=[str(x) for x in item.get("axis_schedule", []) if isinstance(x, str)],
                )
                if Path(image_path).exists():
                    rows.append(rec)
                else:
                    missing.append(image_path)

        if "stepaware" in include_modes:
            for item in sampling.get("stepaware_manifest", []):
                image_path = str(item.get("path", ""))
                rec = EvalRecord(
                    prompt_index=p_idx,
                    prompt=prompt,
                    mode="stepaware",
                    axis=None,
                    seed=int(item.get("seed", -1)),
                    image_path=image_path,
                    axis_schedule=[str(x) for x in item.get("axis_schedule", []) if isinstance(x, str)],
                )
                if Path(image_path).exists():
                    rows.append(rec)
                else:
                    missing.append(image_path)
    rows.sort(key=lambda r: (r.prompt_index, r.mode, r.axis or "", r.seed))
    return rows, missing


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


def _safe_stats(vals: list[float]) -> dict[str, float | int | None]:
    if not vals:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(vals),
        "mean": float(statistics.fmean(vals)),
        "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def _group_backend_stats(rows: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        key = key_fn(row)
        grouped.setdefault(key, {})
        for backend, val in row.get("scores", {}).items():
            if isinstance(val, (float, int)):
                grouped[key].setdefault(backend, []).append(float(val))
    out: dict[str, dict[str, Any]] = {}
    for key, by_backend in grouped.items():
        out[key] = {b: _safe_stats(v) for b, v in by_backend.items()}
    return out


def _format_score(v: Any) -> str:
    if isinstance(v, (float, int)):
        return f"{float(v):.6f}"
    return "n/a"


def main() -> None:
    args = parse_args()
    args.run_dir = str(Path(args.run_dir).expanduser().resolve())
    if args.out_json is None:
        args.out_json = str(Path(args.run_dir) / "reward_validation.json")
    if args.out_tsv is None:
        args.out_tsv = str(Path(args.run_dir) / "reward_validation.tsv")
    if args.out_final_rewards_txt is None:
        args.out_final_rewards_txt = str(Path(args.run_dir) / "reward_final_output.txt")
    if args.out_simple_summary_txt is None:
        args.out_simple_summary_txt = str(Path(args.run_dir) / "reward_summary_simple.txt")

    records, missing = _load_prompt_bank_records(args.run_dir, set(args.include_modes))
    if not records:
        raise RuntimeError(f"No images found under run_dir={args.run_dir}")

    scorers: dict[str, UnifiedRewardScorer] = {}
    init_errors: dict[str, str] = {}
    for backend in args.backends:
        try:
            scorers[str(backend)] = _build_scorer(args, str(backend))
        except Exception as exc:
            init_errors[str(backend)] = f"{type(exc).__name__}: {exc}"

    if init_errors and not args.allow_missing_backends:
        bad = "; ".join(f"{k}: {v}" for k, v in init_errors.items())
        raise RuntimeError(f"Failed to initialize requested backends: {bad}")

    out_rows: list[dict[str, Any]] = []
    eval_errors: dict[str, dict[str, Any]] = {}
    for i, rec in enumerate(records, start=1):
        if i % 50 == 0 or i == 1:
            print(f"[eval-axis] scoring {i}/{len(records)} ...")
        row = {
            "prompt_index": int(rec.prompt_index),
            "prompt": rec.prompt,
            "mode": rec.mode,
            "axis": rec.axis,
            "seed": int(rec.seed),
            "image_path": rec.image_path,
            "axis_schedule": rec.axis_schedule,
            "scores": {},
        }
        with Image.open(rec.image_path) as img:
            img = img.convert("RGB")
            for backend, scorer in scorers.items():
                try:
                    row["scores"][backend] = float(scorer.score(rec.prompt, img))
                except Exception as exc:
                    row["scores"][backend] = None
                    eval_errors.setdefault(backend, {"count": 0, "examples": []})
                    eval_errors[backend]["count"] += 1
                    if len(eval_errors[backend]["examples"]) < 3:
                        eval_errors[backend]["examples"].append(
                            {
                                "image_path": rec.image_path,
                                "prompt_index": int(rec.prompt_index),
                                "mode": rec.mode,
                                "axis": rec.axis,
                                "seed": int(rec.seed),
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    if not args.allow_missing_backends:
                        raise
        out_rows.append(row)

    overall = _group_backend_stats(out_rows, key_fn=lambda _: "overall").get("overall", {})
    by_mode = _group_backend_stats(out_rows, key_fn=lambda r: str(r.get("mode", "unknown")))
    by_prompt = _group_backend_stats(out_rows, key_fn=lambda r: f"p{int(r.get('prompt_index', -1)):04d}")
    by_axis_fixed = _group_backend_stats(
        [r for r in out_rows if r.get("mode") == "fixed" and r.get("axis") is not None],
        key_fn=lambda r: str(r.get("axis")),
    )

    aggregate = {
        "run_dir": args.run_dir,
        "backends_requested": list(args.backends),
        "backends_initialized": sorted(list(scorers.keys())),
        "reward_device": str(args.reward_device),
        "include_modes": list(args.include_modes),
        "num_images_found": len(records),
        "num_images_scored": len(out_rows),
        "num_missing_images": len(missing),
        "missing_image_examples": missing[:20],
        "backend_init_errors": init_errors,
        "backend_eval_errors": eval_errors,
        "overall_backend_stats": overall,
        "backend_stats_by_mode": by_mode,
        "backend_stats_by_axis_fixed": by_axis_fixed,
        "backend_stats_by_prompt": by_prompt,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"aggregate": aggregate, "rows": out_rows}, f, indent=2, ensure_ascii=False)

    out_tsv = Path(args.out_tsv)
    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        header = [
            "prompt_index",
            "mode",
            "axis",
            "seed",
            "image_path",
            "axis_schedule",
        ] + [f"score_{b}" for b in args.backends]
        w.writerow(header)
        for row in out_rows:
            w.writerow(
                [
                    int(row["prompt_index"]),
                    row["mode"],
                    row["axis"] if row["axis"] is not None else "",
                    int(row["seed"]),
                    row["image_path"],
                    " ".join(row.get("axis_schedule", [])),
                ] + [row.get("scores", {}).get(b) for b in args.backends]
            )

    final_txt = Path(args.out_final_rewards_txt)
    with open(final_txt, "w", encoding="utf-8") as f:
        for row in out_rows:
            score_str = " ".join(f"{b}={_format_score(row.get('scores', {}).get(b))}" for b in args.backends)
            axis = row["axis"] if row["axis"] is not None else "-"
            line = (
                f"p{int(row['prompt_index']):04d} mode={row['mode']} axis={axis} seed={int(row['seed'])} "
                f"{score_str} image={row['image_path']}"
            )
            f.write(line + "\n")

    summary_txt = Path(args.out_simple_summary_txt)
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"run_dir: {args.run_dir}\n")
        f.write(f"num_images_scored: {aggregate['num_images_scored']}\n")
        f.write(f"num_missing_images: {aggregate['num_missing_images']}\n")
        for backend in args.backends:
            st = overall.get(backend, {})
            f.write(
                f"{backend}: count={st.get('count', 0)} "
                f"mean={_format_score(st.get('mean'))} "
                f"std={_format_score(st.get('std'))} "
                f"min={_format_score(st.get('min'))} "
                f"max={_format_score(st.get('max'))}\n"
            )

    print(
        f"[eval-axis] done found={aggregate['num_images_found']} "
        f"scored={aggregate['num_images_scored']} missing={aggregate['num_missing_images']}"
    )
    for backend in args.backends:
        stats = overall.get(backend, {})
        mean = stats.get("mean")
        count = stats.get("count", 0)
        if mean is None:
            print(f"[eval-axis] {backend}: count={count} mean=n/a")
        else:
            print(f"[eval-axis] {backend}: count={count} mean={float(mean):.6f}")
    print(f"[eval-axis] json={out_json}")
    print(f"[eval-axis] tsv={out_tsv}")
    print(f"[eval-axis] final_rewards={final_txt}")
    print(f"[eval-axis] summary_simple={summary_txt}")


if __name__ == "__main__":
    main()
