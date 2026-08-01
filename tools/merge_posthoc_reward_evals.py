#!/usr/bin/env python3
"""Merge one-backend post-evaluations and write a sweep summary.

The reward models used by this project do not all share a compatible Python
environment.  ``post_eval_extra_rewards.sh`` therefore evaluates one backend
at a time.  This script joins those files by image identity into the standard
``best_images_multi_reward*.json`` artifacts consumed by the reporting tools.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


METHOD_LABELS = {
    "baseline": "Distilled Baseline",
    "das": "DAS",
    "fksteering": "FK-Steering",
    "bon": "BoN",
    "bon_fixed_rewrite": "Fixed-Rewrite BoN-8",
    "beam": "Beam",
    "sop": "SoP",
    "ga": "GA",
    "dts": "DTS",
    "dts_star": "DTS*",
    "dynamic_cfg_x0": "Dynamic CFG",
    "bon_mcts": "ActDiff",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _row_key(row: dict[str, Any]) -> tuple[int, str, int]:
    return (
        int(row.get("prompt_index", -1)),
        str(row.get("slug", "")),
        int(row.get("sample_index", 0)),
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _backend_stats(
    rows: list[dict[str, Any]], backends: list[str]
) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    for backend in backends:
        values = [
            float(row["scores"][backend])
            for row in rows
            if isinstance(row.get("scores", {}).get(backend), (int, float))
            and math.isfinite(float(row["scores"][backend]))
        ]
        out[backend] = {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "std": (
                statistics.pstdev(values)
                if len(values) > 1
                else 0.0 if values else None
            ),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return out


def _model_fields(method_dir: Path, root: Path) -> tuple[str, str]:
    relative_parts = set(method_dir.relative_to(root).parts)
    if "flux_schnell" in relative_parts:
        return "flux_schnell", "Flux-Schnell"
    if "senseflow_large" in relative_parts:
        return "senseflow_large", "SenseFlow-SD3.5-Large"
    if "senseflow_medium" in relative_parts:
        return "senseflow_medium", "SenseFlow-SD3.5-Medium"
    if "sd35_base" in relative_parts:
        return "sd35_base", "SD3.5-Base"
    if "sid" in relative_parts:
        return "sid", "SiD-SD3.5"
    if "multi_step_baseline" in relative_parts:
        return "sd35_base", "SD3.5-Base"
    return "sid", "SiD-SD3.5"


def _reward_arm(method_dir: Path, root: Path) -> str:
    relative_parts = set(method_dir.relative_to(root).parts)
    for value in ("imagereward", "hpsv3", "multi_reward"):
        if value in relative_parts:
            return value
    return ""


def _mean_logged_nfe(method_dir: Path) -> float | None:
    values: list[float] = []
    for log_path in sorted((method_dir / "logs").glob("rank_*.jsonl")):
        if log_path.name.endswith("_rewrite_examples.jsonl"):
            continue
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get("nfe")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
    if not values:
        for summary_path in sorted(method_dir.glob("rank_*/summary.json")):
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                continue
            for prompt_row in payload:
                for sample in prompt_row.get("samples", []):
                    value = sample.get("diagnostics", {}).get("nfe_total")
                    if isinstance(value, (int, float)) and math.isfinite(
                        float(value)
                    ):
                        values.append(float(value))
    return statistics.fmean(values) if values else None


def _merge_method(
    method_dir: Path,
    root: Path,
    backends: list[str],
    strict: bool,
    expected_count: int | None = None,
) -> dict[str, Any] | None:
    source_paths = {
        backend: method_dir / f"best_images_{backend}.json"
        for backend in backends
    }
    missing = [backend for backend, path in source_paths.items() if not path.is_file()]
    if missing:
        message = f"{method_dir}: missing backend files: {', '.join(missing)}"
        if strict:
            raise FileNotFoundError(message)
        print(f"[merge] WARN {message}")
        return None

    rows_by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
    expected_keys: set[tuple[int, str, int]] | None = None
    for backend in backends:
        payload = _read_json(source_paths[backend])
        backend_rows = payload.get("rows", [])
        if not isinstance(backend_rows, list):
            raise ValueError(f"{source_paths[backend]}: rows must be a list")
        current_keys = {_row_key(row) for row in backend_rows}
        if expected_keys is None:
            expected_keys = current_keys
        elif current_keys != expected_keys:
            missing_here = sorted((expected_keys or set()) - current_keys)
            extra_here = sorted(current_keys - (expected_keys or set()))
            raise ValueError(
                f"{method_dir}: {backend} image keys differ; "
                f"missing={missing_here[:3]} extra={extra_here[:3]}"
            )
        for row in backend_rows:
            key = _row_key(row)
            if key not in rows_by_key:
                rows_by_key[key] = dict(row)
                rows_by_key[key]["scores"] = {}
            merged = rows_by_key[key]
            if str(merged.get("prompt", "")) != str(row.get("prompt", "")):
                raise ValueError(f"{method_dir}: prompt mismatch for image key {key}")
            if str(merged.get("image_path", "")) != str(row.get("image_path", "")):
                raise ValueError(f"{method_dir}: image-path mismatch for image key {key}")
            value = row.get("scores", {}).get(backend)
            if strict and not isinstance(value, (int, float)):
                raise ValueError(
                    f"{method_dir}: missing numeric {backend} score for {key}"
                )
            merged["scores"][backend] = value

    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(
            f"{method_dir}: found {len(rows)} selected images, "
            f"expected {expected_count}"
        )
    stats = _backend_stats(rows, backends)
    method = method_dir.name
    model_id, model_name = _model_fields(method_dir, root)
    reward_arm = _reward_arm(method_dir, root)
    label = (
        "Multi-step Baseline"
        if model_id == "sd35_base" and method == "baseline"
        else METHOD_LABELS.get(method, method)
    )
    aggregate = {
        "layout": "flux" if model_id == "flux_schnell" else "sd35",
        "model_id": model_id,
        "model_name": model_name,
        "reward_arm": reward_arm,
        "method": method,
        "method_label": label,
        "method_out": str(method_dir.resolve()),
        "backends_requested": backends,
        "num_images_found": len(rows),
        "num_images_scored": len(rows),
        "backend_stats": stats,
        "source_files": [str(source_paths[b].resolve()) for b in backends],
    }
    _atomic_json(
        method_dir / "best_images_multi_reward.json",
        {"aggregate": aggregate, "rows": rows},
    )
    _atomic_json(method_dir / "best_images_multi_reward_aggregate.json", aggregate)

    generation = _read_json(method_dir / "aggregate_ddp.json")
    summary: dict[str, Any] = {
        "model_id": model_id,
        "model_name": model_name,
        "reward_arm": reward_arm,
        "method": method,
        "method_label": label,
        "prompt_count": generation.get("num_samples", len(rows)),
        "search_reward": generation.get("search_reward", "vqascore"),
        "mean_nfe": _mean_logged_nfe(method_dir),
        "mean_search_score": generation.get("mean_search_score"),
        "run_path": str(method_dir.resolve()),
    }
    for backend in backends:
        summary[f"eval_{backend}_mean"] = stats[backend]["mean"]
        summary[f"eval_{backend}_count"] = stats[backend]["count"]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["imagereward", "hpsv3", "pickscore", "vqascore"],
    )
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument(
        "--include-models",
        nargs="+",
        default=[],
        help="Only merge method directories beneath these model IDs.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Only merge method directories beneath run_<run-id>.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    backends = [str(value).strip().lower() for value in args.backends]
    method_dirs = sorted(
        path.parent for path in root.rglob("aggregate_ddp.json")
    )
    if args.include_models:
        included = set(args.include_models)
        method_dirs = [
            path
            for path in method_dirs
            if included.intersection(path.relative_to(root).parts)
        ]
    if args.run_id:
        run_component = f"run_{args.run_id}"
        method_dirs = [
            path
            for path in method_dirs
            if run_component in path.relative_to(root).parts
        ]
    if not method_dirs:
        raise SystemExit(f"no aggregate_ddp.json files found under {root}")

    summaries = []
    for method_dir in method_dirs:
        row = _merge_method(
            method_dir,
            root,
            backends,
            bool(args.strict),
            expected_count=args.expected_count,
        )
        if row is not None:
            summaries.append(row)
            print(f"[merge] {row['model_name']} / {row['method_label']}")

    summary_path = (
        args.summary_csv.expanduser().resolve()
        if args.summary_csv
        else root / "vqa_algorithm_sweep_summary.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_id",
        "model_name",
        "reward_arm",
        "method",
        "method_label",
        "prompt_count",
        "search_reward",
        "mean_nfe",
        "mean_search_score",
    ]
    for backend in backends:
        fields.extend([f"eval_{backend}_mean", f"eval_{backend}_count"])
    fields.append("run_path")
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    os.replace(temporary, summary_path)
    print(f"[merge] wrote {len(summaries)} rows: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
