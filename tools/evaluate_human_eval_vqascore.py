#!/usr/bin/env python3
"""Post-evaluate standardized human-eval images with CLIP-FlanT5 VQAScore.

Every image is scored against the original prompt stored in groups.csv.  The
script is resumable, checkpoints results atomically, writes an aggregate JSON
report, and produces a manifest copy with a vqascore column.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


METHODS = ("baseline", "bon", "das", "bon_mcts")
RESULT_FIELDS = (
    "group_id",
    "model_id",
    "prompt_id",
    "algorithm_id",
    "image_path",
    "vqascore",
    "vqascore_model",
    "reward_prompt",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _atomic_csv(path: Path, fields: list[str] | tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _result_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["group_id"]), str(row["algorithm_id"])


def _validate_groups(
    groups: list[dict[str, str]],
    output_dir: Path,
    expected_groups: int | None,
) -> list[dict[str, Any]]:
    required = {"group_id", "model_id", "prompt_id", "prompt"} | {
        f"{method}_path" for method in METHODS
    }
    if not groups:
        raise ValueError("groups.csv contains no rows")
    missing_fields = required - set(groups[0])
    if missing_fields:
        raise ValueError(f"groups.csv missing fields: {sorted(missing_fields)}")
    if expected_groups is not None and len(groups) != expected_groups:
        raise ValueError(f"expected {expected_groups} groups, found {len(groups)}")

    group_ids = [row["group_id"] for row in groups]
    duplicates = sorted(key for key, count in Counter(group_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate group_id values: {duplicates[:10]}")

    prepared: list[dict[str, Any]] = []
    missing_images: list[str] = []
    for row in groups:
        if not row["prompt"]:
            raise ValueError(f"group {row['group_id']} has an empty original prompt")
        expected_group_id = f"{row['model_id']}__{row['prompt_id']}"
        if row["group_id"] != expected_group_id:
            raise ValueError(
                f"group_id mismatch: {row['group_id']!r} != {expected_group_id!r}"
            )
        paths: dict[str, Path] = {}
        for method in METHODS:
            stored_path = Path(row[f"{method}_path"])
            resolved = stored_path if stored_path.is_absolute() else output_dir / stored_path
            resolved = resolved.resolve()
            if not resolved.is_file():
                missing_images.append(str(resolved))
            paths[method] = resolved
        prepared.append({"row": row, "paths": paths})
    if missing_images:
        examples = "\n".join(f"  {path}" for path in missing_images[:20])
        raise FileNotFoundError(
            f"{len(missing_images)} group images are missing; examples:\n{examples}"
        )
    return prepared


def _load_existing_results(
    path: Path,
    valid_keys: set[tuple[str, str]],
    model_name: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = _read_csv(path)
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = _result_key(row)
        if key not in valid_keys:
            raise ValueError(f"existing VQAScore result has unknown key: {key}")
        if key in output:
            raise ValueError(f"duplicate existing VQAScore result: {key}")
        if row.get("vqascore_model") != model_name:
            raise ValueError(
                f"existing result model {row.get('vqascore_model')!r} does not match {model_name!r}"
            )
        try:
            value = float(row["vqascore"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid existing VQAScore for {key}: {row.get('vqascore')!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"non-finite existing VQAScore for {key}: {value}")
        row["vqascore"] = value
        output[key] = row
    return output


def _ordered_results(
    prepared_groups: list[dict[str, Any]],
    records: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = []
    for item in prepared_groups:
        group_id = item["row"]["group_id"]
        for method in METHODS:
            key = (group_id, method)
            if key in records:
                ordered.append(records[key])
    return ordered


def _write_joined_manifest(
    manifest_path: Path,
    out_path: Path,
    records: dict[tuple[str, str], dict[str, Any]],
) -> None:
    rows = _read_csv(manifest_path)
    if not rows:
        raise ValueError("manifest.csv contains no rows")
    fields = list(rows[0])
    if "vqascore" not in fields:
        fields.append("vqascore")
    if "vqascore_model" not in fields:
        fields.append("vqascore_model")
    missing = []
    for row in rows:
        key = (row["group_id"], row["algorithm_id"])
        result = records.get(key)
        if result is None:
            missing.append(key)
            continue
        row["vqascore"] = result["vqascore"]
        row["vqascore_model"] = result["vqascore_model"]
    if missing:
        raise ValueError(f"manifest join is missing {len(missing)} VQAScore rows")
    _atomic_csv(out_path, fields, rows)


def _summary(
    rows: list[dict[str, Any]],
    model_name: str,
    expected_count: int,
) -> dict[str, Any]:
    by_model: dict[str, list[float]] = defaultdict(list)
    by_algorithm: dict[str, list[float]] = defaultdict(list)
    by_model_algorithm: dict[str, list[float]] = defaultdict(list)
    values = []
    for row in rows:
        value = float(row["vqascore"])
        values.append(value)
        by_model[str(row["model_id"])].append(value)
        by_algorithm[str(row["algorithm_id"])].append(value)
        by_model_algorithm[f"{row['model_id']}__{row['algorithm_id']}"].append(value)

    def stats(items: list[float]) -> dict[str, Any]:
        return {
            "count": len(items),
            "mean": statistics.fmean(items),
            "std": statistics.pstdev(items) if len(items) > 1 else 0.0,
            "min": min(items),
            "max": max(items),
        }

    return {
        "passed": len(rows) == expected_count,
        "vqascore_model": model_name,
        "reward_prompt": "original_prompt",
        "expected_images": expected_count,
        "scored_images": len(rows),
        "overall": stats(values) if values else None,
        "by_model": {key: stats(value) for key, value in sorted(by_model.items())},
        "by_algorithm": {key: stats(value) for key, value in sorted(by_algorithm.items())},
        "by_model_algorithm": {
            key: stats(value) for key, value in sorted(by_model_algorithm.items())
        },
    }


def evaluate_vqascore(
    output_dir: str | Path,
    groups_csv: str | Path | None = None,
    out_csv: str | Path | None = None,
    model_name: str = "clip-flant5-xxl",
    resume: bool = True,
    save_every: int = 1,
    expected_groups: int | None = None,
    write_manifest: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    groups_path = Path(groups_csv).resolve() if groups_csv else output_dir / "groups.csv"
    result_path = Path(out_csv).resolve() if out_csv else output_dir / "vqascore_results.csv"
    summary_path = result_path.with_name("vqascore_summary.json")
    groups = _read_csv(groups_path)
    prepared = _validate_groups(groups, output_dir, expected_groups)
    valid_keys = {
        (item["row"]["group_id"], method) for item in prepared for method in METHODS
    }
    records = (
        _load_existing_results(result_path, valid_keys, model_name)
        if resume
        else {}
    )
    complete_groups = {
        item["row"]["group_id"]
        for item in prepared
        if all((item["row"]["group_id"], method) in records for method in METHODS)
    }
    pending = [item for item in prepared if item["row"]["group_id"] not in complete_groups]
    if pending:
        try:
            import torch
            import t2v_metrics
        except ImportError as exc:
            raise RuntimeError(
                "VQAScore requires the legacy CLIP-FlanT5 release. Install it "
                "in a dedicated environment with: pip install 't2v-metrics==3.0'"
            ) from exc
        try:
            scorer = t2v_metrics.VQAScore(model=model_name)
        except Exception as exc:
            raise RuntimeError(
                f"failed to initialize VQAScore model {model_name!r}. "
                "For CLIP-FlanT5 use: pip install 't2v-metrics==3.0'"
            ) from exc

        for index, item in enumerate(pending, 1):
            row = item["row"]
            image_paths = [str(item["paths"][method]) for method in METHODS]
            with torch.inference_mode():
                tensor = scorer(images=image_paths, texts=[row["prompt"]])
                values = tensor[:, 0].detach().float().cpu().tolist()
            if len(values) != len(METHODS):
                raise RuntimeError(
                    f"VQAScore returned {len(values)} values for {len(METHODS)} images"
                )
            for method, stored_path, value in zip(
                METHODS,
                (row[f"{method}_path"] for method in METHODS),
                values,
            ):
                value = float(value)
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite VQAScore for {row['group_id']}/{method}")
                records[(row["group_id"], method)] = {
                    "group_id": row["group_id"],
                    "model_id": row["model_id"],
                    "prompt_id": row["prompt_id"],
                    "algorithm_id": method,
                    "image_path": stored_path,
                    "vqascore": value,
                    "vqascore_model": model_name,
                    "reward_prompt": "original_prompt",
                }
            if index % max(1, int(save_every)) == 0:
                _atomic_csv(
                    result_path,
                    RESULT_FIELDS,
                    _ordered_results(prepared, records),
                )

    ordered = _ordered_results(prepared, records)
    expected_images = len(prepared) * len(METHODS)
    if len(ordered) != expected_images:
        raise RuntimeError(f"VQAScore results contain {len(ordered)}/{expected_images} images")
    _atomic_csv(result_path, RESULT_FIELDS, ordered)
    report = _summary(ordered, model_name, expected_images)
    _atomic_json(summary_path, report)
    if write_manifest:
        _write_joined_manifest(
            output_dir / "manifest.csv",
            output_dir / "manifest_with_vqascore.csv",
            records,
        )
    return {
        "passed": report["passed"],
        "results_csv": str(result_path),
        "summary_json": str(summary_path),
        "manifest_with_vqascore": (
            str(output_dir / "manifest_with_vqascore.csv") if write_manifest else None
        ),
        "scored_images": len(ordered),
        "resumed_images": len(complete_groups) * len(METHODS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--groups-csv", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--model", default="clip-flant5-xxl")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--expected-groups", type=int, default=None)
    parser.add_argument(
        "--write-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write manifest_with_vqascore.csv without modifying manifest.csv.",
    )
    args = parser.parse_args()
    result = evaluate_vqascore(
        output_dir=args.output_dir,
        groups_csv=args.groups_csv,
        out_csv=args.out_csv,
        model_name=str(args.model),
        resume=bool(args.resume),
        save_every=max(1, int(args.save_every)),
        expected_groups=args.expected_groups,
        write_manifest=bool(args.write_manifest),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
