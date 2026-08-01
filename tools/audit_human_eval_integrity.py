#!/usr/bin/env python3
"""Read-only integrity audit for a local pairwise human-evaluation package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "pairwise_human_eval.yaml"
TASK_FIELDS = {
    "task_id", "model_id", "prompt_id", "prompt", "left_image",
    "right_image", "left_method", "right_method", "competitor",
    "left_image_path", "right_image_path",
}
RESPONSE_FIELDS = {"annotator_id", "task_id", "choice", "timestamp"}
CHOICES = {"left", "right", "tie", "skip"}


def _yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for this audit") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve(strict=False)),
        "bytes": "", "width": "", "height": "", "format": "",
        "sha256": "", "status": "missing",
    }
    if not path.is_file():
        return record
    try:
        digest = _sha256(path)
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            image_format = image.format or ""
        record.update({
            "bytes": path.stat().st_size,
            "width": width,
            "height": height,
            "format": image_format,
            "sha256": digest,
            "status": "ok",
        })
    except Exception as exc:  # PIL intentionally reports several exception types.
        record["status"] = f"unreadable:{type(exc).__name__}:{exc}"
    return record


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _configured_path(root: Path, config: dict[str, Any], key: str, fallback: str) -> Path:
    value = config.get("paths", {}).get(key, fallback)
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def audit(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = args.root.expanduser().resolve()
    config = _yaml(args.config.expanduser().resolve())
    models = [str(row["id"]) for row in config.get("models", [])]
    method_files = {
        str(method): str(filename)
        for method, filename in config.get("method_files", {}).items()
    }
    expected_prompt_count = int(config.get("expected_prompt_count", 40))
    expected_model_count = int(config.get("expected_model_count", len(models)))
    prompts_path = root / "prompts.csv"
    images_root = root / "images"
    tasks_path = _configured_path(
        root, config, "tasks", "tasks/pairwise_tasks.jsonl"
    )
    responses_path = _configured_path(
        root, config, "responses", "responses/responses.csv"
    )

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def error(kind: str, message: str, **details: Any) -> None:
        errors.append({"kind": kind, "message": message, **details})

    def warning(kind: str, message: str, **details: Any) -> None:
        warnings.append({"kind": kind, "message": message, **details})

    prompts: list[dict[str, str]] = []
    if not prompts_path.is_file():
        error("missing_prompts", f"missing {prompts_path}")
    else:
        try:
            prompts = _csv(prompts_path)
        except Exception as exc:
            error("unreadable_prompts", str(exc))
        if prompts and not {"prompt_id", "prompt"}.issubset(prompts[0]):
            error("prompt_schema", "prompts.csv needs prompt_id,prompt columns")
            prompts = []
    prompt_by_id: dict[str, str] = {}
    for index, row in enumerate(prompts):
        prompt_id = str(row.get("prompt_id", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        if not prompt_id or not prompt:
            error("empty_prompt", f"empty prompt row {index + 2}")
            continue
        if prompt_id in prompt_by_id:
            error("duplicate_prompt_id", f"duplicate prompt_id {prompt_id}")
        prompt_by_id[prompt_id] = prompt
    if len(prompts) != expected_prompt_count:
        error(
            "prompt_count",
            f"found {len(prompts)} prompts; expected {expected_prompt_count}",
        )
    if len(models) != expected_model_count:
        error(
            "model_config_count",
            f"config lists {len(models)} models; expected {expected_model_count}",
        )
    if not models or not method_files:
        error("config_schema", "config must define models and method_files")

    inventory: list[dict[str, Any]] = []
    for model in models:
        for prompt_id in prompt_by_id:
            hashes: dict[str, list[str]] = {}
            dimensions: set[tuple[Any, Any]] = set()
            for method, filename in method_files.items():
                path = images_root / model / prompt_id / filename
                info = _image_record(path)
                row = {
                    "model_id": model,
                    "prompt_id": prompt_id,
                    "method": method,
                    **info,
                }
                inventory.append(row)
                if info["status"] == "missing":
                    error(
                        "missing_image", f"missing {model}/{prompt_id}/{filename}",
                        model_id=model, prompt_id=prompt_id, method=method,
                    )
                elif info["status"] != "ok":
                    error(
                        "unreadable_image", info["status"], path=str(path),
                        model_id=model, prompt_id=prompt_id, method=method,
                    )
                else:
                    hashes.setdefault(str(info["sha256"]), []).append(method)
                    dimensions.add((info["width"], info["height"]))
            duplicate_methods = [values for values in hashes.values() if len(values) > 1]
            if duplicate_methods:
                warning(
                    "duplicate_method_images",
                    f"byte-identical method outputs in {model}/{prompt_id}",
                    groups=duplicate_methods,
                )
            if len(dimensions) > 1:
                warning(
                    "mixed_dimensions",
                    f"source dimensions differ in {model}/{prompt_id}",
                    dimensions=sorted(dimensions),
                )

    if images_root.is_dir():
        unexpected_models = sorted(
            path.name for path in images_root.iterdir()
            if path.is_dir() and path.name not in models
        )
        if unexpected_models:
            warning("unexpected_models", "extra model folders found", values=unexpected_models)

    tasks: list[dict[str, Any]] = []
    if tasks_path.is_file():
        try:
            tasks = _jsonl(tasks_path)
        except Exception as exc:
            error("task_parse", str(exc))
        task_ids: set[str] = set()
        task_keys: Counter[tuple[str, str, str]] = Counter()
        opaque_images: set[str] = set()
        for index, task in enumerate(tasks, 1):
            missing_fields = sorted(TASK_FIELDS - set(task))
            if missing_fields:
                error("task_schema", f"task line {index} missing fields", fields=missing_fields)
                continue
            task_id = str(task["task_id"])
            if task_id in task_ids:
                error("duplicate_task_id", f"duplicate task_id {task_id}")
            task_ids.add(task_id)
            model = str(task["model_id"])
            prompt_id = str(task["prompt_id"])
            competitor = str(task["competitor"])
            task_keys[(model, prompt_id, competitor)] += 1
            if model not in models or prompt_id not in prompt_by_id:
                error("task_unknown_group", f"unknown group in {task_id}")
            elif str(task["prompt"]) != prompt_by_id[prompt_id]:
                error("task_prompt_mismatch", f"prompt mismatch in {task_id}")
            pair = {str(task["left_method"]), str(task["right_method"])}
            if pair != {"actdiff", competitor} or competitor not in {"baseline", "bon", "das"}:
                error("task_comparison", f"invalid method pair in {task_id}", pair=sorted(pair))
            for side in ("left", "right"):
                opaque = str(task[f"{side}_image"])
                if opaque in opaque_images:
                    error("duplicate_opaque_image_id", f"duplicate image ID {opaque}")
                opaque_images.add(opaque)
                method = str(task[f"{side}_method"])
                canonical = images_root / model / prompt_id / method_files.get(method, "")
                recorded = Path(str(task[f"{side}_image_path"])).expanduser()
                if not recorded.is_absolute():
                    recorded = root / recorded
                if recorded.resolve(strict=False) != canonical.resolve(strict=False):
                    kind = "stale_task_image_path" if canonical.is_file() else "task_image_path_mismatch"
                    error(
                        kind,
                        f"{task_id} {side} path does not point to this local package",
                        recorded=str(recorded), expected=str(canonical),
                    )
        expected_keys = {
            (model, prompt_id, competitor)
            for model in models
            for prompt_id in prompt_by_id
            for competitor in ("baseline", "bon", "das")
            if (images_root / model / prompt_id / method_files.get("actdiff", "")).is_file()
            and (images_root / model / prompt_id / method_files.get(competitor, "")).is_file()
        }
        actual_keys = set(task_keys)
        missing_keys = expected_keys - actual_keys
        duplicate_keys = [key for key, count in task_keys.items() if count != 1]
        if missing_keys:
            error("missing_tasks", f"missing {len(missing_keys)} constructible tasks")
        if duplicate_keys:
            error("duplicate_tasks", f"found {len(duplicate_keys)} duplicate task keys")
    elif args.require_tasks:
        error("missing_tasks_file", f"missing {tasks_path}")
    else:
        warning("missing_tasks_file", f"task audit skipped; {tasks_path} not found")

    if responses_path.is_file():
        try:
            responses = _csv(responses_path)
        except Exception as exc:
            error("response_parse", str(exc))
            responses = []
        if responses and not RESPONSE_FIELDS.issubset(responses[0]):
            error("response_schema", "responses.csv has missing columns")
        known_tasks = {str(task.get("task_id", "")) for task in tasks}
        seen_responses: set[tuple[str, str]] = set()
        for index, response in enumerate(responses, 2):
            key = (str(response.get("annotator_id", "")), str(response.get("task_id", "")))
            if not all(key) or not str(response.get("timestamp", "")):
                error("empty_response_field", f"empty response field at row {index}")
            if str(response.get("choice", "")) not in CHOICES:
                error("invalid_choice", f"invalid choice at row {index}")
            if tasks and key[1] not in known_tasks:
                error("unknown_response_task", f"unknown task at row {index}: {key[1]}")
            if key in seen_responses:
                warning("duplicate_response", f"multiple responses for {key}")
            seen_responses.add(key)

    if args.reference_inventory:
        reference_rows = _csv(args.reference_inventory.expanduser().resolve())
        reference = {
            (row["model_id"], row["prompt_id"], row["method"]): row.get("sha256", "")
            for row in reference_rows
        }
        current = {
            (row["model_id"], row["prompt_id"], row["method"]): row.get("sha256", "")
            for row in inventory
        }
        for key in sorted(set(reference) | set(current)):
            if reference.get(key) != current.get(key):
                error(
                    "reference_hash_mismatch",
                    f"inventory differs for {'/'.join(key)}",
                    reference=reference.get(key, "<missing>"),
                    current=current.get(key, "<missing>"),
                )

    expected_images = len(models) * len(prompt_by_id) * len(method_files)
    valid_images = sum(row["status"] == "ok" for row in inventory)
    missing_only = all(item["kind"] in {"missing_image", "prompt_count"} for item in errors)
    valid = not errors or (args.allow_incomplete and missing_only)
    report = {
        "valid": valid,
        "complete": not errors,
        "allow_incomplete": args.allow_incomplete,
        "root": str(root),
        "config": str(args.config.expanduser().resolve()),
        "prompt_count": len(prompts),
        "model_count": len(models),
        "method_count": len(method_files),
        "expected_images": expected_images,
        "valid_images": valid_images,
        "missing_images": sum(row["status"] == "missing" for row in inventory),
        "unreadable_images": sum(
            row["status"] not in {"ok", "missing"} for row in inventory
        ),
        "task_count": len(tasks),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }
    return report, inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--reference-inventory", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--require-tasks", action="store_true")
    args = parser.parse_args()
    report, inventory = audit(args)
    root = args.root.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir else root / "integrity_reports" / (
            "human_eval_local_check"
            if args.reference_inventory else "human_eval_base"
        )
    )
    inventory_path = out_dir / "image_inventory.csv"
    report_path = out_dir / "integrity_report.json"
    _atomic_csv(
        inventory_path,
        inventory,
        [
            "model_id", "prompt_id", "method", "path", "bytes", "width",
            "height", "format", "sha256", "status",
        ],
    )
    _atomic_json(report_path, report)
    status = "OK" if report["valid"] else "FAIL"
    print(
        f"[human-eval] {status} prompts={report['prompt_count']} "
        f"images={report['valid_images']}/{report['expected_images']} "
        f"tasks={report['task_count']} errors={report['error_count']} "
        f"warnings={report['warning_count']}"
    )
    for item in report["errors"][:10]:
        print(f"  ERROR {item['kind']}: {item['message']}")
    if report["error_count"] > 10:
        print(f"  ... {report['error_count'] - 10} more errors in {report_path}")
    print(f"[human-eval] report={report_path}")
    print(f"[human-eval] inventory={inventory_path}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
