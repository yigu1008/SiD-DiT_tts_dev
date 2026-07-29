#!/usr/bin/env python3
"""Minimal ActDiff pairwise human-evaluation pipeline.

This tool validates an existing four-method image tree, builds only ActDiff
versus baseline/BoN/DAS tasks, serves a blinded local annotation page, and
computes tie-adjusted pairwise human win rates. It does not generate images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import random
import shutil
import threading
from collections import defaultdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from PIL import Image


METHODS = ("baseline", "bon", "das", "actdiff")
COMPETITORS = ("baseline", "bon", "das")
CHOICES = ("left", "right", "tie", "skip")
RESPONSE_FIELDS = ("annotator_id", "task_id", "choice", "timestamp")
RESULT_FIELDS = (
    "model",
    "comparison",
    "wins",
    "losses",
    "ties",
    "valid_responses",
    "win_rate",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: python -m pip install pyyaml") from exc
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a YAML mapping: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _opaque(seed: int, *parts: str, length: int = 20) -> str:
    text = "|".join([str(seed), *parts])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _models(config: dict[str, Any]) -> list[dict[str, str]]:
    raw = config.get("models")
    if not isinstance(raw, list):
        raise ValueError("config models must be a list")
    models: list[dict[str, str]] = []
    for value in raw:
        if isinstance(value, str):
            models.append({"id": value, "display_name": value})
        elif isinstance(value, dict) and value.get("id"):
            models.append(
                {
                    "id": str(value["id"]),
                    "display_name": str(value.get("display_name") or value["id"]),
                }
            )
        else:
            raise ValueError(f"invalid model entry: {value!r}")
    ids = [model["id"] for model in models]
    if len(ids) != len(set(ids)):
        raise ValueError("config contains duplicate model IDs")
    expected = int(config.get("expected_model_count", 5))
    if len(models) != expected:
        raise ValueError(f"expected {expected} models, found {len(models)}")
    return models


def _settings(
    config_path: Path,
    root_override: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    config_path = config_path.expanduser().resolve()
    config = _read_yaml(config_path)
    root_value = root_override or Path(str(config.get("study_root", ".")))
    root = root_value.expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    root = root.resolve()
    paths = config.get("paths") or {}
    if not isinstance(paths, dict):
        raise ValueError("config paths must be a mapping")

    def resolve(name: str, default: str) -> Path:
        value = Path(str(paths.get(name, default))).expanduser()
        return value.resolve() if value.is_absolute() else (root / value).resolve()

    resolved = {
        "root": root,
        "prompts": resolve("prompts", "prompts.csv"),
        "images": resolve("images", "images"),
        "tasks": resolve("tasks", "tasks/pairwise_tasks.jsonl"),
        "validation_report": resolve(
            "validation_report", "tasks/validation_report.json"
        ),
        "build_report": resolve("build_report", "tasks/task_build_report.json"),
        "responses": resolve("responses", "responses/responses.csv"),
        "winrates": resolve("winrates", "results/winrates.csv"),
        "markdown": resolve("markdown", "results/winrates.md"),
        "legacy_import_report": resolve(
            "legacy_import_report", "tasks/legacy_import_report.json"
        ),
    }
    return config, resolved


def _prompts(path: Path, expected_count: int) -> list[dict[str, str]]:
    rows = _read_csv(path)
    if not rows or not {"prompt_id", "prompt"}.issubset(rows[0]):
        raise ValueError(f"{path} must contain prompt_id,prompt")
    normalized = [
        {"prompt_id": str(row["prompt_id"]), "prompt": str(row["prompt"])}
        for row in rows
    ]
    ids = [row["prompt_id"] for row in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate prompt IDs in {path}")
    if any(not row["prompt"].strip() for row in normalized):
        raise ValueError(f"empty prompt text in {path}")
    if len(normalized) != expected_count:
        raise ValueError(
            f"expected {expected_count} prompts, found {len(normalized)} in {path}"
        )
    return normalized


def _method_files(config: dict[str, Any]) -> dict[str, str]:
    raw = config.get("method_files") or {name: f"{name}.png" for name in METHODS}
    if not isinstance(raw, dict) or set(raw) != set(METHODS):
        raise ValueError(f"method_files must contain exactly: {', '.join(METHODS)}")
    result = {method: str(raw[method]) for method in METHODS}
    if len(set(result.values())) != len(METHODS):
        raise ValueError("method image filenames must be unique")
    return result


def _image_path(
    images_dir: Path,
    model_id: str,
    prompt_id: str,
    filename: str,
) -> Path:
    return images_dir / model_id / prompt_id / filename


def validate_inputs(
    config_path: Path,
    *,
    root_override: Path | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    config, paths = _settings(config_path, root_override)
    models = _models(config)
    expected_prompts = int(config.get("expected_prompt_count", 40))
    errors: list[dict[str, str]] = []
    try:
        prompts = _prompts(paths["prompts"], expected_prompts)
    except (OSError, ValueError) as exc:
        prompts = []
        errors.append({"kind": "prompts", "path": str(paths["prompts"]), "error": str(exc)})
    method_files = _method_files(config)
    valid_images = 0
    dimensions: dict[str, list[int]] = {}
    for model in models:
        for prompt in prompts:
            for method in METHODS:
                image_path = _image_path(
                    paths["images"],
                    model["id"],
                    prompt["prompt_id"],
                    method_files[method],
                )
                if not image_path.is_file():
                    errors.append(
                        {
                            "kind": "missing_image",
                            "model_id": model["id"],
                            "prompt_id": prompt["prompt_id"],
                            "method": method,
                            "path": str(image_path),
                        }
                    )
                    continue
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                    with Image.open(image_path) as image:
                        width, height = image.size
                    dimensions[f"{model['id']}/{prompt['prompt_id']}/{method}"] = [
                        width,
                        height,
                    ]
                    valid_images += 1
                except (OSError, ValueError) as exc:
                    errors.append(
                        {
                            "kind": "unreadable_image",
                            "model_id": model["id"],
                            "prompt_id": prompt["prompt_id"],
                            "method": method,
                            "path": str(image_path),
                            "error": str(exc),
                        }
                    )
    expected_images = len(models) * expected_prompts * len(METHODS)
    report = {
        "valid": not errors and valid_images == expected_images,
        "model_count": len(models),
        "prompt_count": len(prompts),
        "method_count": len(METHODS),
        "expected_images": expected_images,
        "valid_images": valid_images,
        "error_count": len(errors),
        "errors": errors,
        "dimensions": dimensions,
    }
    if write_report:
        _write_json(paths["validation_report"], report)
    return report


def _resolve_legacy_image(row: dict[str, str], manifest_dir: Path) -> Path:
    attempted: list[Path] = []
    for field in ("summary_image_path", "source_image_path"):
        raw = str(row.get(field, "")).strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = manifest_dir / path
        attempted.append(path)
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"no source image for {row.get('image_id')}: "
        + ", ".join(str(path) for path in attempted)
    )


def import_legacy(
    config_path: Path,
    manifest_path: Path,
    *,
    root_override: Path | None = None,
    overwrite: bool = False,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Copy existing legacy-summary images into the pairwise input layout."""
    config, paths = _settings(config_path, root_override)
    models = _models(config)
    prompts = _prompts(paths["prompts"], int(config.get("expected_prompt_count", 40)))
    method_files = _method_files(config)
    manifest_path = manifest_path.expanduser().resolve()
    rows = _read_csv(manifest_path)
    by_cell: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            str(row.get("model_id", "")),
            str(row.get("prompt_id", "")),
            str(row.get("algorithm_id", "")),
        )
        if key in by_cell:
            raise ValueError(f"duplicate legacy manifest cell: {key}")
        by_cell[key] = row
    # bon_mcts is only a legacy source identifier. The study method is actdiff.
    legacy_method = {
        "baseline": "baseline",
        "bon": "bon",
        "das": "das",
        "actdiff": "bon_mcts",
    }
    imported = reused = 0
    errors: list[dict[str, str]] = []
    for model in models:
        for prompt in prompts:
            for method in METHODS:
                source_method = legacy_method[method]
                key = (model["id"], prompt["prompt_id"], source_method)
                row = by_cell.get(key)
                if row is None:
                    errors.append(
                        {
                            "kind": "missing_legacy_cell",
                            "model_id": model["id"],
                            "prompt_id": prompt["prompt_id"],
                            "method": method,
                        }
                    )
                    continue
                if row.get("status") != "complete":
                    errors.append(
                        {
                            "kind": "incomplete_legacy_cell",
                            "model_id": model["id"],
                            "prompt_id": prompt["prompt_id"],
                            "method": method,
                            "status": str(row.get("status", "")),
                        }
                    )
                    continue
                if row.get("original_prompt_c0") != prompt["prompt"]:
                    errors.append(
                        {
                            "kind": "prompt_mismatch",
                            "model_id": model["id"],
                            "prompt_id": prompt["prompt_id"],
                            "method": method,
                        }
                    )
                    continue
                try:
                    source = _resolve_legacy_image(row, manifest_path.parent)
                    with Image.open(source) as image:
                        image.verify()
                    source_digest = _sha256(source)
                except (OSError, ValueError) as exc:
                    errors.append(
                        {
                            "kind": "invalid_legacy_image",
                            "model_id": model["id"],
                            "prompt_id": prompt["prompt_id"],
                            "method": method,
                            "error": str(exc),
                        }
                    )
                    continue
                destination = _image_path(
                    paths["images"],
                    model["id"],
                    prompt["prompt_id"],
                    method_files[method],
                )
                if destination.is_file() and _sha256(destination) == source_digest:
                    reused += 1
                    continue
                if destination.exists() and not overwrite:
                    errors.append(
                        {
                            "kind": "destination_conflict",
                            "model_id": model["id"],
                            "prompt_id": prompt["prompt_id"],
                            "method": method,
                            "path": str(destination),
                        }
                    )
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".tmp")
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
                imported += 1
    report = {
        "valid": not errors,
        "allow_incomplete": allow_incomplete,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "destination_images": str(paths["images"]),
        "legacy_method_mapping": legacy_method,
        "imported_images": imported,
        "reused_images": reused,
        "error_count": len(errors),
        "errors": errors,
    }
    _write_json(paths["legacy_import_report"], report)
    return report


def build_tasks(
    config_path: Path,
    *,
    root_override: Path | None = None,
    overwrite: bool = False,
    allow_incomplete: bool = False,
    seed_override: int | None = None,
) -> dict[str, Any]:
    config, paths = _settings(config_path, root_override)
    validation = validate_inputs(
        config_path, root_override=root_override, write_report=True
    )
    if not validation["valid"] and not allow_incomplete:
        raise RuntimeError(
            f"input validation failed with {validation['error_count']} errors; "
            f"see {paths['validation_report']}"
        )
    if paths["tasks"].exists() and not overwrite:
        raise FileExistsError(f"tasks already exist: {paths['tasks']}")
    models = _models(config)
    prompts = _prompts(paths["prompts"], int(config.get("expected_prompt_count", 40)))
    method_files = _method_files(config)
    seed = (
        int(seed_override)
        if seed_override is not None
        else int(config.get("random_seed", 20260729))
    )
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []

    for model in models:
        for competitor in COMPETITORS:
            valid_prompts: list[dict[str, str]] = []
            for prompt in prompts:
                required_paths = {
                    "actdiff": _image_path(
                        paths["images"],
                        model["id"],
                        prompt["prompt_id"],
                        method_files["actdiff"],
                    ),
                    competitor: _image_path(
                        paths["images"],
                        model["id"],
                        prompt["prompt_id"],
                        method_files[competitor],
                    ),
                }
                invalid: list[str] = []
                for method, image_path in required_paths.items():
                    if not image_path.is_file():
                        invalid.append(f"{method}:missing")
                        continue
                    try:
                        with Image.open(image_path) as image:
                            image.verify()
                    except (OSError, ValueError):
                        invalid.append(f"{method}:unreadable")
                if invalid:
                    exclusions.append(
                        {
                            "model_id": model["id"],
                            "prompt_id": prompt["prompt_id"],
                            "competitor": competitor,
                            "reason": ",".join(invalid),
                        }
                    )
                else:
                    valid_prompts.append(prompt)
            prompt_order = list(valid_prompts)
            rng.shuffle(prompt_order)
            side_flags = [False] * (len(valid_prompts) // 2) + [True] * (
                len(valid_prompts) // 2
            )
            if len(valid_prompts) % 2:
                side_flags.append(bool(rng.getrandbits(1)))
            rng.shuffle(side_flags)
            for prompt, actdiff_on_right in zip(prompt_order, side_flags):
                left_method = competitor if actdiff_on_right else "actdiff"
                right_method = "actdiff" if actdiff_on_right else competitor
                task_token = _opaque(
                    seed,
                    model["id"],
                    prompt["prompt_id"],
                    competitor,
                )
                task_id = f"task_{task_token}"
                left_image = f"image_{_opaque(seed, task_token, 'left')}"
                right_image = f"image_{_opaque(seed, task_token, 'right')}"
                left_path = _image_path(
                    paths["images"],
                    model["id"],
                    prompt["prompt_id"],
                    method_files[left_method],
                )
                right_path = _image_path(
                    paths["images"],
                    model["id"],
                    prompt["prompt_id"],
                    method_files[right_method],
                )
                tasks.append(
                    {
                        "task_id": task_id,
                        "model_id": model["id"],
                        "prompt_id": prompt["prompt_id"],
                        "prompt": prompt["prompt"],
                        "left_image": left_image,
                        "right_image": right_image,
                        "left_method": left_method,
                        "right_method": right_method,
                        "competitor": competitor,
                        "left_image_path": str(left_path),
                        "right_image_path": str(right_path),
                    }
                )
    rng.shuffle(tasks)
    expected = len(models) * len(prompts) * len(COMPETITORS)
    if not allow_incomplete and len(tasks) != expected:
        raise AssertionError(f"expected {expected} tasks, constructed {len(tasks)}")
    if not tasks:
        raise RuntimeError("no complete ActDiff pairwise tasks could be constructed")
    _write_jsonl(paths["tasks"], tasks)
    side_counts: dict[str, dict[str, int]] = {}
    for model in models:
        for competitor in COMPETITORS:
            rows = [
                task
                for task in tasks
                if task["model_id"] == model["id"]
                and task["competitor"] == competitor
            ]
            side_counts[f"{model['id']}/actdiff_vs_{competitor}"] = {
                "left": sum(task["left_method"] == "actdiff" for task in rows),
                "right": sum(task["right_method"] == "actdiff" for task in rows),
            }
    report = {
        "valid": len(tasks) == expected,
        "allow_incomplete": allow_incomplete,
        "random_seed": seed,
        "task_count": len(tasks),
        "expected_task_count": expected,
        "excluded_task_count": len(exclusions),
        "exclusions": exclusions,
        "comparison_count_per_model_prompt": len(COMPETITORS),
        "comparisons": [f"actdiff vs {name}" for name in COMPETITORS],
        "side_counts": side_counts,
        "tasks_path": str(paths["tasks"]),
    }
    _write_json(paths["build_report"], report)
    return report


ANNOTATION_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pairwise Image Evaluation</title>
<style>
  :root { color-scheme: light; font-family: Arial, sans-serif; }
  body { margin: 0 auto; max-width: 1500px; padding: 20px; background: #f5f5f5; }
  header { display: flex; justify-content: space-between; gap: 20px; }
  #prompt { font-size: 20px; line-height: 1.35; background: white; padding: 16px;
            border-radius: 8px; min-height: 54px; }
  .question { text-align: center; font-weight: 600; margin: 16px 0; }
  .images { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  figure { margin: 0; background: white; padding: 10px; border-radius: 8px; }
  figure img { width: 100%; height: min(62vh, 680px); object-fit: contain; display: block; }
  figcaption { text-align: center; font-size: 22px; font-weight: 700; padding-top: 8px; }
  .buttons { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 18px; }
  button { border: 0; border-radius: 8px; padding: 14px 8px; font-size: 16px;
           cursor: pointer; background: #26384a; color: white; }
  button:hover { background: #37546f; }
  button:disabled { opacity: .45; cursor: default; }
  #status { text-align: center; padding: 16px; font-size: 18px; }
  .shortcut { opacity: .7; font-size: 13px; display: block; margin-top: 3px; }
</style>
</head>
<body>
<header><h2>Pairwise image evaluation</h2><div id="progress">Loading…</div></header>
<div id="prompt"></div>
<div class="question">Which image better satisfies the text prompt while remaining visually coherent and high quality?</div>
<div class="images">
  <figure><img id="left" alt="Left image"><figcaption>Left</figcaption></figure>
  <figure><img id="right" alt="Right image"><figcaption>Right</figcaption></figure>
</div>
<div class="buttons">
  <button data-choice="left">Left is better<span class="shortcut">←</span></button>
  <button data-choice="right">Right is better<span class="shortcut">→</span></button>
  <button data-choice="tie">About the same<span class="shortcut">↑</span></button>
  <button data-choice="skip">Skip / broken image<span class="shortcut">↓</span></button>
</div>
<div id="status"></div>
<script>
let currentTask = null;
const buttons = [...document.querySelectorAll("button")];
function disabled(value) { buttons.forEach(button => button.disabled = value); }
async function loadTask() {
  disabled(true);
  const response = await fetch("/api/task");
  const data = await response.json();
  document.getElementById("progress").textContent = `${data.completed} / ${data.total}`;
  if (data.done) {
    currentTask = null;
    document.getElementById("prompt").textContent = "All tasks are complete.";
    document.getElementById("left").removeAttribute("src");
    document.getElementById("right").removeAttribute("src");
    document.getElementById("status").textContent = "Responses have been saved.";
    return;
  }
  currentTask = data.task_id;
  document.getElementById("prompt").textContent = data.prompt;
  document.getElementById("left").src = data.left_image;
  document.getElementById("right").src = data.right_image;
  document.getElementById("status").textContent = "";
  disabled(false);
}
async function choose(choice) {
  if (!currentTask) return;
  disabled(true);
  const response = await fetch("/api/respond", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({task_id: currentTask, choice})
  });
  if (!response.ok) {
    document.getElementById("status").textContent = await response.text();
    disabled(false);
    return;
  }
  await loadTask();
}
buttons.forEach(button => button.addEventListener("click", () => choose(button.dataset.choice)));
document.addEventListener("keydown", event => {
  const choices = {ArrowLeft: "left", ArrowRight: "right", ArrowUp: "tie", ArrowDown: "skip"};
  if (choices[event.key]) { event.preventDefault(); choose(choices[event.key]); }
});
loadTask().catch(error => document.getElementById("status").textContent = error);
</script>
</body>
</html>
"""


class AnnotationStore:
    def __init__(self, tasks: list[dict[str, Any]], responses: Path, annotator: str):
        self.tasks = tasks
        self.by_id = {str(task["task_id"]): task for task in tasks}
        if len(self.by_id) != len(tasks):
            raise ValueError("task file contains duplicate task IDs")
        self.responses = responses
        self.annotator = annotator
        self.lock = threading.Lock()
        self.completed = self._completed()
        self.image_paths: dict[str, Path] = {}
        for task in tasks:
            self.image_paths[str(task["left_image"])] = Path(task["left_image_path"])
            self.image_paths[str(task["right_image"])] = Path(task["right_image_path"])

    def _completed(self) -> set[str]:
        completed: set[str] = set()
        if not self.responses.is_file():
            return completed
        for row in _read_csv(self.responses):
            if row.get("annotator_id") == self.annotator and row.get("choice") in CHOICES:
                completed.add(str(row.get("task_id", "")))
        return completed

    def next_task(self) -> dict[str, Any]:
        with self.lock:
            for task in self.tasks:
                if task["task_id"] not in self.completed:
                    return {
                        "done": False,
                        "completed": len(self.completed),
                        "total": len(self.tasks),
                        "task_id": task["task_id"],
                        "prompt": task["prompt"],
                        "left_image": f"/image/{task['left_image']}",
                        "right_image": f"/image/{task['right_image']}",
                    }
            return {
                "done": True,
                "completed": len(self.completed),
                "total": len(self.tasks),
            }

    def record(self, task_id: str, choice: str) -> None:
        if task_id not in self.by_id:
            raise ValueError("unknown task ID")
        if choice not in CHOICES:
            raise ValueError("choice must be left, right, tie, or skip")
        with self.lock:
            self.responses.parent.mkdir(parents=True, exist_ok=True)
            new_file = not self.responses.exists()
            with self.responses.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=RESPONSE_FIELDS, lineterminator="\n"
                )
                if new_file:
                    writer.writeheader()
                writer.writerow(
                    {
                        "annotator_id": self.annotator,
                        "task_id": task_id,
                        "choice": choice,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                handle.flush()
                os.fsync(handle.fileno())
            self.completed.add(task_id)


def serve_annotations(
    config_path: Path,
    *,
    root_override: Path | None = None,
    host_override: str | None = None,
    port_override: int | None = None,
) -> None:
    config, paths = _settings(config_path, root_override)
    if not paths["tasks"].is_file():
        raise FileNotFoundError(
            f"pairwise task file is missing: {paths['tasks']}\n"
            "Prepare it from existing legacy outputs first:\n"
            f"  HUMAN_EVAL_ROOT={paths['root']} "
            "bash tools/prepare_pairwise_human_eval.sh --allow-incomplete"
        )
    tasks = _read_jsonl(paths["tasks"])
    annotator = str(config.get("annotator_id", "")).strip()
    if not annotator:
        raise ValueError("config annotator_id must be nonempty")
    store = AnnotationStore(tasks, paths["responses"], annotator)

    class Handler(BaseHTTPRequestHandler):
        def _json(self, value: Any, status: int = 200) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                payload = ANNOTATION_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
                return
            if path == "/api/task":
                self._json(store.next_task())
                return
            if path.startswith("/image/"):
                image_id = path.removeprefix("/image/")
                image_path = store.image_paths.get(image_id)
                if image_path is None or not image_path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                payload = image_path.read_bytes()
                content_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "private, max-age=3600")
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/respond":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                value = json.loads(self.rfile.read(length))
                store.record(str(value["task_id"]), str(value["choice"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, status=400)
                return
            self._json({"saved": True})

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[human-eval] {self.address_string()} {format % args}")

    server_config = config.get("server") or {}
    host = host_override or str(server_config.get("host", "127.0.0.1"))
    port = port_override or int(server_config.get("port", 8000))
    print(f"Serving blinded pairwise evaluation at http://{host}:{port}")
    print(f"Annotator: {annotator}")
    print(f"Responses: {paths['responses']}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def _metric(rows: list[tuple[str, str]]) -> dict[str, Any]:
    wins = sum(outcome == "win" for _, outcome in rows)
    losses = sum(outcome == "loss" for _, outcome in rows)
    ties = sum(outcome == "tie" for _, outcome in rows)
    valid = wins + losses + ties
    rate = (wins + 0.5 * ties) / valid if valid else None
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "valid_responses": valid,
        "win_rate": "" if rate is None else f"{rate:.6f}",
    }


def compute_winrates(
    config_path: Path,
    *,
    root_override: Path | None = None,
) -> dict[str, Any]:
    config, paths = _settings(config_path, root_override)
    models = _models(config)
    tasks = _read_jsonl(paths["tasks"])
    task_by_id = {str(task["task_id"]): task for task in tasks}
    if not paths["responses"].is_file():
        raise FileNotFoundError(f"responses not found: {paths['responses']}")
    annotator = str(config.get("annotator_id", "")).strip()
    latest: dict[str, dict[str, str]] = {}
    ignored_annotators = 0
    for response in _read_csv(paths["responses"]):
        if response.get("annotator_id") != annotator:
            ignored_annotators += 1
            continue
        task_id = str(response.get("task_id", ""))
        choice = str(response.get("choice", ""))
        if task_id not in task_by_id:
            raise ValueError(f"response references unknown task: {task_id}")
        if choice not in CHOICES:
            raise ValueError(f"invalid response choice for {task_id}: {choice}")
        latest[task_id] = response

    outcomes: list[dict[str, str]] = []
    skips = 0
    for task_id, response in latest.items():
        task = task_by_id[task_id]
        choice = response["choice"]
        if choice == "skip":
            skips += 1
            continue
        if choice == "tie":
            outcome = "tie"
        else:
            selected_method = task[f"{choice}_method"]
            outcome = "win" if selected_method == "actdiff" else "loss"
        outcomes.append(
            {
                "model_id": str(task["model_id"]),
                "competitor": str(task["competitor"]),
                "outcome": outcome,
            }
        )

    result_rows: list[dict[str, Any]] = []
    for model in models:
        model_outcomes = [
            (row["competitor"], row["outcome"])
            for row in outcomes
            if row["model_id"] == model["id"]
        ]
        for competitor in COMPETITORS:
            subset = [row for row in model_outcomes if row[0] == competitor]
            result_rows.append(
                {
                    "model": model["display_name"],
                    "comparison": f"actdiff vs {competitor}",
                    **_metric(subset),
                }
            )
        result_rows.append(
            {
                "model": model["display_name"],
                "comparison": "overall",
                **_metric(model_outcomes),
            }
        )
    aggregate = [(row["competitor"], row["outcome"]) for row in outcomes]
    for competitor in COMPETITORS:
        subset = [row for row in aggregate if row[0] == competitor]
        result_rows.append(
            {
                "model": "Aggregate",
                "comparison": f"actdiff vs {competitor}",
                **_metric(subset),
            }
        )
    result_rows.append(
        {
            "model": "Aggregate",
            "comparison": "overall",
            **_metric(aggregate),
        }
    )
    _write_csv(paths["winrates"], result_rows, RESULT_FIELDS)

    rates = {
        (str(row["model"]), str(row["comparison"])): row["win_rate"]
        for row in result_rows
    }

    def percent(value: Any) -> str:
        return "—" if value in ("", None) else f"{100 * float(value):.1f}%"

    lines = [
        "| Model | vs. baseline | vs. BoN | vs. DAS | Overall |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in [*models, {"display_name": "Aggregate"}]:
        name = model["display_name"]
        lines.append(
            f"| {name} | "
            f"{percent(rates.get((name, 'actdiff vs baseline')))} | "
            f"{percent(rates.get((name, 'actdiff vs bon')))} | "
            f"{percent(rates.get((name, 'actdiff vs das')))} | "
            f"{percent(rates.get((name, 'overall')))} |"
        )
    paths["markdown"].parent.mkdir(parents=True, exist_ok=True)
    paths["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "annotator_id": annotator,
        "task_count": len(tasks),
        "answered_tasks": len(latest),
        "valid_responses": len(outcomes),
        "skips": skips,
        "ignored_other_annotator_rows": ignored_annotators,
        "winrates_csv": str(paths["winrates"]),
        "markdown": str(paths["markdown"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pairwise_human_eval.yaml"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Override study_root from the YAML configuration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validator = subparsers.add_parser("validate")
    validator.add_argument("--allow-incomplete", action="store_true")
    importer = subparsers.add_parser(
        "import-legacy",
        help="Copy existing legacy-manifest images; map bon_mcts to actdiff.",
    )
    importer.add_argument("--manifest", required=True, type=Path)
    importer.add_argument("--overwrite", action="store_true")
    importer.add_argument("--allow-incomplete", action="store_true")
    builder = subparsers.add_parser("build-tasks")
    builder.add_argument("--overwrite", action="store_true")
    builder.add_argument("--allow-incomplete", action="store_true")
    builder.add_argument("--seed", type=int, default=None)
    server = subparsers.add_parser("serve")
    server.add_argument("--host", default=None)
    server.add_argument("--port", type=int, default=None)
    subparsers.add_parser("winrates")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        result = validate_inputs(args.config, root_override=args.root)
        print(json.dumps(result, indent=2))
        return 0 if result["valid"] or args.allow_incomplete else 1
    if args.command == "import-legacy":
        result = import_legacy(
            args.config,
            args.manifest,
            root_override=args.root,
            overwrite=args.overwrite,
            allow_incomplete=args.allow_incomplete,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["valid"] or args.allow_incomplete else 1
    if args.command == "build-tasks":
        result = build_tasks(
            args.config,
            root_override=args.root,
            overwrite=args.overwrite,
            allow_incomplete=args.allow_incomplete,
            seed_override=args.seed,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "serve":
        serve_annotations(
            args.config,
            root_override=args.root,
            host_override=args.host,
            port_override=args.port,
        )
        return 0
    if args.command == "winrates":
        result = compute_winrates(args.config, root_override=args.root)
        print(json.dumps(result, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
