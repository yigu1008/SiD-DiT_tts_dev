#!/usr/bin/env python3
"""Summarize human-evaluation images and logs under legacy_runs.

The collector never rescored images and never infers NFE. For each
model/algorithm/prompt cell it selects the newest run containing both a final
image and its log row, falling back to an older run only when a newer run is
incomplete. Post-evaluation rewards are accepted only when their prompt text
exactly matches the original prompt c0 from prompts.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "human_eval_genai40_v1.yaml"
ALGORITHMS = ("baseline", "bon", "das", "bon_mcts")
MODE_FOR = {"baseline": "base", "bon": "bon", "das": "bon", "bon_mcts": "mcts"}
SD35_SUFFIX_FOR = {"baseline": "base", "bon": "bon", "das": "bon", "bon_mcts": "mcts"}
FLUX_SUFFIXES = {
    "baseline": ("baseline",),
    "bon": ("bon",),
    "das": ("bon", "das"),
    "bon_mcts": ("mcts", "bon_mcts"),
}
MODEL_DISPLAY_NAMES = {
    "flux_schnell": "Flux-Schnell",
    "sid": "SiD-SD3.5",
    "sd35_base": "SD3.5-Large",
    "senseflow_large": "SenseFlow-SD3.5-Large",
    "senseflow_medium": "SenseFlow-SD3.5-Medium",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the human-eval config") from exc
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def _read_prompts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"prompt_id", "prompt"}.issubset(rows[0]):
        raise ValueError(f"{path} must contain prompt_id and prompt columns")
    expected_ids = [f"p{index:03d}" for index in range(len(rows))]
    actual_ids = [str(row["prompt_id"]) for row in rows]
    if actual_ids != expected_ids:
        raise ValueError(
            f"{path} prompt IDs must be contiguous from p000; found {actual_ids[:3]}"
        )
    return rows


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                value = dict(value)
                value["_source_record_path"] = str(path.resolve())
                value["_source_line"] = line_number
                yield value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _recursive_number(value: Any, keys: tuple[str, ...]) -> tuple[float | None, str | None]:
    if not isinstance(value, dict):
        return None, None
    for key in keys:
        number = _as_float(value.get(key))
        if number is not None:
            return number, key
    for key, child in value.items():
        if isinstance(child, dict):
            number, source = _recursive_number(child, keys)
            if number is not None:
                return number, f"{key}.{source}"
    return None, None


def _selected_seed(row: dict[str, Any] | None) -> tuple[int | None, str | None]:
    if not row:
        return None, None
    diagnostics = row.get("search_diagnostics") or row.get("diagnostics")
    number, source = _recursive_number(
        diagnostics,
        ("winner_seed", "chosen_seed", "selected_seed"),
    )
    if number is not None:
        return int(number), f"diagnostics.{source}"
    seed = _as_int(row.get("seed"))
    return seed, "seed" if seed is not None else None


def _logged_nfe(row: dict[str, Any] | None) -> tuple[int | None, str | None]:
    if not row:
        return None, None
    direct = _as_int(row.get("nfe"))
    if direct is not None:
        return direct, "nfe"
    diagnostics = row.get("search_diagnostics") or row.get("diagnostics")
    number, source = _recursive_number(
        diagnostics,
        ("total_nfe", "nfe_total", "actual_nfe", "nfe"),
    )
    if number is not None:
        return int(number), f"diagnostics.{source}"
    return None, None


def _image_info(path: Path) -> tuple[int | None, int | None, str | None, str | None]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        return int(width), int(height), digest.hexdigest(), None
    except Exception as exc:
        return None, None, None, f"{type(exc).__name__}: {exc}"


def _find_sd35_image(method_dir: Path, prompt_index: int, algorithm: str) -> Path | None:
    suffix = SD35_SUFFIX_FOR[algorithm]
    image_dir = method_dir / "images"
    for pad in (5, 6, 4, 3, 2):
        candidate = image_dir / f"p{prompt_index:0{pad}d}_{suffix}.png"
        if candidate.is_file():
            return candidate.resolve()
    for candidate in sorted(image_dir.glob(f"p*_{suffix}.png")):
        stem = candidate.name[1:].rsplit("_", 1)[0]
        try:
            if int(stem) == prompt_index:
                return candidate.resolve()
        except ValueError:
            continue
    return None


def _find_flux_image(
    rank_dir: Path,
    slug: str,
    sample_index: int,
    algorithm: str,
    search_method: str,
) -> Path | None:
    suffixes = list(FLUX_SUFFIXES[algorithm])
    if search_method:
        suffixes.append(search_method)
    for suffix in dict.fromkeys(suffixes):
        candidate = rank_dir / f"{slug}_s{sample_index}_{suffix}.png"
        if candidate.is_file():
            return candidate.resolve()
    candidates = sorted(rank_dir.glob(f"{slug}_s{sample_index}_*.png"))
    for candidate in candidates:
        lower = candidate.name.lower()
        if lower.endswith("_comparison.png"):
            continue
        if algorithm != "baseline" and lower.endswith("_baseline.png"):
            continue
        if algorithm == "baseline" and not lower.endswith("_baseline.png"):
            continue
        return candidate.resolve()
    return None


def _find_flux_image_without_summary(
    method_dir: Path,
    prompt_index: int,
    algorithm: str,
) -> Path | None:
    slug = f"p{prompt_index:04d}"
    for rank_dir in sorted(method_dir.glob("rank_*")):
        if not rank_dir.is_dir():
            continue
        image = _find_flux_image(rank_dir, slug, 0, algorithm, "")
        if image is not None:
            return image
    return None


def _load_sd35_cells(run_dir: Path, algorithm: str) -> dict[int, dict[str, Any]]:
    method_dir = run_dir / algorithm
    mode = MODE_FOR[algorithm]
    rows: dict[int, dict[str, Any]] = {}
    duplicate_counts: Counter[int] = Counter()
    for log_path in sorted((method_dir / "logs").glob("rank_*.jsonl")):
        if log_path.name.endswith("_rewrite_examples.jsonl"):
            continue
        for row in _read_jsonl(log_path):
            if str(row.get("mode", "")) != mode:
                continue
            prompt_index = _as_int(row.get("prompt_index"))
            if prompt_index is None:
                continue
            duplicate_counts[prompt_index] += 1
            row["_source_run_path"] = str(run_dir.resolve())
            row["_source_method_path"] = str(method_dir.resolve())
            row["_source_image_path"] = _find_sd35_image(
                method_dir, prompt_index, algorithm
            )
            rows[prompt_index] = row
    for prompt_index, row in rows.items():
        row["_duplicate_log_rows"] = duplicate_counts[prompt_index]
    return rows


def _load_flux_cells(run_dir: Path, algorithm: str) -> dict[int, dict[str, Any]]:
    method_dir = run_dir / algorithm
    rows: dict[int, dict[str, Any]] = {}
    duplicate_counts: Counter[int] = Counter()
    for summary_path in sorted(method_dir.glob("rank_*/summary.json")):
        payload = _read_json(summary_path)
        if not isinstance(payload, list):
            continue
        rank_dir = summary_path.parent
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            slug = str(entry.get("slug", ""))
            digits = "".join(character for character in slug if character.isdigit())
            prompt_index = _as_int(digits)
            if prompt_index is None:
                continue
            samples = entry.get("samples") or []
            if not isinstance(samples, list) or not samples:
                continue
            sample_index = 0
            sample = samples[sample_index] if isinstance(samples[sample_index], dict) else {}
            search_method = str(entry.get("search_method", "")).strip().lower()
            score_key = "baseline_score" if algorithm == "baseline" else "search_score"
            row = {
                "prompt_index": prompt_index,
                "prompt": entry.get("prompt"),
                "seed": sample.get("seed"),
                "mode": MODE_FOR[algorithm],
                "score": sample.get(score_key),
                "baseline_score": sample.get("baseline_score"),
                "delta_vs_base": (
                    0.0 if algorithm == "baseline" else sample.get("delta_score")
                ),
                "actions": sample.get("actions", []),
                "search_diagnostics": sample.get("diagnostics"),
                "_source_record_path": str(summary_path.resolve()),
                "_source_line": None,
                "_source_run_path": str(run_dir.resolve()),
                "_source_method_path": str(method_dir.resolve()),
                "_source_image_path": _find_flux_image(
                    rank_dir, slug, sample_index, algorithm, search_method
                ),
            }
            duplicate_counts[prompt_index] += 1
            rows[prompt_index] = row
    for prompt_index, row in rows.items():
        row["_duplicate_log_rows"] = duplicate_counts[prompt_index]
    return rows


def _load_reward_rows(method_dir: Path) -> dict[int, list[dict[str, Any]]]:
    by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    path = method_dir / "best_images_multi_reward.json"
    payload = _read_json(path)
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt_index = _as_int(row.get("prompt_index"))
        if prompt_index is not None:
            value = dict(row)
            value["_source_reward_path"] = str(path.resolve())
            by_prompt[prompt_index].append(value)
    return by_prompt


def _reward_for_c0(
    reward_rows: list[dict[str, Any]],
    c0: str,
) -> tuple[dict[str, float], str | None, bool | None]:
    if not reward_rows:
        return {}, None, None
    exact = [row for row in reward_rows if str(row.get("prompt", "")) == c0]
    if not exact:
        return {}, str(reward_rows[-1].get("_source_reward_path", "")), False
    row = exact[-1]
    scores = row.get("scores") or {}
    clean = {
        str(name): float(value)
        for name, value in scores.items()
        if isinstance(value, (int, float))
    }
    return clean, str(row.get("_source_reward_path", "")), True


def _load_seed_map(root: Path, model_id: str, algorithm: str) -> dict[int, int]:
    path = root / "legacy_runs" / "_seed_maps" / model_id / f"{algorithm}.json"
    payload = _read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("seeds"), dict):
        payload = payload["seeds"]
    if not isinstance(payload, dict):
        return {}
    result = {}
    for key, value in payload.items():
        index = _as_int(key)
        seed = _as_int(value)
        if index is not None and seed is not None:
            result[index] = seed
    return result


def _materialize_image(
    source: Path,
    destination: Path,
    mode: str,
) -> Path | None:
    if mode == "none":
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    if mode == "symlink":
        target = os.path.relpath(source, start=destination.parent)
        temporary.symlink_to(target)
    elif mode == "copy":
        shutil.copy2(source, temporary)
    else:
        raise ValueError(f"unsupported materialization mode: {mode}")
    os.replace(temporary, destination)
    return destination.absolute()


def _discover_runs(root: Path, model_id: str, algorithm: str) -> list[Path]:
    parent = root / "legacy_runs" / model_id / algorithm
    return sorted(
        (path.resolve() for path in parent.glob("run_*") if path.is_dir()),
        key=lambda path: path.name,
    )


def summarize(
    root: Path,
    output_dir: Path,
    config_path: Path,
    prompts_path: Path,
    model_filter: set[str] | None = None,
    algorithm_filter: set[str] | None = None,
    materialize: str = "symlink",
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    config = _load_yaml(config_path)
    prompts = _read_prompts(prompts_path)
    model_specs = {
        str(model["model_id"]): dict(model)
        for model in config.get("models", [])
        if isinstance(model, dict) and model.get("model_id")
    }
    model_ids = [
        model_id
        for model_id in model_specs
        if model_filter is None or model_id in model_filter
    ]
    algorithms = [
        str(algorithm)
        for algorithm in config.get("algorithms", ALGORITHMS)
        if algorithm in ALGORITHMS
        and (algorithm_filter is None or algorithm in algorithm_filter)
    ]
    if model_filter:
        unknown = model_filter - set(model_specs)
        if unknown:
            raise ValueError(f"unknown model IDs: {sorted(unknown)}")
    if algorithm_filter:
        unknown = algorithm_filter - set(ALGORITHMS)
        if unknown:
            raise ValueError(f"unknown algorithms: {sorted(unknown)}")

    discovered_runs: dict[str, list[str]] = {}
    selected_run_counts: Counter[str] = Counter()
    manifest_rows: list[dict[str, Any]] = []
    json_records: list[dict[str, Any]] = []
    image_info_cache: dict[
        Path, tuple[int | None, int | None, str | None, str | None]
    ] = {}

    def cached_image_info(
        path: Path,
    ) -> tuple[int | None, int | None, str | None, str | None]:
        if path not in image_info_cache:
            image_info_cache[path] = _image_info(path)
        return image_info_cache[path]

    for model_id in model_ids:
        is_flux = model_id == "flux_schnell"
        for algorithm in algorithms:
            runs = _discover_runs(root, model_id, algorithm)
            pair_key = f"{model_id}/{algorithm}"
            discovered_runs[pair_key] = [str(path) for path in runs]
            run_cells: dict[Path, dict[int, dict[str, Any]]] = {}
            reward_cells: dict[Path, dict[int, list[dict[str, Any]]]] = {}
            for run in runs:
                cells = (
                    _load_flux_cells(run, algorithm)
                    if is_flux
                    else _load_sd35_cells(run, algorithm)
                )
                run_cells[run] = cells
                reward_cells[run] = _load_reward_rows(run / algorithm)
            seed_map = _load_seed_map(root, model_id, algorithm)

            for prompt_index, prompt in enumerate(prompts):
                c0 = str(prompt["prompt"])
                chosen_run: Path | None = None
                chosen_row: dict[str, Any] | None = None
                # Prefer newest complete image+record pair.
                for run in reversed(runs):
                    candidate = run_cells[run].get(prompt_index)
                    image = candidate.get("_source_image_path") if candidate else None
                    if candidate and isinstance(image, Path) and image.is_file():
                        info = cached_image_info(image)
                        if info[3] is not None:
                            continue
                        chosen_run, chosen_row = run, candidate
                        break
                # Preserve image-only evidence if a legacy log is absent.
                if chosen_row is None:
                    image_only_candidates: list[tuple[Path, Path, Path]] = []
                    for run in reversed(runs):
                        method_dir = run / algorithm
                        image = (
                            _find_flux_image_without_summary(
                                method_dir, prompt_index, algorithm
                            )
                            if is_flux
                            else _find_sd35_image(method_dir, prompt_index, algorithm)
                        )
                        if image is not None:
                            image_only_candidates.append((run, method_dir, image))
                            if cached_image_info(image)[3] is not None:
                                continue
                            chosen_run = run
                            chosen_row = {
                                "prompt_index": prompt_index,
                                "_source_run_path": str(run),
                                "_source_method_path": str(method_dir),
                                "_source_image_path": image,
                                "_source_record_path": None,
                                "_duplicate_log_rows": 0,
                            }
                            break
                    if chosen_row is None and image_only_candidates:
                        run, method_dir, image = image_only_candidates[0]
                        chosen_run = run
                        chosen_row = {
                            "prompt_index": prompt_index,
                            "_source_run_path": str(run),
                            "_source_method_path": str(method_dir),
                            "_source_image_path": image,
                            "_source_record_path": None,
                            "_duplicate_log_rows": 0,
                        }

                source_image = (
                    chosen_row.get("_source_image_path") if chosen_row else None
                )
                source_image = source_image if isinstance(source_image, Path) else None
                width = height = None
                image_sha256 = image_error = None
                if source_image is not None:
                    width, height, image_sha256, image_error = cached_image_info(
                        source_image
                    )
                log_prompt = (
                    str(chosen_row.get("prompt", ""))
                    if chosen_row and chosen_row.get("prompt") is not None
                    else None
                )
                prompt_match = log_prompt == c0 if log_prompt is not None else None
                rewards, reward_path, reward_prompt_match = _reward_for_c0(
                    reward_cells.get(chosen_run, {}).get(prompt_index, [])
                    if chosen_run
                    else [],
                    c0,
                )
                selected_seed, selected_seed_source = _selected_seed(chosen_row)
                total_nfe, nfe_source = _logged_nfe(chosen_row)

                if source_image is None:
                    status = "missing_image"
                elif image_error:
                    status = "corrupt_image"
                elif chosen_row is None or not chosen_row.get("_source_record_path"):
                    status = "log_missing"
                elif log_prompt is None:
                    status = "prompt_unverifiable"
                elif prompt_match is False:
                    status = "prompt_mismatch"
                else:
                    status = "complete"

                summary_image: Path | None = None
                if source_image is not None and image_error is None:
                    summary_image = _materialize_image(
                        source_image,
                        output_dir
                        / "images"
                        / model_id
                        / str(prompt["prompt_id"])
                        / f"{algorithm}.png",
                        materialize,
                    )
                if chosen_run is not None:
                    selected_run_counts[str(chosen_run)] += 1

                record = {
                    "image_id": f"{model_id}__{prompt['prompt_id']}__{algorithm}",
                    "group_id": f"{model_id}__{prompt['prompt_id']}",
                    "model_id": model_id,
                    "model_name": MODEL_DISPLAY_NAMES.get(model_id, model_id),
                    "algorithm_id": algorithm,
                    "prompt_index": prompt_index,
                    "prompt_id": str(prompt["prompt_id"]),
                    "source_id": prompt.get("source_id"),
                    "difficulty": prompt.get("difficulty"),
                    "original_prompt_c0": c0,
                    "logged_prompt": log_prompt,
                    "logged_prompt_matches_c0": prompt_match,
                    "planned_method_seed": seed_map.get(prompt_index),
                    "logged_root_seed": _as_int(chosen_row.get("seed")) if chosen_row else None,
                    "selected_candidate_seed": selected_seed,
                    "selected_candidate_seed_source": selected_seed_source,
                    "total_nfe": total_nfe,
                    "nfe_source": nfe_source,
                    "objective_score": _as_float(chosen_row.get("score")) if chosen_row else None,
                    "baseline_score": _as_float(chosen_row.get("baseline_score")) if chosen_row else None,
                    "delta_vs_baseline": _as_float(chosen_row.get("delta_vs_base")) if chosen_row else None,
                    "reward_scores": rewards,
                    "reward_prompt_matches_c0": reward_prompt_match,
                    "actions": chosen_row.get("actions", []) if chosen_row else [],
                    "source_run_path": str(chosen_run) if chosen_run else None,
                    "source_record_path": chosen_row.get("_source_record_path") if chosen_row else None,
                    "source_record_line": chosen_row.get("_source_line") if chosen_row else None,
                    "source_reward_path": reward_path,
                    "source_image_path": str(source_image) if source_image else None,
                    "summary_image_path": str(summary_image) if summary_image else None,
                    "width": width,
                    "height": height,
                    "image_sha256": image_sha256,
                    "image_error": image_error,
                    "duplicate_log_rows": int(chosen_row.get("_duplicate_log_rows", 0)) if chosen_row else 0,
                    "status": status,
                }
                json_records.append(record)
                manifest_rows.append(
                    {
                        **record,
                        "actions_json": json.dumps(record["actions"], separators=(",", ":")),
                        "reward_scores_json": json.dumps(
                            record["reward_scores"], sort_keys=True, separators=(",", ":")
                        ),
                        "imagereward": rewards.get("imagereward"),
                        "hpsv3": rewards.get("hpsv3"),
                        "pickscore": rewards.get("pickscore"),
                        "vqascore": rewards.get("vqascore"),
                    }
                )

    coverage_rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        for algorithm in algorithms:
            rows = [
                row
                for row in json_records
                if row["model_id"] == model_id and row["algorithm_id"] == algorithm
            ]
            complete = [row for row in rows if row["status"] == "complete"]
            coverage_rows.append(
                {
                    "model_id": model_id,
                    "model_name": MODEL_DISPLAY_NAMES.get(model_id, model_id),
                    "algorithm_id": algorithm,
                    "expected_prompts": len(prompts),
                    "complete_images": len(complete),
                    "images_found": sum(row["source_image_path"] is not None for row in rows),
                    "logs_found": sum(row["source_record_path"] is not None for row in rows),
                    "nfe_logged": sum(row["total_nfe"] is not None for row in rows),
                    "c0_reward_rows": sum(row["reward_prompt_matches_c0"] is True for row in rows),
                    "missing_prompt_ids": ",".join(
                        row["prompt_id"] for row in rows if row["status"] != "complete"
                    ),
                    "status_counts_json": json.dumps(
                        dict(sorted(Counter(row["status"] for row in rows).items())),
                        sort_keys=True,
                    ),
                    "run_count": len(discovered_runs[f"{model_id}/{algorithm}"]),
                }
            )

    group_rows: list[dict[str, Any]] = []
    records_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in json_records:
        records_by_group[str(record["group_id"])].append(record)
    for group_id in sorted(records_by_group):
        rows = records_by_group[group_id]
        first = rows[0]
        by_algorithm = {str(row["algorithm_id"]): row for row in rows}
        group_row: dict[str, Any] = {
            "group_id": group_id,
            "model_id": first["model_id"],
            "model_name": first["model_name"],
            "prompt_id": first["prompt_id"],
            "source_id": first["source_id"],
            "original_prompt_c0": first["original_prompt_c0"],
            "complete": all(
                by_algorithm.get(algorithm, {}).get("status") == "complete"
                for algorithm in algorithms
            ),
        }
        for algorithm in algorithms:
            record = by_algorithm.get(algorithm, {})
            group_row[f"{algorithm}_path"] = (
                record.get("summary_image_path") or record.get("source_image_path")
            )
            group_row[f"{algorithm}_status"] = record.get("status", "missing")
        group_rows.append(group_row)

    prompt_sets = {}
    per_model_algorithm = {}
    for model_id in model_ids:
        per_algorithm = {}
        for algorithm in algorithms:
            complete_ids = sorted(
                row["prompt_id"]
                for row in json_records
                if row["model_id"] == model_id
                and row["algorithm_id"] == algorithm
                and row["status"] == "complete"
            )
            per_algorithm[algorithm] = complete_ids
        per_model_algorithm[model_id] = per_algorithm
        prompt_sets[model_id] = sorted(
            {
                prompt_id
                for ids in per_algorithm.values()
                for prompt_id in ids
            }
        )
    distinct_generated_sets = {
        tuple(values) for values in prompt_sets.values()
    }
    expected_prompt_ids = [str(prompt["prompt_id"]) for prompt in prompts]
    prompt_audit = {
        "intended_policy": (
            "Every configured model and algorithm receives the same ordered "
            "prompts.csv selection. GPU shards are execution partitions only."
        ),
        "expected_prompt_ids": expected_prompt_ids,
        "reserve_prompts_used": False,
        "seed_policy": (
            "Prompt text is shared across models. Deterministic seed maps are "
            "model- and algorithm-specific; multi-root methods may select a "
            "different winning seed."
        ),
        "generated_prompt_ids_by_model": prompt_sets,
        "complete_prompt_ids_by_model_algorithm": per_model_algorithm,
        "observed_model_prompt_sets_identical": len(distinct_generated_sets) <= 1,
        "all_models_cover_expected_prompt_set": all(
            values == expected_prompt_ids for values in prompt_sets.values()
        ),
        "all_model_algorithm_cells_cover_expected_prompt_set": all(
            ids == expected_prompt_ids
            for per_algorithm in per_model_algorithm.values()
            for ids in per_algorithm.values()
        ),
    }
    unused_runs = sorted(
        run
        for runs in discovered_runs.values()
        for run in runs
        if selected_run_counts[run] == 0
    )
    summary = {
        "schema_version": "1.0",
        "root": str(root),
        "output_dir": str(output_dir),
        "config_path": str(config_path.resolve()),
        "prompts_path": str(prompts_path.resolve()),
        "prompt_count": len(prompts),
        "models": model_ids,
        "algorithms": algorithms,
        "materialization": materialize,
        "selection_rule": (
            "Per model/algorithm/prompt: newest run with both decodable final "
            "image and matching log row; older-run fallback only for missing cells."
        ),
        "reward_rule": (
            "Attach existing reward scores only when reward-row prompt exactly "
            "equals original_prompt_c0; no rescoring is performed."
        ),
        "nfe_rule": "Use explicit log/diagnostic NFE fields only; never infer NFE.",
        "status_counts": dict(sorted(Counter(row["status"] for row in json_records).items())),
        "record_count": len(json_records),
        "discovered_runs": discovered_runs,
        "unused_runs": unused_runs,
        "prompt_audit": prompt_audit,
        "records": json_records,
    }

    manifest_fields = [
        "image_id", "group_id", "model_id", "model_name", "algorithm_id",
        "prompt_index", "prompt_id", "source_id", "difficulty",
        "original_prompt_c0", "logged_prompt", "logged_prompt_matches_c0",
        "planned_method_seed", "logged_root_seed", "selected_candidate_seed",
        "selected_candidate_seed_source", "total_nfe", "nfe_source",
        "objective_score", "baseline_score", "delta_vs_baseline",
        "imagereward", "hpsv3", "pickscore", "vqascore",
        "reward_prompt_matches_c0", "reward_scores_json", "actions_json",
        "source_run_path", "source_record_path", "source_record_line",
        "source_reward_path", "source_image_path", "summary_image_path",
        "width", "height", "image_sha256", "image_error",
        "duplicate_log_rows", "status",
    ]
    coverage_fields = [
        "model_id", "model_name", "algorithm_id", "expected_prompts",
        "complete_images", "images_found", "logs_found", "nfe_logged",
        "c0_reward_rows", "missing_prompt_ids", "status_counts_json", "run_count",
    ]
    group_fields = [
        "group_id", "model_id", "model_name", "prompt_id", "source_id",
        "original_prompt_c0", "complete",
    ]
    for algorithm in algorithms:
        group_fields.extend([f"{algorithm}_path", f"{algorithm}_status"])
    _atomic_csv(output_dir / "legacy_manifest.csv", manifest_rows, manifest_fields)
    _atomic_csv(output_dir / "legacy_coverage.csv", coverage_rows, coverage_fields)
    _atomic_csv(output_dir / "legacy_groups.csv", group_rows, group_fields)
    _atomic_json(output_dir / "legacy_manifest.json", summary)
    _atomic_json(output_dir / "prompt_subset_audit.json", prompt_audit)
    return {
        "manifest_csv": str(output_dir / "legacy_manifest.csv"),
        "manifest_json": str(output_dir / "legacy_manifest.json"),
        "coverage_csv": str(output_dir / "legacy_coverage.csv"),
        "groups_csv": str(output_dir / "legacy_groups.csv"),
        "prompt_subset_audit": str(output_dir / "prompt_subset_audit.json"),
        "status_counts": summary["status_counts"],
        "record_count": len(json_records),
        "all_models_cover_expected_prompt_set": prompt_audit[
            "all_models_cover_expected_prompt_set"
        ],
        "all_model_algorithm_cells_cover_expected_prompt_set": prompt_audit[
            "all_model_algorithm_cells_cover_expected_prompt_set"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Human-eval root containing prompts.csv and legacy_runs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <root>/legacy_summary.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--prompts-file", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=None)
    parser.add_argument(
        "--materialize",
        choices=("none", "symlink", "copy"),
        default="symlink",
        help="How to build the compact image tree; symlink avoids PNG duplication.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    config = args.config
    if config is None:
        config = root / "run_config.yaml"
        if not config.is_file():
            config = DEFAULT_CONFIG
    prompts = args.prompts_file or (root / "prompts.csv")
    output_dir = args.output_dir or (root / "legacy_summary")
    result = summarize(
        root=root,
        output_dir=output_dir,
        config_path=config,
        prompts_path=prompts,
        model_filter=set(args.models) if args.models else None,
        algorithm_filter=set(args.algorithms) if args.algorithms else None,
        materialize=args.materialize,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
