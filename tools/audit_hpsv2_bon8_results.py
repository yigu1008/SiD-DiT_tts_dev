#!/usr/bin/env python3
"""Audit completeness and invariants of the HPSv2 fixed-rewrite BoN-8 sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_MODELS = ("sid", "senseflow_large", "sd35_base", "flux_schnell")
DEFAULT_ARMS = ("imagereward", "hpsv3", "multi_reward")
DEFAULT_EVALS = ("imagereward", "hpsv3", "pickscore", "hpsv2")
SEARCH_REWARD = {
    "imagereward": "imagereward",
    "hpsv3": "hpsv3",
    "multi_reward": "composite_hpsv3_ir",
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _read_prompts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"prompt_id", "prompt"}.issubset(rows[0]):
        raise ValueError(f"{path}: expected prompt_id,prompt columns")
    return [
        {"prompt_id": str(row["prompt_id"]).strip(), "prompt": str(row["prompt"]).strip()}
        for row in rows
    ]


def _prompt_index(row: dict[str, Any]) -> int | None:
    value = row.get("prompt_index")
    try:
        return int(value)
    except (TypeError, ValueError):
        slug = str(row.get("slug", ""))
        match = re.search(r"(\d+)", slug)
        return int(match.group(1)) if match else None


def _eval_key(row: dict[str, Any]) -> tuple[int, str, int]:
    return (
        _prompt_index(row) if _prompt_index(row) is not None else -1,
        str(row.get("slug", "")),
        int(row.get("sample_index", 0) or 0),
    )


def _generation_records(method_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((method_dir / "logs").glob("rank_*.jsonl")):
        if path.name.endswith("_rewrite_examples.jsonl"):
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if isinstance(value, dict) and _prompt_index(value) is not None:
                records.append(value)
    if records:
        return records
    for path in sorted(method_dir.glob("rank_*/summary.json")):
        value = _json(path)
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected a JSON list")
        records.extend(row for row in value if isinstance(row, dict))
    return records


def _logged_bon_n(row: dict[str, Any]) -> int | None:
    diagnostics = row.get("search_diagnostics") or row.get("diagnostics")
    if isinstance(diagnostics, dict):
        try:
            return int(diagnostics.get("bon_n"))
        except (TypeError, ValueError):
            pass
    samples = row.get("samples")
    if isinstance(samples, list) and len(samples) == 1 and isinstance(samples[0], dict):
        diagnostics = samples[0].get("diagnostics")
        if isinstance(diagnostics, dict):
            try:
                return int(diagnostics.get("bon_n"))
            except (TypeError, ValueError):
                pass
    return None


def _score(row: dict[str, Any], backend: str) -> float | None:
    scores = row.get("scores")
    value = scores.get(backend) if isinstance(scores, dict) else row.get("score")
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _check_image(path: Path, cache: dict[str, str]) -> str:
    key = str(path)
    if key in cache:
        return cache[key]
    if not path.is_file():
        cache[key] = "missing"
        return cache[key]
    try:
        with Image.open(path) as image:
            image.verify()
        cache[key] = "ok"
    except Exception as exc:
        cache[key] = f"unreadable:{type(exc).__name__}:{exc}"
    return cache[key]


def audit(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = args.root.expanduser().resolve()
    manifest_path = root / "study_manifest.json"
    global_errors: list[dict[str, Any]] = []
    global_warnings: list[dict[str, Any]] = []

    def global_error(kind: str, message: str, **details: Any) -> None:
        global_errors.append({"kind": kind, "message": message, **details})

    manifest: dict[str, Any] = {}
    if not manifest_path.is_file():
        global_error("missing_manifest", f"missing {manifest_path}")
    else:
        value = _json(manifest_path)
        if isinstance(value, dict):
            manifest = value
        else:
            global_error("manifest_schema", "study_manifest.json must be an object")

    run_id = args.run_id or str(manifest.get("run_id", ""))
    if not run_id:
        global_error("missing_run_id", "pass --run-id or provide it in study_manifest.json")
    models = args.models or [
        str(row.get("model_id"))
        for row in manifest.get("models", [])
        if isinstance(row, dict) and row.get("model_id")
    ] or list(DEFAULT_MODELS)
    arms = args.reward_arms or list(manifest.get("reward_arms", {}).keys()) or list(DEFAULT_ARMS)
    eval_backends = args.eval_backends

    prompts_path = root / "prompts.csv"
    prompts: list[dict[str, str]] = []
    if not prompts_path.is_file():
        global_error("missing_prompts", f"missing {prompts_path}")
    else:
        try:
            prompts = _read_prompts(prompts_path)
        except Exception as exc:
            global_error("prompt_parse", str(exc))
    prompt_texts = [row["prompt"] for row in prompts]
    expected = args.expected_prompts
    if expected is None:
        expected = int(manifest.get("prompt_count", len(prompts)))
    if len(prompts) != expected:
        global_error("prompt_count", f"found {len(prompts)} prompts; expected {expected}")
    if len(set(row["prompt_id"] for row in prompts)) != len(prompts):
        global_error("duplicate_prompt_id", "prompts.csv contains duplicate prompt IDs")
    if len(set(prompt_texts)) != len(prompt_texts):
        global_error("duplicate_prompt_text", "prompts.csv contains duplicate prompt text")
    if int(manifest.get("candidate_count", 8)) != 8:
        global_error("candidate_count", "study manifest does not specify BoN-8")
    if int(manifest.get("rewrite_count_per_prompt", 1)) != 1:
        global_error("rewrite_count", "study manifest does not specify one fixed rewrite")

    rewrite_path = root / "fixed_rewrite_cache.json"
    rewrites: dict[str, Any] = {}
    if not rewrite_path.is_file():
        global_error("missing_rewrite_cache", f"missing {rewrite_path}")
    else:
        value = _json(rewrite_path)
        if isinstance(value, dict):
            rewrites = value
        else:
            global_error("rewrite_schema", "fixed_rewrite_cache.json must be an object")
    if rewrites:
        if set(rewrites) != set(prompt_texts):
            global_error(
                "rewrite_prompt_set",
                "rewrite cache keys do not exactly match original prompts",
                missing=len(set(prompt_texts) - set(rewrites)),
                extra=len(set(rewrites) - set(prompt_texts)),
            )
        bad_rewrites = [
            prompt for prompt in prompt_texts
            if not isinstance(rewrites.get(prompt), list)
            or len(rewrites.get(prompt, [])) != 1
            or not str(rewrites[prompt][0]).strip()
            or str(rewrites[prompt][0]).strip() == prompt
        ]
        if bad_rewrites:
            global_error(
                "rewrite_invariant",
                f"{len(bad_rewrites)} prompts lack one distinct fixed rewrite",
            )

    for model in models:
        seed_path = root / "seed_maps" / f"{model}.json"
        if not seed_path.is_file():
            global_error("missing_seed_map", f"missing {seed_path}", model_id=model)
            continue
        seed_map = _json(seed_path)
        seeds = seed_map.get("seeds", {}) if isinstance(seed_map, dict) else {}
        if (
            not isinstance(seeds, dict)
            or set(seeds) != {str(index) for index in range(expected)}
            or len(set(seeds.values())) != expected
        ):
            global_error(
                "seed_map_coverage",
                f"invalid seed coverage for {model}", model_id=model,
            )

    image_cache: dict[str, str] = {}
    cells: list[dict[str, Any]] = []
    for model in models:
        for arm in arms:
            method_dir = root / model / arm / f"run_{run_id}" / "bon_fixed_rewrite"
            cell_errors: list[str] = []
            aggregate_path = method_dir / "aggregate_ddp.json"
            generated = 0
            search_reward = ""
            if not aggregate_path.is_file():
                cell_errors.append("missing aggregate_ddp.json")
            else:
                aggregate = _json(aggregate_path)
                if not isinstance(aggregate, dict):
                    cell_errors.append("aggregate_ddp.json is not an object")
                else:
                    generated = int(aggregate.get("num_samples", 0) or 0)
                    search_reward = str(aggregate.get("search_reward", ""))
                    if generated != expected:
                        cell_errors.append(f"generated={generated}, expected={expected}")
                    expected_reward = SEARCH_REWARD.get(arm)
                    if expected_reward and search_reward != expected_reward:
                        cell_errors.append(
                            f"search_reward={search_reward!r}, expected={expected_reward!r}"
                        )

            try:
                generation_records = _generation_records(method_dir)
            except Exception as exc:
                generation_records = []
                cell_errors.append(f"generation records unreadable: {exc}")
            indices = [_prompt_index(row) for row in generation_records]
            valid_indices = [index for index in indices if index is not None]
            if generation_records and (
                len(valid_indices) != expected or set(valid_indices) != set(range(expected))
            ):
                cell_errors.append(
                    f"generation record coverage={len(set(valid_indices))}/{expected}"
                )
            for row in generation_records:
                index = _prompt_index(row)
                if index is None or not 0 <= index < len(prompt_texts):
                    continue
                if str(row.get("prompt", "")) != prompt_texts[index]:
                    cell_errors.append(f"generation prompt mismatch at index {index}")
                    break
            logged_bon8 = sum(_logged_bon_n(row) == 8 for row in generation_records)
            if logged_bon8 != expected:
                cell_errors.append(
                    f"logged BoN-8 diagnostics={logged_bon8}/{expected}"
                )

            variant_files = sorted(method_dir.glob("rank_*/p*_variants.txt"))
            variant_indices: set[int] = set()
            bad_variants = 0
            for path in variant_files:
                match = re.fullmatch(r"p(\d+)_variants\.txt", path.name)
                if match is None:
                    bad_variants += 1
                    continue
                index = int(match.group(1))
                if index in variant_indices or not 0 <= index < len(prompt_texts):
                    bad_variants += 1
                    continue
                variant_indices.add(index)
                lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                expected_rewrite = (
                    str(rewrites[prompt_texts[index]][0]).strip()
                    if prompt_texts[index] in rewrites else None
                )
                effective = lines[0].split(":", 1)[1].strip() if len(lines) == 1 and ":" in lines[0] else None
                if effective is None or effective != expected_rewrite:
                    bad_variants += 1
            if len(variant_indices) != expected or bad_variants:
                cell_errors.append(
                    f"fixed prompt banks={len(variant_indices)}/{expected}, bad={bad_variants}"
                )

            eval_counts: dict[str, int] = {}
            eval_keys: set[tuple[int, str, int]] | None = None
            if not args.generation_only:
                for backend in eval_backends:
                    eval_path = method_dir / f"best_images_{backend}.json"
                    rows: list[dict[str, Any]] = []
                    if not eval_path.is_file():
                        cell_errors.append(f"missing best_images_{backend}.json")
                    else:
                        value = _json(eval_path)
                        raw_rows = value.get("rows", []) if isinstance(value, dict) else []
                        rows = raw_rows if isinstance(raw_rows, list) else []
                    eval_counts[backend] = len(rows)
                    keys = {_eval_key(row) for row in rows}
                    if len(rows) != expected or len(keys) != expected:
                        cell_errors.append(
                            f"eval {backend} rows/unique={len(rows)}/{len(keys)}, expected={expected}"
                        )
                    if eval_keys is None:
                        eval_keys = keys
                    elif keys != eval_keys:
                        cell_errors.append(f"eval {backend} selected-image keys differ")
                    for row in rows:
                        index = _prompt_index(row)
                        if index is None or not 0 <= index < len(prompt_texts):
                            cell_errors.append(f"eval {backend} has invalid prompt index")
                            break
                        if str(row.get("prompt", "")) != prompt_texts[index]:
                            cell_errors.append(
                                f"eval {backend} is not scored against c0 at index {index}"
                            )
                            break
                        if _score(row, backend) is None:
                            cell_errors.append(f"eval {backend} lacks numeric score at index {index}")
                            break
                        image_path = Path(str(row.get("image_path", ""))).expanduser()
                        image_status = _check_image(image_path, image_cache)
                        if image_status != "ok":
                            cell_errors.append(
                                f"eval {backend} image {image_status} at index {index}: {image_path}"
                            )
                            break

            cells.append({
                "model_id": model,
                "reward_arm": arm,
                "generated": generated,
                "expected": expected,
                "search_reward": search_reward,
                "logged_bon8": logged_bon8,
                "prompt_banks": len(variant_indices),
                **{f"eval_{backend}": eval_counts.get(backend, "") for backend in eval_backends},
                "error_count": len(cell_errors),
                "status": "OK" if not cell_errors else "INCOMPLETE",
                "errors": " | ".join(cell_errors),
                "method_dir": str(method_dir),
            })

    incomplete_cells = sum(row["status"] != "OK" for row in cells)
    complete = not global_errors and incomplete_cells == 0
    valid = complete or args.allow_incomplete
    report = {
        "valid": valid,
        "complete": complete,
        "allow_incomplete": args.allow_incomplete,
        "generation_only": args.generation_only,
        "root": str(root),
        "run_id": run_id,
        "models": models,
        "reward_arms": arms,
        "eval_backends": eval_backends,
        "expected_prompts": expected,
        "candidate_count": 8,
        "expected_cells": len(models) * len(arms),
        "complete_cells": len(cells) - incomplete_cells,
        "incomplete_cells": incomplete_cells,
        "global_error_count": len(global_errors),
        "global_errors": global_errors,
        "global_warnings": global_warnings,
        "cells": cells,
    }
    return report, cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--reward-arms", nargs="+", default=None)
    parser.add_argument("--eval-backends", nargs="+", default=list(DEFAULT_EVALS))
    parser.add_argument("--expected-prompts", type=int, default=None)
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    report, rows = audit(args)
    root = args.root.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir else root / "integrity_reports" / "bon8"
    )
    report_path = out_dir / "bon8_integrity_report.json"
    csv_path = out_dir / "bon8_cell_coverage.csv"
    fields = [
        "model_id", "reward_arm", "generated", "expected", "search_reward",
        "logged_bon8", "prompt_banks", *[f"eval_{name}" for name in args.eval_backends],
        "error_count", "status", "errors", "method_dir",
    ]
    _atomic_json(report_path, report)
    _atomic_csv(csv_path, rows, fields)
    status = "OK" if report["complete"] else "INCOMPLETE"
    print(
        f"[bon8] {status} cells={report['complete_cells']}/{report['expected_cells']} "
        f"prompts={report['expected_prompts']} generation_only={report['generation_only']}"
    )
    for row in rows:
        eval_text = ",".join(
            f"{backend}:{row[f'eval_{backend}']}" for backend in args.eval_backends
        )
        print(
            f"  {row['model_id']}/{row['reward_arm']} "
            f"generated={row['generated']}/{row['expected']} "
            f"bon8={row['logged_bon8']} prompt_banks={row['prompt_banks']} "
            f"evals={eval_text} {row['status']}"
        )
    for item in report["global_errors"]:
        print(f"  ERROR {item['kind']}: {item['message']}")
    print(f"[bon8] report={report_path}")
    print(f"[bon8] coverage={csv_path}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
