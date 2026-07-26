#!/usr/bin/env python3
"""Thin orchestration/export layer for the existing generation suites.

This is not a new sampler.  It converts the prepared prompt CSV to the
one-prompt-per-line format consumed by the existing suites, invokes those
suites one model/algorithm at a time with deterministic seed maps, and copies
only selected final images into the human-evaluation layout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


REPO = Path(__file__).resolve().parents[1]
ALGORITHMS = ("baseline", "bon", "das", "bon_mcts")
METHOD_MODE = {"baseline": "base", "bon": "bon", "das": "bon", "bon_mcts": "mcts"}
DEFAULT_CANDIDATE_COUNTS = {"baseline": 1, "bon": 16, "das": 16, "bon_mcts": 8}
DEFAULT_REWARD_COMPONENTS = ("imagereward", "hpsv3", "pickscore")
IMAGE_SUFFIXES_SD35 = {
    "baseline": ("base",),
    "bon": ("bon",),
    "das": ("bon",),
    "bon_mcts": ("mcts",),
}


def stable_seed(study_id: str, model_id: str, prompt_id: str, algorithm_id: str, base_seed: int) -> int:
    payload = f"{study_id}\0{model_id}\0{prompt_id}\0{algorithm_id}\0{base_seed}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return 1 + value % 2_147_483_646


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("human_eval_runner.py requires PyYAML to read the YAML config") from exc
    with path.open(encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a YAML mapping: {path}")
    return value


def _read_prompts(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {"prompt_id", "source_id", "prompt", "difficulty"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"prompts CSV missing fields: {sorted(missing)}")
    if len(rows) != 40:
        raise ValueError(f"prompts CSV must contain exactly 40 prompts, found {len(rows)}")
    ids = [r["prompt_id"] for r in rows]
    if ids != [f"p{i:03d}" for i in range(40)]:
        raise ValueError("prompts CSV prompt_id values must be p000..p039 in order")
    if len({r["prompt"] for r in rows}) != len(rows):
        raise ValueError("prompts CSV contains duplicate exact prompt text")
    return rows


def _atomic_copy_image(source: Path, destination: Path) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    try:
        with Image.open(temporary) as image:
            image.load()
            dimensions = tuple(image.size)
            if image.mode not in {"RGB", "RGBA", "L"}:
                image.convert("RGB").save(temporary)
                dimensions = tuple(Image.open(temporary).size)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, destination)
    return dimensions


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _copy_if_different(source: Path, destination: Path) -> None:
    """Copy a study input unless it is already at the export destination."""
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _infer_conda_base(environment: dict[str, str]) -> Path | None:
    """Infer the active Conda installation root without assuming /opt/conda."""
    explicit = str(environment.get("REWARD_ENV_CONDA_BASE", "")).strip()
    if explicit:
        return Path(explicit).expanduser()

    conda_exe = str(environment.get("CONDA_EXE", "")).strip()
    if conda_exe:
        candidate = Path(conda_exe).expanduser().resolve().parent.parent
        if (candidate / "bin" / "conda").is_file():
            return candidate

    conda_prefix = str(environment.get("CONDA_PREFIX", "")).strip()
    if conda_prefix:
        prefix = Path(conda_prefix).expanduser().resolve()
        candidate = prefix.parent.parent if prefix.parent.name == "envs" else prefix
        if (candidate / "bin" / "conda").is_file():
            return candidate

    executable = Path(sys.executable).resolve()
    for parent in executable.parents:
        if parent.name == "envs" and (parent.parent / "bin" / "conda").is_file():
            return parent.parent
    candidate = executable.parent.parent
    if (candidate / "bin" / "conda").is_file():
        return candidate
    return None


def _seed_plan(config: dict[str, Any], prompts: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    study_id = str(config["study_id"])
    base_seed = int(config["generation_base_seed"])
    plan: dict[str, dict[str, dict[str, int]]] = {}
    for model in config["models"]:
        model_id = str(model["model_id"])
        plan[model_id] = {}
        for prompt in prompts:
            pid = str(prompt["prompt_id"])
            initial = stable_seed(study_id, model_id, pid, "initial", base_seed)
            plan[model_id][pid] = {"initial_seed": initial}
            for algorithm in config["algorithms"]:
                plan[model_id][pid][f"{algorithm}_seed"] = stable_seed(
                    study_id, model_id, pid, str(algorithm), base_seed
                )
    return plan


def _write_seed_map(path: Path, model_id: str, algorithm: str, prompts: list[dict[str, Any]], plan: dict[str, dict[str, dict[str, int]]]) -> None:
    # Existing runners address prompts by zero-based prompt index.  The map is
    # keyed by that index while the compact metadata retains the stable p-ID.
    seeds = {}
    for index, prompt in enumerate(prompts):
        pid = str(prompt["prompt_id"])
        key = "initial_seed" if algorithm in {"baseline", "bon", "das"} else f"{algorithm}_seed"
        seeds[str(index)] = int(plan[model_id][pid][key])
    _atomic_json(path, {"model_id": model_id, "algorithm_id": algorithm, "seeds": seeds})


def _load_jsonl_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("rank_*.jsonl")):
        if "rewrite_examples" in path.name:
            continue
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _source_row(run_root: Path, prompt_index: int, algorithm: str) -> dict[str, Any] | None:
    expected = METHOD_MODE[algorithm]
    rows = [r for r in _load_jsonl_rows(run_root) if int(r.get("prompt_index", -1)) == prompt_index]
    exact = [r for r in rows if str(r.get("mode", "")) == expected]
    if exact:
        return exact[-1]
    if rows:
        return rows[-1]
    # Flux's compact summary is the fallback when rank JSONL is not emitted.
    for summary_path in sorted(run_root.rglob("summary.json")):
        try:
            entries = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for entry in entries if isinstance(entries, list) else []:
            if int(str(entry.get("slug", "p-1")).lstrip("p")) != prompt_index:
                continue
            sample = (entry.get("samples") or [{}])[0]
            if algorithm == "baseline":
                return {
                    "prompt_index": prompt_index,
                    "prompt": entry.get("prompt"),
                    "seed": sample.get("seed"),
                    "mode": "base",
                    "score": sample.get("baseline_score"),
                    "nfe": None,
                }
            return {
                "prompt_index": prompt_index,
                "prompt": entry.get("prompt"),
                "seed": sample.get("seed"),
                "mode": expected,
                "score": sample.get("search_score"),
                "baseline_score": sample.get("baseline_score"),
                "delta_vs_base": sample.get("delta_score"),
                "actions": sample.get("actions", []),
                "search_diagnostics": sample.get("diagnostics"),
                "nfe": None,
            }
    return None


def _find_source_image(run_root: Path, prompt_index: int, algorithm: str, model_id: str) -> Path | None:
    if model_id == "flux_schnell":
        slug = f"p{prompt_index:04d}"
        names = [f"{slug}_s0_{suffix}.png" for suffix in ("baseline", "bon", "mcts")]
        if algorithm == "baseline":
            names = [f"{slug}_s0_baseline.png"]
        elif algorithm == "bon_mcts":
            names = [f"{slug}_s0_mcts.png"]
        else:
            names = [f"{slug}_s0_bon.png"]
    else:
        slug = f"p{prompt_index:05d}"
        names = [f"{slug}_{suffix}.png" for suffix in IMAGE_SUFFIXES_SD35[algorithm]]
    for name in names:
        matches = sorted(run_root.rglob(name))
        if matches:
            return matches[-1]
    return None


def _latest_run(root: Path) -> Path | None:
    runs = sorted(p for p in root.glob("run_*") if p.is_dir())
    return runs[-1] if runs else None


def _selected_candidate_seed(row: dict[str, Any] | None) -> int | None:
    if not row:
        return None
    diagnostics = row.get("search_diagnostics") or row.get("diagnostics") or {}
    if isinstance(diagnostics, dict):
        for key in ("winner_seed", "chosen_seed", "selected_seed", "seed"):
            if diagnostics.get(key) is not None:
                try:
                    return int(diagnostics[key])
                except (TypeError, ValueError):
                    pass
        for value in diagnostics.values():
            if isinstance(value, dict):
                candidate = _selected_candidate_seed({"search_diagnostics": value})
                if candidate is not None:
                    return candidate
    try:
        return int(row["seed"]) if row.get("seed") is not None else None
    except (TypeError, ValueError):
        return None


def _multi_reward_scores(
    run_root: Path | None,
    prompt_index: int,
    original_prompt: str,
) -> dict[str, float]:
    if run_root is None:
        return {}
    for path in sorted(run_root.rglob("best_images_multi_reward.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        matches = [
            row for row in rows
            if (
                isinstance(row, dict)
                and int(row.get("prompt_index", -1)) == prompt_index
                and row.get("prompt") == original_prompt
            )
        ]
        if not matches:
            continue
        scores = matches[-1].get("scores", {})
        if isinstance(scores, dict):
            return {
                str(name): float(value)
                for name, value in scores.items()
                if isinstance(value, (int, float))
            }
    return {}


def _write_manifest(
    output_dir: Path,
    study_id: str,
    prompts: list[dict[str, Any]],
    models: list[dict[str, Any]],
    algorithms: list[str],
    plan: dict[str, dict[str, dict[str, int]]],
    source_roots: dict[tuple[str, str], Path | None],
    reward_backend: str,
    reward_components: list[str],
    reward_normalization: dict[str, Any],
    overwrite: bool = False,
) -> None:
    manifest_fields = [
        "image_id", "group_id", "model_id", "prompt_id", "source_id", "prompt", "difficulty",
        "algorithm_id", "image_path", "metadata_path", "initial_seed", "method_seed",
        "selected_candidate_seed", "total_nfe", "candidate_count", "reward_model", "selected_reward",
        "imagereward", "hpsv3", "pickscore",
        "width", "height", "status",
    ]
    manifest_rows: list[dict[str, Any]] = []
    prompt_by_id = {str(p["prompt_id"]): p for p in prompts}
    for model in models:
        model_id = str(model["model_id"])
        for prompt in prompts:
            pid = str(prompt["prompt_id"])
            for algorithm in algorithms:
                source_root = source_roots.get((model_id, algorithm))
                row = _source_row(source_root, int(pid[1:]), algorithm) if source_root else None
                source_image = _find_source_image(source_root, int(pid[1:]), algorithm, model_id) if source_root else None
                component_scores = _multi_reward_scores(
                    source_root, int(pid[1:]), str(prompt["prompt"])
                )
                image_rel = Path("final") / model_id / pid / f"{algorithm}.png"
                metadata_rel = Path("metadata") / model_id / pid / f"{algorithm}.json"
                image_path = output_dir / image_rel
                metadata_path = output_dir / metadata_rel
                status = "pending"
                width = height = None
                existing_meta: dict[str, Any] | None = None
                if image_path.is_file() and metadata_path.is_file():
                    try:
                        existing_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        existing_meta = None
                if existing_meta and (
                    existing_meta.get("model_id") == model_id
                    and existing_meta.get("prompt_id") == pid
                    and existing_meta.get("algorithm_id") == algorithm
                    and existing_meta.get("original_prompt") == prompt["prompt"]
                    and existing_meta.get("search_reward_backend") == reward_backend
                    and all(
                        isinstance((existing_meta.get("reward_scores") or {}).get(name), (int, float))
                        for name in reward_components
                    )
                ):
                    with Image.open(image_path) as image:
                        image.load()
                        width, height = image.size
                    status = "complete"
                elif source_image and source_image.is_file():
                    if (image_path.exists() or metadata_path.exists()) and not overwrite:
                        raise RuntimeError(
                            f"existing incomplete/mismatched output would be overwritten: {image_path}; "
                            "set overwrite: true in the study config to allow replacement"
                        )
                    width, height = _atomic_copy_image(source_image, image_path)
                    status = "complete"
                    # A stale metadata file must never supply scores or config
                    # for a newly generated image.
                    existing_meta = None
                meta = {
                    "study_id": study_id,
                    "group_id": f"{model_id}__{pid}",
                    "model_id": model_id,
                    "prompt_id": pid,
                    "source_id": prompt["source_id"],
                    "algorithm_id": algorithm,
                    "original_prompt": prompt["prompt"],
                    "conditioning_prompt": (row or {}).get("conditioning_prompt", prompt["prompt"]),
                    "prompt_variants": (row or {}).get("prompt_variants", []),
                    "initial_seed": plan[model_id][pid]["initial_seed"],
                    "method_seed": plan[model_id][pid][f"{algorithm}_seed"],
                    "selected_candidate_seed": (existing_meta or {}).get("selected_candidate_seed", _selected_candidate_seed(row)),
                    "total_nfe": (existing_meta or {}).get("total_nfe", (row or {}).get("nfe")),
                    "candidate_count": (existing_meta or {}).get("candidate_count", ((row or {}).get("search_diagnostics") or {}).get("candidate_count") if isinstance((row or {}).get("search_diagnostics"), dict) else DEFAULT_CANDIDATE_COUNTS[algorithm]),
                    "reward_model": (existing_meta or {}).get("reward_model", (row or {}).get("reward_model") or reward_backend),
                    "selected_reward": (existing_meta or {}).get("selected_reward", (row or {}).get("score")),
                    "search_reward_backend": reward_backend,
                    "reward_scores": (existing_meta or {}).get("reward_scores", component_scores),
                    "reward_components": reward_components,
                    "reward_normalization": reward_normalization,
                    "reward_prompt_text": prompt["prompt"],
                    "width": width,
                    "height": height,
                    "image_path": image_rel.as_posix(),
                    "source_run_path": str(source_root) if source_root else None,
                    "source_image_path": str(source_image) if source_image else None,
                    "model_config": model.get("model_config_id"),
                    "algorithm_config": algorithm,
                    "actions": (existing_meta or {}).get("actions", (row or {}).get("actions", [])),
                    "search_diagnostics": (existing_meta or {}).get("search_diagnostics", (row or {}).get("search_diagnostics")),
                    "reward_prompt": "original_prompt",
                }
                if status == "complete" and not existing_meta:
                    _atomic_json(metadata_path, meta)
                manifest_rows.append({
                    "image_id": f"{model_id}__{pid}__{algorithm}",
                    "group_id": f"{model_id}__{pid}",
                    "model_id": model_id,
                    "prompt_id": pid,
                    "source_id": prompt["source_id"],
                    "prompt": prompt["prompt"],
                    "difficulty": prompt["difficulty"],
                    "algorithm_id": algorithm,
                    "image_path": image_rel.as_posix(),
                    "metadata_path": metadata_rel.as_posix(),
                    "initial_seed": plan[model_id][pid]["initial_seed"],
                    "method_seed": plan[model_id][pid][f"{algorithm}_seed"],
                    "selected_candidate_seed": meta["selected_candidate_seed"],
                    "total_nfe": meta["total_nfe"],
                    "candidate_count": meta["candidate_count"],
                    "reward_model": meta["reward_model"],
                    "selected_reward": meta["selected_reward"],
                    "imagereward": meta["reward_scores"].get("imagereward"),
                    "hpsv3": meta["reward_scores"].get("hpsv3"),
                    "pickscore": meta["reward_scores"].get("pickscore"),
                    "width": width,
                    "height": height,
                    "status": status,
                })
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    groups: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        groups.setdefault(row["group_id"], {
            "group_id": row["group_id"], "model_id": row["model_id"], "prompt_id": row["prompt_id"],
            "source_id": row["source_id"], "prompt": row["prompt"],
        })[f"{row['algorithm_id']}_path"] = row["image_path"]
    group_fields = ["group_id", "model_id", "prompt_id", "source_id", "prompt"] + [f"{a}_path" for a in algorithms]
    with (output_dir / "groups.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=group_fields, lineterminator="\n")
        writer.writeheader()
        for group_id in sorted(groups):
            writer.writerow(groups[group_id])


def validate_layout(output_dir: str | Path, expected_prompt_count: int = 40) -> dict[str, Any]:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.csv"
    groups_path = output_dir / "groups.csv"
    result: dict[str, Any] = {
        "passed": False, "expected_images": 5 * expected_prompt_count * 4, "found_images": 0,
        "expected_groups": 5 * expected_prompt_count, "complete_groups": 0, "missing": [],
        "corrupt": [], "dimension_mismatches": [], "metadata_missing": [],
        "duplicate_image_ids": [], "duplicate_group_algorithm": [],
        "prompt_mismatches": [], "reward_score_missing": [],
        "reward_prompt_mismatches": [], "models": [], "algorithms": [],
    }
    if not manifest_path.is_file() or not groups_path.is_file():
        result["missing"].append("manifest.csv or groups.csv")
        _atomic_json(output_dir / "validation.json", result)
        return result
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with groups_path.open(newline="", encoding="utf-8") as f:
        groups = list(csv.DictReader(f))
    result["found_images"] = sum(1 for r in rows if r.get("status") == "complete")
    result["models"] = sorted({r.get("model_id") for r in rows})
    result["algorithms"] = sorted({r.get("algorithm_id") for r in rows})
    result["expected_images"] = 5 * expected_prompt_count * 4
    result["expected_groups"] = 5 * expected_prompt_count
    prompt_file = output_dir / "prompts.csv"
    if prompt_file.is_file():
        with prompt_file.open(newline="", encoding="utf-8-sig") as f:
            prompt_rows = {r.get("prompt_id"): r.get("prompt") for r in csv.DictReader(f)}
        result["prompt_mismatches"] = sorted(
            r.get("prompt_id") for r in rows if prompt_rows.get(r.get("prompt_id")) != r.get("prompt")
        )
    image_ids = [r.get("image_id") for r in rows]
    result["duplicate_image_ids"] = sorted(k for k, v in Counter(image_ids).items() if v > 1)
    group_alg = [(r.get("group_id"), r.get("algorithm_id")) for r in rows]
    result["duplicate_group_algorithm"] = sorted(k for k, v in Counter(group_alg).items() if v > 1)
    dims_by_group: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in rows:
        image = output_dir / str(row.get("image_path", ""))
        metadata = output_dir / str(row.get("metadata_path", ""))
        if not image.is_file():
            result["missing"].append(row.get("image_path"))
            continue
        if not metadata.is_file():
            result["metadata_missing"].append(row.get("metadata_path"))
        else:
            try:
                payload = json.loads(metadata.read_text(encoding="utf-8"))
                component_names = payload.get("reward_components") or []
                scores = payload.get("reward_scores") or {}
                missing_scores = [
                    name for name in component_names
                    if not isinstance(scores.get(name), (int, float))
                ]
                if missing_scores:
                    result["reward_score_missing"].append(
                        f"{row.get('metadata_path')}: {','.join(missing_scores)}"
                    )
                if (
                    payload.get("reward_prompt") != "original_prompt"
                    or payload.get("reward_prompt_text") != payload.get("original_prompt")
                ):
                    result["reward_prompt_mismatches"].append(row.get("metadata_path"))
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                result["reward_score_missing"].append(
                    f"{row.get('metadata_path')}: {type(exc).__name__}: {exc}"
                )
        try:
            with Image.open(image) as im:
                im.load()
                dims_by_group[str(row.get("group_id"))].add(tuple(im.size))
        except Exception as exc:
            result["corrupt"].append(f"{row.get('image_path')}: {type(exc).__name__}: {exc}")
    result["dimension_mismatches"] = sorted(k for k, v in dims_by_group.items() if len(v) > 1)
    complete = 0
    for group in groups:
        paths = [group.get(f"{a}_path") for a in ALGORITHMS]
        if all(p and (output_dir / p).is_file() for p in paths):
            complete += 1
    result["complete_groups"] = complete
    model_counts = Counter(r.get("model_id") for r in rows if r.get("status") == "complete")
    algorithm_counts = Counter(r.get("algorithm_id") for r in rows if r.get("status") == "complete")
    result["images_per_model"] = dict(model_counts)
    result["images_per_algorithm"] = dict(algorithm_counts)
    result["passed"] = (
        len(rows) == result["expected_images"]
        and len(groups) == result["expected_groups"]
        and result["found_images"] == result["expected_images"]
        and result["complete_groups"] == result["expected_groups"]
        and not result["missing"] and not result["corrupt"]
        and not result["metadata_missing"] and not result["dimension_mismatches"]
        and not result["reward_score_missing"] and not result["reward_prompt_mismatches"]
        and not result["duplicate_image_ids"] and not result["duplicate_group_algorithm"]
        and len(result["models"]) == 5 and result["algorithms"] == sorted(ALGORITHMS)
        and len({r.get("prompt_id") for r in rows}) == expected_prompt_count
        and not result["prompt_mismatches"]
        and all(v == expected_prompt_count * 4 for v in model_counts.values())
        and all(v == expected_prompt_count * 5 for v in algorithm_counts.values())
    )
    _atomic_json(output_dir / "validation.json", result)
    return result


def _run_one_model_algorithm(config: dict[str, Any], model: dict[str, Any], algorithm: str, prompts_txt: Path, seed_map: Path, output_dir: Path) -> Path:
    runner = REPO / str(model["runner"])
    if not runner.is_file():
        raise FileNotFoundError(runner)
    model_id = str(model["model_id"])
    legacy_root = output_dir / "legacy_runs" / model_id / algorithm
    env = os.environ.copy()
    configured_conda_base = str(config.get("reward_env_conda_base", "")).strip()
    if configured_conda_base:
        env["REWARD_ENV_CONDA_BASE"] = configured_conda_base
    else:
        inferred_conda_base = _infer_conda_base(env)
        if inferred_conda_base is not None:
            env["REWARD_ENV_CONDA_BASE"] = str(inferred_conda_base)
    env["REWARD_ENV_NAME"] = str(
        config.get("reward_env_name", env.get("REWARD_ENV_NAME", "reward"))
    )
    env.update({
        "PROMPT_FILE": str(prompts_txt),
        "METHODS": algorithm,
        "OUT_ROOT": str(legacy_root),
        "SEED_MAP_FILE": str(seed_map),
        "SAVE_IMAGES": "1",
        "SAVE_BEST_IMAGES": "1",
        "SAVE_VARIANTS": "1",
        "SAVE_FIRST_K": "-1",
        "EVAL_BEST_IMAGES": "1",
        "REWARD_BACKEND": str(config.get("reward_backend", "imagereward")),
        "EVAL_BACKENDS": " ".join(
            str(x) for x in config.get("reward_components", DEFAULT_REWARD_COMPONENTS)
        ),
        "EVAL_ALLOW_MISSING_BACKENDS": (
            "0" if bool(config.get("require_all_reward_components", True)) else "1"
        ),
        "USE_REWARD_SERVER": "1" if bool(config.get("use_reward_server", False)) else "0",
        "REWARD_SERVER_BACKENDS": " ".join(
            str(x) for x in config.get(
                "reward_server_backends",
                config.get("reward_components", DEFAULT_REWARD_COMPONENTS),
            )
        ),
        "REWARD_SERVER_REQUIRE_ALL": (
            "1" if bool(config.get("require_all_reward_components", True)) else "0"
        ),
        "WIDTH": str((config.get("generation") or {}).get("width", 1024)),
        "HEIGHT": str((config.get("generation") or {}).get("height", 1024)),
        "START_INDEX": "0",
        "END_INDEX": "-1",
    })
    normalization_env = {
        "imagereward": ("COMPOSITE_IR_LO", "COMPOSITE_IR_HI"),
        "hpsv3": ("COMPOSITE_HPSV3_LO", "COMPOSITE_HPSV3_HI"),
        "pickscore": ("COMPOSITE_PICKSCORE_LO", "COMPOSITE_PICKSCORE_HI"),
        "hpsv2": ("COMPOSITE_HPSV2_LO", "COMPOSITE_HPSV2_HI"),
    }
    for name, bounds in (config.get("reward_normalization") or {}).items():
        if name not in normalization_env or not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"invalid reward normalization for {name!r}: {bounds!r}")
        lo_env, hi_env = normalization_env[name]
        env[lo_env] = str(float(bounds[0]))
        env[hi_env] = str(float(bounds[1]))
    if model_id == "flux_schnell":
        env["MODEL_ID"] = str(model["model_config_id"])
    else:
        env["SD35_BACKEND"] = str(model["backend"])
    subprocess.run(["bash", str(runner)], cwd=REPO, env=env, check=True)
    run = _latest_run(legacy_root)
    if not run:
        raise RuntimeError(f"existing runner completed but no run_* directory was found under {legacy_root}")
    return run


def _combination_complete(
    output_dir: Path,
    model_id: str,
    algorithm: str,
    prompts: list[dict[str, Any]],
    reward_backend: str,
    reward_components: list[str],
) -> bool:
    for prompt in prompts:
        pid = str(prompt["prompt_id"])
        image = output_dir / "final" / model_id / pid / f"{algorithm}.png"
        metadata = output_dir / "metadata" / model_id / pid / f"{algorithm}.json"
        if not image.is_file() or not metadata.is_file():
            return False
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            with Image.open(image) as opened:
                opened.verify()
            if not (
                payload.get("model_id") == model_id
                and payload.get("prompt_id") == pid
                and payload.get("algorithm_id") == algorithm
                and payload.get("original_prompt") == prompt["prompt"]
                and payload.get("search_reward_backend") == reward_backend
                and all(
                    isinstance((payload.get("reward_scores") or {}).get(name), (int, float))
                    for name in reward_components
                )
            ):
                return False
        except Exception:
            return False
    return True


def prepare_layout(config_path: Path, prompts_file: Path, output_dir: Path, execute: bool = False, prompt_limit: int | None = None, resume_override: bool | None = None) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if resume_override is not None:
        config["resume"] = resume_override
    prompts = _read_prompts(prompts_file)
    models = list(config.get("models", []))
    algorithms = list(config.get("algorithms", []))
    reward_backend = str(config.get("reward_backend", "imagereward"))
    reward_components = [str(x) for x in config.get("reward_components", DEFAULT_REWARD_COMPONENTS)]
    reward_normalization = dict(config.get("reward_normalization") or {})
    if reward_backend == "composite_3":
        expected_components = set(DEFAULT_REWARD_COMPONENTS)
        if set(reward_components) != expected_components:
            raise ValueError(
                "reward_backend=composite_3 requires exactly "
                f"{sorted(expected_components)}, found {reward_components}"
            )
        if not bool(config.get("require_all_reward_components", True)):
            raise ValueError("reward_backend=composite_3 must require all reward components")
    if len(models) != 5:
        raise ValueError(f"configuration must specify exactly five models, found {len(models)}")
    if tuple(algorithms) != ALGORITHMS:
        raise ValueError(f"configuration algorithms must be {ALGORITHMS}, found {algorithms}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_if_different(prompts_file, output_dir / "prompts.csv")
    for companion in ("prompts_reserve.csv", "prompt_processing_report.json"):
        companion_path = prompts_file.with_name(companion)
        if companion_path.is_file():
            _copy_if_different(companion_path, output_dir / companion)
    try:
        import yaml

        effective_config = dict(config)
        effective_config["prompts_file"] = "prompts.csv"
        effective_config["output_dir"] = "."
        effective_config["source_config"] = str(config_path)
        (output_dir / "run_config.yaml").write_text(
            yaml.safe_dump(effective_config, sort_keys=False), encoding="utf-8"
        )
    except ImportError:
        shutil.copyfile(config_path, output_dir / "run_config.yaml")
    run_prompts = prompts[:prompt_limit] if prompt_limit is not None else prompts
    plan = _seed_plan(config, run_prompts)
    # Seed maps are runner inputs, not compact per-image metadata.
    seed_dir = output_dir / "legacy_runs" / "_seed_maps"
    source_roots: dict[tuple[str, str], Path | None] = {}
    with tempfile.TemporaryDirectory(prefix="human_eval_prompts_") as temp_dir:
        temp_dir = Path(temp_dir)
        prompts_txt = temp_dir / "prompts.txt"
        prompts_txt.write_text("\n".join(str(p["prompt"]) for p in run_prompts) + "\n", encoding="utf-8")
        for model in models:
            model_id = str(model["model_id"])
            for algorithm in algorithms:
                seed_map = seed_dir / model_id / f"{algorithm}.json"
                seed_map.parent.mkdir(parents=True, exist_ok=True)
                _write_seed_map(seed_map, model_id, algorithm, run_prompts, plan)
                source_roots[(model_id, algorithm)] = None
                if execute:
                    if bool(config.get("resume", True)) and _combination_complete(
                        output_dir, model_id, algorithm, run_prompts,
                        reward_backend, reward_components,
                    ):
                        print(f"[resume] complete: {model_id}/{algorithm}")
                    else:
                        source_roots[(model_id, algorithm)] = _run_one_model_algorithm(
                            config, model, algorithm, prompts_txt, seed_map, output_dir
                        )
    _write_manifest(
        output_dir, str(config["study_id"]), run_prompts, models, algorithms, plan, source_roots,
        reward_backend, reward_components, reward_normalization,
        overwrite=bool(config.get("overwrite", False)),
    )
    validation = validate_layout(output_dir, expected_prompt_count=len(run_prompts))
    return {"output_dir": str(output_dir), "validation": validation, "executed": execute}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prompts-file", required=True, type=Path, help="Prepared prompts.csv")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help="Invoke the existing generation suites")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Use the first two prompts and write output under <output-dir>/smoke_test")
    parser.add_argument("--expected-prompts", type=int, default=40, help="Expected prompt count for --validate-only")
    args = parser.parse_args()
    if args.validate_only:
        result = validate_layout(args.output_dir, expected_prompt_count=args.expected_prompts)
    else:
        output_dir = args.output_dir / "smoke_test" if args.smoke_test else args.output_dir
        result = prepare_layout(
            args.config, args.prompts_file, output_dir, execute=args.execute,
            prompt_limit=2 if args.smoke_test else None, resume_override=args.resume,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("validation", result).get("passed", False) else (0 if not args.execute else 1)


if __name__ == "__main__":
    raise SystemExit(main())
