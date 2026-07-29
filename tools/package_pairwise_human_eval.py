#!/usr/bin/env python3
"""Build a blinded, reproducibly shuffled pairwise human-evaluation package.

The input is ``legacy_manifest.csv`` produced by
``tools/summarize_human_eval_legacy.py``. Comparisons are formed only within
the same model/prompt group. Public files contain opaque A/B image names and
the original prompt c0; method identities, seeds, rewards, and source paths
are retained only in the private answer key.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


DEFAULT_ALGORITHMS = ("baseline", "bon", "das", "bon_mcts")
PUBLIC_FIELDS = (
    "trial_id",
    "prompt_id",
    "original_prompt_c0",
    "image_a",
    "image_b",
    "preference",
    "confidence",
    "notes",
)
PRIVATE_FIELDS = (
    "trial_id",
    "group_id",
    "model_id",
    "model_name",
    "prompt_id",
    "source_id",
    "difficulty",
    "original_prompt_c0",
    "algorithm_a",
    "algorithm_b",
    "image_id_a",
    "image_id_b",
    "source_image_a",
    "source_image_b",
    "source_sha256_a",
    "source_sha256_b",
    "packaged_sha256_a",
    "packaged_sha256_b",
    "logged_root_seed_a",
    "logged_root_seed_b",
    "same_logged_root_seed",
    "selected_candidate_seed_a",
    "selected_candidate_seed_b",
    "planned_method_seed_a",
    "planned_method_seed_b",
    "total_nfe_a",
    "total_nfe_b",
    "reward_scores_json_a",
    "reward_scores_json_b",
    "reward_prompt_matches_c0_a",
    "reward_prompt_matches_c0_b",
    "actions_json_a",
    "actions_json_b",
    "comparison_id",
    "anchor_algorithm",
    "opponent_algorithm",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
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


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_sha256(path: Path) -> str:
    return _sha256(path)


def _resolve_image(row: dict[str, str], manifest_dir: Path) -> Path:
    attempts: list[Path] = []
    for field in ("summary_image_path", "source_image_path"):
        raw = str(row.get(field, "")).strip()
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = manifest_dir / candidate
        attempts.append(candidate)
        if candidate.is_file():
            return candidate.resolve()
    joined = ", ".join(str(path) for path in attempts) or "<no image path>"
    raise FileNotFoundError(f"image not found for {row.get('image_id')}: {joined}")


def _validate_image(path: Path) -> tuple[int, int, str]:
    digest = _sha256(path)
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
    return width, height, digest


def _safe_extension(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix in {".png", ".jpg", ".jpeg", ".webp"} else ".png"


def _materialize(source: Path, target: Path, mode: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        if mode == "hardlink":
            os.link(source, temporary)
        else:
            shutil.copy2(source, temporary)
        _, _, digest = _validate_image(temporary)
        os.replace(temporary, target)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _bool_text(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def _same_nonempty(left: str, right: str) -> bool | None:
    left = str(left).strip()
    right = str(right).strip()
    if not left or not right:
        return None
    return left == right


def _row_status_ok(row: dict[str, str]) -> bool:
    return str(row.get("status", "")).strip().lower() == "complete"


def _comparison_specs(
    algorithms: set[str],
    anchor: str,
    opponents: list[str],
    all_pairs: bool,
) -> list[tuple[str, str]]:
    if all_pairs:
        ordered = [name for name in DEFAULT_ALGORITHMS if name in algorithms]
        ordered.extend(sorted(algorithms.difference(ordered)))
        return [
            (ordered[left], ordered[right])
            for left in range(len(ordered))
            for right in range(left + 1, len(ordered))
        ]
    if anchor not in algorithms:
        raise ValueError(f"anchor algorithm {anchor!r} is absent from the manifest")
    missing = sorted(set(opponents).difference(algorithms))
    if missing:
        raise ValueError(f"opponent algorithms absent from manifest: {', '.join(missing)}")
    if anchor in opponents:
        raise ValueError("anchor cannot also be an opponent")
    if len(opponents) != len(set(opponents)):
        raise ValueError("opponents must not contain duplicates")
    return [(anchor, opponent) for opponent in opponents]


def _balanced_swaps(
    trials: list[dict[str, Any]],
    rng: random.Random,
) -> dict[int, bool]:
    """Randomize sides while keeping each model/comparison stratum balanced."""
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, trial in enumerate(trials):
        strata[(trial["model_id"], trial["comparison_id"])].append(index)
    swaps: dict[int, bool] = {}
    for key in sorted(strata):
        indices = list(strata[key])
        rng.shuffle(indices)
        false_count = len(indices) // 2
        true_count = len(indices) // 2
        if len(indices) % 2:
            if rng.getrandbits(1):
                true_count += 1
            else:
                false_count += 1
        flags = [False] * false_count + [True] * true_count
        rng.shuffle(flags)
        swaps.update(zip(indices, flags))
    return swaps


def _instructions() -> str:
    return """# Pairwise image-preference evaluation

For each row, read the original prompt and inspect image **A** and image **B**.
Record exactly one preference:

- `A`: A better fulfills the prompt and has better overall visual quality.
- `B`: B better fulfills the prompt and has better overall visual quality.
- `Tie`: neither image is meaningfully preferable.

Judge prompt faithfulness first, including requested objects, attributes,
relations, and composition. Then consider artifacts and overall visual quality.
Do not infer the generating method from style or filenames. `confidence` and
`notes` are optional. Keep the private answer key away from annotators.
"""


def package_pairwise(
    manifest_path: Path,
    output_dir: Path,
    *,
    seed: int,
    anchor: str = "bon_mcts",
    opponents: list[str] | None = None,
    all_pairs: bool = False,
    models: set[str] | None = None,
    materialize: str = "copy",
    allow_incomplete: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    opponents = opponents or ["baseline", "bon", "das"]
    rows = _read_csv(manifest_path)
    if not rows:
        raise ValueError(f"empty manifest: {manifest_path}")
    required = {
        "group_id",
        "model_id",
        "algorithm_id",
        "prompt_id",
        "original_prompt_c0",
        "status",
    }
    missing_fields = sorted(required.difference(rows[0]))
    if missing_fields:
        raise ValueError(f"manifest is missing columns: {', '.join(missing_fields)}")

    if models:
        rows = [row for row in rows if row["model_id"] in models]
        missing_models = sorted(models.difference({row["model_id"] for row in rows}))
        if missing_models:
            raise ValueError(f"models absent from manifest: {', '.join(missing_models)}")
    algorithms = {row["algorithm_id"] for row in rows}
    specs = _comparison_specs(algorithms, anchor, opponents, all_pairs)

    by_group: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    duplicate_cells: list[str] = []
    for row in rows:
        group_id = row["group_id"]
        algorithm = row["algorithm_id"]
        if algorithm in by_group[group_id]:
            duplicate_cells.append(f"{group_id}/{algorithm}")
        by_group[group_id][algorithm] = row
    if duplicate_cells:
        raise ValueError(f"duplicate manifest cells: {', '.join(duplicate_cells[:10])}")

    trials: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    image_cache: dict[Path, tuple[int, int, str]] = {}
    for group_id in sorted(by_group):
        cells = by_group[group_id]
        for first_algorithm, second_algorithm in specs:
            comparison_id = f"{first_algorithm}_vs_{second_algorithm}"
            pair_rows = [cells.get(first_algorithm), cells.get(second_algorithm)]
            reason = ""
            if pair_rows[0] is None or pair_rows[1] is None:
                reason = "missing_manifest_cell"
            elif not all(_row_status_ok(row) for row in pair_rows if row is not None):
                reason = "incomplete_manifest_cell"
            elif any(
                pair_rows[0].get(field) != pair_rows[1].get(field)
                for field in (
                    "group_id",
                    "model_id",
                    "prompt_id",
                    "original_prompt_c0",
                )
            ):
                reason = "group_metadata_mismatch"
            if reason:
                exclusions.append(
                    {
                        "group_id": group_id,
                        "comparison_id": comparison_id,
                        "reason": reason,
                    }
                )
                continue
            assert pair_rows[0] is not None and pair_rows[1] is not None
            try:
                paths = [
                    _resolve_image(row, manifest_path.parent) for row in pair_rows
                ]
                for path in paths:
                    if path not in image_cache:
                        image_cache[path] = _validate_image(path)
            except (FileNotFoundError, OSError) as exc:
                exclusions.append(
                    {
                        "group_id": group_id,
                        "comparison_id": comparison_id,
                        "reason": f"invalid_image: {exc}",
                    }
                )
                continue
            trials.append(
                {
                    "group_id": group_id,
                    "model_id": pair_rows[0]["model_id"],
                    "comparison_id": comparison_id,
                    "anchor_algorithm": first_algorithm if not all_pairs else "",
                    "opponent_algorithm": second_algorithm if not all_pairs else "",
                    "rows": pair_rows,
                    "paths": paths,
                }
            )

    if exclusions and not allow_incomplete:
        preview = "; ".join(
            f"{item['group_id']}/{item['comparison_id']}: {item['reason']}"
            for item in exclusions[:8]
        )
        raise RuntimeError(
            f"{len(exclusions)} requested comparisons are incomplete. "
            f"Rerun with --allow-incomplete to log and skip them. {preview}"
        )
    if not trials:
        raise RuntimeError("no complete pairwise trials were found")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_dir}; use --overwrite to replace it"
        )

    rng = random.Random(seed)
    swaps = _balanced_swaps(trials, rng)
    for index, trial in enumerate(trials):
        if swaps[index]:
            trial["rows"].reverse()
            trial["paths"].reverse()
    rng.shuffle(trials)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp.", dir=output_dir.parent)
    )
    public_dir = temporary_root / "public"
    private_dir = temporary_root / "private"
    public_images = public_dir / "images"
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    side_counts: Counter[tuple[str, str, str, str]] = Counter()
    duplicate_image_pairs = 0
    try:
        for number, trial in enumerate(trials, 1):
            trial_id = f"trial_{number:06d}"
            left_row, right_row = trial["rows"]
            left_source, right_source = trial["paths"]
            left_rel = Path("images") / f"{trial_id}_A{_safe_extension(left_source)}"
            right_rel = Path("images") / f"{trial_id}_B{_safe_extension(right_source)}"
            left_digest = _materialize(
                left_source, public_dir / left_rel, materialize
            )
            right_digest = _materialize(
                right_source, public_dir / right_rel, materialize
            )
            duplicate_image_pairs += int(left_digest == right_digest)
            public_rows.append(
                {
                    "trial_id": trial_id,
                    "prompt_id": left_row["prompt_id"],
                    "original_prompt_c0": left_row["original_prompt_c0"],
                    "image_a": left_rel.as_posix(),
                    "image_b": right_rel.as_posix(),
                    "preference": "",
                    "confidence": "",
                    "notes": "",
                }
            )
            same_root = _same_nonempty(
                left_row.get("logged_root_seed", ""),
                right_row.get("logged_root_seed", ""),
            )
            private_rows.append(
                {
                    "trial_id": trial_id,
                    "group_id": trial["group_id"],
                    "model_id": left_row["model_id"],
                    "model_name": left_row.get("model_name", ""),
                    "prompt_id": left_row["prompt_id"],
                    "source_id": left_row.get("source_id", ""),
                    "difficulty": left_row.get("difficulty", ""),
                    "original_prompt_c0": left_row["original_prompt_c0"],
                    "algorithm_a": left_row["algorithm_id"],
                    "algorithm_b": right_row["algorithm_id"],
                    "image_id_a": left_row.get("image_id", ""),
                    "image_id_b": right_row.get("image_id", ""),
                    "source_image_a": str(left_source),
                    "source_image_b": str(right_source),
                    "source_sha256_a": image_cache[left_source][2],
                    "source_sha256_b": image_cache[right_source][2],
                    "packaged_sha256_a": left_digest,
                    "packaged_sha256_b": right_digest,
                    "logged_root_seed_a": left_row.get("logged_root_seed", ""),
                    "logged_root_seed_b": right_row.get("logged_root_seed", ""),
                    "same_logged_root_seed": _bool_text(same_root),
                    "selected_candidate_seed_a": left_row.get(
                        "selected_candidate_seed", ""
                    ),
                    "selected_candidate_seed_b": right_row.get(
                        "selected_candidate_seed", ""
                    ),
                    "planned_method_seed_a": left_row.get("planned_method_seed", ""),
                    "planned_method_seed_b": right_row.get("planned_method_seed", ""),
                    "total_nfe_a": left_row.get("total_nfe", ""),
                    "total_nfe_b": right_row.get("total_nfe", ""),
                    "reward_scores_json_a": left_row.get("reward_scores_json", ""),
                    "reward_scores_json_b": right_row.get("reward_scores_json", ""),
                    "reward_prompt_matches_c0_a": left_row.get(
                        "reward_prompt_matches_c0", ""
                    ),
                    "reward_prompt_matches_c0_b": right_row.get(
                        "reward_prompt_matches_c0", ""
                    ),
                    "actions_json_a": left_row.get("actions_json", ""),
                    "actions_json_b": right_row.get("actions_json", ""),
                    "comparison_id": trial["comparison_id"],
                    "anchor_algorithm": trial["anchor_algorithm"],
                    "opponent_algorithm": trial["opponent_algorithm"],
                }
            )
            side_counts[
                (
                    left_row["model_id"],
                    trial["comparison_id"],
                    left_row["algorithm_id"],
                    "A",
                )
            ] += 1
            side_counts[
                (
                    right_row["model_id"],
                    trial["comparison_id"],
                    right_row["algorithm_id"],
                    "B",
                )
            ] += 1

        _atomic_csv(public_dir / "pairwise_trials.csv", public_rows, PUBLIC_FIELDS)
        _atomic_csv(
            private_dir / "pairwise_answer_key.csv", private_rows, PRIVATE_FIELDS
        )
        _atomic_csv(
            private_dir / "excluded_trials.csv",
            exclusions,
            ("group_id", "comparison_id", "reason"),
        )
        (public_dir / "INSTRUCTIONS.md").write_text(_instructions(), encoding="utf-8")
        side_rows = [
            {
                "model_id": model,
                "comparison_id": comparison,
                "algorithm_id": algorithm,
                "side": side,
                "count": count,
            }
            for (model, comparison, algorithm, side), count in sorted(side_counts.items())
        ]
        report = {
            "schema_version": "1.0",
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _manifest_sha256(manifest_path),
            "blind_seed": seed,
            "construction_rule": (
                "Compare only images sharing model_id and group_id; randomly "
                "balance A/B placement within each model/comparison stratum; "
                "then shuffle all trials with the recorded seed."
            ),
            "prompt_rule": (
                "Expose and evaluate against original_prompt_c0 only. Rewritten "
                "prompts and reward scores are never included in public files."
            ),
            "seed_claim": (
                "No same-seed claim is made. Logged root and selected-candidate "
                "seeds are retained in the private answer key."
            ),
            "materialization": materialize,
            "all_pairs": all_pairs,
            "anchor_algorithm": "" if all_pairs else anchor,
            "opponents": [] if all_pairs else opponents,
            "comparison_specs": [list(spec) for spec in specs],
            "models": sorted({trial["model_id"] for trial in trials}),
            "trial_count": len(trials),
            "image_count": 2 * len(trials),
            "excluded_trial_count": len(exclusions),
            "duplicate_image_pair_count": duplicate_image_pairs,
            "side_counts": side_rows,
        }
        _atomic_json(private_dir / "package_report.json", report)
        (temporary_root / "README.md").write_text(
            "# Blinded pairwise human-evaluation package\n\n"
            "Give annotators only the `public/` directory. Keep `private/` "
            "separate until responses are frozen. Image A/B placement and "
            f"trial order were shuffled with seed `{seed}`.\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temporary_root, output_dir)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    return {
        "output_dir": str(output_dir),
        "public_manifest": str(output_dir / "public" / "pairwise_trials.csv"),
        "private_answer_key": str(
            output_dir / "private" / "pairwise_answer_key.csv"
        ),
        "report": str(output_dir / "private" / "package_report.json"),
        "trial_count": len(trials),
        "image_count": 2 * len(trials),
        "excluded_trial_count": len(exclusions),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to legacy_summary/legacy_manifest.csv.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--anchor", default="bon_mcts")
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["baseline", "bon", "das"],
        help="Algorithms compared independently against --anchor.",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Package every algorithm pair instead of anchor-vs-opponents.",
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--materialize",
        choices=("copy", "hardlink"),
        default="copy",
        help="copy is portable; hardlink saves disk on one filesystem.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Skip missing/corrupt pairs and log every exclusion.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = package_pairwise(
        args.manifest,
        args.output_dir,
        seed=args.seed,
        anchor=args.anchor,
        opponents=args.opponents,
        all_pairs=args.all_pairs,
        models=set(args.models) if args.models else None,
        materialize=args.materialize,
        allow_incomplete=args.allow_incomplete,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
