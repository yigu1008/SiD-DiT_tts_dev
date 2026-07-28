#!/usr/bin/env python3
"""Prepare one deterministic, balanced GenAI-Bench subset for every method."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DIFFICULTIES = ("basic", "advanced")
QUARTILES = (1, 2, 3, 4)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _stable_seed(study_id: str, prompt_id: str, base_seed: int) -> int:
    payload = f"{study_id}\0{prompt_id}\0{base_seed}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return 1 + value % 2_147_483_646


def prepare(
    source: Path,
    output_csv: Path,
    output_txt: Path,
    seed_map: Path,
    manifest_path: Path,
    subset_size: int,
    subset_seed: int,
    generation_base_seed: int,
    study_id: str,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    with source.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"prompt_id", "prompt", "difficulty", "length_quartile"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{source} must contain columns {sorted(required)}")
    if subset_size <= 0 or subset_size % 8:
        raise ValueError("subset size must be a positive multiple of 8")
    if len({str(row["prompt_id"]) for row in rows}) != len(rows):
        raise ValueError(f"{source} contains duplicate prompt_id values")

    pools: dict[tuple[str, int], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for source_index, row in enumerate(rows):
        prompt = str(row["prompt"]).strip()
        difficulty = str(row["difficulty"]).strip().lower()
        try:
            quartile = int(row["length_quartile"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid length_quartile for {row.get('prompt_id')}"
            ) from exc
        if not prompt or "\n" in prompt or "\r" in prompt:
            raise ValueError(
                f"{row.get('prompt_id')}: prompt must be nonempty and one line"
            )
        if difficulty not in DIFFICULTIES or quartile not in QUARTILES:
            raise ValueError(
                f"{row.get('prompt_id')}: invalid stratum {difficulty}/q{quartile}"
            )
        pools[(difficulty, quartile)].append((source_index, row))

    per_cell = subset_size // 8
    shortages = {
        f"{difficulty}/q{quartile}": len(pools[(difficulty, quartile)])
        for difficulty in DIFFICULTIES
        for quartile in QUARTILES
        if len(pools[(difficulty, quartile)]) < per_cell
    }
    if shortages:
        raise ValueError(
            f"not enough prompts for {per_cell} samples per stratum: {shortages}"
        )

    rng = random.Random(subset_seed)
    selected: list[tuple[int, dict[str, str]]] = []
    for difficulty in DIFFICULTIES:
        for quartile in QUARTILES:
            cell = list(pools[(difficulty, quartile)])
            rng.shuffle(cell)
            selected.extend(cell[:per_cell])
    selected.sort(key=lambda item: item[0])
    selected_ids = {str(row["prompt_id"]) for _, row in selected}

    fields = list(rows[0].keys())
    if "sweep_prompt_index" not in fields:
        fields.append("sweep_prompt_index")
    csv_lines: list[str] = []
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    prompt_records = []
    seeds: dict[str, int] = {}
    for local_index, (source_index, source_row) in enumerate(selected):
        row: dict[str, Any] = dict(source_row)
        row["sweep_prompt_index"] = local_index
        writer.writerow(row)
        prompt_id = str(row["prompt_id"])
        root_seed = _stable_seed(study_id, prompt_id, generation_base_seed)
        seeds[str(local_index)] = root_seed
        prompt_records.append(
            {
                "sweep_prompt_index": local_index,
                "source_row_index": source_index,
                "prompt_id": prompt_id,
                "prompt": str(row["prompt"]),
                "difficulty": str(row["difficulty"]).lower(),
                "length_quartile": int(row["length_quartile"]),
                "root_seed": root_seed,
            }
        )
    csv_lines.append(buffer.getvalue())
    _atomic_text(output_csv, "".join(csv_lines))
    _atomic_text(
        output_txt,
        "".join(record["prompt"] + "\n" for record in prompt_records),
    )

    seed_payload = {
        "study_id": study_id,
        "seed_rule": "stable_sha256(study_id,prompt_id,generation_base_seed)",
        "generation_base_seed": generation_base_seed,
        "shared_base_seed_across_methods": True,
        "root_selection_rule": (
            "Each method receives the same prompt-level base seed. Multi-root "
            "methods derive deterministic candidate-root pools from it and may "
            "select different winning roots."
        ),
        "seeds": seeds,
    }
    _atomic_text(
        seed_map,
        json.dumps(seed_payload, indent=2, ensure_ascii=False) + "\n",
    )

    manifest = {
        "study_id": study_id,
        "source_file": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "subset_csv": str(output_csv.resolve()),
        "subset_csv_sha256": hashlib.sha256(output_csv.read_bytes()).hexdigest(),
        "subset_size": subset_size,
        "subset_seed": subset_seed,
        "selection_rule": (
            f"Seeded uniform sampling without replacement of {per_cell} prompt(s) "
            "from each of the 8 difficulty x length-quartile strata; selected "
            "rows are restored to source-file order."
        ),
        "same_subset_across_methods": True,
        "shared_base_seed_across_methods": True,
        "selected_output_seed_warning": (
            "Do not describe outputs as same-seed when a method selects across "
            "multiple candidate roots; report its winner seed from diagnostics."
        ),
        "reward_prompt_rule": (
            "Search and evaluation always score against the exact original "
            "GenAI-Bench prompt c0 stored in this manifest."
        ),
        "prompts": prompt_records,
        "exclusions": [
            {
                "prompt_id": str(row["prompt_id"]),
                "reason": "not_selected_by_balanced_random_subset",
            }
            for row in rows
            if str(row["prompt_id"]) not in selected_ids
        ],
    }
    _atomic_text(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-txt", required=True, type=Path)
    parser.add_argument("--seed-map", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--subset-size", type=int, default=16)
    parser.add_argument("--subset-seed", type=int, default=20260728)
    parser.add_argument("--generation-base-seed", type=int, default=12345)
    parser.add_argument("--study-id", default="sid_vqascore_algorithm_sweep")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare(
        source=args.input,
        output_csv=args.output_csv.expanduser().resolve(),
        output_txt=args.output_txt.expanduser().resolve(),
        seed_map=args.seed_map.expanduser().resolve(),
        manifest_path=args.manifest.expanduser().resolve(),
        subset_size=int(args.subset_size),
        subset_seed=int(args.subset_seed),
        generation_base_seed=int(args.generation_base_seed),
        study_id=str(args.study_id),
    )
    print(
        f"[subset] selected {manifest['subset_size']} prompts with "
        f"seed={manifest['subset_seed']}"
    )
    print(f"[subset] CSV: {manifest['subset_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
