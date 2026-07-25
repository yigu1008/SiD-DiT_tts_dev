#!/usr/bin/env python3
"""Prepare a deterministic, stratified GenAI-Bench prompt subset.

This module intentionally has no model-generation dependencies.  It accepts
CSV, JSONL, or Parquet metadata and writes the exact original prompt text to a
portable CSV plus a reserve set and processing report.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_SEED = 20260725
DIFFICULTIES = ("basic", "advanced")
QUARTILES = (1, 2, 3, 4)


@dataclass(frozen=True)
class PromptRecord:
    source: str
    source_id: str
    prompt: str
    normalized_prompt: str
    difficulty: str
    basic_skills: tuple[str, ...]
    advanced_skills: tuple[str, ...]
    word_count: int
    character_count: int
    length_quartile: int


@dataclass(frozen=True)
class ProcessingSummary:
    input_row_count: int
    valid_row_count: int
    duplicate_count: int
    basic_pool_size: int
    advanced_pool_size: int
    selected_count: int
    reserve_count: int
    counts_by_difficulty_and_length_quartile: dict[str, int]
    skill_frequencies: dict[str, dict[str, int]]
    sampling_seed: int
    prompts_csv_sha256: str
    output_path: str
    reserve_path: str
    report_path: str


def _field(row: Mapping[str, Any], aliases: Iterable[str]) -> Any:
    """Read a flattened or nested field using case-insensitive aliases."""
    aliases = tuple(aliases)
    lowered = {str(k).strip().casefold(): v for k, v in row.items()}
    for alias in aliases:
        if alias.casefold() in lowered:
            return lowered[alias.casefold()]
    for alias in aliases:
        current: Any = row
        ok = True
        for part in alias.split("."):
            if not isinstance(current, Mapping):
                ok = False
                break
            match = next((k for k in current if str(k).casefold() == part.casefold()), None)
            if match is None:
                ok = False
                break
            current = current[match]
        if ok:
            return current
    return None


def _parse_skills(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        values = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        values: Any = None
        for parser in (json.loads, ast.literal_eval):
            try:
                candidate = parser(text)
                if isinstance(candidate, (list, tuple, set)):
                    values = candidate
                    break
            except (ValueError, SyntaxError, json.JSONDecodeError):
                pass
        if values is None:
            values = re.split(r"[,|]", text)
    else:
        values = (value,)
    output = []
    for item in values:
        text = str(item).strip()
        if text:
            output.append(text)
    return tuple(dict.fromkeys(output))


def _normalize_prompt(prompt: str) -> str:
    # Only this copy is normalized; PromptRecord.prompt remains byte-for-byte
    # equivalent to the input string after CSV/JSON decoding.
    return re.sub(r"\s+", " ", prompt).strip().casefold()


def _read_rows(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open(encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise ValueError(f"JSONL row {line_no} is not an object")
                    rows.append(value)
        return rows
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            return pq.read_table(path).to_pylist()
        except ImportError:
            try:
                import pandas as pd

                return pd.read_parquet(path).to_dict(orient="records")
            except ImportError as exc:
                raise RuntimeError("Parquet input requires pyarrow or pandas") from exc
    raise ValueError(f"Unsupported input format {path.suffix!r}; use CSV, JSONL, or Parquet")


def _quartile_assign(records: list[PromptRecord]) -> list[PromptRecord]:
    grouped: dict[str, list[PromptRecord]] = defaultdict(list)
    for record in records:
        grouped[record.difficulty].append(record)
    output: list[PromptRecord] = []
    for difficulty in DIFFICULTIES:
        pool = sorted(
            grouped[difficulty],
            key=lambda r: (r.word_count, r.character_count, r.normalized_prompt, r.source_id),
        )
        n = len(pool)
        for index, record in enumerate(pool):
            quartile = min(4, (index * 4) // n + 1)
            output.append(
                PromptRecord(
                    source=record.source,
                    source_id=record.source_id,
                    prompt=record.prompt,
                    normalized_prompt=record.normalized_prompt,
                    difficulty=record.difficulty,
                    basic_skills=record.basic_skills,
                    advanced_skills=record.advanced_skills,
                    word_count=record.word_count,
                    character_count=record.character_count,
                    length_quartile=quartile,
                )
            )
    return output


def _csv_row(record: PromptRecord, prompt_id: str, seed: int) -> dict[str, Any]:
    return {
        "prompt_id": prompt_id,
        "source": record.source,
        "source_id": record.source_id,
        "prompt": record.prompt,
        "difficulty": record.difficulty,
        "basic_skills": "|".join(record.basic_skills),
        "advanced_skills": "|".join(record.advanced_skills),
        "word_count": record.word_count,
        "character_count": record.character_count,
        "length_quartile": record.length_quartile,
        "sampling_seed": seed,
    }


def _write_prompt_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "prompt_id", "source", "source_id", "prompt", "difficulty", "basic_skills",
        "advanced_skills", "word_count", "character_count", "length_quartile", "sampling_seed",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _validate_selected(selected: list[PromptRecord], reserve: list[PromptRecord], num_prompts: int) -> None:
    if len(selected) != num_prompts:
        raise RuntimeError(f"selected prompt count is {len(selected)}, expected {num_prompts}")
    if len(selected) != 40:
        raise RuntimeError("this study requires exactly 40 selected prompts")
    if sum(r.difficulty == "basic" for r in selected) != 20:
        raise RuntimeError("selected set must contain exactly 20 basic prompts")
    if sum(r.difficulty == "advanced" for r in selected) != 20:
        raise RuntimeError("selected set must contain exactly 20 advanced prompts")
    selected_keys = {r.normalized_prompt for r in selected}
    if len(selected_keys) != len(selected):
        raise RuntimeError("duplicate normalized prompts appear in selected set")
    if selected_keys & {r.normalized_prompt for r in reserve}:
        raise RuntimeError("selected and reserve prompts overlap")
    counts = Counter((r.difficulty, r.length_quartile) for r in selected)
    for difficulty in DIFFICULTIES:
        for quartile in QUARTILES:
            if counts[(difficulty, quartile)] != 5:
                raise RuntimeError(
                    f"cell {difficulty}/q{quartile} has {counts[(difficulty, quartile)]} prompts, expected 5"
                )


def prepare_prompts(
    input_path: str | Path,
    output_path: str | Path,
    num_prompts: int = 40,
    num_reserve: int = 16,
    seed: int = DEFAULT_SEED,
) -> ProcessingSummary:
    """Prepare the selected and reserve prompt files.

    The selection is deterministic for fixed input rows and seed.  Prompt text
    is never rewritten; normalization is used only for duplicate detection.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if num_prompts != 40:
        raise ValueError("num_prompts must be exactly 40 for human_eval_genai40_v1")
    if num_reserve != 16:
        raise ValueError("num_reserve must be exactly 16 for human_eval_genai40_v1")
    rows = _read_rows(input_path)
    if not rows:
        raise ValueError("input metadata contains no rows")

    raw_records: list[PromptRecord] = []
    seen_prompts: set[str] = set()
    duplicate_count = 0
    source_name = input_path.name
    for row_number, row in enumerate(rows, 1):
        prompt_value = _field(row, ("Prompt", "prompt", "text"))
        source_id_value = _field(row, ("Index", "index", "source_id"))
        if prompt_value is None or source_id_value is None:
            if not raw_records:
                raise ValueError(
                    "could not find prompt/source-ID fields; expected Prompt/prompt/text and Index/index/source_id"
                )
            continue
        prompt = str(prompt_value)
        if not prompt.strip():
            continue
        normalized = _normalize_prompt(prompt)
        if not normalized:
            continue
        if normalized in seen_prompts:
            duplicate_count += 1
            continue
        seen_prompts.add(normalized)
        basic = _parse_skills(_field(row, ("Tags.basic_skills", "basic_skills")))
        advanced = _parse_skills(_field(row, ("Tags.advanced_skills", "advanced_skills")))
        raw_records.append(
            PromptRecord(
                source=source_name,
                source_id=str(source_id_value),
                prompt=prompt,
                normalized_prompt=normalized,
                difficulty="advanced" if advanced else "basic",
                basic_skills=basic,
                advanced_skills=advanced,
                word_count=len(prompt.split()),
                character_count=len(prompt),
                length_quartile=0,
            )
        )
    if not raw_records:
        raise ValueError("no valid prompt rows remain after filtering")

    records = _quartile_assign(raw_records)
    pools: dict[tuple[str, int], list[PromptRecord]] = defaultdict(list)
    for record in records:
        pools[(record.difficulty, record.length_quartile)].append(record)
    missing_cells = {
        f"{difficulty}/q{quartile}": len(pools[(difficulty, quartile)])
        for difficulty in DIFFICULTIES
        for quartile in QUARTILES
        if len(pools[(difficulty, quartile)]) < 5
    }
    if missing_cells:
        raise ValueError(f"difficulty/length cell has fewer than five prompts: {missing_cells}")

    rng = random.Random(seed)
    selected: list[PromptRecord] = []
    selected_keys: set[str] = set()
    for difficulty in DIFFICULTIES:
        for quartile in QUARTILES:
            cell = list(pools[(difficulty, quartile)])
            rng.shuffle(cell)
            picks = cell[:5]
            selected.extend(picks)
            selected_keys.update(r.normalized_prompt for r in picks)

    remaining = [r for r in records if r.normalized_prompt not in selected_keys]
    if len(remaining) < num_reserve:
        raise ValueError(f"only {len(remaining)} non-selected prompts remain; cannot reserve {num_reserve}")
    rng.shuffle(remaining)
    reserve = remaining[:num_reserve]
    selected.sort(key=lambda r: (r.difficulty, r.length_quartile, r.source_id, r.normalized_prompt))
    reserve.sort(key=lambda r: (r.difficulty, r.length_quartile, r.source_id, r.normalized_prompt))
    _validate_selected(selected, reserve, num_prompts)

    selected_rows = [_csv_row(record, f"p{index:03d}", seed) for index, record in enumerate(selected)]
    reserve_rows = [_csv_row(record, f"r{index:03d}", seed) for index, record in enumerate(reserve)]
    _write_prompt_csv(output_path, selected_rows)
    reserve_path = output_path.with_name("prompts_reserve.csv")
    report_path = output_path.with_name("prompt_processing_report.json")
    _write_prompt_csv(reserve_path, reserve_rows)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()

    cell_counts = Counter((r.difficulty, r.length_quartile) for r in selected)
    skills = {
        "basic_skills": dict(sorted(Counter(skill for r in selected for skill in r.basic_skills).items())),
        "advanced_skills": dict(sorted(Counter(skill for r in selected for skill in r.advanced_skills).items())),
    }
    summary = ProcessingSummary(
        input_row_count=len(rows),
        valid_row_count=len(raw_records),
        duplicate_count=duplicate_count,
        basic_pool_size=sum(r.difficulty == "basic" for r in records),
        advanced_pool_size=sum(r.difficulty == "advanced" for r in records),
        selected_count=len(selected),
        reserve_count=len(reserve),
        counts_by_difficulty_and_length_quartile={
            f"{difficulty}/q{quartile}": cell_counts[(difficulty, quartile)]
            for difficulty in DIFFICULTIES for quartile in QUARTILES
        },
        skill_frequencies=skills,
        sampling_seed=seed,
        prompts_csv_sha256=digest,
        output_path=str(output_path),
        reserve_path=str(reserve_path),
        report_path=str(report_path),
    )
    report_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-prompts", type=int, default=40)
    parser.add_argument("--num-reserve", type=int, default=16)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    summary = prepare_prompts(args.input, args.output, args.num_prompts, args.num_reserve, args.seed)
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
