#!/usr/bin/env python3
"""Summarize generation and reward inference cost from existing run logs.

The report separates terminal image candidates, objective evaluations, reward
backend forward calls, generator NFE, wall time, and GPU-hours. Missing NFE or
memory measurements remain missing; they are never reconstructed from a
nominal candidate budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


BACKENDS_FOR_OBJECTIVE = {
    "imagereward": ("imagereward",),
    "vqascore": ("vqascore",),
    "composite_ir_vqa": ("imagereward", "vqascore"),
    "composite_3": ("imagereward", "hpsv3", "pickscore"),
    "composite_hpsv3_ir": ("imagereward", "hpsv3"),
    "composite_ir_ps": ("imagereward", "pickscore"),
}
ARM_OBJECTIVES = {
    "imagereward": "imagereward",
    "vqascore": "vqascore",
    "ir_vqa_equal": "composite_ir_vqa",
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


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


def _method_dirs(root: Path) -> list[Path]:
    found: set[Path] = set()
    for log_dir in root.rglob("logs"):
        if any(
            path.is_file() and not path.name.endswith("_rewrite_examples.jsonl")
            for path in log_dir.glob("rank_*.jsonl")
        ):
            found.add(log_dir.parent.resolve())
    for summary in root.rglob("rank_*/summary.json"):
        found.add(summary.parent.parent.resolve())
    return sorted(found)


def _load_sd35_rows(method_dir: Path) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for path in sorted((method_dir / "logs").glob("rank_*.jsonl")):
        if path.name.endswith("_rewrite_examples.jsonl"):
            continue
        for row in _read_jsonl(path):
            prompt_index = _as_int(row.get("prompt_index"))
            if prompt_index is None:
                continue
            by_key[(prompt_index, str(row.get("mode", "")))] = row
    return list(by_key.values())


def _load_flux_rows(method_dir: Path) -> list[dict[str, Any]]:
    rows = []
    algorithm = method_dir.name
    for path in sorted(method_dir.glob("rank_*/summary.json")):
        payload = _read_json(path)
        if not isinstance(payload, list):
            continue
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            slug = str(entry.get("slug", ""))
            digits = "".join(character for character in slug if character.isdigit())
            prompt_index = _as_int(digits)
            samples = entry.get("samples") or []
            if prompt_index is None or not isinstance(samples, list):
                continue
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                score_key = "baseline_score" if algorithm == "baseline" else "search_score"
                rows.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt": entry.get("prompt"),
                        "mode": (
                            "base"
                            if algorithm == "baseline"
                            else "mcts"
                            if "mcts" in algorithm
                            else "bon"
                            if algorithm in {"bon", "das"}
                            else algorithm
                        ),
                        "score": sample.get(score_key),
                        "nfe": sample.get("nfe"),
                        "search_diagnostics": sample.get("diagnostics"),
                    }
                )
    return rows


def _candidate_account(
    row: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    """Return conservative per-prompt terminal/reward accounting."""
    mode = str(row.get("mode", "")).lower()
    diagnostics = row.get("search_diagnostics") or row.get("diagnostics") or {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}

    if mode == "base" or method == "baseline":
        return {
            "search_terminal_candidates": 0,
            "comparison_baseline_candidates": 1,
            "terminal_candidates_total": 1,
            "objective_evaluations": 1,
            "accounting_exact": True,
            "accounting_note": "one scored baseline output",
        }

    if mode == "bon" or method in {"bon", "das"}:
        n = _as_int(diagnostics.get("bon_n"))
        if n is None:
            candidate_seeds = diagnostics.get("candidate_seeds")
            n = len(candidate_seeds) if isinstance(candidate_seeds, list) else None
        if n is None:
            return {
                "search_terminal_candidates": None,
                "comparison_baseline_candidates": 1,
                "terminal_candidates_total": None,
                "objective_evaluations": None,
                "accounting_exact": False,
                "accounting_note": "BoN candidate count missing from diagnostics",
            }
        total = 1 + int(n)
        return {
            "search_terminal_candidates": int(n),
            "comparison_baseline_candidates": 1,
            "terminal_candidates_total": total,
            "objective_evaluations": total,
            "accounting_exact": True,
            "accounting_note": "comparison baseline plus scored BoN/DAS candidates",
        }

    bon_mcts = diagnostics.get("bon_mcts")
    if isinstance(bon_mcts, dict):
        prescreen = bon_mcts.get("prescreen_ranked")
        prescreen_count = (
            len(prescreen)
            if isinstance(prescreen, list)
            else _as_int(bon_mcts.get("prescreen_n"))
        )
        refine = bon_mcts.get("tree_refine") or bon_mcts.get("mcts_refine")
        refine = refine if isinstance(refine, list) else []
        sims = [
            _as_int(item.get("n_sims_used"))
            for item in refine
            if isinstance(item, dict)
        ]
        exact = prescreen_count is not None and bool(refine) and all(
            value is not None for value in sims
        )
        if not exact:
            return {
                "search_terminal_candidates": None,
                "comparison_baseline_candidates": 1,
                "terminal_candidates_total": None,
                "objective_evaluations": None,
                "accounting_exact": False,
                "accounting_note": "BoN-MCTS prescreen/refine diagnostics incomplete",
            }
        sparse = diagnostics.get("sparse_noise_refine")
        if isinstance(sparse, dict) and sparse.get("enabled"):
            # The diagnostics retain sparse-refine detail only for the winning
            # root, so a run-wide total cannot be reconstructed faithfully.
            return {
                "search_terminal_candidates": None,
                "comparison_baseline_candidates": 1,
                "terminal_candidates_total": None,
                "objective_evaluations": None,
                "accounting_exact": False,
                "accounting_note": (
                    "sparse-noise refinement enabled; diagnostics do not retain "
                    "all refined-root rollout counts"
                ),
            }
        terminal_search = (
            int(prescreen_count)
            + sum(int(value) for value in sims if value is not None)
            + len(refine)  # one exploit replay per refined root
        )
        step_reward_unknown = "step_reward" in method
        return {
            "search_terminal_candidates": terminal_search,
            "comparison_baseline_candidates": 1,
            "terminal_candidates_total": terminal_search + 1,
            "objective_evaluations": (
                None if step_reward_unknown else terminal_search + 1
            ),
            "accounting_exact": bool(exact and not step_reward_unknown),
            "accounting_note": (
                "terminal count includes prescreens, MCTS rollouts, one exploit "
                "per refined root, and comparison baseline"
                + (
                    "; per-step shaping reward calls are not recoverable"
                    if step_reward_unknown
                    else ""
                )
            ),
        }

    return {
        "search_terminal_candidates": None,
        "comparison_baseline_candidates": 1,
        "terminal_candidates_total": None,
        "objective_evaluations": None,
        "accounting_exact": False,
        "accounting_note": (
            "unsupported or incomplete algorithm diagnostics; sampled MCTS "
            "history is not treated as a complete rollout count"
        ),
    }


def _infer_objective(root: Path, method_dir: Path) -> str | None:
    try:
        relative = method_dir.relative_to(root)
    except ValueError:
        relative = method_dir
    for part in relative.parts:
        if part in ARM_OBJECTIVES:
            return ARM_OBJECTIVES[part]
    aggregate = _read_json(method_dir / "aggregate_ddp.json")
    if isinstance(aggregate, dict) and aggregate.get("search_reward"):
        return str(aggregate["search_reward"])
    return None


def _post_eval_counts(method_dir: Path) -> dict[str, int]:
    payload = _read_json(method_dir / "best_images_multi_reward.json")
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    counts: Counter[str] = Counter()
    for row in rows:
        scores = row.get("scores") if isinstance(row, dict) else None
        if not isinstance(scores, dict):
            continue
        for backend, value in scores.items():
            if isinstance(value, (int, float)):
                counts[str(backend)] += 1
    return dict(counts)


def _prompt_rewriter_account(root: Path) -> dict[str, Any]:
    manifests = sorted(root.glob("study_manifest_*.json"))
    for path in reversed(manifests):
        payload = _read_json(path)
        control = payload.get("prompt_variant_control") if isinstance(payload, dict) else None
        if not isinstance(control, dict):
            continue
        if not bool(control.get("use_qwen")):
            return {
                "calls": 0,
                "exact": True,
                "note": "USE_QWEN=0; deterministic local variants use no rewriter model",
            }
        cache_path = Path(str(control.get("shared_rewrites_file", "")))
        cache = _read_json(cache_path)
        if isinstance(cache, dict):
            return {
                "calls": len(cache),
                "exact": False,
                "note": (
                    "Qwen enabled; cache entry count is an upper-bound/proxy because "
                    "pre-existing cache hits are not separately logged"
                ),
            }
        return {
            "calls": None,
            "exact": False,
            "note": "Qwen enabled but rewrite API/cache-hit calls were not logged",
        }
    return {
        "calls": None,
        "exact": False,
        "note": "No study manifest records whether Qwen rewriting was enabled",
    }


def summarize(
    root: Path,
    output_dir: Path,
    generation_gpus: int,
    reward_gpus: int,
    generation_gpu_hour_price: float | None,
    reward_gpu_hour_price: float | None,
    memory_summary: Path | None,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    run_rows: list[dict[str, Any]] = []
    backend_totals: Counter[str] = Counter()

    for method_dir in _method_dirs(root):
        rows = _load_sd35_rows(method_dir)
        layout = "sd35"
        if not rows:
            rows = _load_flux_rows(method_dir)
            layout = "flux"
        if not rows:
            continue
        method = method_dir.name
        accounts = [_candidate_account(row, method) for row in rows]
        exact_accounts = [account for account in accounts if account["accounting_exact"]]
        objective_values = [
            int(account["objective_evaluations"])
            for account in accounts
            if account["objective_evaluations"] is not None
        ]
        terminal_values = [
            int(account["terminal_candidates_total"])
            for account in accounts
            if account["terminal_candidates_total"] is not None
        ]
        search_terminal_values = [
            int(account["search_terminal_candidates"])
            for account in accounts
            if account["search_terminal_candidates"] is not None
        ]
        nfe_values = [
            int(value)
            for value in (_as_int(row.get("nfe")) for row in rows)
            if value is not None
        ]
        objective = _infer_objective(root, method_dir)
        components = BACKENDS_FOR_OBJECTIVE.get(str(objective), ())
        search_backend_calls: Counter[str] = Counter()
        if len(objective_values) == len(rows):
            objective_total = sum(objective_values)
            for backend in components:
                search_backend_calls[backend] += objective_total
        else:
            objective_total = None
        post_eval = _post_eval_counts(method_dir)
        backend_calls = Counter(search_backend_calls)
        backend_calls.update(post_eval)
        backend_totals.update(backend_calls)

        aggregate = _read_json(method_dir / "aggregate_ddp.json")
        elapsed = (
            _as_float(aggregate.get("elapsed_sec"))
            if isinstance(aggregate, dict)
            else None
        )
        generation_gpu_hours = (
            elapsed * generation_gpus / 3600.0 if elapsed is not None else None
        )
        reward_gpu_hours = (
            elapsed * reward_gpus / 3600.0 if elapsed is not None else None
        )
        dollar_cost = None
        if (
            generation_gpu_hours is not None
            and reward_gpu_hours is not None
            and generation_gpu_hour_price is not None
            and reward_gpu_hour_price is not None
        ):
            dollar_cost = (
                generation_gpu_hours * generation_gpu_hour_price
                + reward_gpu_hours * reward_gpu_hour_price
            )
        run_rows.append(
            {
                "run_path": str(method_dir.parent),
                "method_path": str(method_dir),
                "layout": layout,
                "algorithm": method,
                "objective": objective,
                "completed_prompts": len(rows),
                "exactly_accounted_prompts": len(exact_accounts),
                "search_terminal_candidates": (
                    sum(search_terminal_values)
                    if len(search_terminal_values) == len(rows)
                    else None
                ),
                "comparison_baseline_candidates": len(rows),
                "terminal_candidates_total": (
                    sum(terminal_values) if len(terminal_values) == len(rows) else None
                ),
                "objective_evaluations": objective_total,
                "search_backend_calls_json": json.dumps(
                    dict(sorted(search_backend_calls.items())), sort_keys=True
                ),
                "post_eval_backend_calls_json": json.dumps(
                    dict(sorted(post_eval.items())), sort_keys=True
                ),
                "backend_calls_total_json": json.dumps(
                    dict(sorted(backend_calls.items())), sort_keys=True
                ),
                "generator_nfe_logged": (
                    sum(nfe_values) if len(nfe_values) == len(rows) else None
                ),
                "nfe_logged_prompts": len(nfe_values),
                "elapsed_sec": elapsed,
                "generation_gpu_hours": generation_gpu_hours,
                "reward_gpu_hours": reward_gpu_hours,
                "total_gpu_hours": (
                    generation_gpu_hours + reward_gpu_hours
                    if generation_gpu_hours is not None and reward_gpu_hours is not None
                    else None
                ),
                "estimated_dollar_cost": dollar_cost,
                "accounting_notes": " | ".join(
                    sorted({str(account["accounting_note"]) for account in accounts})
                ),
            }
        )

    prompt_rewriter = _prompt_rewriter_account(root)
    memory = _read_json(memory_summary) if memory_summary else None
    total_gpu_hours_values = [
        float(row["total_gpu_hours"])
        for row in run_rows
        if row["total_gpu_hours"] is not None
    ]
    dollar_values = [
        float(row["estimated_dollar_cost"])
        for row in run_rows
        if row["estimated_dollar_cost"] is not None
    ]
    summary = {
        "schema_version": "1.0",
        "root": str(root),
        "definitions": {
            "terminal_candidate": (
                "A decoded terminal image scored by the search objective. "
                "Intermediate predicted-clean step images are excluded."
            ),
            "objective_evaluation": (
                "One call to the configured scalar search objective. A composite "
                "objective may invoke multiple reward backends."
            ),
            "backend_evaluation": (
                "One image scored by one reward backend. A composite objective "
                "therefore contributes one evaluation to each component backend; "
                "separately reported post-evaluation scores are included."
            ),
            "generator_nfe_logged": (
                "The explicit per-row generator NFE counter. Comparison-baseline "
                "NFE may be excluded when the runner resets its counter before search."
            ),
            "gpu_hours": (
                "Method elapsed time multiplied by configured generation/reward GPU "
                "counts; model startup outside method timing is excluded."
            ),
        },
        "prompt_rewriter": prompt_rewriter,
        "memory": (
            memory
            if memory is not None
            else {
                "available": False,
                "note": (
                    "Peak memory was not logged by historical runs. Start "
                    "tools/monitor_gpu_usage.py for sampled peak memory."
                ),
            }
        ),
        "aggregate": {
            "method_count": len(run_rows),
            "completed_prompt_rows": sum(int(row["completed_prompts"]) for row in run_rows),
            "terminal_candidates_total": (
                sum(int(row["terminal_candidates_total"]) for row in run_rows)
                if run_rows and all(row["terminal_candidates_total"] is not None for row in run_rows)
                else None
            ),
            "objective_evaluations": (
                sum(int(row["objective_evaluations"]) for row in run_rows)
                if run_rows and all(row["objective_evaluations"] is not None for row in run_rows)
                else None
            ),
            "reward_backend_calls": dict(sorted(backend_totals.items())),
            "generator_nfe_logged": (
                sum(int(row["generator_nfe_logged"]) for row in run_rows)
                if run_rows and all(row["generator_nfe_logged"] is not None for row in run_rows)
                else None
            ),
            "total_gpu_hours": (
                sum(total_gpu_hours_values)
                if len(total_gpu_hours_values) == len(run_rows) and run_rows
                else None
            ),
            "estimated_dollar_cost": (
                sum(dollar_values)
                if len(dollar_values) == len(run_rows) and run_rows
                else None
            ),
        },
        "runs": run_rows,
    }
    fields = [
        "run_path", "method_path", "layout", "algorithm", "objective",
        "completed_prompts", "exactly_accounted_prompts",
        "search_terminal_candidates", "comparison_baseline_candidates",
        "terminal_candidates_total", "objective_evaluations",
        "search_backend_calls_json", "post_eval_backend_calls_json",
        "backend_calls_total_json", "generator_nfe_logged", "nfe_logged_prompts",
        "elapsed_sec", "generation_gpu_hours", "reward_gpu_hours",
        "total_gpu_hours", "estimated_dollar_cost", "accounting_notes",
    ]
    _atomic_csv(output_dir / "inference_costs.csv", run_rows, fields)
    _atomic_json(output_dir / "inference_costs.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--generation-gpus", type=int, default=3)
    parser.add_argument("--reward-gpus", type=int, default=1)
    parser.add_argument("--generation-gpu-hour-price", type=float, default=None)
    parser.add_argument("--reward-gpu-hour-price", type=float, default=None)
    parser.add_argument(
        "--memory-summary",
        type=Path,
        default=None,
        help="Optional summary JSON produced by monitor_gpu_usage.py.",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir or (root / "cost_summary")
    summary = summarize(
        root,
        output_dir,
        max(0, int(args.generation_gpus)),
        max(0, int(args.reward_gpus)),
        args.generation_gpu_hour_price,
        args.reward_gpu_hour_price,
        args.memory_summary,
    )
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))
    print(f"wrote {Path(output_dir) / 'inference_costs.csv'}")
    print(f"wrote {Path(output_dir) / 'inference_costs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
