#!/usr/bin/env python3
"""Bundle existing prompts where ActDiff has the largest VQAScore margin.

This is a read-only selector with respect to experiment outputs: it never runs
generation or a reward model and never modifies source images.  It discovers
the existing GenAI-200 VQAScore run layout, ranks ActDiff against the strongest
available non-ActDiff method for each prompt, and copies a few comparison images
plus auditable CSV/JSON/Markdown metadata into a separate bundle directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


MODEL_LABELS = {
    "flux_schnell": "Flux-Schnell",
    "sid": "SiD-SD3.5",
    "senseflow_large": "SenseFlow-SD3.5-Large",
    "sd35_base": "SD3.5-Base",
}

METHOD_LABELS = {
    "baseline": "Baseline",
    "das": "DAS",
    "fksteering": "FK-Steering",
    "bon": "BoN",
    "beam": "Beam",
    "sop": "SoP",
    "ga": "GA",
    "dts": "DTS",
    "dts_star": "DTS*",
    "dynamic_cfg_x0": "Dynamic CFG",
    "bon_mcts": "ActDiff",
}

SCORE_FILES = (
    "best_images_vqascore.json",
    "best_images_multi_reward.json",
    "best_images_imagereward.json",
    "best_images_hpsv3.json",
    "best_images_pickscore.json",
    "best_images_hpsv2.json",
)


@dataclass
class PromptResult:
    prompt_index: int
    prompt: str
    image_path: Path | None
    vqascore: float | None
    score_source: str


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _finite(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _valid_image(path: Path | None) -> bool:
    if path is None or not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _rows(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return []
    return (row for row in rows if isinstance(row, dict))


def _load_method(method_dir: Path) -> dict[str, PromptResult]:
    """Load selected-image metadata and VQAScores, preferring post-hoc scores."""
    aggregate_path = method_dir / "aggregate_ddp.json"
    if not aggregate_path.is_file():
        return {}
    aggregate = _json(aggregate_path)
    if str(aggregate.get("search_reward", "vqascore")) != "vqascore":
        return {}

    # DDP rank-local files can reuse prompt_index/slug values.  The exact c0 is
    # the stable cross-method identity and is also the required reward prompt.
    results: dict[str, PromptResult] = {}
    for filename in SCORE_FILES:
        path = method_dir / filename
        if not path.is_file():
            continue
        for row in _rows(_json(path)):
            try:
                prompt_index = int(row.get("prompt_index", -1))
            except (TypeError, ValueError):
                continue
            if prompt_index < 0:
                continue
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                continue
            current = results.get(prompt)
            raw_image = str(row.get("image_path", "")).strip()
            image_path = Path(raw_image).expanduser() if raw_image else None
            score = _finite(row.get("scores", {}).get("vqascore"))
            if current is None:
                results[prompt] = PromptResult(
                    prompt_index=prompt_index,
                    prompt=prompt,
                    image_path=image_path,
                    vqascore=score,
                    score_source=(str(path) if score is not None else ""),
                )
                continue
            if not current.prompt and prompt:
                current.prompt = prompt
            if current.image_path is None and image_path is not None:
                current.image_path = image_path
            if current.vqascore is None and score is not None:
                current.vqascore = score
                current.score_source = str(path)

    # Search logs are the fallback when the post-hoc VQAScore file was not
    # retained.  These runs declare search_reward=vqascore in aggregate_ddp.
    for log_path in sorted((method_dir / "logs").glob("rank_*.jsonl")):
        if log_path.name.endswith("_rewrite_examples.jsonl"):
            continue
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                prompt_index = int(row.get("prompt_index", -1))
            except (TypeError, ValueError):
                continue
            if prompt_index < 0:
                continue
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                continue
            score = _finite(row.get("score"))
            current = results.get(prompt)
            if current is None:
                results[prompt] = PromptResult(
                    prompt_index=prompt_index,
                    prompt=prompt,
                    image_path=None,
                    vqascore=score,
                    score_source=(str(log_path) if score is not None else ""),
                )
                continue
            if not current.prompt and prompt:
                current.prompt = prompt
            if current.vqascore is None and score is not None:
                current.vqascore = score
                current.score_source = str(log_path)
    return results


def _model_run_root(human_eval_root: Path, run_id: str, model_id: str) -> Path:
    if model_id == "sid":
        return (
            human_eval_root
            / "sid_vqascore_algorithm_sweep"
            / run_id
            / "sid"
            / f"run_{run_id}"
        )
    return (
        human_eval_root
        / "vqascore_remaining_models"
        / run_id
        / model_id
        / f"run_{run_id}"
    )


def _safe_method_name(method: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in method)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def select_model(
    model_id: str,
    run_root: Path,
    output_dir: Path,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    method_dirs = sorted(
        path for path in run_root.iterdir()
        if path.is_dir() and (path / "aggregate_ddp.json").is_file()
    ) if run_root.is_dir() else []
    by_method = {path.name: _load_method(path) for path in method_dirs}
    by_method = {method: rows for method, rows in by_method.items() if rows}
    actdiff = by_method.get("bon_mcts", {})
    controls = {
        method: rows for method, rows in by_method.items() if method != "bon_mcts"
    }

    candidates: list[dict[str, Any]] = []
    exclusions: dict[str, int] = {}

    def exclude(reason: str) -> None:
        exclusions[reason] = exclusions.get(reason, 0) + 1

    for prompt, act in sorted(actdiff.items()):
        prompt_index = int(act.prompt_index)
        if act.vqascore is None:
            exclude("actdiff_vqascore_missing")
            continue
        if not act.prompt:
            exclude("actdiff_original_prompt_missing")
            continue
        if not _valid_image(act.image_path):
            exclude("actdiff_image_missing_or_unreadable")
            continue
        valid_controls: list[tuple[str, PromptResult]] = []
        for method, rows in controls.items():
            control = rows.get(prompt)
            if control is None or control.vqascore is None:
                continue
            if control.prompt != act.prompt:
                continue
            if not _valid_image(control.image_path):
                continue
            valid_controls.append((method, control))
        if not valid_controls:
            exclude("no_complete_prompt_matched_control")
            continue
        strongest_method, strongest = max(
            valid_controls,
            key=lambda item: (float(item[1].vqascore), item[0]),
        )
        margin = float(act.vqascore) - float(strongest.vqascore)
        if margin <= 0:
            exclude("actdiff_not_above_strongest_control")
            continue
        baseline = controls.get("baseline", {}).get(prompt)
        baseline_score = (
            float(baseline.vqascore)
            if baseline is not None and baseline.vqascore is not None
            else None
        )
        candidates.append({
            "model_id": model_id,
            "model_name": MODEL_LABELS.get(model_id, model_id),
            "prompt_index": prompt_index,
            "prompt": act.prompt,
            "actdiff_vqascore": float(act.vqascore),
            "strongest_control": strongest_method,
            "strongest_control_label": METHOD_LABELS.get(strongest_method, strongest_method),
            "strongest_control_vqascore": float(strongest.vqascore),
            "delta_vs_strongest": margin,
            "baseline_vqascore": baseline_score,
            "delta_vs_baseline": (
                float(act.vqascore) - baseline_score if baseline_score is not None else None
            ),
            "actdiff_source_image": str(act.image_path),
            "control_source_image": str(strongest.image_path),
            "baseline_source_image": (
                str(baseline.image_path)
                if baseline is not None and _valid_image(baseline.image_path)
                else ""
            ),
            "actdiff_score_source": act.score_source,
            "control_score_source": strongest.score_source,
            "run_root": str(run_root),
        })

    candidates.sort(
        key=lambda row: (-float(row["delta_vs_strongest"]), int(row["prompt_index"]))
    )
    selected = candidates[:top_k]
    model_out = output_dir / model_id
    for rank, row in enumerate(selected, 1):
        prompt_tag = f"p{int(row['prompt_index']):04d}"
        case_dir = model_out / f"rank_{rank:02d}_{prompt_tag}"
        act_out = case_dir / "actdiff.png"
        control_out = case_dir / (
            f"strongest_control_{_safe_method_name(str(row['strongest_control']))}.png"
        )
        _copy(Path(str(row["actdiff_source_image"])), act_out)
        _copy(Path(str(row["control_source_image"])), control_out)
        baseline_out = ""
        if row["baseline_source_image"]:
            target = case_dir / "baseline.png"
            _copy(Path(str(row["baseline_source_image"])), target)
            baseline_out = str(target)
        row["selection_rank"] = rank
        row["actdiff_bundle_image"] = str(act_out)
        row["control_bundle_image"] = str(control_out)
        row["baseline_bundle_image"] = baseline_out

    report = {
        "model_id": model_id,
        "model_name": MODEL_LABELS.get(model_id, model_id),
        "run_root": str(run_root),
        "methods_discovered": sorted(by_method),
        "actdiff_rows": len(actdiff),
        "positive_margin_rows": len(candidates),
        "selected_rows": len(selected),
        "top_k": top_k,
        "exclusions": exclusions,
    }
    return selected, report


def _write_outputs(
    output_dir: Path,
    selected: list[dict[str, Any]],
    reports: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_id", "model_name", "selection_rank", "prompt_index", "prompt",
        "actdiff_vqascore", "strongest_control", "strongest_control_label",
        "strongest_control_vqascore", "delta_vs_strongest", "baseline_vqascore",
        "delta_vs_baseline", "actdiff_source_image", "control_source_image",
        "baseline_source_image", "actdiff_bundle_image", "control_bundle_image",
        "baseline_bundle_image", "actdiff_score_source", "control_score_source",
        "run_root",
    ]
    with (output_dir / "selected_vqa_examples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    (output_dir / "selected_vqa_examples.json").write_text(
        json.dumps({"reports": reports, "selected": selected}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Existing VQAScore-guided high-margin examples",
        "",
        "Selection rule: rank prompts by `ActDiff VQAScore - strongest available control VQAScore`; retain only positive margins and exact original-prompt matches.",
        "",
    ]
    for row in selected:
        lines.extend([
            f"## {row['model_name']} — rank {row['selection_rank']}",
            "",
            f"**Prompt:** {row['prompt']}",
            "",
            f"- ActDiff: {float(row['actdiff_vqascore']):.6f}",
            f"- Strongest control ({row['strongest_control_label']}): {float(row['strongest_control_vqascore']):.6f}",
            f"- Margin: {float(row['delta_vs_strongest']):+.6f}",
            f"- ActDiff image: `{row['actdiff_bundle_image']}`",
            f"- Control image: `{row['control_bundle_image']}`",
            "",
        ])
    (output_dir / "selected_vqa_examples.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-eval-root", required=True, type=Path)
    parser.add_argument("--run-id", default="genai200_v1")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_LABELS),
        choices=sorted(MODEL_LABELS),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-missing-models", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    human_eval_root = args.human_eval_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else human_eval_root / "vqa_actdiff_examples" / args.run_id
    )
    selected: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for model_id in args.models:
        run_root = _model_run_root(human_eval_root, args.run_id, model_id)
        if not run_root.is_dir():
            message = f"missing model run: {run_root}"
            if not args.allow_missing_models:
                raise FileNotFoundError(message)
            print(f"[vqa-examples] WARN {message}")
            reports.append({
                "model_id": model_id,
                "model_name": MODEL_LABELS[model_id],
                "run_root": str(run_root),
                "missing": True,
                "selected_rows": 0,
            })
            continue
        model_selected, report = select_model(
            model_id=model_id,
            run_root=run_root,
            output_dir=output_dir,
            top_k=int(args.top_k),
        )
        selected.extend(model_selected)
        reports.append(report)
        print(
            f"[vqa-examples] {MODEL_LABELS[model_id]}: "
            f"positive={report['positive_margin_rows']} selected={report['selected_rows']}"
        )

    _write_outputs(output_dir, selected, reports)
    print(f"[vqa-examples] bundle: {output_dir}")
    print(f"[vqa-examples] selected rows: {len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
