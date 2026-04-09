from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

import sampling_unified_sd35 as su
import sd35_ddp_experiment as base

AXIS_REWRITE_ORDER = ("faithful", "composition", "subject", "background", "detail", "style")
AXIS_REWRITE_STYLES = {
    "composition": (
        "Axis=composition. Change framing/crop/viewpoint/camera distance/spatial arrangement. "
        "Do not significantly change subject identity or background semantics."
    ),
    "subject": (
        "Axis=subject. Sharpen visible subject attributes: face, hair, clothing, pose, held props. "
        "Do not mainly change framing or background."
    ),
    "background": (
        "Axis=background. Enrich environment/layout/depth/background objects. "
        "Keep subject identity and shot type stable."
    ),
    "detail": (
        "Axis=detail. Add local textures/materials/fine visible attributes "
        "(wrinkles, reflections, embroidery, folds, strands). "
        "Do not change composition or large scene structure."
    ),
    "style": (
        "Axis=style. Change rendering treatment/palette/mood/photographic style "
        "while keeping core scene semantics stable."
    ),
}


def _parse_axis_flags(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--axis_target_size",
        type=int,
        default=6,
        help="Number of axis variants including faithful/original. Clamped to [1,6].",
    )
    return parser.parse_known_args(argv)


def _axis_target_size(args: argparse.Namespace) -> int:
    size = int(getattr(args, "axis_target_size", 6))
    if size <= 0:
        size = 6
    return max(1, min(size, len(AXIS_REWRITE_ORDER)))


def _dedup_variants(values: list[str], fallback: str, max_items: int | None = None) -> list[str]:
    out: list[str] = []
    for value in values:
        vv = su.sanitize_rewrite_text(value, fallback)
        if vv not in out:
            out.append(vv)
        if max_items is not None and len(out) >= int(max_items):
            break
    return out if out else [fallback]


def _axis_heuristic_rewrite(prompt: str, axis: str) -> str:
    suffix_map = {
        "faithful": "Preserve the same scene and identity with minimal wording cleanup.",
        "composition": "Adjust framing and viewpoint to clarify spatial arrangement and depth.",
        "subject": "Clarify visible subject traits: facial details, hair, clothing, pose, and held props.",
        "background": "Enrich environment details, layout depth cues, and consistent background objects.",
        "detail": "Add concrete fine details: textures, material reflections, seams, folds, and strands.",
        "style": "Adjust rendering style, palette, and mood while keeping scene semantics unchanged.",
    }
    return su.sanitize_rewrite_text(f"{prompt}. {suffix_map.get(axis, 'Refine visible details.')}", prompt)


def _read_axis_cache_entry(entry: Any, prompt: str, target: int) -> list[str] | None:
    if isinstance(entry, list):
        return _dedup_variants([str(v) for v in entry], prompt, max_items=target)

    if isinstance(entry, dict):
        # Direct payload with pre-rendered variants list.
        raw_variants = entry.get("variants")
        if isinstance(raw_variants, list):
            return _dedup_variants([str(v) for v in raw_variants], prompt, max_items=target)

        # Direct axis mapping: {"faithful": "...", "composition": "...", ...}
        axis_map = {str(k).strip().lower(): str(v) for k, v in entry.items() if isinstance(k, str)}
        if any(ax in axis_map for ax in AXIS_REWRITE_ORDER):
            vals = [prompt]
            for axis in AXIS_REWRITE_ORDER[1:]:
                if axis in axis_map:
                    vals.append(axis_map[axis])
            return _dedup_variants(vals, prompt, max_items=target)

        # Alternate payload shape: {"records":[{"axis":"...","prompt":"..."}]}
        recs = entry.get("records")
        if isinstance(recs, list):
            by_axis: dict[str, str] = {}
            for rec in recs:
                if not isinstance(rec, dict):
                    continue
                axis = str(rec.get("axis", "")).strip().lower()
                text = str(rec.get("prompt", "")).strip()
                if axis and text:
                    by_axis[axis] = text
            if any(ax in by_axis for ax in AXIS_REWRITE_ORDER):
                vals = [prompt]
                for axis in AXIS_REWRITE_ORDER[1:]:
                    if axis in by_axis:
                        vals.append(by_axis[axis])
                return _dedup_variants(vals, prompt, max_items=target)

    return None


def _generate_axis_variants(args: argparse.Namespace, prompt: str) -> list[str]:
    target = _axis_target_size(args)
    vals: list[str] = [prompt]
    if bool(getattr(args, "no_qwen", False)):
        for axis in AXIS_REWRITE_ORDER[1:]:
            vals.append(_axis_heuristic_rewrite(prompt, axis))
        return _dedup_variants(vals, prompt, max_items=target)

    for axis in AXIS_REWRITE_ORDER[1:]:
        style = AXIS_REWRITE_STYLES.get(axis, "")
        rewritten = su.sanitize_rewrite_text(su.qwen_rewrite(args, prompt, style), prompt)
        if rewritten == prompt:
            rewritten = _axis_heuristic_rewrite(prompt, axis)
        vals.append(rewritten)

    dedup = _dedup_variants(vals, prompt, max_items=None)
    if len(dedup) < target:
        for axis in AXIS_REWRITE_ORDER[1:]:
            cand = _axis_heuristic_rewrite(prompt, axis)
            if cand not in dedup:
                dedup.append(cand)
            if len(dedup) >= target:
                break
    return dedup[:target]


def _generate_variants_axis(args: argparse.Namespace, prompt: str, cache: dict[str, Any]) -> list[str]:
    target = _axis_target_size(args)
    entry = cache.get(prompt)
    if entry is not None:
        from_cache = _read_axis_cache_entry(entry, prompt, target)
        if from_cache is not None:
            return _dedup_variants(from_cache, prompt, max_items=target)
    return _generate_axis_variants(args, prompt)


def _make_patched_parse_args(
    original_parse_args: Callable[[], argparse.Namespace],
) -> Callable[[], argparse.Namespace]:
    def _patched_parse_args() -> argparse.Namespace:
        axis_args, remaining = _parse_axis_flags(sys.argv[1:])
        original_argv = sys.argv
        sys.argv = [original_argv[0]] + remaining
        try:
            args = original_parse_args()
        finally:
            sys.argv = original_argv

        if not hasattr(args, "x0_sampler"):
            setattr(args, "x0_sampler", False)
        setattr(args, "axis_target_size", int(axis_args.axis_target_size))
        return args

    return _patched_parse_args


def main() -> None:
    original_parse_args = base.parse_args
    original_generate_variants = base.generate_variants
    original_su_generate_variants = su.generate_variants

    base.parse_args = _make_patched_parse_args(original_parse_args)
    base.generate_variants = _generate_variants_axis
    su.generate_variants = _generate_variants_axis
    try:
        base.main()
    finally:
        base.parse_args = original_parse_args
        base.generate_variants = original_generate_variants
        su.generate_variants = original_su_generate_variants


if __name__ == "__main__":
    main()
