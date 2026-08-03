#!/usr/bin/env python3
"""Check whether one post-hoc reward file matches current generated records."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluate_best_images_multi_reward import collect_records  # noqa: E402


def _key(value: Any) -> tuple[int, str, int, str]:
    image_path = value.image_path if hasattr(value, "image_path") else value.get("image_path", "")
    return (
        int(value.prompt_index if hasattr(value, "prompt_index") else value.get("prompt_index", -1)),
        str(value.slug if hasattr(value, "slug") else value.get("slug", "")),
        int(value.sample_index if hasattr(value, "sample_index") else value.get("sample_index", 0)),
        str(Path(str(image_path)).expanduser().resolve()),
    )


def check_complete(
    *,
    layout: str,
    method_out: Path,
    method: str,
    backend: str,
    eval_json: Path,
    expected_count: int,
) -> tuple[bool, str]:
    if not eval_json.is_file():
        return False, "evaluation JSON is missing"
    try:
        payload = json.loads(eval_json.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        return False, f"invalid evaluation JSON: {type(exc).__name__}"
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return False, "evaluation rows is not a list"
    if len(rows) != expected_count:
        return False, f"evaluation row count {len(rows)} != {expected_count}"

    records, missing = collect_records(layout, str(method_out), method)
    if missing:
        return False, f"current generation has {len(missing)} missing image paths"
    if len(records) != expected_count:
        return False, f"current record count {len(records)} != {expected_count}"
    current = {_key(record): record for record in records}
    existing = {_key(row): row for row in rows}
    if len(current) != len(records) or len(existing) != len(rows):
        return False, "duplicate image keys"
    if current.keys() != existing.keys():
        return False, "image keys differ from current generation"

    for key, record in current.items():
        row = existing[key]
        if str(row.get("prompt", "")) != str(record.prompt):
            return False, f"prompt mismatch at {key}"
        recorded_path = Path(str(row.get("image_path", ""))).expanduser().resolve()
        current_path = Path(str(record.image_path)).expanduser().resolve()
        if recorded_path != current_path:
            return False, f"image-path mismatch at {key}"
        value = row.get("scores", {}).get(backend)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return False, f"missing/non-finite {backend} score at {key}"
    return True, "keys, prompts, paths, and scores match current generation"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", required=True, choices=["sd35", "flux", "sana"])
    parser.add_argument("--method-out", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--eval-json", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    args = parser.parse_args()
    complete, reason = check_complete(
        layout=args.layout,
        method_out=args.method_out.expanduser().resolve(),
        method=args.method,
        backend=args.backend,
        eval_json=args.eval_json.expanduser().resolve(),
        expected_count=args.expected_count,
    )
    state = "complete" if complete else "stale"
    print(f"[posthoc-resume] {state}: {args.method}/{args.backend}: {reason}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
