#!/usr/bin/env python3
"""Audit model/method/prompt/evaluation coverage for a VQAScore sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--expected-prompts", required=True, type=int)
    parser.add_argument(
        "--eval-backends",
        nargs="+",
        default=["imagereward", "hpsv3", "pickscore", "vqascore"],
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    failures = 0
    for model in args.models:
        for method in args.methods:
            if args.run_id:
                method_dir = root / model / f"run_{args.run_id}" / method
                candidates = [method_dir]
            else:
                candidates = sorted((root / model).glob(f"run_*/{method}"))
                method_dir = candidates[-1] if candidates else root / model / "<missing>" / method
            aggregate_path = method_dir / "aggregate_ddp.json"
            generated = 0
            search_reward = ""
            if aggregate_path.is_file():
                aggregate = _read_json(aggregate_path)
                generated = int(aggregate.get("num_samples", 0) or 0)
                search_reward = str(aggregate.get("search_reward", ""))
            row: dict[str, Any] = {
                "model_id": model,
                "method": method,
                "generated": generated,
                "expected": args.expected_prompts,
                "search_reward": search_reward,
                "method_dir": str(method_dir),
            }
            cell_ok = generated == args.expected_prompts and search_reward == "vqascore"
            for backend in args.eval_backends:
                eval_path = method_dir / f"best_images_{backend}.json"
                count = 0
                if eval_path.is_file():
                    payload = _read_json(eval_path)
                    values = payload.get("rows", [])
                    count = len(values) if isinstance(values, list) else 0
                row[f"eval_{backend}"] = count
                cell_ok = cell_ok and count == args.expected_prompts
            row["status"] = "OK" if cell_ok else "MISSING"
            if not cell_ok:
                failures += 1
            rows.append(row)

    fields = [
        "model_id", "method", "generated", "expected", "search_reward",
        *[f"eval_{backend}" for backend in args.eval_backends],
        "status", "method_dir",
    ]
    out_csv = (
        args.out_csv.expanduser().resolve()
        if args.out_csv
        else root / "vqascore_coverage.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_csv.with_name(out_csv.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(out_csv)

    print("model\tmethod\tgenerated\tevals\tstatus")
    for row in rows:
        eval_text = ",".join(
            f"{backend}:{row[f'eval_{backend}']}" for backend in args.eval_backends
        )
        print(
            f"{row['model_id']}\t{row['method']}\t"
            f"{row['generated']}/{row['expected']}\t{eval_text}\t{row['status']}"
        )
    print(f"[coverage] cells={len(rows)} failures={failures} csv={out_csv}")
    return 1 if failures and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
