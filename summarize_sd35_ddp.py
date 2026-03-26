from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SD3.5 DDP run logs.")
    parser.add_argument("--log_dir", required=True, help="Directory containing rank_*.jsonl")
    parser.add_argument("--out_dir", default=None, help="Output dir for aggregate summary (default: log_dir parent)")
    return parser.parse_args()


def load_rows(log_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(log_dir)):
        if not (name.startswith("rank_") and name.endswith(".jsonl")):
            continue
        if name.endswith("_rewrite_examples.jsonl"):
            continue
        path = os.path.join(log_dir, name)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_mode = defaultdict(list)
    by_prompt = defaultdict(dict)
    for row in rows:
        if "mode" not in row:
            continue
        by_mode[row["mode"]].append(row)
        by_prompt[row["prompt_index"]][row["mode"]] = row

    mode_stats: Dict[str, Dict[str, Any]] = {}
    for mode, mode_rows in by_mode.items():
        scores = [r["score"] for r in mode_rows]
        deltas = [r.get("delta_vs_base", 0.0) for r in mode_rows]
        mode_stats[mode] = {
            "count": len(mode_rows),
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "mean_delta_vs_base": float(np.mean(deltas)) if deltas else 0.0,
            "std_delta_vs_base": float(np.std(deltas)) if deltas else 0.0,
        }

    best_mode_counts = defaultdict(int)
    for _, mode_map in by_prompt.items():
        best_mode = max(mode_map.items(), key=lambda kv: kv[1]["score"])[0]
        best_mode_counts[best_mode] += 1

    return {
        "num_rows": len(rows),
        "num_prompts": len(by_prompt),
        "mode_stats": mode_stats,
        "best_mode_counts": dict(best_mode_counts),
    }


def write_outputs(out_dir: str, summary: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "aggregate_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = []
    lines.append("Mode\tCount\tMeanScore\tMeanDeltaVsBase\tStdDeltaVsBase")
    for mode in sorted(summary["mode_stats"]):
        s = summary["mode_stats"][mode]
        lines.append(
            f"{mode}\t{s['count']}\t{s['mean_score']:.6f}\t"
            f"{s['mean_delta_vs_base']:+.6f}\t{s['std_delta_vs_base']:.6f}"
        )
    lines.append("")
    lines.append("Best mode counts:")
    for mode, count in sorted(summary["best_mode_counts"].items()):
        lines.append(f"{mode}: {count}")
    with open(os.path.join(out_dir, "aggregate_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.log_dir)
    summary = summarize(rows)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.log_dir))
    write_outputs(out_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
