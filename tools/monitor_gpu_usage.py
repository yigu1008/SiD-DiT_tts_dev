#!/usr/bin/env python3
"""Sample nvidia-smi and report observed peak GPU memory/utilization."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path


QUERY = (
    "index,uuid,memory.used,memory.total,utilization.gpu,power.draw"
)


def _sample() -> list[dict[str, float | int | str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={QUERY}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append(
            {
                "gpu_index": int(parts[0]),
                "gpu_uuid": parts[1],
                "memory_used_mib": float(parts[2]),
                "memory_total_mib": float(parts[3]),
                "utilization_percent": float(parts[4]),
                "power_watts": None if parts[5] in {"N/A", "[N/A]"} else float(parts[5]),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-indices", nargs="+", type=int, default=None)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "gpu_usage_samples.csv"
    summary_path = output_dir / "gpu_usage_summary.json"
    selected = set(args.gpu_indices) if args.gpu_indices else None
    started = time.time()
    samples: list[dict[str, float | int | str | None]] = []
    print(
        f"[gpu-monitor] sampling every {args.interval:.1f}s -> {samples_path}; "
        "press Ctrl-C to stop and write summary",
        flush=True,
    )
    try:
        while True:
            now = time.time()
            for row in _sample():
                if selected is not None and int(row["gpu_index"]) not in selected:
                    continue
                samples.append({"unix_time": now, **row})
            if args.duration is not None and now - started >= args.duration:
                break
            time.sleep(max(0.2, float(args.interval)))
    except KeyboardInterrupt:
        pass

    fields = [
        "unix_time", "gpu_index", "gpu_uuid", "memory_used_mib",
        "memory_total_mib", "utilization_percent", "power_watts",
    ]
    with samples_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(samples)
    by_gpu: dict[int, list[dict[str, float | int | str | None]]] = defaultdict(list)
    for row in samples:
        by_gpu[int(row["gpu_index"])].append(row)
    gpu_summary = {}
    for index, rows in sorted(by_gpu.items()):
        power_rows = [float(row["power_watts"]) for row in rows if row["power_watts"] is not None]
        duration = max(0.0, float(rows[-1]["unix_time"]) - float(rows[0]["unix_time"]))
        gpu_summary[str(index)] = {
            "samples": len(rows),
            "observed_duration_sec": duration,
            "peak_memory_used_mib": max(float(row["memory_used_mib"]) for row in rows),
            "memory_total_mib": max(float(row["memory_total_mib"]) for row in rows),
            "mean_utilization_percent": (
                sum(float(row["utilization_percent"]) for row in rows) / len(rows)
            ),
            "peak_utilization_percent": max(float(row["utilization_percent"]) for row in rows),
            "mean_power_watts": sum(power_rows) / len(power_rows) if power_rows else None,
            "observed_energy_wh": (
                (sum(power_rows) / len(power_rows)) * duration / 3600.0
                if power_rows
                else None
            ),
        }
    payload = {
        "available": bool(samples),
        "started_unix": started,
        "ended_unix": time.time(),
        "note": (
            "Observed sampled peaks cover only the monitor interval and are not "
            "retroactive CUDA allocator peak measurements."
        ),
        "gpus": gpu_summary,
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[gpu-monitor] wrote {samples_path}")
    print(f"[gpu-monitor] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
