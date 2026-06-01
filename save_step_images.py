#!/usr/bin/env python3
"""Generate images and dump x_0 after each denoising step.

Standalone — does NOT read rank_*.jsonl or diagnostics.  Just runs a single
deterministic trajectory per prompt with a fixed CFG (or per-step CFG list)
and saves the decoded x_0 image after every step.

Usage examples:

    # Static CFG = 1.5 across all 4 steps:
    python save_step_images.py \
        --backend sid \
        --prompts_file /data/ygu/dpg_bench_prompts.txt \
        --n_prompts 20 --seed 42 --cfg 1.5 \
        --out_dir /data/ygu/runs/dpg_sid_20260531/sid_step_images_fresh

    # Adaptive CFG schedule (one CFG per step) reused for all prompts:
    python save_step_images.py \
        --backend sid \
        --prompts_file /data/ygu/dpg_bench_prompts.txt \
        --n_prompts 20 --seed 42 \
        --cfg_schedule "1.0,1.5,2.0,1.5" \
        --out_dir /data/ygu/runs/dpg_sid_20260531/sid_step_images_fresh

    # Adaptive per-prompt schedule from a CSV: one row per prompt, comma-
    # separated CFG values (must have same number of rows as prompts):
    python save_step_images.py \
        --backend sid \
        --prompts_file /data/ygu/dpg_bench_prompts.txt \
        --n_prompts 20 --seed 42 \
        --cfg_schedule_file /data/ygu/sid_cfgs.csv \
        --out_dir <out>

Output layout:
    <out_dir>/prompt_NNNN/step_K_cfgX.XX.png
    <out_dir>/prompt_NNNN/final.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _read_prompts(p: Path, n: int) -> list[str]:
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if n > 0:
        lines = lines[:n]
    return lines


def _parse_schedule(spec: str | None, default_cfg: float, steps: int) -> list[float]:
    if not spec:
        return [float(default_cfg)] * steps
    vals = [float(x) for x in spec.split(",") if x.strip()]
    if len(vals) < steps:
        vals += [vals[-1]] * (steps - len(vals))
    return vals[:steps]


def _read_per_prompt_schedules(p: Path) -> list[list[float]]:
    out: list[list[float]] = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        out.append([float(x) for x in ln.split(",") if x.strip()])
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--backend", required=True,
                   choices=["sid", "senseflow_large", "sd35_base", "flux_schnell"])
    p.add_argument("--prompts_file", required=True, type=Path)
    p.add_argument("--n_prompts", type=int, default=-1,
                   help="Use only the first N prompts from the file (-1 = all).")
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg", type=float, default=1.5,
                   help="Static CFG used for every step (when --cfg_schedule not set).")
    p.add_argument("--cfg_schedule", default=None,
                   help="Comma-separated CFG values, one per step. Reused for ALL prompts.")
    p.add_argument("--cfg_schedule_file", default=None, type=Path,
                   help="CSV file: one row of comma-separated CFGs per prompt.")
    p.add_argument("--steps", type=int, default=None,
                   help="Override #denoising steps (default: backend default).")
    p.add_argument("--variant", type=int, default=0,
                   help="Prompt variant index used for every step (0 = canonical).")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    args = p.parse_args()

    prompts = _read_prompts(args.prompts_file, args.n_prompts)
    if args.start_index > 0:
        prompts = prompts[args.start_index:]
    if not prompts:
        print(f"[FATAL] no prompts loaded from {args.prompts_file}", file=sys.stderr)
        sys.exit(1)
    print(f"# backend={args.backend}  n_prompts={len(prompts)}  out_dir={args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_prompt_sched: list[list[float]] | None = None
    if args.cfg_schedule_file:
        per_prompt_sched = _read_per_prompt_schedules(args.cfg_schedule_file)
        if len(per_prompt_sched) < len(prompts):
            print(f"[FATAL] cfg_schedule_file has {len(per_prompt_sched)} rows; need >= {len(prompts)}",
                  file=sys.stderr)
            sys.exit(1)

    # Import the replay helper from the existing module — same engine, just
    # called with a CFG schedule we build ourselves.
    if args.backend == "flux_schnell":
        from replay_winner_step_images import _replay_flux as _replay
        n_steps_default = 4
        def call(prompt, seed, actions, prompt_dir):
            _replay(prompt, seed, actions, prompt_dir,
                    height=args.height, width=args.width,
                    steps=args.steps or n_steps_default)
    else:
        from replay_winner_step_images import _replay_sd35 as _replay
        # Per-backend defaults match _replay_sd35 logic.
        defaults = {"sid": 4, "senseflow_large": 4, "sd35_base": 28}
        n_steps_default = defaults.get(args.backend, 4)
        def call(prompt, seed, actions, prompt_dir):
            _replay(args.backend, prompt, seed, actions, prompt_dir,
                    height=args.height, width=args.width,
                    steps=args.steps or n_steps_default)

    n_steps = args.steps or n_steps_default
    for i, prompt_text in enumerate(prompts):
        if per_prompt_sched is not None:
            sched = per_prompt_sched[i]
            if len(sched) < n_steps:
                sched = sched + [sched[-1]] * (n_steps - len(sched))
            sched = sched[:n_steps]
        else:
            sched = _parse_schedule(args.cfg_schedule, args.cfg, n_steps)
        actions = [(args.variant, float(cfg)) for cfg in sched]
        prompt_dir = args.out_dir / f"prompt_{i:04d}"
        print(f"\n  prompt {i}  seed={args.seed}  cfgs={sched}  text=\"{prompt_text[:80]}\"")
        try:
            call(prompt_text, args.seed, actions, prompt_dir)
        except Exception as exc:
            print(f"    [ERR] prompt {i}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

    print(f"\n[done] images under {args.out_dir}")


if __name__ == "__main__":
    main()
