#!/usr/bin/env python3
"""Replay each prompt's MCTS winning trajectory and save the decoded x_0
image after every denoising step.

For each prompt:
  1. Read winning action sequence from lookahead_node_logs in rank_*.jsonl
  2. Re-run that exact (variant, cfg) sequence step-by-step on the pipeline
  3. Decode and save x_0 after each step

Output: <out_dir>/prompt_<idx>/step_<k>_cfg<X.XX>.png + final.png

Usage (cluster, with /mnt/data mounted):
    python replay_winner_step_images.py \
        --run_root /mnt/data/v-yigu/all_in_one/flux-newcfg/composite/flux_schnell/seed42 \
        --method bon_mcts \
        --backend flux_schnell \
        --prompts 0 5 17 \
        --out_dir /mnt/data/v-yigu/_tree_archive/step_images_flux

Supported backends:
    sid, senseflow_large, sd35_base   -> uses sampling_unified_sd35.encode_variants + transformer_step
    flux_schnell                       -> uses sampling_flux_unified.flux_transformer_step

Note: this script LOADS THE FULL PIPELINE on the GPU.  Each prompt takes
~10-20s on an A100 (4-step models) or ~60-90s (sd35_base 28-step).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


def _load_row(run_root: Path, method: str, prompt_idx: int) -> dict | None:
    """Locate the JSONL row for (method, prompt_idx).

    Tries multiple glob patterns to handle different on-disk layouts:
      A) <root>/<method>/<run_ts>/rank_*.jsonl          (cluster default)
      B) <root>/run_<ts>/<method>/logs/rank_*.jsonl     (a6000 wrapper)
      C) <root>/**/rank_*.jsonl                         (catch-all)
    """
    globs = [
        f"**/{method}/**/rank_*.jsonl",
        f"**/rank_*.jsonl",
    ]
    seen: set = set()
    candidates: list[Path] = []
    for g in globs:
        for jp in run_root.glob(g):
            if jp in seen:
                continue
            seen.add(jp)
            candidates.append(jp)
    if not candidates:
        print(f"    [debug] no rank_*.jsonl under {run_root} (tried {globs})")
        return None
    for jp in candidates:
        try:
            with open(jp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if int(row.get("prompt_index", -1)) != int(prompt_idx):
                        continue
                    if row.get("mode") and row["mode"] != method:
                        continue
                    return row
        except Exception:
            continue
    return None


def _load_diag(run_root: Path, method: str, prompt_idx: int) -> tuple[dict | None, str | None]:
    """Return (diag-like-dict, prompt_text) for the given prompt.

    The dict is augmented with a synthetic 'actions' field (list of [v, cfg, cs])
    so callers can fall back to the chosen action sequence even when the run
    didn't emit lookahead_node_logs (e.g. sd35_ddp_experiment.py path).
    """
    row = _load_row(run_root, method, prompt_idx)
    if row is None:
        return None, None
    d = row.get("diagnostics") or {}
    if isinstance(d.get("bon_mcts"), dict):
        d = d["bon_mcts"]
    # Always attach the per-step action sequence + winning seed if present.
    if "actions" not in d and isinstance(row.get("actions"), list):
        d = {**d, "actions": row["actions"]}
    if "winner_seed" not in d and row.get("seed") is not None:
        d = {**d, "winner_seed": int(row["seed"])}
    return d, row.get("prompt", "")


def _winning_action_sequence(diag: dict) -> list[tuple[int, float]]:
    """For each step, return the (variant_idx, cfg) MCTS picked.

    Two sources, in priority order:
      1. lookahead_node_logs (per-sim per-step decisions) -- best, but only
         emitted by sampling_unified_sd35_lookahead_reweighting.run_mcts_lookahead.
      2. SearchResult.actions [(v, cfg, cs), ...] -- emitted by ALL search
         methods (greedy/mcts/sop/...), so this is the universal fallback.
    """
    logs = diag.get("lookahead_node_logs") or diag.get("node_logs") or []
    if logs:
        visit: Counter = Counter()
        for r in logs:
            act = r.get("chosen_action") or {}
            step = int(r.get("step_idx", -1))
            if step < 0:
                continue
            key = (step, int(act.get("variant_idx", 0)), round(float(act.get("cfg", 0.0)), 4))
            visit[key] += 1
        by_step: dict[int, list[tuple]] = {}
        for (step, v, cfg), n in visit.items():
            by_step.setdefault(step, []).append((n, v, cfg))
        out: list[tuple[int, float]] = []
        for step in sorted(by_step.keys()):
            cands = sorted(by_step[step], key=lambda t: t[0], reverse=True)
            n, v, cfg = cands[0]
            out.append((int(v), float(cfg)))
        if out:
            return out
    # Fallback: SearchResult.actions field.
    actions = diag.get("actions") or []
    out2: list[tuple[int, float]] = []
    for a in actions:
        try:
            v, cfg = int(a[0]), float(a[1])
            out2.append((v, cfg))
        except Exception:
            continue
    return out2


# ---- Per-backend replay helpers ------------------------------------------

def _replay_sd35(
    backend: str, prompt: str, seed: int, actions: list[tuple[int, float]],
    out_dir: Path, height: int = 1024, width: int = 1024, steps: int | None = None,
) -> None:
    import torch
    from PIL import Image as _Image  # noqa
    from sampling_unified_sd35 import (
        encode_variants, load_pipeline, make_latents, step_schedule,
        _prepare_latents, transformer_step, _apply_step, _final_decode_tensor,
        decode_to_pil, _BACKEND_CONFIGS,
    )

    if backend not in _BACKEND_CONFIGS:
        raise SystemExit(f"unknown sd35 backend: {backend}")
    cfg_backend = _BACKEND_CONFIGS[backend]
    args = argparse.Namespace(
        backend=backend,
        model_id=cfg_backend["model_id"],
        transformer_id=cfg_backend.get("transformer_id"),
        transformer_subfolder=cfg_backend.get("transformer_subfolder"),
        sigmas=cfg_backend.get("sigmas"),
        dtype=cfg_backend["dtype"],
        gen_batch_size=cfg_backend.get("gen_batch_size", 1),
        steps=steps or (len(cfg_backend["sigmas"]) if cfg_backend.get("sigmas") else 4),
        height=height, width=width,
        euler_sampler=cfg_backend.get("euler_sampler", False),
        x0_sampler=cfg_backend.get("x0_sampler", False),
        baseline_cfg=cfg_backend.get("baseline_cfg", 1.0),
    )
    ctx = load_pipeline(args)
    # Encode at least the variants we need (variant indices may be > 0).
    n_variants = max(1, max((v for v, _ in actions), default=0) + 1)
    variants = [prompt] + [prompt] * (n_variants - 1)  # plain replication
    emb = encode_variants(ctx, variants)

    use_euler = bool(args.euler_sampler)
    latents = make_latents(ctx, seed, args.height, args.width, emb.cond_text[0].dtype)
    dx = torch.zeros_like(latents)
    sched = step_schedule(ctx.device, latents.dtype, args.steps, args.sigmas, euler=use_euler)

    out_dir.mkdir(parents=True, exist_ok=True)
    for j, (t_flat, t_4d, step_dt) in enumerate(sched):
        v, cfg = actions[min(j, len(actions) - 1)]
        latents = _prepare_latents(latents, dx, torch.randn_like(latents), t_4d, j, use_euler)
        flow = transformer_step(args, ctx, latents, emb, int(v), t_flat, float(cfg))
        latents, dx = _apply_step(latents, flow, dx, t_4d, step_dt, use_euler, args.x0_sampler)
        # Save x_0 after this step.
        img = decode_to_pil(ctx, dx)
        img.save(out_dir / f"step_{j}_cfg{cfg:.2f}_v{v}.png")
        print(f"    step {j}: cfg={cfg:.2f} v={v} -> {out_dir/f'step_{j}_cfg{cfg:.2f}_v{v}.png'}", flush=True)
    final = decode_to_pil(ctx, _final_decode_tensor(latents, dx, use_euler))
    final.save(out_dir / "final.png")
    print(f"    final -> {out_dir/'final.png'}", flush=True)


def _replay_flux(
    prompt: str, seed: int, actions: list[tuple[int, float]],
    out_dir: Path, height: int = 1024, width: int = 1024, steps: int = 4,
) -> None:
    import torch
    from sampling_flux_unified import (
        load_flux_context, encode_prompt_for_flux, make_initial_latents,
        flux_transformer_step, _pred_x0, _compute_dt, _final_decode_tensor,
        decode_to_pil, build_t_schedule,
    )

    args = argparse.Namespace(
        backend="flux",
        model_id="black-forest-labs/FLUX.1-schnell",
        steps=steps, height=height, width=width,
        baseline_guidance_scale=0.0,
        euler_sampler=False,
        x0_sampler=False,
        sigmas=None,
    )
    ctx = load_flux_context(args)
    n_variants = max(1, max((v for v, _ in actions), default=0) + 1)
    embeds = [encode_prompt_for_flux(ctx, prompt) for _ in range(n_variants)]

    init = make_initial_latents(ctx, seed, args.height, args.width, batch_size=1)
    dx = torch.zeros_like(init)
    latents = init.clone()
    t_values = build_t_schedule(args.steps, getattr(args, "sigmas", None))

    out_dir.mkdir(parents=True, exist_ok=True)
    for j, t_val in enumerate(t_values):
        v, cfg = actions[min(j, len(actions) - 1)]
        t_4d = torch.tensor(float(t_val), device=ctx.device, dtype=init.dtype).view(1, 1, 1, 1)
        if j > 0:
            noise = torch.randn_like(init)
            latents = (1.0 - t_4d) * dx + t_4d * noise
        flow = flux_transformer_step(ctx, latents, embeds[int(v)], float(t_val), float(cfg))
        dx = _pred_x0(latents, t_4d, flow, args.x0_sampler)
        # Save x_0 after this step.
        img = decode_to_pil(ctx, dx)
        img.save(out_dir / f"step_{j}_cfg{cfg:.2f}_v{v}.png")
        print(f"    step {j}: cfg={cfg:.2f} v={v} -> {out_dir/f'step_{j}_cfg{cfg:.2f}_v{v}.png'}", flush=True)
    final = decode_to_pil(ctx, _final_decode_tensor(latents, dx, False))
    final.save(out_dir / "final.png")
    print(f"    final -> {out_dir/'final.png'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_root", required=True, type=Path)
    p.add_argument("--method", default="bon_mcts")
    p.add_argument("--backend", required=True,
                   choices=["sid", "senseflow_large", "sd35_base", "flux_schnell"])
    p.add_argument("--prompts", nargs="+", type=int, default=None)
    p.add_argument("--prompt_range", default=None)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    args = p.parse_args()

    if args.prompts is None and args.prompt_range:
        a, b = args.prompt_range.split(":")
        args.prompts = list(range(int(a), int(b)))
    if args.prompts is None:
        args.prompts = list(range(0, 10))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"# backend = {args.backend}  out_dir = {args.out_dir}")

    for pi in args.prompts:
        diag, prompt_text = _load_diag(args.run_root, args.method, pi)
        if diag is None:
            print(f"  prompt {pi}: [no diagnostics found]")
            continue
        actions = _winning_action_sequence(diag)
        if not actions:
            print(f"  prompt {pi}: [no winning action sequence -- old run without lookahead_node_logs]")
            continue
        winner_seed = int(diag.get("winner_seed", 42))
        print(f"\n  prompt {pi}  seed={winner_seed}  text=\"{prompt_text}\"")
        print(f"    actions: {actions}")
        prompt_dir = args.out_dir / f"prompt_{pi:04d}"
        if args.backend == "flux_schnell":
            _replay_flux(prompt_text, winner_seed, actions, prompt_dir,
                         height=args.height, width=args.width)
        else:
            _replay_sd35(args.backend, prompt_text, winner_seed, actions, prompt_dir,
                         height=args.height, width=args.width)


if __name__ == "__main__":
    main()
