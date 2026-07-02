#!/usr/bin/env python3
"""Fixed (cfg x prompt-variant) grid sampler with a reward recorder.

For every prompt, enumerate the full 2-D rectangle of actions
  variant in {0..n_variants-1}  x  cfg in --cfg_scales
generating one image per cell with that CONSTANT action across all steps
(no search), scoring it against the ORIGINAL prompt, and recording the reward.
The result is a per-prompt reward surface over the cfg axis and the prompt axis
-- the clean, cheap way to see the cfg x prompt interaction (1+1>2): does the
best joint (cfg, variant) beat the best-cfg-alone + best-variant-alone?

Single process, one GPU (no torchrun). Uses the same primitives as the per-step
MI diagnostic: su.load_pipeline / load_reward_model / generate_variants /
encode_variants / run_schedule_actions.

Grid knobs via env: GRID_SEEDS (default --seed), GRID_START/GRID_END (prompt
slice), GRID_SAVE_IMAGES=1. Everything else via the standard su CLI, e.g.:

  REWARD_SERVER_URL=http://localhost:5119 CUDA_VISIBLE_DEVICES=1 \
  python run_cfg_prompt_grid.py --backend sid --steps 4 \
    --prompt_file .../backend_sid.txt --rewrites_file .../rewrites_qwen.json \
    --n_variants 3 --cfg_scales 1.0 1.25 1.5 1.75 2.0 2.25 2.5 \
    --reward_backend composite_3 --out_dir /data/ygu/cfg_prompt_grid/exp1

IMPORTANT: for composite_* / hpsv3 rewards, set REWARD_SERVER_URL to an already-
running reward server (reward_unified routes scoring there). Without it the
reward models load IN-PROCESS on the pipeline GPU and hpsv3 OOMs. --reward_api_base
is unrelated (it's the UnifiedReward LLM API), do NOT use it for the server.
"""
from __future__ import annotations

import csv
import json
import os

import sampling_unified_sd35 as su


def main() -> None:
    args = su.parse_args()  # full sampler config + backend defaults
    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    seeds = [int(s) for s in os.environ.get("GRID_SEEDS", str(int(args.seed))).split()]
    start = int(os.environ.get("GRID_START", "0"))
    end_env = os.environ.get("GRID_END", "")
    save_images = os.environ.get("GRID_SAVE_IMAGES", "1") == "1"   # default ON
    img_dir = os.path.join(out_dir, "images")
    if save_images:
        os.makedirs(img_dir, exist_ok=True)
    msl = int(getattr(args, "max_sequence_length", 256) or 256)
    steps = int(args.steps)
    cfg_bank = [float(c) for c in args.cfg_scales]
    n_variants = int(args.n_variants)

    if not args.prompt_file or not os.path.exists(args.prompt_file):
        raise SystemExit(f"--prompt_file required and must exist (got {args.prompt_file!r})")
    prompts = [ln.strip() for ln in open(args.prompt_file, encoding="utf-8") if ln.strip()]
    end = int(end_env) if end_env else len(prompts)
    sel = list(range(start, min(end, len(prompts))))

    rewrite_cache: dict = {}
    if args.rewrites_file and os.path.exists(args.rewrites_file):
        rewrite_cache = json.load(open(args.rewrites_file, encoding="utf-8"))
        print(f"[grid] loaded rewrites for {len(rewrite_cache)} prompts")

    # composite_* / hpsv3 must be SERVED (loading them locally alongside the
    # pipeline OOMs). Verify the server is up AND actually serving the needed
    # backends up front -- otherwise reward_unified silently falls back to
    # loading them in-process and hpsv3 OOMs mid-run.
    _need = {"composite_3": ["imagereward", "hpsv3", "pickscore"],
             "composite_hpsv3_ir": ["imagereward", "hpsv3"],
             "composite_all4": ["imagereward", "hpsv3", "pickscore", "hpsv2"],
             "hpsv3": ["hpsv3"]}.get(str(args.reward_backend), [])
    if _need:
        url = os.environ.get("REWARD_SERVER_URL", "").strip()
        if not url:
            raise SystemExit(
                f"reward_backend={args.reward_backend} needs a reward server, but REWARD_SERVER_URL "
                "is unset -> reward models load in-process on the pipeline GPU and OOM. Boot one and set it."
            )
        import json as _json
        import urllib.request as _u
        try:
            health = _json.loads(_u.urlopen(f"{url.rstrip('/')}/health", timeout=5).read().decode())
            served = set(health.get("backends", []))
        except Exception as e:
            raise SystemExit(f"reward server at {url} is unreachable ({e}). Boot it before running the grid.")
        missing = [b for b in _need if b not in served]
        if missing:
            raise SystemExit(
                f"reward server at {url} serves {sorted(served)} but is MISSING {missing} "
                f"(needed for {args.reward_backend}). The server likely failed to load {missing} "
                "(often OOM/leaked GPU memory). Free the server GPU (kill leftover procs / reset) and "
                "reboot reward_server.py with --backends hpsv3 imagereward pickscore, then re-run."
            )

    print(f"[grid] backend={args.backend} reward={args.reward_backend} steps={steps} "
          f"cfg_bank={cfg_bank} n_variants={n_variants} seeds={seeds} prompts={len(sel)} "
          f"reward_server={os.environ.get('REWARD_SERVER_URL', '<in-process>')}")
    ctx = su.load_pipeline(args)
    reward_model = su.load_reward_model(args, ctx.device)

    grid_csv = os.path.join(out_dir, "cfg_prompt_grid.csv")
    fields = ["prompt_index", "seed", "variant", "cfg", "reward", "variant_text"]
    written = 0
    with open(grid_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for pi in sel:
            prompt = prompts[pi]
            variants = su.generate_variants(args, prompt, rewrite_cache)
            if len(variants) < n_variants:
                variants = variants + [prompt] * (n_variants - len(variants))
            variants = variants[:n_variants]
            emb = su.encode_variants(ctx, variants, max_sequence_length=msl)
            for seed in seeds:
                for vi in range(n_variants):
                    for cfg in cfg_bank:
                        actions = [(int(vi), float(cfg), 0.0)] * steps
                        res = su.run_schedule_actions(
                            args, ctx, emb, reward_model, prompt, int(seed), actions,
                            deterministic_noise=True, score_prompt=prompt,  # score vs ORIGINAL prompt
                        )
                        w.writerow({
                            "prompt_index": pi, "seed": int(seed), "variant": int(vi),
                            "cfg": float(cfg), "reward": float(res.score),
                            "variant_text": variants[vi][:200],
                        })
                        written += 1
                        if save_images:
                            res.image.save(os.path.join(img_dir, f"p{pi:05d}_s{seed}_v{vi}_cfg{cfg:.2f}.png"))
                        del res
                f.flush()
            print(f"[grid] p{pi:05d} done ({written} cells so far)")
    print(f"[grid] wrote {grid_csv}  ({written} cells)")


if __name__ == "__main__":
    main()
