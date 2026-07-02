#!/usr/bin/env python3
"""Per-step CFG intervention grid: bump cfg at ONE step (baseline cfg elsewhere),
record reward -> a (step x cfg) reward rectangle per prompt.

A constant cfg (run_cfg_prompt_grid.py) can miss the effect where cfg=baseline is
best globally yet raising cfg at a specific intermediate step improves reward.
Here the action schedule is baseline everywhere except step k, which gets cfg c:
  actions = [(variant, baseline_cfg, 0)] * steps ; actions[k] = (variant, c, 0)
Plus the all-baseline reference (step=-1). Variant is fixed (GRID_VARIANT, default 0)
so the only thing changing is cfg-at-one-step.

Single process, one GPU. composite_* / hpsv3 must be SERVED (REWARD_SERVER_URL).
Env: GRID_SEEDS, GRID_START/GRID_END, GRID_VARIANT, GRID_SAVE_IMAGES(=1).
"""
from __future__ import annotations

import csv
import json
import os

import sampling_unified_sd35 as su


def main() -> None:
    args = su.parse_args()
    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    seeds = [int(s) for s in os.environ.get("GRID_SEEDS", str(int(args.seed))).split()]
    start = int(os.environ.get("GRID_START", "0"))
    end_env = os.environ.get("GRID_END", "")
    save_images = os.environ.get("GRID_SAVE_IMAGES", "1") == "1"
    variant = int(os.environ.get("GRID_VARIANT", "0"))
    msl = int(getattr(args, "max_sequence_length", 256) or 256)
    steps = int(args.steps)
    cfg_bank = [float(c) for c in args.cfg_scales]
    bcfg = float(args.baseline_cfg)

    _need = {"composite_3": ["imagereward", "hpsv3", "pickscore"],
             "composite_hpsv3_ir": ["imagereward", "hpsv3"],
             "composite_all4": ["imagereward", "hpsv3", "pickscore", "hpsv2"],
             "hpsv3": ["hpsv3"]}.get(str(args.reward_backend), [])
    if _need:
        url = os.environ.get("REWARD_SERVER_URL", "").strip()
        if not url:
            raise SystemExit(f"reward_backend={args.reward_backend} needs REWARD_SERVER_URL (else hpsv3 loads in-process and OOMs).")
        import urllib.request as _u
        try:
            served = set(json.loads(_u.urlopen(f"{url.rstrip('/')}/health", timeout=5).read().decode()).get("backends", []))
        except Exception as e:
            raise SystemExit(f"reward server at {url} unreachable ({e}).")
        miss = [b for b in _need if b not in served]
        if miss:
            raise SystemExit(f"reward server at {url} serves {sorted(served)}, MISSING {miss} for {args.reward_backend}.")

    prompts = [ln.strip() for ln in open(args.prompt_file, encoding="utf-8") if ln.strip()]
    end = int(end_env) if end_env else len(prompts)
    sel = list(range(start, min(end, len(prompts))))
    rewrite_cache: dict = {}
    if args.rewrites_file and os.path.exists(args.rewrites_file):
        rewrite_cache = json.load(open(args.rewrites_file, encoding="utf-8"))

    print(f"[cfg-step] backend={args.backend} reward={args.reward_backend} steps={steps} "
          f"cfg_bank={cfg_bank} baseline_cfg={bcfg} variant={variant} seeds={seeds} prompts={len(sel)}")
    ctx = su.load_pipeline(args)
    reward_model = su.load_reward_model(args, ctx.device)
    img_dir = os.path.join(out_dir, "images")
    if save_images:
        os.makedirs(img_dir, exist_ok=True)

    grid_csv = os.path.join(out_dir, "cfg_step_grid.csv")
    with open(grid_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_index", "seed", "step", "cfg", "reward", "is_baseline"])
        w.writeheader()
        for pi in sel:
            prompt = prompts[pi]
            variants = su.generate_variants(args, prompt, rewrite_cache)
            v = min(variant, len(variants) - 1)
            emb = su.encode_variants(ctx, variants, max_sequence_length=msl)
            for seed in seeds:
                base_actions = [(v, bcfg, 0.0)] * steps
                res0 = su.run_schedule_actions(args, ctx, emb, reward_model, prompt, int(seed),
                                               base_actions, deterministic_noise=True, score_prompt=prompt)
                w.writerow({"prompt_index": pi, "seed": int(seed), "step": -1, "cfg": bcfg,
                            "reward": float(res0.score), "is_baseline": 1})
                if save_images:
                    res0.image.save(os.path.join(img_dir, f"p{pi:05d}_s{seed}_baseline.png"))
                for k in range(steps):
                    for cfg in cfg_bank:
                        if abs(cfg - bcfg) < 1e-9:
                            continue  # == all-baseline, already recorded
                        actions = list(base_actions)
                        actions[k] = (v, float(cfg), 0.0)
                        res = su.run_schedule_actions(args, ctx, emb, reward_model, prompt, int(seed),
                                                      actions, deterministic_noise=True, score_prompt=prompt)
                        w.writerow({"prompt_index": pi, "seed": int(seed), "step": k, "cfg": float(cfg),
                                    "reward": float(res.score), "is_baseline": 0})
                        if save_images:
                            res.image.save(os.path.join(img_dir, f"p{pi:05d}_s{seed}_step{k}_cfg{cfg:.2f}.png"))
                        del res
                f.flush()
            print(f"[cfg-step] p{pi:05d} done")
    print(f"[cfg-step] wrote {grid_csv}")


if __name__ == "__main__":
    main()
