#!/usr/bin/env python3
"""
Diagnostic script for two issues:
  1) SenseFlow producing garbage (reward ~ -2.x)
  2) SD3.5 base MCTS deteriorating vs baseline

Usage:
  # Test SenseFlow only (needs ~35GB VRAM)
  python debug_diagnose.py --test senseflow --device cuda

  # Test SD3.5 base only
  python debug_diagnose.py --test sd35base --device cuda

  # Test both
  python debug_diagnose.py --test all --device cuda

  # Quick CPU smoke test (no real images, just checks flow)
  python debug_diagnose.py --test sd35base --device cpu --steps 2 --skip_reward
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_transformer_step_cfg():
    """Verify transformer_step handles cfg=0.0, 1.0, 4.5 correctly."""
    print("\n" + "=" * 60)
    print("TEST: transformer_step CFG routing")
    print("=" * 60)

    import sampling_unified_sd35 as su

    # Check source code to verify cfg=0.0 goes to conditional-only branch
    import inspect
    src = inspect.getsource(su.transformer_step)
    if "cfg == 0.0" in src or "cfg == 1.0 or cfg == 0.0" in src:
        print("  OK: cfg=0.0 routes to conditional-only branch")
    else:
        print("  BUG: cfg=0.0 falls through to CFG branch (returns unconditional!)")
        # Show the relevant line
        for i, line in enumerate(src.split("\n")):
            if "cfg ==" in line or "cfg >" in line:
                print(f"    line {i}: {line.strip()}")
    print()


def test_senseflow(device, reward_backend="imagereward", prompt="a cat sitting on a windowsill"):
    """Test SenseFlow SD3.5-Large: load, generate, score."""
    print("\n" + "=" * 60)
    print("TEST: SenseFlow SD3.5-Large")
    print("=" * 60)

    from diffusers import StableDiffusion3Pipeline
    from diffusers.models.transformers import SD3Transformer2DModel

    dtype = torch.bfloat16

    # Load transformer from SenseFlow
    print("  Loading SenseFlow transformer...")
    transformer = SD3Transformer2DModel.from_pretrained(
        "domiso/SenseFlow",
        subfolder="SenseFlow-SD35L/transformer",
        torch_dtype=dtype,
    )
    print(f"  Transformer loaded: {type(transformer).__name__}")

    # Load pipeline
    print("  Loading SD3.5-Large pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        transformer=transformer,
        torch_dtype=dtype,
    ).to(device)

    pipe.transformer.eval().requires_grad_(False)
    pipe.vae.eval().requires_grad_(False)

    # Test 1: Use diffusers pipeline directly (ground truth)
    print("\n  --- Test 1: diffusers pipe() call (guidance_scale=0.0) ---")
    gen = torch.Generator(device=device).manual_seed(42)
    t0 = time.time()
    result = pipe(
        prompt,
        num_inference_steps=2,
        guidance_scale=0.0,
        generator=gen,
    )
    t1 = time.time()
    img_pipe = result.images[0]
    img_pipe.save("debug_senseflow_pipe.png")
    print(f"  Generated in {t1 - t0:.1f}s, saved debug_senseflow_pipe.png")

    # Test 2: Manual SenseFlow loop (our code path)
    print("\n  --- Test 2: Manual denoising loop (x0_sampler=True, cfg=0.0) ---")
    sigmas = [1.0, 0.75]
    latent_c = pipe.transformer.config.in_channels
    latents = torch.randn(1, latent_c, 128, 128, device=device, dtype=dtype,
                          generator=torch.Generator(device=device).manual_seed(42))

    # Encode prompt
    enc_out = pipe.encode_prompt(prompt, prompt, "")
    pe, pp = enc_out[0], enc_out[-1]
    print(f"  encode_prompt returned {len(enc_out)} values")
    print(f"  pe shape: {pe.shape}, pp shape: {pp.shape}")

    # Also get unconditional embeddings for comparison
    enc_uncond = pipe.encode_prompt("", "", "")
    pe_uncond = enc_uncond[0]

    dx = torch.zeros_like(latents)
    noise = latents.clone()

    for i, sigma in enumerate(sigmas):
        t_val = sigma
        t_flat = torch.full((1,), t_val, device=device, dtype=dtype)

        # Re-noise
        if i == 0:
            cur_latents = (1.0 - t_val) * dx + t_val * noise
        else:
            cur_noise = torch.randn_like(dx)
            cur_latents = (1.0 - t_val) * dx + t_val * cur_noise

        print(f"\n  Step {i}: sigma={sigma}")
        print(f"    latents: mean={cur_latents.mean():.4f} std={cur_latents.std():.4f}")

        # Conditional-only forward (cfg=0.0 should route here)
        out_cond = pipe.transformer(
            hidden_states=cur_latents,
            encoder_hidden_states=pe,
            pooled_projections=pp,
            timestep=1000.0 * t_flat,
            return_dict=False,
        )[0]
        print(f"    cond output: mean={out_cond.mean():.4f} std={out_cond.std():.4f}")

        # Unconditional forward (what cfg=0.0 was wrongly returning before)
        out_uncond = pipe.transformer(
            hidden_states=cur_latents,
            encoder_hidden_states=pe_uncond.expand_as(pe),
            pooled_projections=torch.zeros_like(pp),
            timestep=1000.0 * t_flat,
            return_dict=False,
        )[0]
        print(f"    uncond output: mean={out_uncond.mean():.4f} std={out_uncond.std():.4f}")

        # Compare
        diff = (out_cond - out_uncond).abs().mean()
        print(f"    |cond - uncond|: {diff:.4f}")
        if diff < 0.01:
            print(f"    WARNING: cond and uncond are nearly identical!")

        # x0_sampler: output IS x0
        dx = out_cond  # NOT latents - t*flow

    # Decode
    shift = getattr(pipe.vae.config, "shift_factor", 0.0)
    image = pipe.vae.decode(
        (dx / pipe.vae.config.scaling_factor) + shift,
        return_dict=False,
    )[0]
    img_manual = pipe.image_processor.postprocess(image, output_type="pil")[0]
    img_manual.save("debug_senseflow_manual.png")
    print(f"\n  Saved debug_senseflow_manual.png")

    # Score both
    try:
        from reward_unified import UnifiedRewardScorer
        scorer = UnifiedRewardScorer(device="cpu", backend=reward_backend)
        score_pipe = float(scorer.score(prompt, img_pipe))
        score_manual = float(scorer.score(prompt, img_manual))
        print(f"\n  SCORES:")
        print(f"    pipe() call:     {score_pipe:.4f}")
        print(f"    manual loop:     {score_manual:.4f}")
        print(f"    delta:           {score_manual - score_pipe:+.4f}")
        if score_pipe < -1.0:
            print(f"    PROBLEM: pipe() score is {score_pipe:.4f} — likely garbage image")
        if score_manual < -1.0:
            print(f"    PROBLEM: manual score is {score_manual:.4f} — likely garbage image")
    except Exception as e:
        print(f"\n  Could not score: {e}")

    del pipe, transformer
    torch.cuda.empty_cache()


def test_sd35_base(device, reward_backend="imagereward", prompt="a cat sitting on a windowsill",
                   steps=28, n_sims=10, skip_reward=False):
    """Test SD3.5 base: baseline vs MCTS, diagnose scoring mismatch."""
    print("\n" + "=" * 60)
    print("TEST: SD3.5 Base — Baseline vs MCTS diagnosis")
    print("=" * 60)

    import sampling_unified_sd35 as su

    dtype = torch.bfloat16
    model_id = "stabilityai/stable-diffusion-3.5-large"

    if device == "cpu" and skip_reward:
        print("  CPU + skip_reward: testing flow only (no model loading)")
        # Verify step_schedule produces correct sigmas
        sched = su.step_schedule("cpu", dtype, steps, euler=True, shift=3.0)
        print(f"\n  Step schedule (euler=True, shift=3.0, steps={steps}):")
        for i, (t_flat, t_4d, dt) in enumerate(sched):
            sigma = float(t_flat.item())
            print(f"    step {i:2d}: sigma={sigma:.6f}  dt={dt:.6f}")
        print(f"  Final sigma should be near 0: {float(sched[-1][0].item()):.6f}")

        # Check dt sums to ~ -first_sigma
        total_dt = sum(dt for _, _, dt in sched)
        first_sigma = float(sched[0][0].item())
        print(f"  sum(dt)={total_dt:.6f}, -first_sigma={-first_sigma:.6f}")
        print(f"  Match: {'YES' if abs(total_dt + first_sigma) < 0.01 else 'NO — PROBLEM!'}")
        return

    from diffusers import StableDiffusion3Pipeline

    print(f"  Loading SD3.5-Large pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    pipe.transformer.eval().requires_grad_(False)
    pipe.vae.eval().requires_grad_(False)

    # Load reward
    scorer = None
    if not skip_reward:
        try:
            from reward_unified import UnifiedRewardScorer
            scorer = UnifiedRewardScorer(device="cpu", backend=reward_backend)
            print(f"  Reward model: {scorer.describe()}")
        except Exception as e:
            print(f"  Could not load reward: {e}")

    # Encode prompt
    enc_out = pipe.encode_prompt(prompt, prompt, "")
    pe, pp = enc_out[0], enc_out[-1]
    # Uncond
    enc_uncond = pipe.encode_prompt("", "", "")
    pe_uncond, pp_uncond = enc_uncond[0], enc_uncond[-1]

    latent_c = pipe.transformer.config.in_channels
    shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0)
    scaling_factor = pipe.vae.config.scaling_factor

    def decode(x):
        image = pipe.vae.decode((x / scaling_factor) + shift_factor, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")[0]

    def score(img):
        if scorer is None:
            return 0.0
        return float(scorer.score(prompt, img))

    seed = 42
    cfg = 4.5

    # ── Test 1: diffusers pipe() call (ground truth) ──
    print(f"\n  --- Test 1: pipe() call (cfg={cfg}, steps={steps}) ---")
    gen = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    result = pipe(prompt, num_inference_steps=steps, guidance_scale=cfg, generator=gen)
    t1 = time.time()
    img_pipe = result.images[0]
    img_pipe.save("debug_sd35base_pipe.png")
    score_pipe = score(img_pipe)
    print(f"  Generated in {t1 - t0:.1f}s, score={score_pipe:.4f}")

    # ── Test 2: manual Euler loop ──
    print(f"\n  --- Test 2: Manual Euler loop (cfg={cfg}, steps={steps}) ---")
    sched = su.step_schedule(device, dtype, steps, euler=True, shift=3.0)
    latents = torch.randn(1, latent_c, 128, 128, device=device, dtype=dtype,
                          generator=torch.Generator(device=device).manual_seed(seed))

    x0_estimates = []  # track x0 estimates at each step
    for i, (t_flat, t_4d, dt) in enumerate(sched):
        # CFG: uncond + cond
        flow_both = pipe.transformer(
            hidden_states=torch.cat([latents, latents]),
            encoder_hidden_states=torch.cat([pe_uncond.expand_as(pe), pe]),
            pooled_projections=torch.cat([pp_uncond.expand_as(pp), pp]),
            timestep=1000.0 * torch.cat([t_flat, t_flat]),
            return_dict=False,
        )[0]
        flow_u, flow_c = flow_both.chunk(2)
        flow = flow_u + cfg * (flow_c - flow_u)

        # x0 estimate at this step
        x0_est = latents - float(t_flat.item()) * flow
        x0_estimates.append(x0_est.clone())

        # Euler step
        latents = latents + dt * flow

        if i % 7 == 0 or i == steps - 1:
            sigma = float(t_flat.item())
            lat_std = float(latents.std().item())
            x0_std = float(x0_est.std().item())
            print(f"    step {i:2d}: sigma={sigma:.4f} dt={dt:.6f} "
                  f"lat_std={lat_std:.4f} x0_std={x0_std:.4f}")

    img_euler = decode(latents)
    img_euler.save("debug_sd35base_euler.png")
    score_euler = score(img_euler)

    # Also decode from x0 estimate at last step
    img_x0_last = decode(x0_estimates[-1])
    img_x0_last.save("debug_sd35base_x0_last.png")
    score_x0_last = score(img_x0_last)

    # Decode x0 at mid-step
    mid = steps // 2
    img_x0_mid = decode(x0_estimates[mid])
    img_x0_mid.save(f"debug_sd35base_x0_step{mid}.png")
    score_x0_mid = score(img_x0_mid)

    print(f"\n  SCORES:")
    print(f"    pipe() call:          {score_pipe:.4f}")
    print(f"    manual Euler final:   {score_euler:.4f}")
    print(f"    x0 estimate (last):   {score_x0_last:.4f}")
    print(f"    x0 estimate (mid):    {score_x0_mid:.4f}")
    print(f"    pipe vs euler delta:  {score_euler - score_pipe:+.4f}")
    print(f"    euler vs x0_last:     {score_x0_last - score_euler:+.4f}")

    if abs(score_euler - score_pipe) > 0.3:
        print(f"    WARNING: pipe() and manual Euler differ significantly!")
        print(f"    This means our Euler implementation may be wrong.")

    if abs(score_x0_last - score_euler) > 0.1:
        print(f"    NOTE: x0_last and Euler final differ by {score_x0_last - score_euler:+.4f}")
        print(f"    If MCTS was scoring x0 estimates, this mismatch explains deterioration.")

    # ── Test 3: per-step x0 vs Euler comparison ──
    print(f"\n  --- Test 3: x0 estimate quality over steps ---")
    for i in [0, steps // 4, steps // 2, 3 * steps // 4, steps - 1]:
        img_xi = decode(x0_estimates[i])
        s = score(img_xi)
        sigma = float(sched[i][0].item())
        print(f"    step {i:2d} (sigma={sigma:.4f}): x0_score={s:.4f}")

    # ── Test 4: Simple MCTS-like test ──
    print(f"\n  --- Test 4: Baseline vs random CFG search (mini-MCTS check) ---")
    cfgs_to_test = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]
    best_cfg = None
    best_score = -float("inf")
    for test_cfg in cfgs_to_test:
        lat = torch.randn(1, latent_c, 128, 128, device=device, dtype=dtype,
                          generator=torch.Generator(device=device).manual_seed(seed))
        for i, (t_flat, t_4d, dt) in enumerate(sched):
            flow_both = pipe.transformer(
                hidden_states=torch.cat([lat, lat]),
                encoder_hidden_states=torch.cat([pe_uncond.expand_as(pe), pe]),
                pooled_projections=torch.cat([pp_uncond.expand_as(pp), pp]),
                timestep=1000.0 * torch.cat([t_flat, t_flat]),
                return_dict=False,
            )[0]
            flow_u, flow_c = flow_both.chunk(2)
            flow = flow_u + test_cfg * (flow_c - flow_u)
            lat = lat + dt * flow

        img = decode(lat)
        s = score(img)
        mark = ""
        if s > best_score:
            best_score = s
            best_cfg = test_cfg
            mark = " <- best"
        print(f"    cfg={test_cfg:.1f}: score={s:.4f}{mark}")

    print(f"\n  Best fixed-CFG: {best_cfg} with score={best_score:.4f}")
    print(f"  Baseline (cfg=4.5): {score_euler:.4f}")
    print(f"  If MCTS < baseline, the search is hurting — check action selection logic.")

    del pipe
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Diagnose SenseFlow + SD3.5 base issues")
    parser.add_argument("--test", choices=["senseflow", "sd35base", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reward_backend", default="imagereward")
    parser.add_argument("--prompt", default="a cat sitting on a windowsill")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--n_sims", type=int, default=10)
    parser.add_argument("--skip_reward", action="store_true")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Always run the source code check (no GPU needed)
    test_transformer_step_cfg()

    if args.test in ("senseflow", "all"):
        test_senseflow(args.device, args.reward_backend, args.prompt)

    if args.test in ("sd35base", "all"):
        test_sd35_base(args.device, args.reward_backend, args.prompt,
                       steps=args.steps, n_sims=args.n_sims, skip_reward=args.skip_reward)

    print("\n" + "=" * 60)
    print("DONE — check debug_*.png files for visual inspection")
    print("=" * 60)


if __name__ == "__main__":
    main()
