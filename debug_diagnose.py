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

    from diffusers import StableDiffusion3Pipeline, AutoencoderKL
    from diffusers.models.transformers import SD3Transformer2DModel

    dtype = torch.bfloat16
    model_id = "stabilityai/stable-diffusion-3.5-large"

    import gc

    # Aggressively clear GPU from any prior tests
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"  GPU before start: {torch.cuda.memory_allocated(device) / 1e9:.1f} GB allocated, "
              f"{torch.cuda.memory_reserved(device) / 1e9:.1f} GB reserved")

    # ── Step 1: Load pipeline WITHOUT transformer to encode prompts on CPU ──
    # This avoids loading the 8GB base transformer entirely
    print("  Loading SD3.5-Large pipeline (transformer=None, CPU only)...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        transformer=None,
        torch_dtype=dtype,
    )

    print("  Encoding prompt on CPU...")
    enc_out = pipe.encode_prompt(prompt, prompt, prompt)
    pe_sf = enc_out[0].clone().to(dtype=dtype)
    pp_sf = enc_out[-1].clone().to(dtype=dtype)
    enc_uncond_sf = pipe.encode_prompt("", "", "")
    pe_uncond_sf = enc_uncond_sf[0].clone().to(dtype=dtype)

    # Destroy entire pipeline (frees text encoders ~20GB CPU RAM)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Freed pipeline (text encoders only, no transformer was loaded).")

    # ── Step 2: Load SenseFlow transformer → GPU ──
    print(f"  Loading SenseFlow transformer...")
    transformer = SD3Transformer2DModel.from_pretrained(
        "domiso/SenseFlow",
        subfolder="SenseFlow-SD35L/transformer",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    transformer.eval().requires_grad_(False)
    # Verify dtype — force bf16 if checkpoint loaded as fp32
    first_param = next(transformer.parameters())
    if first_param.dtype != dtype:
        print(f"  WARNING: transformer loaded as {first_param.dtype}, converting to {dtype}")
        transformer = transformer.to(dtype=dtype)
    param_gb = sum(p.numel() * p.element_size() for p in transformer.parameters()) / 1e9
    print(f"  Transformer: {type(transformer).__name__} ({param_gb:.1f} GB, dtype={first_param.dtype})")
    if torch.cuda.is_available():
        free_gb = (torch.cuda.get_device_properties(device).total_mem - torch.cuda.memory_allocated(device)) / 1e9
        print(f"  GPU before transformer load: {torch.cuda.memory_allocated(device) / 1e9:.1f} GB used, {free_gb:.1f} GB free")
        if param_gb > free_gb * 0.9:
            print(f"  WARNING: transformer ({param_gb:.1f} GB) may not fit in free GPU ({free_gb:.1f} GB)")
            print(f"  Tip: kill other GPU processes or use CUDA_VISIBLE_DEVICES=<other_gpu>")
    transformer.to(device)
    if torch.cuda.is_available():
        print(f"  GPU after transformer load: {torch.cuda.memory_allocated(device) / 1e9:.1f} GB")

    # ── Step 3: Load VAE separately (~0.3GB) ──
    print(f"  Loading VAE...")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae.eval().requires_grad_(False)
    vae.to(device)
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    from diffusers.image_processor import VaeImageProcessor
    image_processor = VaeImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1))
    vae_shift = getattr(vae.config, "shift_factor", 0.0)
    vae_scaling = vae.config.scaling_factor

    pe_sf, pp_sf = pe_sf.to(device), pp_sf.to(device)
    pe_uncond_sf = pe_uncond_sf.to(device)
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Test 1: Manual loop with SenseFlow sigmas (ground truth equivalent)
    print("\n  --- Test 1: Manual SenseFlow loop (guidance_scale=0.0) ---")
    t0 = time.time()
    latent_c = transformer.config.in_channels
    sf_latents = torch.randn(1, latent_c, 128, 128, device=device, dtype=dtype,
                             generator=torch.Generator(device=device).manual_seed(42))
    sf_sigmas = [1.0, 0.75]
    sf_dx = torch.zeros_like(sf_latents)
    sf_noise = sf_latents.clone()
    for i, sigma in enumerate(sf_sigmas):
        t_val = sigma
        t_flat = torch.full((1,), t_val, device=device, dtype=dtype)
        if i == 0:
            cur_latents = (1.0 - t_val) * sf_dx + t_val * sf_noise
        else:
            cur_noise = torch.randn_like(sf_dx)
            cur_latents = (1.0 - t_val) * sf_dx + t_val * cur_noise
        out = transformer(
            hidden_states=cur_latents,
            encoder_hidden_states=pe_sf,
            pooled_projections=pp_sf,
            timestep=1000.0 * t_flat,
            return_dict=False,
        )[0]
        sf_dx = out  # x0_sampler: output IS x0
    image = vae.decode((sf_dx / vae_scaling) + vae_shift, return_dict=False)[0]
    img_cond = image_processor.postprocess(image, output_type="pil")[0]
    t1 = time.time()
    img_cond.save("debug_senseflow_cond.png")
    print(f"  Generated in {t1 - t0:.1f}s, saved debug_senseflow_cond.png")

    # Test 2: Same but with unconditional (what the old buggy cfg=0.0 did)
    print("\n  --- Test 2: Unconditional loop (simulating old cfg=0.0 bug) ---")
    sf_dx2 = torch.zeros_like(sf_latents)
    sf_noise2 = torch.randn(1, latent_c, 128, 128, device=device, dtype=dtype,
                            generator=torch.Generator(device=device).manual_seed(42))
    for i, sigma in enumerate(sf_sigmas):
        t_val = sigma
        t_flat = torch.full((1,), t_val, device=device, dtype=dtype)
        if i == 0:
            cur_latents = (1.0 - t_val) * sf_dx2 + t_val * sf_noise2
        else:
            cur_noise = torch.randn_like(sf_dx2)
            cur_latents = (1.0 - t_val) * sf_dx2 + t_val * cur_noise

        # Conditional
        out_cond = transformer(
            hidden_states=cur_latents,
            encoder_hidden_states=pe_sf,
            pooled_projections=pp_sf,
            timestep=1000.0 * t_flat,
            return_dict=False,
        )[0]

        # Unconditional
        out_uncond = transformer(
            hidden_states=cur_latents,
            encoder_hidden_states=pe_uncond_sf.expand_as(pe_sf),
            pooled_projections=torch.zeros_like(pp_sf),
            timestep=1000.0 * t_flat,
            return_dict=False,
        )[0]

        diff = (out_cond - out_uncond).abs().mean()
        print(f"    step {i} (sigma={sigma}): |cond - uncond| = {diff:.4f}")
        if diff < 0.01:
            print(f"      WARNING: nearly identical — text conditioning has no effect")

        # Old bug: cfg=0.0 → flow_u + 0*(flow_c - flow_u) = flow_u (unconditional)
        sf_dx2 = out_uncond  # simulating the bug

    image2 = vae.decode((sf_dx2 / vae_scaling) + vae_shift, return_dict=False)[0]
    img_uncond = image_processor.postprocess(image2, output_type="pil")[0]
    img_uncond.save("debug_senseflow_uncond.png")

    # Score both
    try:
        from reward_unified import UnifiedRewardScorer
        scorer = UnifiedRewardScorer(device="cpu", backend=reward_backend)
        score_cond = float(scorer.score(prompt, img_cond))
        score_uncond = float(scorer.score(prompt, img_uncond))
        print(f"\n  SCORES:")
        print(f"    conditional (fixed):     {score_cond:.4f}")
        print(f"    unconditional (old bug): {score_uncond:.4f}")
        print(f"    delta:                   {score_cond - score_uncond:+.4f}")
        if score_uncond < -1.0:
            print(f"    CONFIRMED: unconditional gives garbage ({score_uncond:.4f})")
            print(f"    This was the cfg=0.0 bug — it returned unconditional output")
        if score_cond > 0.0:
            print(f"    CONFIRMED: conditional works ({score_cond:.4f})")
    except Exception as e:
        print(f"\n  Could not score: {e}")

    del transformer, vae
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

    print(f"  Loading SD3.5-Large pipeline (memory-optimized for 48GB)...")
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
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

    # Encode prompt on CPU before moving to GPU (saves ~14GB)
    print("  Encoding prompt on CPU...")
    enc_out = pipe.encode_prompt(prompt, prompt, "")
    pe = enc_out[0].to(dtype=dtype)
    pp = enc_out[-1].to(dtype=dtype)
    enc_uncond = pipe.encode_prompt("", "", "")
    pe_uncond = enc_uncond[0].to(dtype=dtype)
    pp_uncond = enc_uncond[-1].to(dtype=dtype)

    # Free text encoders to save ~14GB GPU
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    pipe.text_encoder_3 = None
    pipe.tokenizer = None
    pipe.tokenizer_2 = None
    pipe.tokenizer_3 = None
    import gc; gc.collect()

    # Move only transformer + VAE to GPU
    pipe.transformer.to(device)
    pipe.vae.to(device)
    pe, pp = pe.to(device), pp.to(device)
    pe_uncond, pp_uncond = pe_uncond.to(device), pp_uncond.to(device)
    print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    latent_c = pipe.transformer.config.in_channels
    vae = pipe.vae
    image_processor = pipe.image_processor
    shift_factor = getattr(vae.config, "shift_factor", 0.0)
    scaling_factor = vae.config.scaling_factor

    def decode(x):
        image = vae.decode((x / scaling_factor) + shift_factor, return_dict=False)[0]
        return image_processor.postprocess(image, output_type="pil")[0]

    def score(img):
        if scorer is None:
            return 0.0
        return float(scorer.score(prompt, img))

    seed = 42
    cfg = 4.5

    # ── Test 1: Manual Euler baseline ──
    print(f"\n  --- Test 1: Manual Euler loop (cfg={cfg}, steps={steps}) ---")
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
    print(f"    manual Euler final:   {score_euler:.4f}")
    print(f"    x0 estimate (last):   {score_x0_last:.4f}")
    print(f"    x0 estimate (mid):    {score_x0_mid:.4f}")
    print(f"    euler vs x0_last:     {score_x0_last - score_euler:+.4f}")

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


def test_sd35_mcts(device, reward_backend="imagereward",
                   prompt="a cat sitting on a windowsill",
                   steps=28, n_sims=20):
    """Run actual MCTS code path with detailed per-sim logging."""
    print("\n" + "=" * 60)
    print("TEST: SD3.5 Base MCTS — full code path diagnosis")
    print("=" * 60)

    import sampling_unified_sd35 as su
    import sampling_sd35_base as sb

    dtype = torch.bfloat16
    model_id = "stabilityai/stable-diffusion-3.5-large"

    from diffusers import StableDiffusion3Pipeline

    print("  Loading pipeline (memory-optimized for 48GB)...")
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
    # Encode prompt on CPU first, then offload text encoders
    pipe.transformer.eval().requires_grad_(False)
    pipe.vae.eval().requires_grad_(False)

    # Load reward on CPU before moving pipeline to GPU
    from reward_unified import UnifiedRewardScorer
    scorer = UnifiedRewardScorer(device="cpu", backend=reward_backend)
    print(f"  Reward: {scorer.describe()}")

    # Encode prompt while text encoders are still on CPU (saves ~14GB GPU)
    print("  Encoding prompt on CPU...")
    _enc_out = pipe.encode_prompt(prompt, prompt, "")
    _cached_pe = _enc_out[0].to(dtype=dtype)
    _cached_pp = _enc_out[-1].to(dtype=dtype)
    _enc_uncond = pipe.encode_prompt("", "", "")
    _cached_pe_uncond = _enc_uncond[0].to(dtype=dtype)
    _cached_pp_uncond = _enc_uncond[-1].to(dtype=dtype)

    # Free text encoders (~14GB)
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    pipe.text_encoder_3 = None
    pipe.tokenizer = None
    pipe.tokenizer_2 = None
    pipe.tokenizer_3 = None
    import gc; gc.collect(); torch.cuda.empty_cache()

    # Now move only transformer + VAE to GPU (~16GB + 0.3GB)
    pipe.transformer.to(device)
    pipe.vae.to(device)
    print(f"  GPU memory after offload: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    latent_c = pipe.transformer.config.in_channels
    ctx = su.PipelineContext(pipe=pipe, device=device, dtype=dtype, latent_c=latent_c)

    # Move cached embeddings to GPU
    _cached_pe = _cached_pe.to(device)
    _cached_pp = _cached_pp.to(device)
    _cached_pe_uncond = _cached_pe_uncond.to(device)
    _cached_pp_uncond = _cached_pp_uncond.to(device)

    # Build args manually (mimics sampling_sd35_base defaults)
    args = argparse.Namespace(
        backend="sd35_base",
        model_id=model_id,
        steps=steps,
        cfg_scales=[3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0],
        baseline_cfg=4.5,
        correction_strengths=[0.0],
        x0_sampler=False,
        euler_sampler=True,
        sigmas=None,
        height=1024,
        width=1024,
        time_scale=1000.0,
        n_sims=n_sims,
        ucb_c=1.41,
        n_variants=1,
        seed=42,
        gen_batch_size=1,
        # Lookahead args
        lookahead_mode="rollout_tree_prior",
        lookahead_u_t_def="latent_delta_rms",
        lookahead_tau=0.35,
        lookahead_c_puct=1.20,
        lookahead_u_ref=0.0,
        lookahead_w_cfg=1.0,
        lookahead_w_variant=0.25,
        lookahead_w_q=0.20,
        lookahead_w_explore=0.05,
        lookahead_cfg_width_min=3,
        lookahead_cfg_width_max=7,
        lookahead_cfg_anchor_count=2,
        lookahead_min_visits_for_center=3,
        lookahead_log_action_topk=12,
        # Dynamic CFG args
        mcts_cfg_mode="adaptive",
        mcts_cfg_root_bank=[3.5, 5.0, 7.0],
        mcts_cfg_anchors=[3.5, 7.0],
        mcts_cfg_step_anchor_count=2,
        mcts_cfg_min_parent_visits=3,
        mcts_cfg_round_ndigits=6,
        mcts_cfg_log_action_topk=12,
        # Interp
        mcts_interp_family="none",
        mcts_n_interp=1,
        # Qwen
        rewrites_file=None,
    )

    seed = 42
    variants = [prompt]

    # Build EmbeddingContext from pre-computed embeddings (text encoders already freed)
    emb = su.EmbeddingContext(
        cond_text=[_cached_pe],
        cond_pooled=[_cached_pp],
        uncond_text=_cached_pe_uncond,
        uncond_pooled=_cached_pp_uncond,
    )

    # ── Step 0: Baseline ──
    print(f"\n  --- Baseline (cfg=4.5, steps={steps}) ---")
    ctx.nfe = 0
    base_img, base_score = su.run_baseline(
        args, ctx, emb, scorer, prompt, seed, cfg_scale=4.5,
    )
    base_img.save("debug_mcts_baseline.png")
    print(f"  Baseline score: {base_score:.4f}  NFE: {ctx.nfe}")

    # ── Step 1: Run MCTS with verbose logging ──
    print(f"\n  --- MCTS (n_sims={n_sims}, steps={steps}) ---")
    ctx.nfe = 0
    ctx.correction_nfe = 0

    sched = su.step_schedule(ctx.device, dtype, steps, euler=True, shift=3.0)
    latents0 = su.make_latents(ctx, seed, args.height, args.width, dtype)

    # Quick check: what does scoring the initial noise give us?
    noise_img = su.decode_to_pil(ctx, latents0)
    noise_score = float(scorer.score(prompt, noise_img))
    print(f"  Pure noise score: {noise_score:.4f} (sanity check)")

    # Run one full Euler trajectory with each CFG to establish reference
    print(f"\n  --- CFG sweep (single trajectory each) ---")
    cfg_scores = {}
    for test_cfg in [3.5, 4.5, 5.5, 7.0]:
        lat = latents0.clone()
        for i, (t_flat, t_4d, dt) in enumerate(sched):
            flow = su.transformer_step(args, ctx, lat, emb, 0, t_flat, test_cfg)
            lat = lat + dt * flow
        img = su.decode_to_pil(ctx, lat)
        s = float(scorer.score(prompt, img))
        cfg_scores[test_cfg] = s
        print(f"    cfg={test_cfg:.1f}: score={s:.4f}")

    # Now run actual MCTS
    print(f"\n  --- Running MCTS ---")
    ctx.nfe = 0
    t0 = time.time()
    mcts_result = sb.run_mcts_sd35base(
        args, ctx, emb, scorer, prompt, variants, seed,
    )
    t1 = time.time()
    mcts_result.image.save("debug_mcts_result.png")
    print(f"\n  MCTS completed in {t1 - t0:.1f}s")
    print(f"  MCTS score:     {mcts_result.score:.4f}")
    print(f"  Baseline score: {base_score:.4f}")
    print(f"  Delta:          {mcts_result.score - base_score:+.4f}")
    print(f"  NFE:            {ctx.nfe}")

    # Print action trajectory
    print(f"\n  MCTS action trajectory:")
    for i, (v, c, cs) in enumerate(mcts_result.actions):
        print(f"    step {i:2d}: variant={v} cfg={c:.2f}")

    # Print diagnostics if available
    diag = mcts_result.diagnostics or {}
    if "history" in diag:
        print(f"\n  MCTS convergence history:")
        for h in diag["history"]:
            print(f"    sim {h['sim']:3d}: best={h['best_score']:.4f} "
                  f"root_visits={h['root_visits']}")

    if "u_t_stats" in diag:
        u = diag["u_t_stats"]
        print(f"\n  u_t stats: mean={u['mean']:.4f} std={u['std']:.4f} "
              f"min={u['min']:.4f} max={u['max']:.4f}")

    if "final_cfg_trajectory" in diag:
        print(f"  Final CFG trajectory: {diag['final_cfg_trajectory']}")
    if "exploit_cfg_trajectory" in diag:
        print(f"  Exploit CFG trajectory: {diag['exploit_cfg_trajectory']}")

    # ── Verdict ──
    print(f"\n  {'=' * 50}")
    print(f"  VERDICT:")
    if mcts_result.score > base_score:
        print(f"  OK: MCTS improved over baseline by {mcts_result.score - base_score:+.4f}")
    else:
        print(f"  PROBLEM: MCTS WORSE than baseline by {mcts_result.score - base_score:+.4f}")
        best_cfg_val = max(cfg_scores, key=cfg_scores.get)
        print(f"  Best fixed-CFG was {best_cfg_val} ({cfg_scores[best_cfg_val]:.4f})")
        if mcts_result.score < cfg_scores[best_cfg_val]:
            print(f"  MCTS is even worse than best fixed-CFG — search is broken")
        print(f"  Possible causes:")
        print(f"    1. Scoring mismatch (x0 vs Euler latent)")
        print(f"    2. Too few simulations (n_sims={n_sims})")
        print(f"    3. Action space too large / CFG bank issues")
        print(f"    4. Rollout randomness dominating signal")

    del pipe
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Diagnose SenseFlow + SD3.5 base issues")
    parser.add_argument("--test", choices=["senseflow", "sd35base", "sd35mcts", "all"], default="all")
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

    import gc

    if args.test in ("sd35base", "all"):
        test_sd35_base(args.device, args.reward_backend, args.prompt,
                       steps=args.steps, n_sims=args.n_sims, skip_reward=args.skip_reward)
        gc.collect(); torch.cuda.empty_cache()

    if args.test in ("sd35mcts", "all"):
        test_sd35_mcts(args.device, args.reward_backend, args.prompt,
                       steps=args.steps, n_sims=args.n_sims)
        gc.collect(); torch.cuda.empty_cache()

    if args.test in ("senseflow", "all"):
        test_senseflow(args.device, args.reward_backend, args.prompt)
        gc.collect(); torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("DONE — check debug_*.png files for visual inspection")
    print("=" * 60)


if __name__ == "__main__":
    main()
