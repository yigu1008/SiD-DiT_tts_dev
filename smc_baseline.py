"""
smc_reward_sana.py
==================
Inference-time reward-tilted generation baselines for SANA 0.6B (flow-match).
Runs three methods on the same prompts/seeds and logs per-prompt reward scores.

Methods
-------
1. best_of_n     — generate N images per prompt, keep highest-reward one.
                   (Best-of-N; trivial baseline)

2. svdd_pm       — SVDD-PM (Li et al. 2024, arXiv:2408.08252).
                   At each denoising step branch into M particles,
                   score each via one-step Tweedie x̂₀, resample to 1
                   particle via softmax(r/α) weights.
                   Cost: M × T transformer forwards per image.

3. smc           — Feynman-Kac SMC with reward-tilted intermediate targets
                   (Wu et al. NeurIPS 2023 / DAS arXiv:2501.05803).
                   K particles run full denoising in parallel.
                   At each step t compute importance weight
                       w_t^(k) ∝ exp(λ_t · r̂(x̂₀_t^(k)))
                   with geometric tempering λ_t = (1+γ)^t − 1.
                   Systematic resampling when ESS < ess_threshold × K.
                   Select highest-weight particle at end.
                   Cost: K × T transformer forwards per image.

Flow-match x̂₀ formula (SANA/FLUX style):
    x̂₀ = x_t - σ_t · v_θ(x_t, c, t)
    where σ_t = pipe.scheduler.sigmas[t_idx]

All methods use the same CFG forward pass and VAE decode as the training
scripts (pipeline-faithful, with timestep_scale and variance split).

Usage
-----
# Compare all three on 32 prompts
python smc_reward_sana.py \\
    --prompt_file prompts.txt \\
    --methods best_of_n svdd_pm smc \\
    --n_prompts 32 --n_seeds 1 \\
    --bon_n 8 --svdd_m 8 --smc_k 8 \\
    --output_dir ./smc_outputs

# Fast smoke test (1 prompt each)
python smc_reward_sana.py \\
    --prompt "A majestic lion at sunset." \\
    --methods best_of_n svdd_pm smc \\
    --bon_n 4 --svdd_m 4 --smc_k 4 \\
    --output_dir ./smc_test

# With optional neg embed from ReNeg training
python smc_reward_sana.py \\
    --prompt_file prompts.txt \\
    --neg_ckpt neg_embed_ckpts/neg_embed_best.pt \\
    --methods smc --smc_k 8 \\
    --output_dir ./smc_reneg

References
----------
- Wu et al. (2023) "Practical and Asymptotically Exact Conditional Sampling
  in Diffusion Models" (TDS) NeurIPS 2023.
- Li et al. (2024) "Derivative-Free Guidance ... with Soft Value-Based
  Decoding" (SVDD) arXiv:2408.08252.
- Kim et al. (2025) "Test-time Alignment of Diffusion Models without Reward
  Over-optimization" (DAS) arXiv:2501.05803.
- Singhal et al. (2025) / F-SMC: "Navigating the Exploration-Exploitation
  Tradeoff in Inference-Time Scaling of Diffusion Models" arXiv:2508.12361.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from diffusers import SanaPipeline
from diffusers.utils import logging as dlogging
dlogging.set_verbosity_error()


# ─────────────────────────────────────────────────────────────────────────────
# Reward model
# ─────────────────────────────────────────────────────────────────────────────

def load_reward_model(device):
    try:
        import hpsv2 as hm
        m = hm.utils.initialize_model().to(device).eval()
        for p in m.parameters(): p.requires_grad_(False)
        return m, "hpsv2"
    except ImportError:
        pass
    try:
        import ImageReward as RM
        m = RM.load("ImageReward-v1.0", device=str(device)).eval()
        for p in m.parameters(): p.requires_grad_(False)
        return m, "imagereward"
    except ImportError:
        pass
    return None, None


_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073])
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

def _prep(imgs, device):
    """imgs: [B,3,H,W] in [0,1]"""
    x = F.interpolate(imgs, size=(224, 224), mode="bilinear", align_corners=False)
    return (x - _CLIP_MEAN.to(device).view(1,3,1,1)) / \
               _CLIP_STD.to(device).view(1,3,1,1)


@torch.no_grad()
def score_batch(reward_model, backend, prompts: list[str],
                imgs_01: torch.Tensor, device) -> torch.Tensor:
    """
    Score a batch of images.
    imgs_01: [B,3,H,W] float in [0,1]
    Returns: [B] float tensor of reward scores.
    """
    B = imgs_01.shape[0]
    if backend == "hpsv2":
        import open_clip
        toks = open_clip.tokenize(prompts).to(device)
        tf   = reward_model.encode_text(toks)
        tf   = tf / tf.norm(dim=-1, keepdim=True)
        vf   = reward_model.encode_image(_prep(imgs_01, device))
        vf   = vf / vf.norm(dim=-1, keepdim=True)
        return (vf * tf).sum(dim=-1)
    elif backend == "imagereward":
        tok = reward_model.blip.tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=35, return_tensors="pt",
        ).to(device)
        # ImageReward score_gard expects preprocessed images
        scores = []
        for i in range(B):
            img_t = _prep(imgs_01[i:i+1], device)
            s = reward_model.score_gard(
                tok.input_ids[i:i+1], tok.attention_mask[i:i+1], img_t)
            scores.append(s.squeeze())
        return torch.stack(scores)
    # fallback: mean pixel value (useless but won't crash)
    return imgs_01.mean(dim=[1,2,3])


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline helpers (identical to training scripts)
# ─────────────────────────────────────────────────────────────────────────────

def reset_scheduler(pipe, denoise_steps, device):
    pipe.scheduler.set_timesteps(denoise_steps, device=device)
    pipe.scheduler.model_outputs = [None] * pipe.scheduler.config.solver_order
    if hasattr(pipe.scheduler, '_step_index'):
        pipe.scheduler._step_index = None
    if hasattr(pipe.scheduler, '_begin_index'):
        pipe.scheduler._begin_index = None


def cfg_forward(pipe, x_t, t, cond_e, cond_m, neg_e, neg_m,
                cfg_scale, latent_channels):
    """
    x_t:   [B, C, H, W]  — may have B > 1 (all particles at once)
    Returns CFG velocity prediction [B, C, H, W] float32.
    """
    B      = x_t.shape[0]
    mdtype = pipe.transformer.dtype
    ts     = getattr(pipe.transformer.config, "timestep_scale", 1.0)
    t_in   = (t * ts).expand(2 * B)

    # broadcast cond/neg to match batch B
    ce = cond_e.expand(B, -1, -1).to(mdtype)
    cm = cond_m.expand(B, -1)
    ne = neg_e.expand(B, -1, -1).to(mdtype)
    nm = neg_m.expand(B, -1)

    out = pipe.transformer(
        hidden_states          = torch.cat([x_t, x_t]).to(mdtype),
        encoder_hidden_states  = torch.cat([ne, ce]),
        encoder_attention_mask = torch.cat([nm, cm]),
        timestep               = t_in,
        return_dict            = False,
    )[0].float()

    if pipe.transformer.config.out_channels // 2 == latent_channels:
        out = out.chunk(2, dim=1)[0]

    out_u, out_c = out.chunk(2, dim=0)
    return out_u + cfg_scale * (out_c - out_u)


def predict_x0(v, x_t, sigma):
    """Flow-match one-step x̂₀: x̂₀ = x_t - σ·v"""
    return x_t - sigma * v


@torch.no_grad()
def vae_decode_batch(pipe, latents: torch.Tensor) -> torch.Tensor:
    """latents: [B,C,H,W] → imgs [B,3,H',W'] in [0,1]"""
    vdt = next(pipe.vae.parameters()).dtype
    z   = (latents / pipe.vae.config.scaling_factor).to(vdt)
    imgs = pipe.vae.decode(z).sample.float()
    return (imgs.clamp(-1, 1) + 1.0) / 2.0


@torch.no_grad()
def encode_prompt(pipe, prompt: str, device):
    (ce, cm, ne, nm) = pipe.encode_prompt(
        prompt=[prompt],
        do_classifier_free_guidance=True,
        negative_prompt=[""],
        num_images_per_prompt=1,
        device=device,
        clean_caption=False,
        max_sequence_length=300,
    )
    return ce.detach(), cm.detach(), ne.detach(), nm.detach()


def load_neg_embed(path, device, dtype):
    """Load ReNeg checkpoint → (neg_e [1,L,D], neg_m [1,L])"""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    return ckpt["neg_embeds"].to(device).to(dtype), \
           ckpt["neg_mask"].to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Systematic resampling (lower variance than multinomial)
# ─────────────────────────────────────────────────────────────────────────────

def systematic_resample(weights: torch.Tensor) -> torch.Tensor:
    """
    weights: [K] normalized (sum=1).
    Returns indices [K] via systematic resampling.
    """
    K   = weights.shape[0]
    cdf = torch.cumsum(weights, dim=0)
    u   = (torch.rand(1, device=weights.device) + torch.arange(K, device=weights.device)) / K
    return torch.searchsorted(cdf, u).clamp(0, K - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Method 1: Best-of-N
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def best_of_n(pipe, prompt, cond_e, cond_m, neg_e, neg_m,
              reward_model, reward_backend,
              device, dtype, cfg_scale, denoise_steps,
              N: int, seed: int) -> tuple[Image.Image, float, dict]:
    """
    Generate N independent images, return the highest-reward one.
    Cost: N × T transformer forwards.
    """
    C  = pipe.transformer.config.in_channels
    H  = W = pipe.transformer.config.sample_size
    lc = C

    torch.manual_seed(seed)
    latents = torch.randn(N, C, H, W, device=device, dtype=dtype)

    reset_scheduler(pipe, denoise_steps, device)
    timesteps = pipe.scheduler.timesteps.clone()
    sigmas    = pipe.scheduler.sigmas.clone()

    # Run N trajectories in parallel
    for i, t in enumerate(timesteps):
        v   = cfg_forward(pipe, latents, t, cond_e, cond_m,
                          neg_e, neg_m, cfg_scale, lc)
        res = pipe.scheduler.step(v, t, latents, return_dict=False)
        latents = (res[0] if isinstance(res, (tuple, list)) else res.prev_sample)

    imgs_01 = vae_decode_batch(pipe, latents)                   # [N,3,H,W]
    scores  = score_batch(reward_model, reward_backend,
                          [prompt] * N, imgs_01, device)        # [N]

    best_idx = scores.argmax().item()
    best_img = imgs_01[best_idx]
    from torchvision.transforms.functional import to_pil_image
    pil = to_pil_image(best_img.cpu().clamp(0, 1))

    stats = {
        "reward_best":   scores[best_idx].item(),
        "reward_mean":   scores.mean().item(),
        "reward_std":    scores.std().item(),
        "reward_all":    scores.tolist(),
    }
    return pil, stats["reward_best"], stats


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: SVDD-PM  (Li et al. 2024)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def svdd_pm(pipe, prompt, cond_e, cond_m, neg_e, neg_m,
            reward_model, reward_backend,
            device, dtype, cfg_scale, denoise_steps,
            M: int, alpha: float, seed: int,
            resample_start_frac: float = 0.3,
) -> tuple[Image.Image, float, dict]:
    """
    SVDD-PM: at each step branch x_t into M particles, score each via
    one-step Tweedie x̂₀, then resample to 1 via softmax(scores/alpha).

    resample_start_frac: don't resample in the first (1-frac) fraction of
    steps — early x̂₀ predictions are blurry and reward scores unreliable
    (Singhal / F-SMC insight). Default 0.3 → start resampling at 30% of T.

    Cost: M × T transformer forwards.
    """
    C  = pipe.transformer.config.in_channels
    H  = W = pipe.transformer.config.sample_size
    lc = C

    torch.manual_seed(seed)
    # Single particle trajectory; we branch at each step
    x_t = torch.randn(1, C, H, W, device=device, dtype=dtype)

    reset_scheduler(pipe, denoise_steps, device)
    timesteps = pipe.scheduler.timesteps.clone()
    sigmas    = pipe.scheduler.sigmas.clone()

    T         = len(timesteps)
    start_idx = int((1.0 - resample_start_frac) * T)

    step_rewards = []

    for i, t in enumerate(timesteps):
        sigma = sigmas[i].item()

        if i < start_idx or M == 1:
            # Before resampling window: just denoise normally
            v   = cfg_forward(pipe, x_t, t, cond_e, cond_m,
                               neg_e, neg_m, cfg_scale, lc)
            res = pipe.scheduler.step(v, t, x_t, return_dict=False)
            x_t = (res[0] if isinstance(res, (tuple, list)) else res.prev_sample)
        else:
            # Branch: M copies of x_t
            x_branch = x_t.expand(M, -1, -1, -1).clone()
            # Add small independent noise perturbation so particles diverge
            # (SVDD-PM: use the model's stochastic step rather than perturbing,
            #  but for a deterministic solver we add ε ~ N(0, σ_branch²))
            sigma_branch = sigma * 0.05
            x_branch = x_branch + sigma_branch * torch.randn_like(x_branch)

            # Score each branch via one-step x̂₀
            v_branch = cfg_forward(pipe, x_branch, t, cond_e, cond_m,
                                   neg_e, neg_m, cfg_scale, lc)
            x0_branch = predict_x0(v_branch, x_branch.float(), sigma)
            imgs_01   = vae_decode_batch(pipe, x0_branch)
            scores    = score_batch(reward_model, reward_backend,
                                    [prompt] * M, imgs_01, device)

            # Softmax resampling → select 1 particle
            w       = torch.softmax(scores / alpha, dim=0)
            idx     = torch.multinomial(w, num_samples=1).item()
            x_t     = x_branch[idx:idx+1]
            step_rewards.append(scores[idx].item())

            # Denoise selected particle one step
            v   = cfg_forward(pipe, x_t, t, cond_e, cond_m,
                               neg_e, neg_m, cfg_scale, lc)
            res = pipe.scheduler.step(v, t, x_t, return_dict=False)
            x_t = (res[0] if isinstance(res, (tuple, list)) else res.prev_sample)

    imgs_01 = vae_decode_batch(pipe, x_t)
    final_score = score_batch(reward_model, reward_backend,
                              [prompt], imgs_01, device)[0].item()
    from torchvision.transforms.functional import to_pil_image
    pil = to_pil_image(imgs_01[0].cpu().clamp(0, 1))

    stats = {
        "reward_final":   final_score,
        "reward_steps":   step_rewards,
        "alpha":          alpha,
        "resample_start": resample_start_frac,
    }
    return pil, final_score, stats


# ─────────────────────────────────────────────────────────────────────────────
# Method 3: SMC with reward-tilted resampling
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def smc_reward(pipe, prompt, cond_e, cond_m, neg_e, neg_m,
               reward_model, reward_backend,
               device, dtype, cfg_scale, denoise_steps,
               K: int, gamma: float, ess_threshold: float,
               seed: int,
               resample_start_frac: float = 0.3,
               select: str = "best",
) -> tuple[Image.Image, float, dict]:
    """
    Feynman-Kac SMC with reward-tilted intermediate targets.
    (Wu/TDS 2023 + DAS 2501.05803 tempering scheme)

    Algorithm:
        Initialise K particles x_T^(1..K) ~ N(0,I)
        log_w^(k) = 0  for all k

        for t = T-1 ... 0:
            1. Propagate: x_t^(k) = denoise_step(x_{t+1}^(k))  for all k
            2. Score:     r̂_t^(k) = reward(VAE(x̂₀(x_t^(k))))
            3. Temper:    λ_t = (1+γ)^(T-t) − 1   [geometric, grows with t]
                          log_w^(k) += λ_t · r̂_t^(k)
            4. Resample:  if ESS < ess_threshold*K and t > start_resample:
                              indices = systematic_resample(softmax(log_w))
                              x_t ← x_t[indices]; log_w ← 0

        Select: particle with highest cumulative log_w (or max final reward).

    Geometric tempering (DAS §3): λ_t = (1+γ)^(step_from_end) − 1
        γ=0.05 → λ grows slowly, mild tilt throughout
        γ=0.2  → λ grows fast, aggressive tilt at end

    resample_start_frac: skip resampling for first (1-frac) of steps.
    Default 0.3 matches F-SMC and DAS recommendation (early x̂₀ noisy).

    Cost: K × T transformer forwards (same as BoN with N=K, but with
    diversity maintained via resampling rather than independent rollouts).
    """
    C  = pipe.transformer.config.in_channels
    H  = W = pipe.transformer.config.sample_size
    lc = C

    torch.manual_seed(seed)
    # Initialise K particles
    particles = torch.randn(K, C, H, W, device=device, dtype=dtype)
    log_w     = torch.zeros(K, device=device)

    reset_scheduler(pipe, denoise_steps, device)
    timesteps = pipe.scheduler.timesteps.clone()
    sigmas    = pipe.scheduler.sigmas.clone()

    T         = len(timesteps)
    start_idx = int((1.0 - resample_start_frac) * T)

    resample_count  = 0
    ess_history     = []
    reward_history  = []

    for i, t in enumerate(timesteps):
        sigma = sigmas[i].item()

        # ── 1. Propagate all K particles ──────────────────────────────────
        v   = cfg_forward(pipe, particles, t, cond_e, cond_m,
                          neg_e, neg_m, cfg_scale, lc)
        res = pipe.scheduler.step(v, t, particles, return_dict=False)
        particles = (res[0] if isinstance(res, (tuple, list))
                     else res.prev_sample).clone()

        if i < start_idx:
            continue   # no scoring/resampling in early steps

        # ── 2. Score via one-step x̂₀ ──────────────────────────────────────
        # Re-use v already computed above (same timestep)
        x0_pred = predict_x0(v, particles.float(), sigma)       # [K,C,H,W]
        imgs_01 = vae_decode_batch(pipe, x0_pred)               # [K,3,H,W]
        scores  = score_batch(reward_model, reward_backend,
                              [prompt] * K, imgs_01, device)    # [K]
        reward_history.append(scores.mean().item())

        # ── 3. Geometric tempering: λ_t = (1+γ)^(step from end) - 1 ──────
        step_from_end = T - 1 - i          # 0 at last step, T-1 at first
        lam = (1.0 + gamma) ** step_from_end - 1.0
        log_w = log_w + lam * scores

        # ── 4. Adaptive resampling based on ESS ──────────────────────────
        # ESS = (Σ w_k)² / Σ w_k²  with w_k = softmax(log_w)
        w_norm = torch.softmax(log_w, dim=0)
        ess    = 1.0 / (w_norm ** 2).sum().item()
        ess_history.append(ess)

        if ess < ess_threshold * K:
            idx       = systematic_resample(w_norm)
            particles = particles[idx].clone()
            log_w     = torch.zeros(K, device=device)
            resample_count += 1

    # ── Final selection ───────────────────────────────────────────────────
    # After last step: use remaining log_w OR just take max final reward
    imgs_01 = vae_decode_batch(pipe, particles)
    final_scores = score_batch(reward_model, reward_backend,
                               [prompt] * K, imgs_01, device)

    if select == "best":
        # pick particle with highest final image reward
        best_idx = final_scores.argmax().item()
    else:
        # pick particle with highest cumulative weight × final score
        combined = torch.softmax(log_w, dim=0) * final_scores
        best_idx = combined.argmax().item()

    from torchvision.transforms.functional import to_pil_image
    pil = to_pil_image(imgs_01[best_idx].cpu().clamp(0, 1))

    stats = {
        "reward_best":      final_scores[best_idx].item(),
        "reward_mean":      final_scores.mean().item(),
        "reward_all":       final_scores.tolist(),
        "resample_count":   resample_count,
        "ess_min":          min(ess_history) if ess_history else 0.0,
        "ess_mean":         sum(ess_history) / len(ess_history) if ess_history else 0.0,
        "reward_traj_mean": reward_history,
        "gamma":            gamma,
    }
    return pil, stats["reward_best"], stats


# ─────────────────────────────────────────────────────────────────────────────
# Grid / output helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_comparison_grid(entries: list[tuple[str, float, Image.Image]],
                         prompt: str, img_size: int = 512) -> Image.Image:
    """entries: [(method_label, score, PIL), ...]"""
    n       = len(entries)
    margin  = 8
    label_h = 30
    prompt_h = 44
    W = img_size * n + margin * (n + 1)
    H = prompt_h + label_h + img_size + margin * 2

    grid = Image.new("RGB", (W, H), (24, 24, 24))
    draw = ImageDraw.Draw(grid)

    try:
        fl = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        fp = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        fl = fp = ImageFont.load_default()

    ptext = prompt if len(prompt) <= 110 else prompt[:107] + "..."
    draw.text((margin, margin + 2), ptext, fill=(210, 210, 210), font=fp)

    for i, (label, score, img) in enumerate(entries):
        x = margin + i * (img_size + margin)
        y = prompt_h
        draw.text((x + 4, y + 4), f"{label}  r={score:.3f}",
                  fill=(255, 215, 80), font=fl)
        img_r = img.resize((img_size, img_size), Image.LANCZOS)
        grid.paste(img_r, (x, y + label_h))

    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device(args.device)
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"Loading SANA '{args.model_id}' ...")
    pipe = SanaPipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    pipe.transformer.eval().requires_grad_(False)
    pipe.vae.eval().requires_grad_(False)
    pipe.text_encoder.eval().requires_grad_(False)

    print(f"Loading reward model ...")
    reward_model, reward_backend = load_reward_model(device)
    if reward_model is None:
        raise RuntimeError("No reward model found. Install hpsv2 or ImageReward.")
    print(f"  → {reward_backend}")

    # Optional: ReNeg negative embed
    neg_e_global = neg_m_global = None
    if args.neg_ckpt:
        neg_e_global, neg_m_global = load_neg_embed(args.neg_ckpt, device, dtype)
        print(f"Loaded ReNeg checkpoint: {args.neg_ckpt}")

    # Collect prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        with open(args.prompt_file) as f:
            all_prompts = [l.strip() for l in f if l.strip()]
        rng     = random.Random(args.seed)
        prompts = rng.sample(all_prompts, min(args.n_prompts, len(all_prompts)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods  = args.methods
    all_results = []   # list of dicts for JSON summary

    print(f"\nRunning {methods} on {len(prompts)} prompt(s), "
          f"{args.n_seeds} seed(s) each\n")

    for p_idx, prompt in enumerate(prompts):
        print(f"[{p_idx+1}/{len(prompts)}] {prompt[:80]}")

        # Encode prompt once
        cond_e, cond_m, null_e, null_m = encode_prompt(pipe, prompt, device)
        neg_e = neg_e_global if neg_e_global is not None else null_e
        neg_m = neg_m_global if neg_m_global is not None else null_m

        for seed in range(args.seed, args.seed + args.n_seeds):
            entries     = []
            result_row  = {"prompt": prompt, "seed": seed}

            # ── Baseline: plain generation ─────────────────────────────────
            if "baseline" in methods:
                t0 = time.time()
                reset_scheduler(pipe, args.denoise_steps, device)
                torch.manual_seed(seed)
                C = pipe.transformer.config.in_channels
                H = W = pipe.transformer.config.sample_size
                lc = C
                x = torch.randn(1, C, H, W, device=device, dtype=dtype)
                for t in pipe.scheduler.timesteps:
                    v   = cfg_forward(pipe, x, t, cond_e, cond_m,
                                      neg_e, neg_m, args.cfg_scale, lc)
                    res = pipe.scheduler.step(v, t, x, return_dict=False)
                    x   = res[0] if isinstance(res, (tuple, list)) else res.prev_sample
                imgs = vae_decode_batch(pipe, x)
                r    = score_batch(reward_model, reward_backend,
                                   [prompt], imgs, device)[0].item()
                from torchvision.transforms.functional import to_pil_image
                pil = to_pil_image(imgs[0].cpu().clamp(0, 1))
                print(f"  baseline      reward={r:.4f}  t={time.time()-t0:.1f}s")
                entries.append(("baseline", r, pil))
                result_row["baseline"] = r

            # ── Best-of-N ─────────────────────────────────────────────────
            if "best_of_n" in methods:
                t0 = time.time()
                pil, r, stats = best_of_n(
                    pipe, prompt, cond_e, cond_m, neg_e, neg_m,
                    reward_model, reward_backend,
                    device, dtype, args.cfg_scale, args.denoise_steps,
                    N=args.bon_n, seed=seed,
                )
                print(f"  BoN(N={args.bon_n:2d})      "
                      f"reward={r:.4f}  mean={stats['reward_mean']:.4f}  "
                      f"t={time.time()-t0:.1f}s")
                entries.append((f"BoN-{args.bon_n}", r, pil))
                result_row["best_of_n"] = stats

            # ── SVDD-PM ────────────────────────────────────────────────────
            if "svdd_pm" in methods:
                t0 = time.time()
                pil, r, stats = svdd_pm(
                    pipe, prompt, cond_e, cond_m, neg_e, neg_m,
                    reward_model, reward_backend,
                    device, dtype, args.cfg_scale, args.denoise_steps,
                    M=args.svdd_m, alpha=args.svdd_alpha, seed=seed,
                    resample_start_frac=args.resample_start_frac,
                )
                print(f"  SVDD-PM(M={args.svdd_m:2d})  "
                      f"reward={r:.4f}  t={time.time()-t0:.1f}s")
                entries.append((f"SVDD-{args.svdd_m}", r, pil))
                result_row["svdd_pm"] = stats

            # ── SMC ────────────────────────────────────────────────────────
            if "smc" in methods:
                t0 = time.time()
                pil, r, stats = smc_reward(
                    pipe, prompt, cond_e, cond_m, neg_e, neg_m,
                    reward_model, reward_backend,
                    device, dtype, args.cfg_scale, args.denoise_steps,
                    K=args.smc_k, gamma=args.smc_gamma,
                    ess_threshold=args.ess_threshold,
                    seed=seed,
                    resample_start_frac=args.resample_start_frac,
                    select=args.smc_select,
                )
                print(f"  SMC(K={args.smc_k:2d})       "
                      f"reward={r:.4f}  mean={stats['reward_mean']:.4f}  "
                      f"resamples={stats['resample_count']}  "
                      f"ESS_min={stats['ess_min']:.1f}  "
                      f"t={time.time()-t0:.1f}s")
                entries.append((f"SMC-{args.smc_k}", r, pil))
                result_row["smc"] = stats

            all_results.append(result_row)

            # ── Save comparison grid ───────────────────────────────────────
            if entries:
                grid = make_comparison_grid(entries, prompt,
                                            img_size=args.img_size)
                safe  = "".join(c if c.isalnum() or c in " -_" else "_"
                                for c in prompt[:50]).strip()
                fname = f"{p_idx:04d}_seed{seed}_{safe}.png"
                grid.save(out_dir / fname)

        print()

    # ── JSON summary ──────────────────────────────────────────────────────
    summary_path = out_dir / "results.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {summary_path}")

    # ── Print aggregate table ──────────────────────────────────────────────
    print("\n── Aggregate rewards ──────────────────────────────────")
    for method in ["baseline", "best_of_n", "svdd_pm", "smc"]:
        if method not in methods and method != "baseline":
            continue
        vals = []
        for row in all_results:
            v = row.get(method)
            if v is None:
                continue
            if isinstance(v, dict):
                v = v.get("reward_best") or v.get("reward_final")
            if v is not None:
                vals.append(v)
        if vals:
            print(f"  {method:12s}  mean={sum(vals)/len(vals):.4f}  "
                  f"n={len(vals)}")
    print("──────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Reward-tilted SMC/SVDD/BoN inference for SANA")

    # Model
    p.add_argument("--model_id",    default="Efficient-Large-Model/Sana_600M_512px_diffusers")
    p.add_argument("--neg_ckpt",    default=None,
                   help="Optional ReNeg checkpoint for neg embedding.")
    p.add_argument("--device",      default="cuda")

    # Prompts
    p.add_argument("--prompt",      default=None)
    p.add_argument("--prompt_file", default=None)
    p.add_argument("--n_prompts",   type=int, default=32)
    p.add_argument("--n_seeds",     type=int, default=1)
    p.add_argument("--seed",        type=int, default=0)

    # Methods to run
    p.add_argument("--methods",     nargs="+",
                   default=["baseline", "best_of_n", "svdd_pm", "smc"],
                   choices=["baseline", "best_of_n", "svdd_pm", "smc"],
                   help="Which methods to run.")

    # Generation
    p.add_argument("--cfg_scale",     type=float, default=4.5)
    p.add_argument("--denoise_steps", type=int,   default=20)

    # Best-of-N
    p.add_argument("--bon_n",       type=int,   default=8,
                   help="N for Best-of-N.")

    # SVDD-PM
    p.add_argument("--svdd_m",      type=int,   default=8,
                   help="M particles per step for SVDD-PM.")
    p.add_argument("--svdd_alpha",  type=float, default=1.0,
                   help="Temperature α for SVDD-PM softmax resampling.")

    # SMC
    p.add_argument("--smc_k",       type=int,   default=8,
                   help="K particles for SMC.")
    p.add_argument("--smc_gamma",   type=float, default=0.1,
                   help="Geometric tempering base γ. λ_t=(1+γ)^step−1.")
    p.add_argument("--ess_threshold", type=float, default=0.5,
                   help="Resample when ESS < threshold * K.")
    p.add_argument("--smc_select",  default="best",
                   choices=["best", "weighted"],
                   help="Final particle selection strategy.")

    # Shared
    p.add_argument("--resample_start_frac", type=float, default=0.3,
                   help="Fraction of steps (from end) in which resampling "
                        "is active. 0.3 → last 30%% of steps. "
                        "Early x̂₀ is blurry; reward is unreliable.")

    # Output
    p.add_argument("--output_dir",  default="./smc_outputs")
    p.add_argument("--img_size",    type=int, default=512)

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())