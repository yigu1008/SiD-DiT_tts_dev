#!/usr/bin/env python3
"""
Quick debug script to test:
  1) Flux Schnell with HPSv2 reward
  2) SenseFlow SD3.5-Large
  3) SenseFlow Flux
  4) NFE scaling curves (reward vs compute)

Usage (1 prompt per GPU, all models):
  python debug_scaling.py --device cuda --prompt "a cat on a rooftop at sunset"

Usage (single model):
  python debug_scaling.py --models flux --device cuda

Usage (on cluster with 8 GPUs, 1 prompt per GPU):
  torchrun --nproc_per_node=8 debug_scaling.py --device cuda
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field

import torch
import numpy as np
from PIL import Image

# ── Model configs ────────────────────────────────────────────────────────────

MODELS = {
    "flux": {
        "pipeline": "flux",
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "transformer_id": None,
        "transformer_subfolder": None,
        "steps": 4,
        "guidance_scale": 0.0,
        "dtype": torch.bfloat16,
    },
    "senseflow_sd35l": {
        "pipeline": "sd35",
        "model_id": "stabilityai/stable-diffusion-3.5-large",
        "transformer_id": "domiso/SenseFlow",
        "transformer_subfolder": "SenseFlow-SD35L/transformer",
        "steps": 2,
        "guidance_scale": 0.0,
        "sigmas": [1.0, 0.75],
        "dtype": torch.bfloat16,
    },
    "senseflow_flux": {
        "pipeline": "flux",
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "transformer_id": "domiso/SenseFlow",
        "transformer_subfolder": "SenseFlow-FLUX",
        "steps": 2,
        "guidance_scale": 0.0,
        "sigmas": [1.0, 0.75],
        "dtype": torch.bfloat16,
    },
}

# NFE budgets to sweep for scaling curves (BoN with N samples)
BON_SIZES = [1, 2, 4, 8, 16, 32]


@dataclass
class ScalingResult:
    model: str
    bon_n: int
    nfe: int
    best_reward: float
    all_rewards: list[float] = field(default_factory=list)
    elapsed: float = 0.0


# ── Pipeline loading ─────────────────────────────────────────────────────────

def load_sd35_pipeline(cfg, device):
    from diffusers import StableDiffusion3Pipeline
    pretrained_kwargs = {"torch_dtype": cfg["dtype"]}
    if cfg["transformer_id"]:
        from diffusers.models.transformers import SD3Transformer2DModel
        tf_kwargs = {"torch_dtype": cfg["dtype"]}
        if cfg["transformer_subfolder"]:
            tf_kwargs["subfolder"] = cfg["transformer_subfolder"]
        print(f"  Loading transformer: {cfg['transformer_id']} subfolder={cfg.get('transformer_subfolder')}")
        pretrained_kwargs["transformer"] = SD3Transformer2DModel.from_pretrained(
            cfg["transformer_id"], **tf_kwargs
        )
    print(f"  Loading pipeline: {cfg['model_id']}")
    pipe = StableDiffusion3Pipeline.from_pretrained(cfg["model_id"], **pretrained_kwargs)
    pipe = pipe.to(device)
    return pipe


def load_flux_pipeline(cfg, device):
    from diffusers import FluxPipeline
    print(f"  Loading pipeline: {cfg['model_id']}")
    pipe = FluxPipeline.from_pretrained(cfg["model_id"], torch_dtype=cfg["dtype"])
    if cfg["transformer_id"]:
        try:
            from diffusers.models.transformers import FluxTransformer2DModel
        except ImportError:
            from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
        tf_kwargs = {"torch_dtype": cfg["dtype"]}
        if cfg["transformer_subfolder"]:
            tf_kwargs["subfolder"] = cfg["transformer_subfolder"]
        print(f"  Loading transformer: {cfg['transformer_id']} subfolder={cfg.get('transformer_subfolder')}")
        pipe.transformer = FluxTransformer2DModel.from_pretrained(
            cfg["transformer_id"], **tf_kwargs
        ).to(device)
    pipe = pipe.to(device)
    return pipe


def load_pipeline(model_name, device):
    cfg = MODELS[model_name]
    print(f"[{model_name}] Loading...")
    if cfg["pipeline"] == "sd35":
        pipe = load_sd35_pipeline(cfg, device)
    else:
        pipe = load_flux_pipeline(cfg, device)
    pipe.transformer.eval().requires_grad_(False)
    pipe.vae.eval().requires_grad_(False)
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    print(f"[{model_name}] Ready on {device}")
    return pipe, cfg


# ── Reward loading ───────────────────────────────────────────────────────────

def load_reward(backend, device):
    from reward_unified import UnifiedRewardScorer
    scorer = UnifiedRewardScorer(device=device, backend=backend)
    print(f"Reward model loaded: {scorer.describe()}")
    return scorer


# ── Generation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_images(pipe, cfg, prompt, n_images, seed, device):
    """Generate n_images for a single prompt, return list of PIL images and total NFE."""
    images = []
    steps = cfg["steps"]
    guidance = cfg["guidance_scale"]
    nfe_per_image = steps * (2 if guidance > 1.0 else 1)

    for i in range(n_images):
        gen = torch.Generator(device=device).manual_seed(seed + i)
        result = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
        )
        images.append(result.images[0])

    total_nfe = nfe_per_image * n_images
    return images, total_nfe


# ── Scaling sweep ────────────────────────────────────────────────────────────

def run_scaling_sweep(pipe, cfg, model_name, prompt, reward_model, seed, device, bon_sizes):
    """Run BoN sweep: for each N, generate N images, pick best by reward. Track NFE scaling."""
    results = []
    # Generate the max N images once, reuse subsets
    max_n = max(bon_sizes)
    print(f"[{model_name}] Generating {max_n} images for scaling sweep...")
    t0 = time.time()
    all_images, _ = generate_images(pipe, cfg, prompt, max_n, seed, device)
    gen_time = time.time() - t0
    print(f"[{model_name}] Generated {max_n} images in {gen_time:.1f}s")

    # Score all images
    print(f"[{model_name}] Scoring {max_n} images...")
    all_rewards = []
    for img in all_images:
        r = float(reward_model.score(prompt, img))
        all_rewards.append(r)

    steps = cfg["steps"]
    nfe_per_image = steps * (2 if cfg["guidance_scale"] > 1.0 else 1)

    for n in bon_sizes:
        subset_rewards = all_rewards[:n]
        best = max(subset_rewards)
        nfe = nfe_per_image * n
        results.append(ScalingResult(
            model=model_name,
            bon_n=n,
            nfe=nfe,
            best_reward=best,
            all_rewards=subset_rewards,
            elapsed=gen_time * n / max_n,
        ))
        print(f"  BoN({n:3d}): NFE={nfe:4d}  best_reward={best:.4f}  mean={np.mean(subset_rewards):.4f}")

    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_scaling_curves(all_results, out_path="scaling_curves.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available, skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    models = sorted(set(r.model for r in all_results))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(models), 3)))

    for idx, model in enumerate(models):
        rs = [r for r in all_results if r.model == model]
        rs.sort(key=lambda r: r.nfe)
        nfes = [r.nfe for r in rs]
        bests = [r.best_reward for r in rs]
        means = [np.mean(r.all_rewards) for r in rs]

        ax1.plot(nfes, bests, "o-", color=colors[idx], label=f"{model} (best)", linewidth=2)
        ax1.plot(nfes, means, "s--", color=colors[idx], label=f"{model} (mean)", alpha=0.5)

        # Log-scale NFE
        ax2.plot(nfes, bests, "o-", color=colors[idx], label=model, linewidth=2)

    ax1.set_xlabel("NFE (number of function evaluations)")
    ax1.set_ylabel("Reward Score")
    ax1.set_title("Scaling: Reward vs NFE")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("NFE (log scale)")
    ax2.set_ylabel("Best Reward Score")
    ax2.set_title("Scaling: Reward vs NFE (log)")
    ax2.set_xscale("log", base=2)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved scaling plot: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Debug: test models + NFE scaling curves")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()),
                        help="Which models to test")
    parser.add_argument("--prompt", default="a cinematic photo of a red panda drinking coffee in a rainy Tokyo alley")
    parser.add_argument("--reward_backend", default="hpsv2",
                        help="Reward backend (hpsv2, imagereward, pickscore, hpsv3)")
    parser.add_argument("--reward_device", default=None,
                        help="Device for reward model (default: same as pipeline)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_bon", type=int, default=32,
                        help="Max BoN size for scaling sweep")
    parser.add_argument("--out_dir", default="debug_scaling_out")
    parser.add_argument("--skip_plot", action="store_true")
    args = parser.parse_args()

    # torchrun support: 1 prompt on this rank's GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        device = f"cuda:{local_rank}"
        # Each rank tests a different model (round-robin)
        args.models = [args.models[i] for i in range(len(args.models)) if i % world_size == local_rank]
        if not args.models:
            print(f"[rank {local_rank}] No models assigned, exiting")
            return
        print(f"[rank {local_rank}] Testing: {args.models}")
    else:
        device = args.device

    reward_device = args.reward_device or ("cpu" if device.startswith("cuda") else device)
    bon_sizes = [n for n in BON_SIZES if n <= args.max_bon]

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = []

    # Load reward model once
    print(f"Loading reward model ({args.reward_backend}) on {reward_device}...")
    reward_model = load_reward(args.reward_backend, reward_device)

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")

        try:
            pipe, cfg = load_pipeline(model_name, device)
        except Exception as e:
            print(f"[{model_name}] FAILED to load: {e}")
            continue

        # Quick sanity: generate 1 image
        print(f"[{model_name}] Sanity check: generating 1 image...")
        images, nfe = generate_images(pipe, cfg, args.prompt, 1, args.seed, device)
        r = float(reward_model.score(args.prompt, images[0]))
        print(f"[{model_name}] Sanity OK: reward={r:.4f} NFE={nfe}")
        images[0].save(os.path.join(args.out_dir, f"{model_name}_sanity.png"))

        # Scaling sweep
        print(f"\n[{model_name}] Running BoN scaling sweep (N={bon_sizes})...")
        results = run_scaling_sweep(
            pipe, cfg, model_name, args.prompt, reward_model,
            args.seed, device, bon_sizes,
        )
        all_results.extend(results)

        # Free GPU memory before next model
        del pipe
        torch.cuda.empty_cache()

    # Save results
    results_path = os.path.join(args.out_dir, f"scaling_results_rank{local_rank}.json")
    with open(results_path, "w") as f:
        json.dump([{
            "model": r.model, "bon_n": r.bon_n, "nfe": r.nfe,
            "best_reward": r.best_reward, "mean_reward": float(np.mean(r.all_rewards)),
            "elapsed": r.elapsed,
        } for r in all_results], f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Plot
    if not args.skip_plot and all_results:
        plot_path = os.path.join(args.out_dir, f"scaling_curves_rank{local_rank}.png")
        plot_scaling_curves(all_results, plot_path)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_name in args.models:
        rs = [r for r in all_results if r.model == model_name]
        if rs:
            baseline = rs[0].best_reward
            best = max(r.best_reward for r in rs)
            print(f"  {model_name}: baseline={baseline:.4f} -> best={best:.4f} (delta={best-baseline:+.4f})")


if __name__ == "__main__":
    main()
