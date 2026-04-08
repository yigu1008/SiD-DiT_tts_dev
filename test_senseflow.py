"""
Test script for SenseFlow models (SD3.5-Large, SD3.5-Medium, FLUX).

Usage:
  python test_senseflow.py --variant sd35l --prompt "a cat sitting on a windowsill"
  python test_senseflow.py --variant sd35m --prompt "a cat sitting on a windowsill"
  python test_senseflow.py --variant flux  --prompt "a cat sitting on a windowsill"

Each variant loads the SenseFlow transformer from domiso/SenseFlow and injects it
into the corresponding base pipeline from diffusers (no SiD dependency).
"""

import argparse
import os
import torch

# ── Variant configs ──────────────────────────────────────────────────────────

VARIANTS = {
    "sd35l": {
        "base_model": "stabilityai/stable-diffusion-3.5-large",
        "senseflow_repo": "domiso/SenseFlow",
        "subfolder": "SenseFlow-SD35L/transformer",
        "pipeline_cls": "StableDiffusion3Pipeline",
        "transformer_cls": "SD3Transformer2DModel",
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
    },
    "sd35m": {
        "base_model": "stabilityai/stable-diffusion-3.5-medium",
        "senseflow_repo": "domiso/SenseFlow",
        "subfolder": "SenseFlow-SD35M/transformer",
        "pipeline_cls": "StableDiffusion3Pipeline",
        "transformer_cls": "SD3Transformer2DModel",
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        # SD35M is missing config.json in the HF repo — we copy it from the
        # base model's transformer before loading.
        "missing_config": True,
    },
    "flux": {
        "base_model": "black-forest-labs/FLUX.1-schnell",
        "senseflow_repo": "domiso/SenseFlow",
        "subfolder": "SenseFlow-FLUX",
        "pipeline_cls": "FluxPipeline",
        "transformer_cls": "FluxTransformer2DModel",
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
    },
}


def load_pipeline(variant_name: str, device: str = "cuda", dtype=torch.bfloat16):
    cfg = VARIANTS[variant_name]

    # ── Import the right classes ─────────────────────────────────────────────
    if cfg["pipeline_cls"] == "StableDiffusion3Pipeline":
        from diffusers import StableDiffusion3Pipeline as PipelineCls
        from diffusers.models.transformers import SD3Transformer2DModel as TransformerCls
    else:
        from diffusers import FluxPipeline as PipelineCls
        try:
            from diffusers.models.transformers import FluxTransformer2DModel as TransformerCls
        except ImportError:
            from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel as TransformerCls

    # ── Handle missing config.json (SD35M) ───────────────────────────────────
    if cfg.get("missing_config"):
        _ensure_transformer_config(cfg, dtype)

    # ── Load transformer ─────────────────────────────────────────────────────
    print(f"Loading SenseFlow transformer: {cfg['senseflow_repo']}  subfolder={cfg['subfolder']}")
    transformer = TransformerCls.from_pretrained(
        cfg["senseflow_repo"],
        subfolder=cfg["subfolder"],
        torch_dtype=dtype,
    )

    # ── Load pipeline with injected transformer ──────────────────────────────
    print(f"Loading base pipeline: {cfg['base_model']}")
    pipe = PipelineCls.from_pretrained(
        cfg["base_model"],
        transformer=transformer,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)
    print(f"Pipeline ready on {device}")
    return pipe, cfg


def _ensure_transformer_config(cfg, dtype):
    """
    SD35M SenseFlow is missing config.json in the HF repo.
    Copy it from the base model's transformer so from_pretrained works.
    """
    try:
        from huggingface_hub import try_to_load_from_cache, snapshot_download
        # Make sure the SenseFlow repo is cached
        sf_path = snapshot_download(cfg["senseflow_repo"])
        local_tf_dir = os.path.join(sf_path, cfg["subfolder"])
        cfg_path = os.path.join(local_tf_dir, "config.json")
        if os.path.isfile(cfg_path):
            return  # already present
        # Get config from base model's transformer
        base_cfg = try_to_load_from_cache(cfg["base_model"], "transformer/config.json")
        if base_cfg and os.path.isfile(base_cfg):
            import shutil
            os.makedirs(local_tf_dir, exist_ok=True)
            shutil.copy2(base_cfg, cfg_path)
            print(f"Copied transformer config.json from {cfg['base_model']} -> {cfg_path}")
        else:
            print(f"[warn] Could not find base model transformer config.json in cache. "
                  f"Run: python -c \"from huggingface_hub import snapshot_download; "
                  f"snapshot_download('{cfg['base_model']}')\" first.")
    except Exception as exc:
        print(f"[warn] _ensure_transformer_config failed: {exc}")


def generate(pipe, cfg, prompt: str, seed: int = 42, out_path: str = "senseflow_test.png"):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    print(f"Generating: \"{prompt}\"  steps={cfg['num_inference_steps']}  cfg={cfg['guidance_scale']}")
    result = pipe(
        prompt,
        num_inference_steps=cfg["num_inference_steps"],
        guidance_scale=cfg["guidance_scale"],
        generator=generator,
    )
    img = result.images[0]
    img.save(out_path)
    print(f"Saved: {out_path}")
    return img


def main():
    parser = argparse.ArgumentParser(description="Test SenseFlow variants")
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), required=True,
                        help="SenseFlow variant to test")
    parser.add_argument("--prompt", default="a cinematic photo of a red panda drinking coffee in a rainy Tokyo alley")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="Output image path (default: senseflow_test_{variant}.png)")
    parser.add_argument("--steps", type=int, default=None, help="Override num_inference_steps")
    parser.add_argument("--cfg", type=float, default=None, help="Override guidance_scale")
    args = parser.parse_args()

    pipe, cfg = load_pipeline(args.variant, device=args.device)
    if args.steps is not None:
        cfg["num_inference_steps"] = args.steps
    if args.cfg is not None:
        cfg["guidance_scale"] = args.cfg
    out_path = args.out or f"senseflow_test_{args.variant}.png"
    generate(pipe, cfg, args.prompt, seed=args.seed, out_path=out_path)


if __name__ == "__main__":
    main()
