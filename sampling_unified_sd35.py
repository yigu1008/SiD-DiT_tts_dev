"""
Unified test-time scaling sampler for SiD SD3.5-large.

Search space per denoising step:
  action = (prompt_variant_idx, cfg_scale)

Supports:
  - greedy search
  - MCTS search
  - GA search
  - SMC search
  - paper UnifiedReward scoring (with blend fallback)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from blend_ops import nlerp_all, slerp_pair
from reward_unified import UnifiedRewardScorer


REWRITE_SYSTEM = (
    "You are a concise image prompt editor. "
    "Given a text-to-image prompt, produce a single minimally-changed rewrite. "
    "Keep the subject and composition identical. "
    "You may only change: lighting descriptors, camera/lens terms, mood adjectives, "
    "or add/remove one small detail. "
    "Output ONLY the rewritten prompt, no explanation, no quotes."
)

REWRITE_STYLES = [
    "Adjust the lighting or time of day slightly.",
    "Swap or add a camera/lens detail.",
    "Change one mood or atmosphere word.",
]

_DEFAULT_CFG_SCALES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
_DEFAULT_BASELINE_CFG = 1.0

_REWRITE_PLACEHOLDER_RE = re.compile(r"^/?\s*<[^>]+>\s*$")
_REWRITE_BAD_TOKENS = {
    "<thin>",
    "</thin>",
    "/<thin>",
    "<think>",
    "</think>",
    "/<think>",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified SD3.5 sampling/search with unified reward.")
    parser.add_argument("--search_method", choices=["greedy", "mcts", "ga", "smc", "bon", "beam", "noise_inject"], default="greedy")

    parser.add_argument(
        "--backend",
        choices=["sid", "sd35_base", "senseflow_large", "senseflow_medium"],
        default=None,
        help="Convenience shortcut: sets --model_id, --transformer_id, --transformer_subfolder, "
             "and --sigmas together. Explicit flags override these defaults.",
    )
    parser.add_argument("--model_id", default=os.environ.get("MODEL_ID"))
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--transformer_id", default=os.environ.get("TRANSFORMER_ID"),
                        help="HuggingFace repo for the transformer (e.g. domiso/SenseFlow).")
    parser.add_argument("--transformer_subfolder", default=None,
                        help="Subfolder within --transformer_id (e.g. SenseFlow-SD35L/transformer).")
    parser.add_argument("--prompt", default="a cinematic portrait of a woman in soft rim light, 85mm, ultra detailed")
    parser.add_argument("--prompt_file", default=None)

    parser.add_argument("--n_variants", type=int, default=3)
    parser.add_argument("--no_qwen", action="store_true")
    parser.add_argument("--qwen_id", default="Qwen/Qwen3-4B")
    parser.add_argument("--qwen_python", default="python3")
    parser.add_argument("--qwen_dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--qwen_timeout_sec", type=float, default=240.0)
    parser.add_argument("--rewrites_file", default=None)

    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--sigmas",
        nargs="+",
        type=float,
        default=None,
        help="Explicit sigma schedule (overrides --backend default).",
    )
    parser.add_argument(
        "--cfg_scales",
        nargs="+",
        type=float,
        default=list(_DEFAULT_CFG_SCALES),
    )
    parser.add_argument("--baseline_cfg", type=float, default=_DEFAULT_BASELINE_CFG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--time_scale", type=float, default=1000.0)
    parser.add_argument("--out_dir", default="./imagereward_sd35_out")

    parser.add_argument("--n_sims", type=int, default=60)
    parser.add_argument("--ucb_c", type=float, default=1.41)
    parser.add_argument(
        "--mcts_key_steps",
        default="",
        help="Comma-separated key step indices for MCTS branching (e.g. '0,7,14,21'). "
             "Empty = branch at every step (original behavior).",
    )
    parser.add_argument(
        "--mcts_key_step_count",
        type=int,
        default=0,
        help="Auto-compute N evenly-spaced key steps. 0 = disabled (use --mcts_key_steps or branch at every step).",
    )
    parser.add_argument(
        "--mcts_fresh_noise_steps",
        default="",
        help="Comma-separated step indices for extra fresh-noise exploration in MCTS (e.g. '0,7,14'). "
             "'all' enables every step.",
    )
    parser.add_argument(
        "--mcts_fresh_noise_samples",
        type=int,
        default=1,
        help="Number of fresh-noise candidates to try at selected MCTS steps. 1 disables extra exploration.",
    )
    parser.add_argument(
        "--mcts_fresh_noise_scale",
        type=float,
        default=1.0,
        help="Scale for additive latent perturbation used by fresh-noise exploration.",
    )
    parser.add_argument(
        "--mcts_fresh_noise_key_steps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also apply fresh-noise exploration at resolved MCTS key steps.",
    )
    parser.add_argument("--bon_n", type=int, default=16,
                        help="Number of independent samples for best-of-N search.")
    parser.add_argument(
        "--noise_inject_mode",
        choices=["seeds_only", "reinject_only", "combined"],
        default="combined",
        help="Sparse noise-injection search mode: seeds-only | reinjection-only | combined.",
    )
    parser.add_argument(
        "--noise_inject_seed_budget",
        type=int,
        default=8,
        help="Number of root seeds to evaluate for seeds-only or combined mode.",
    )
    parser.add_argument(
        "--noise_inject_candidate_steps",
        default="",
        help="Comma-separated candidate reinjection steps (e.g. '1,2'); 'all' uses all steps; empty uses middle step.",
    )
    parser.add_argument(
        "--noise_inject_gamma_bank",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.50],
        help="Latent reinjection magnitude bank gamma for x_t <- x_t + gamma*sigma_t*eps.",
    )
    parser.add_argument(
        "--noise_inject_eps_samples",
        type=int,
        default=4,
        help="Number of eps samples in the reinjection bank per seed.",
    )
    parser.add_argument(
        "--noise_inject_steps_per_rollout",
        type=int,
        default=1,
        help="Reinjection count per rollout (1 or 2) for ablation.",
    )
    parser.add_argument(
        "--noise_inject_include_no_inject",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include a no-injection candidate policy in reinjection search.",
    )
    parser.add_argument(
        "--noise_inject_max_policies",
        type=int,
        default=0,
        help="Cap reinjection policy count per seed (0 = no cap).",
    )
    parser.add_argument(
        "--noise_inject_variant_idx",
        type=int,
        default=0,
        help="Prompt-variant index used by sparse noise-injection search.",
    )
    parser.add_argument(
        "--noise_inject_cfg",
        type=float,
        default=None,
        help="Optional fixed CFG for sparse noise-injection search. If unset, search over cfg_scales.",
    )
    parser.add_argument("--beam_width", type=int, default=4,
                        help="Number of beams to keep per step in beam search.")
    parser.add_argument("--smc_k", type=int, default=8)
    parser.add_argument("--smc_gamma", type=float, default=0.10)
    parser.add_argument("--ess_threshold", type=float, default=0.5)
    parser.add_argument("--resample_start_frac", type=float, default=0.3)
    parser.add_argument("--smc_cfg_scale", type=float, default=1.25)
    parser.add_argument("--smc_variant_idx", type=int, default=0)
    parser.add_argument(
        "--smc_variant_expansion",
        action="store_true",
        default=False,
        help="Enable CFG/prompt-variant expansion at SMC resample points: "
             "fan out each surviving particle into M children with distinct "
             "(variant, cfg, cs) actions from a bank and resample K out of K·M.",
    )
    parser.add_argument(
        "--smc_expansion_variants",
        nargs="+",
        type=int,
        default=[],
        help="Variant indices in the expansion bank. Empty = all emb.cond_text variants.",
    )
    parser.add_argument(
        "--smc_expansion_cfgs",
        nargs="+",
        type=float,
        default=[],
        help="CFG scales in the expansion bank. Empty = --cfg_scales.",
    )
    parser.add_argument(
        "--smc_expansion_cs",
        nargs="+",
        type=float,
        default=[],
        help="Correction strengths in the expansion bank. Empty = --correction_strengths.",
    )
    parser.add_argument(
        "--smc_expansion_factor",
        type=int,
        default=-1,
        help="M = children per surviving particle at resample. -1 = full bank.",
    )
    parser.add_argument(
        "--smc_expansion_proposal",
        choices=["uniform", "score_softmax"],
        default="uniform",
    )
    parser.add_argument("--smc_expansion_tau", type=float, default=1.0)
    parser.add_argument(
        "--smc_expansion_lookahead",
        action="store_true",
        default=False,
        help="Run one extra transformer step with each child's new action and score "
             "its decode before reweighting (expensive, most diversity).",
    )
    parser.add_argument(
        "--correction_strengths",
        nargs="+",
        type=float,
        default=[0.0],
        help="Reward-gradient correction strength values to include as actions (like --cfg_scales). "
             "[0.0] disables correction. E.g. --correction_strengths 0.0 0.5 1.0. "
             "Requires ImageReward backend.",
    )
    parser.add_argument(
        "--x0_sampler",
        action="store_true",
        default=False,
        help="Treat transformer output as a direct x̂₀ prediction instead of flow/velocity. "
             "Set automatically for senseflow backends.",
    )
    parser.add_argument("--ga_population", type=int, default=24)
    parser.add_argument("--ga_generations", type=int, default=8)
    parser.add_argument("--ga_elites", type=int, default=3)
    parser.add_argument("--ga_mutation_prob", type=float, default=0.10)
    parser.add_argument("--ga_tournament_k", type=int, default=3)
    parser.add_argument("--ga_crossover", choices=["uniform", "one_point"], default="uniform")
    parser.add_argument("--ga_selection", choices=["rank", "tournament"], default="rank")
    parser.add_argument("--ga_rank_pressure", type=float, default=1.7)
    parser.add_argument("--ga_log_topk", type=int, default=3)
    parser.add_argument("--ga_eval_batch", type=int, default=1, help="Batch size for GA genome rollout evaluation.")
    parser.add_argument("--ga_phase_constraints", action="store_true")
    parser.add_argument(
        "--reward_model",
        default="CodeGoat24/UnifiedReward-qwen-7b",
        help="Legacy alias for UnifiedReward model id (kept for compatibility).",
    )
    parser.add_argument(
        "--unifiedreward_model",
        default=None,
        help="UnifiedReward model id override. Defaults to --reward_model when unset.",
    )
    parser.add_argument(
        "--image_reward_model",
        default="ImageReward-v1.0",
        help="ImageReward model id/checkpoint name.",
    )
    parser.add_argument(
        "--pickscore_model",
        default="yuvalkirstain/PickScore_v1",
        help="PickScore model id.",
    )
    parser.add_argument(
        "--reward_backend",
        choices=["auto", "unifiedreward", "unified", "imagereward", "pickscore", "hpsv3", "hpsv2", "blend"],
        default="unifiedreward",
    )
    parser.add_argument(
        "--reward_weights",
        nargs=2,
        type=float,
        default=[1.0, 1.0],
        help="Blend backend weights: imagereward hps(v2/v3)",
    )
    parser.add_argument("--reward_api_base", default=None, help="Optional OpenAI-compatible API base for UnifiedReward.")
    parser.add_argument("--reward_api_key", default="unifiedreward")
    parser.add_argument("--reward_api_model", default="UnifiedReward-7b-v1.5")
    parser.add_argument("--reward_max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--reward_prompt_mode",
        choices=["standard", "strict"],
        default="standard",
        help="UnifiedReward prompt template mode.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default=None,
        help="Pipeline/transformer dtype. Defaults to backend setting (float16 for sid/sd35_base, bfloat16 for senseflow).",
    )
    args = parser.parse_args(argv)
    return _apply_backend_defaults(args)


_BACKEND_CONFIGS = {
    # SiD student baked into the HF repo.
    # CFG doubles the transformer batch at every step, so gen_batch_size=1 is the safe default.
    "sid": {
        "model_id": "YGu1998/SiD-DiT-SD3.5-large",
        "transformer_id": None,
        "transformer_subfolder": None,
        "sigmas": None,       # linear schedule driven by --steps
        "dtype": "float16",
        "gen_batch_size": 2,  # CFG doubles → [4,C,H,W] per step at gbs=2; may OOM on 40GB
    },
    # Official SD3.5 Large base model (decoupled from SiD student checkpoint).
    # Uses standard flow-matching Euler ODE solver (NOT SiD re-noising).
    "sd35_base": {
        "model_id": "stabilityai/stable-diffusion-3.5-large",
        "transformer_id": None,
        "transformer_subfolder": None,
        "sigmas": None,       # will use FlowMatchEulerDiscreteScheduler
        "dtype": "bfloat16",  # SD3.5 official uses bfloat16
        "gen_batch_size": 1,  # conservative default for SD3.5L + CFG
        "euler_sampler": True,  # standard Euler ODE stepping, no re-noising
    },
    # SenseFlow Large: 2-step, no CFG → no batch doubling → can use larger gen_batch_size.
    # On a 40 GB GPU: ~16 GB transformer + ~14 GB text-encoders + ~0.3 GB VAE ≈ 30 GB static,
    # leaving ~10 GB for activations.  gen_batch_size=2 → ~4-6 GB activation headroom → safe.
    # Raise to 4 if you have headroom (e.g., after verifying with nvidia-smi).
    "senseflow_large": {
        "model_id": "stabilityai/stable-diffusion-3.5-large",
        "transformer_id": "domiso/SenseFlow",
        "transformer_subfolder": "SenseFlow-SD35L/transformer",
        "sigmas": [1.0, 0.9, 0.75, 0.5],  # official SenseFlow schedule
        "dtype": "bfloat16",  # SenseFlow official scripts use bfloat16
        "gen_batch_size": 2,
        "cfg_scales": [0.0],  # SenseFlow release uses guidance_scale=0.0
        "baseline_cfg": 0.0,
        # SenseFlow transformer outputs flow/velocity (same as base SD3.5).
        # x0 prediction sampler: derive x̂₀ = xt - t*flow, then re-noise
        # latents = (1-t') * x̂₀ + t' * noise for the next step.
        # This gives clean x̂₀ at every intermediate step, making reward
        # scoring meaningful for MCTS/beam/greedy.
        "x0_sampler": False,
        "euler_sampler": False,
    },
    "senseflow_medium": {
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "transformer_id": "domiso/SenseFlow",
        "transformer_subfolder": "SenseFlow-SD35M/transformer",
        "sigmas": [1.0, 0.9, 0.75, 0.5],  # official SenseFlow schedule
        "dtype": "bfloat16",
        "gen_batch_size": 2,
        "cfg_scales": [0.0],
        "baseline_cfg": 0.0,
        "x0_sampler": False,
        "euler_sampler": False,
    },
}


def _apply_backend_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill model_id / transformer_id / sigmas / dtype / gen_batch_size from --backend."""
    cfg = _BACKEND_CONFIGS.get(args.backend or "", {})
    for key, val in cfg.items():
        if key in {"cfg_scales", "baseline_cfg"}:
            continue
        if getattr(args, key, None) is None:
            setattr(args, key, val)
    # Keep legacy behavior for sid, but use SenseFlow defaults when user kept parser defaults.
    if (args.backend or "").startswith("senseflow"):
        if list(getattr(args, "cfg_scales", [])) == list(_DEFAULT_CFG_SCALES):
            args.cfg_scales = list(cfg.get("cfg_scales", [0.0]))
        if float(getattr(args, "baseline_cfg", _DEFAULT_BASELINE_CFG)) == float(_DEFAULT_BASELINE_CFG):
            args.baseline_cfg = float(cfg.get("baseline_cfg", 0.0))
        if not getattr(args, "x0_sampler", False):
            args.x0_sampler = bool(cfg.get("x0_sampler", False))
    # Final fallbacks for fields not covered by any backend config.
    if args.model_id is None:
        args.model_id = _BACKEND_CONFIGS["senseflow_large"]["model_id"]
    if getattr(args, "dtype", None) is None:
        args.dtype = "float16"
    if getattr(args, "gen_batch_size", None) is None:
        args.gen_batch_size = 1
    # Sigmas define the true denoising length (SenseFlow-style sampler).
    if getattr(args, "sigmas", None) is not None:
        args.sigmas = [float(s) for s in args.sigmas]
        args.steps = len(args.sigmas)
    return args


def _resolve_file(path_str: str, label: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(path)


def _resolve_optional_file(path_str: str | None, label: str) -> str | None:
    if path_str is None:
        return None
    return _resolve_file(path_str, label)


def _resolve_out_dir(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.ckpt = _resolve_optional_file(args.ckpt, "ckpt")
    args.prompt_file = _resolve_optional_file(args.prompt_file, "prompt_file")
    args.rewrites_file = _resolve_optional_file(args.rewrites_file, "rewrites_file")
    args.out_dir = _resolve_out_dir(args.out_dir)
    return args


def _unwrap_state_dict(raw: Any, depth: int = 0) -> Any:
    if not isinstance(raw, dict):
        return raw
    dotted = sum(1 for key in raw if "." in str(key))
    if dotted / max(len(raw), 1) > 0.5:
        return raw
    if depth > 4:
        return raw
    for key in ("ema", "ema_model", "model_ema", "model", "state_dict", "generator", "G_state"):
        if key in raw and isinstance(raw[key], dict):
            return _unwrap_state_dict(raw[key], depth + 1)
    return raw


@dataclass
class PipelineContext:
    pipe: Any
    device: str
    dtype: torch.dtype
    latent_c: int
    nfe: int = 0              # incremented by transformer_step: +1 per step (cfg=1.0), +2 (cfg>1.0)
    correction_nfe: int = 0   # incremented by apply_reward_correction: +1 per call


@dataclass
class EmbeddingContext:
    cond_text: list[torch.Tensor]
    cond_pooled: list[torch.Tensor]
    uncond_text: torch.Tensor
    uncond_pooled: torch.Tensor


@dataclass
class SearchResult:
    image: Image.Image
    score: float
    actions: list[tuple[int, float, float]]  # (variant_idx, cfg_scale, correction_strength)
    diagnostics: dict[str, Any] | None = None


def load_pipeline(args: argparse.Namespace) -> PipelineContext:
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Compatibility shim for mixed huggingface_hub versions.
    try:
        import huggingface_hub.constants as hhc

        if not hasattr(hhc, "HF_HOME"):
            cache_root = getattr(hhc, "HUGGINGFACE_HUB_CACHE", None)
            if cache_root:
                hhc.HF_HOME = str(Path(cache_root).expanduser().parent)
            else:
                hhc.HF_HOME = str(Path.home() / ".cache" / "huggingface")
    except Exception:
        pass

    cuda_available = torch.cuda.is_available()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if cuda_available:
        # In torchrun, pin each rank to its own GPU explicitly.
        device = f"cuda:{local_rank}" if world_size > 1 else "cuda"
    else:
        device = "cpu"
        if world_size > 1:
            raise RuntimeError(
                "CUDA is unavailable under torchrun (WORLD_SIZE>1). "
                "This would force SD3.5 fp16 pipeline to CPU and hang/slow heavily. "
                f"WORLD_SIZE={world_size} LOCAL_RANK={local_rank} "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
            )
    dtype_str = getattr(args, "dtype", None) or "float16"
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    dev_count = int(torch.cuda.device_count()) if cuda_available else 0
    print(
        f"Loading SD3.5 pipeline: {args.model_id} "
        f"(device={device} cuda_available={cuda_available} device_count={dev_count} "
        f"local_rank={local_rank} cvd={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} dtype={dtype_str})"
    )
    transformer_id = getattr(args, "transformer_id", None)
    transformer_subfolder = getattr(args, "transformer_subfolder", None)

    pretrained_kwargs: dict = {"torch_dtype": dtype}
    if transformer_id:
        # Load the replacement transformer first and inject it into from_pretrained so
        # diffusers skips loading the base model's transformer from cache.  This lets
        # senseflow backends work even when the original SD3.5 transformer weights are
        # not present in HF_HOME (we only need VAE + text encoders from the base model).
        from diffusers.models.transformers import SD3Transformer2DModel
        tf_kwargs: dict = {"torch_dtype": dtype}
        transformer_load_id = transformer_id
        offline = str(os.environ.get("HF_HUB_OFFLINE", "")).strip().lower() in {"1", "true", "yes", "on"}
        if offline:
            # Avoid hub metadata requests in offline mode by resolving a local snapshot path.
            from huggingface_hub import snapshot_download

            try:
                transformer_load_id = snapshot_download(
                    transformer_id,
                    cache_dir=os.environ.get("HF_HOME"),
                    local_files_only=True,
                )
                tf_kwargs["local_files_only"] = True
                print(f"Resolved offline transformer snapshot: {transformer_load_id}")
            except Exception as exc:
                raise RuntimeError(
                    "Offline mode is enabled, but transformer weights are not cached for "
                    f"{transformer_id}. Pre-download it before setting HF_HUB_OFFLINE=1."
                ) from exc
        if transformer_subfolder:
            tf_kwargs["subfolder"] = transformer_subfolder
        print(f"Loading transformer from {transformer_load_id} subfolder={transformer_subfolder}")
        pretrained_kwargs["transformer"] = SD3Transformer2DModel.from_pretrained(
            transformer_load_id, **tf_kwargs
        ).to(device)

    if (args.backend or "sid") == "sid":
        from sid import SiDSD3Pipeline
        pipe = SiDSD3Pipeline.from_pretrained(args.model_id, **pretrained_kwargs).to(device)
    else:
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, **pretrained_kwargs).to(device)
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    # Normalize text-encoder dtypes explicitly. On some cluster images (Apex fused RMSNorm),
    # partially-fp32 encoder params can trigger Half/Float mismatch at prompt encoding time.
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        enc = getattr(pipe, name, None)
        if enc is not None:
            try:
                enc.to(device=device, dtype=dtype)
            except Exception as exc:
                print(f"[warn] unable to cast {name} to {dtype}: {exc}")
    if args.ckpt:
        print(f"Loading transformer weights from {args.ckpt}")
        raw = torch.load(args.ckpt, map_location=device, weights_only=False)
        state_dict = _unwrap_state_dict(raw)
        if any(str(k).startswith("module.") for k in state_dict):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        missing, unexpected = pipe.transformer.load_state_dict(state_dict, strict=False)
        print(
            f"  loaded={len(state_dict) - len(unexpected)}/{len(state_dict)} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    pipe.transformer.eval()
    latent_c = pipe.transformer.config.in_channels
    return PipelineContext(pipe=pipe, device=device, dtype=dtype, latent_c=latent_c)


def load_reward_model(args: argparse.Namespace, device: str) -> UnifiedRewardScorer:
    unified_model = args.unifiedreward_model if args.unifiedreward_model else args.reward_model
    scorer = UnifiedRewardScorer(
        device=device,
        backend=args.reward_backend,
        image_reward_model=args.image_reward_model,
        pickscore_model=getattr(args, "pickscore_model", "yuvalkirstain/PickScore_v1"),
        unifiedreward_model=unified_model,
        unified_weights=(float(args.reward_weights[0]), float(args.reward_weights[1])),
        unifiedreward_api_base=args.reward_api_base,
        unifiedreward_api_key=args.reward_api_key,
        unifiedreward_api_model=args.reward_api_model,
        max_new_tokens=int(args.reward_max_new_tokens),
        unifiedreward_prompt_mode=args.reward_prompt_mode,
    )
    print(f"Reward: {scorer.describe()}")
    return scorer


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file:
        prompts = [line.strip() for line in open(args.prompt_file) if line.strip()]
    else:
        prompts = [args.prompt]
    if not prompts:
        raise RuntimeError("No prompts found.")
    return prompts


def qwen_rewrite(args: argparse.Namespace, prompt: str, instruction: str) -> str:
    dtype_literal = "torch.bfloat16" if args.qwen_dtype == "bfloat16" else "torch.float16"
    marker = "__SID_QWEN_REWRITE__"
    script = f"""
import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained({repr(args.qwen_id)})
mdl = AutoModelForCausalLM.from_pretrained(
    {repr(args.qwen_id)},
    torch_dtype={dtype_literal},
    device_map="auto")
mdl.eval()
messages = [
    {{"role":"system","content":{repr(REWRITE_SYSTEM)}}},
    {{"role":"user","content":sys.argv[1] + "\\n\\nOriginal prompt: " + sys.argv[2] + " /no_think"}}
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok([text], return_tensors="pt").to(mdl.device)
with torch.no_grad():
    out = mdl.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tok.eos_token_id)
decoded = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()
for line in decoded.splitlines():
    line = line.strip()
    if line:
        print({repr(marker)} + line)
        raise SystemExit(0)
print({repr(marker)} + sys.argv[2])
"""
    try:
        result = subprocess.run(
            [args.qwen_python, "-c", script, instruction, prompt],
            capture_output=True,
            text=True,
            timeout=max(1.0, float(getattr(args, "qwen_timeout_sec", 240.0))),
        )
    except subprocess.TimeoutExpired:
        return prompt
    if result.returncode != 0:
        return prompt
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith(marker):
            continue
        candidate = line[len(marker):].strip()
        if candidate:
            return sanitize_rewrite_text(candidate, prompt)
    return prompt


def sanitize_rewrite_text(candidate: str, fallback: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", str(candidate), flags=re.DOTALL).strip()
    text = text.replace("\ufffd", "")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return fallback
    text = text.strip("`\"' ")
    if any(ord(ch) < 32 for ch in text):
        return fallback
    if len(text) > 220:
        return fallback
    lower = text.lower()
    if lower in _REWRITE_BAD_TOKENS:
        return fallback
    if "nccl info" in lower or "traceback" in lower or "runtimeerror" in lower:
        return fallback
    if _REWRITE_PLACEHOLDER_RE.fullmatch(text):
        return fallback
    if "<" in text and ">" in text and len(text) < 24:
        return fallback
    if all(ord(ch) < 128 for ch in fallback):
        non_ascii = sum(1 for ch in text if ord(ch) >= 128)
        if non_ascii > 0 and (non_ascii / max(1, len(text))) > 0.08:
            return fallback
    if len(text) < 4:
        return fallback
    return text


def _dedup_variants(values: list[str], fallback: str, max_items: int | None = None) -> list[str]:
    out: list[str] = []
    for v in values:
        vv = sanitize_rewrite_text(v, fallback)
        if vv not in out:
            out.append(vv)
        if max_items is not None and len(out) >= int(max_items):
            break
    return out if out else [fallback]


def _legacy_target_size(args: argparse.Namespace) -> int:
    return max(1, int(getattr(args, "n_variants", 0)) + 1)


def _read_legacy_cache_entry(entry: Any, prompt: str, target: int) -> list[str] | None:
    if isinstance(entry, list):
        vals = [str(v) for v in entry]
        return _dedup_variants(vals, prompt, max_items=target)
    if isinstance(entry, dict):
        raw_variants = entry.get("variants")
        if isinstance(raw_variants, list):
            vals = [str(v) for v in raw_variants]
            return _dedup_variants(vals, prompt, max_items=target)
    return None


def _generate_legacy_variants(args: argparse.Namespace, prompt: str) -> list[str]:
    target = _legacy_target_size(args)
    if target <= 1 or bool(getattr(args, "no_qwen", False)):
        return [prompt]
    vals = [prompt]
    n_rewrites = max(0, target - 1)
    styles = (REWRITE_STYLES * ((n_rewrites // len(REWRITE_STYLES)) + 1))[: n_rewrites]
    for style in styles:
        vals.append(sanitize_rewrite_text(qwen_rewrite(args, prompt, style), prompt))
    return _dedup_variants(vals, prompt, max_items=target)


def generate_variants(args: argparse.Namespace, prompt: str, cache: dict[str, Any]) -> list[str]:
    entry = cache.get(prompt)
    if entry is not None:
        from_cache = _read_legacy_cache_entry(entry, prompt, _legacy_target_size(args))
        if from_cache is not None:
            return from_cache
    return _generate_legacy_variants(args, prompt)


def encode_variants(ctx: PipelineContext, variants: list[str], max_sequence_length: int = 256) -> EmbeddingContext:
    cond_text: list[torch.Tensor] = []
    cond_pooled: list[torch.Tensor] = []
    for variant in variants:
        enc_out = ctx.pipe.encode_prompt(
            prompt=variant,
            prompt_2=variant,
            prompt_3=variant,
            device=ctx.device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        # SiDSD3Pipeline returns (pe, pp); StableDiffusion3Pipeline returns (pe, neg_pe, pp)
        pe, pp = (enc_out[0], enc_out[-1])
        cond_text.append(pe.detach())
        cond_pooled.append(pp.detach())

    enc_out = ctx.pipe.encode_prompt(
        prompt="",
        prompt_2="",
        prompt_3="",
        device=ctx.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )
    ue, up = (enc_out[0], enc_out[-1])
    return EmbeddingContext(
        cond_text=cond_text,
        cond_pooled=cond_pooled,
        uncond_text=ue.detach(),
        uncond_pooled=up.detach(),
    )


def _infer_latent_hw(pipe: Any, height: int, width: int) -> tuple[int, int, int]:
    scale = int(getattr(pipe, "vae_scale_factor", 0) or 0)
    if scale <= 1:
        try:
            enc_ch = getattr(pipe.vae.config, "encoder_block_out_channels", None)
            if enc_ch is not None and len(enc_ch) > 1:
                scale = int(2 ** (len(enc_ch) - 1))
        except Exception:
            pass
    if scale <= 1:
        scale = 8
    return max(1, int(height) // scale), max(1, int(width) // scale), scale


def make_latents(ctx: PipelineContext, seed: int, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
    exp_h, exp_w, scale = _infer_latent_hw(ctx.pipe, height, width)
    exec_device = torch.device(ctx.device) if isinstance(ctx.device, str) else ctx.device
    generator = torch.Generator(device=exec_device).manual_seed(seed)
    latents = ctx.pipe.prepare_latents(
        1,
        ctx.latent_c,
        height,
        width,
        dtype,
        exec_device,
        generator,
    )
    got_h, got_w = int(latents.shape[-2]), int(latents.shape[-1])
    if (got_h, got_w) == (int(height), int(width)) and (exp_h, exp_w) != (int(height), int(width)):
        print(
            "Warning: prepare_latents returned pixel-space latent "
            f"{got_h}x{got_w}; forcing latent-space {exp_h}x{exp_w} (scale={scale})."
        )
        latents = torch.randn(
            (1, ctx.latent_c, exp_h, exp_w),
            device=exec_device,
            dtype=dtype,
            generator=generator,
        )
    return latents


@torch.no_grad()
def transformer_step(
    args: argparse.Namespace,
    ctx: PipelineContext,
    latents: torch.Tensor,
    emb: EmbeddingContext,
    variant_idx: int,
    t_flat: torch.Tensor,
    cfg: float,
    stats_out: dict | None = None,
) -> torch.Tensor:
    pe = emb.cond_text[variant_idx]
    pp = emb.cond_pooled[variant_idx]
    n = latents.shape[0]

    if cfg == 1.0 or cfg == 0.0:
        # cfg=1.0: standard conditional-only (no guidance)
        # cfg=0.0: SenseFlow guidance_scale=0 means "no CFG, run conditional only"
        ctx.nfe += 1
        velocity = ctx.pipe.transformer(
            hidden_states=latents,
            encoder_hidden_states=pe.expand(n, -1, -1),
            pooled_projections=pp.expand(n, -1),
            timestep=args.time_scale * t_flat,
            return_dict=False,
        )[0]
        if stats_out is not None:
            stats_out["cfg_delta_rms"] = 0.0
        return velocity

    ctx.nfe += 2  # uncond + cond = 2 evaluations
    flow = ctx.pipe.transformer(
        hidden_states=torch.cat([latents, latents]),
        encoder_hidden_states=torch.cat([emb.uncond_text.expand(n, -1, -1), pe.expand(n, -1, -1)]),
        pooled_projections=torch.cat([emb.uncond_pooled.expand(n, -1), pp.expand(n, -1)]),
        timestep=args.time_scale * torch.cat([t_flat, t_flat]),
        return_dict=False,
    )[0]
    flow_u, flow_c = flow.chunk(2)
    if stats_out is not None:
        diff = (flow_c - flow_u).detach().float()
        stats_out["cfg_delta_rms"] = float(torch.sqrt(torch.mean(diff * diff)).item())
    return flow_u + cfg * (flow_c - flow_u)


@torch.no_grad()
def decode_to_pil(ctx: PipelineContext, dx: torch.Tensor) -> Image.Image:
    shift = getattr(ctx.pipe.vae.config, "shift_factor", 0.0)
    image = ctx.pipe.vae.decode(
        (dx / ctx.pipe.vae.config.scaling_factor) + shift,
        return_dict=False,
    )[0]
    return ctx.pipe.image_processor.postprocess(image, output_type="pil")[0]


def apply_reward_correction(
    ctx: "PipelineContext",
    dx: torch.Tensor,
    prompt: str,
    reward_model: "UnifiedRewardScorer",
    strength: float,
    cfg: float = 1.0,
) -> torch.Tensor:
    """Apply one step of reward-gradient ascent on the predicted x̂₀ (dx).

    Computes dR/d(dx) via a differentiable BLIP forward pass through ImageReward,
    then nudges dx in the reward-gradient direction.  No-op when strength <= 0 or
    ImageReward is not loaded.

    The effective strength is scaled by an NFE weighting factor: a CFG step costs
    2 transformer NFEs vs. 1 for this correction call, so CFG steps get weight 2.0
    (strength is upscaled proportionally to the transformer compute budget spent).
    """
    if strength <= 0.0:
        return dx
    # Scale strength by the transformer-NFE cost of this step relative to one correction call.
    nfe_weight = 2.0 if cfg > 1.0 else 1.0
    strength = strength * nfe_weight
    ctx.correction_nfe += 1
    state = getattr(reward_model, "state", None)
    ir = getattr(state, "imagereward", None) if state is not None else None
    if ir is None:
        return dx

    import torch.nn.functional as F  # noqa: PLC0415

    ir_p = next(ir.parameters())
    ir_device = ir_p.device
    ir_dtype = ir_p.dtype  # respect float16 / bfloat16 IR model
    # Match VAE dtype so the decode forward pass doesn't get a type mismatch.
    vae_dtype = next(ctx.pipe.vae.parameters()).dtype

    # Free fragmented allocations before the expensive autograd pass.
    torch.cuda.empty_cache()

    # Enable VAE gradient checkpointing to trade recompute for activation memory.
    vae_gc_was_enabled = getattr(ctx.pipe.vae, "gradient_checkpointing", False)
    if not vae_gc_was_enabled:
        try:
            ctx.pipe.vae.enable_gradient_checkpointing()
        except Exception:
            pass  # not all VAE versions support it; proceed anyway

    # Decode at half latent resolution — 4× less activation memory.
    # The correction is a nudge; low-res gradient direction is sufficient.
    dx_half = F.interpolate(
        dx.detach().to(dtype=vae_dtype), scale_factor=0.5, mode="bilinear", align_corners=False
    ).requires_grad_(True)

    try:
        with torch.enable_grad():
            shift = getattr(ctx.pipe.vae.config, "shift_factor", 0.0)
            img = ctx.pipe.vae.decode(
                (dx_half / ctx.pipe.vae.config.scaling_factor) + shift,
                return_dict=False,
            )[0]
            img_01 = ((img + 1.0) / 2.0).clamp(0.0, 1.0)
            img_224 = F.interpolate(
                img_01.to(device=ir_device, dtype=ir_dtype), size=(224, 224), mode="bicubic", align_corners=False
            )
            mean = torch.tensor(
                [0.48145466, 0.4578275, 0.40821073], device=ir_device, dtype=ir_dtype
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                [0.26862954, 0.26130258, 0.27577711], device=ir_device, dtype=ir_dtype
            ).view(1, 3, 1, 1)
            img_norm = (img_224 - mean) / std
            image_embeds = ir.blip.visual_encoder(img_norm)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=ir_device)
            text_tok = ir.blip.tokenizer(
                prompt, padding="max_length", truncation=True, max_length=35, return_tensors="pt"
            ).to(ir_device)
            text_out = ir.blip.text_encoder(
                text_tok.input_ids,
                attention_mask=text_tok.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            txt_feat = text_out.last_hidden_state[:, 0, :]
            reward = ir.mlp(txt_feat)
            reward = (reward - ir.mean) / ir.std
            grad_half = torch.autograd.grad(reward.squeeze(), dx_half)[0]
    finally:
        if not vae_gc_was_enabled:
            try:
                ctx.pipe.vae.disable_gradient_checkpointing()
            except Exception:
                pass

    # Upsample gradient back to full latent resolution and apply correction.
    grad = F.interpolate(grad_half.detach(), size=dx.shape[-2:], mode="bilinear", align_corners=False)
    result = (dx + strength * grad.to(dx.device, dtype=dx.dtype)).detach()
    del grad, grad_half, dx_half
    torch.cuda.empty_cache()
    return result


def score_image(reward_model: UnifiedRewardScorer, prompt: str, image: Image.Image) -> float:
    return float(reward_model.score(prompt, image))


def step_schedule(
    device: str,
    dtype: torch.dtype,
    steps: int,
    sigmas: list[float] | None = None,
    euler: bool = False,
    shift: float = 3.0,
) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Return (t_flat, t_4d, dt) triples for each denoising step.

    ``dt`` is the Euler step size (sigma_next - sigma_current), negative because
    sigma decreases.  For SiD/SenseFlow ``dt`` is unused by the caller.

    If ``sigmas`` is provided it is used as the sigma schedule directly
    (SenseFlow style: e.g. [1.0, 0.75] for 2-step Large).
    If ``euler`` is True, use FlowMatchEulerDiscreteScheduler-style shifted
    schedule (proper for SD3.5 base).
    Otherwise a uniform linear schedule over ``steps`` is used (SiD).
    """
    if sigmas is not None:
        sigma_list = [float(s) for s in sigmas]
    elif euler:
        # Replicate FlowMatchEulerDiscreteScheduler with shift (SD3.5 uses shift=3.0)
        # timesteps are linearly spaced, then shifted: sigma = shift * t / (1 + (shift-1) * t)
        ts = torch.linspace(1, 0, steps + 1)  # [1.0, ..., 0.0] with steps+1 points
        sigmas_raw = ts  # base sigmas = t
        # Apply shift: sigma = shift * s / (1 + (shift - 1) * s)
        shifted = shift * sigmas_raw / (1.0 + (shift - 1.0) * sigmas_raw)
        sigma_list = [float(shifted[i]) for i in range(steps)]
        sigma_next_list = [float(shifted[i + 1]) for i in range(steps)]
    else:
        sigma_list = [1.0 - float(i) / float(steps) for i in range(steps)]

    sched: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    for i, s in enumerate(sigma_list):
        t_flat = torch.full((1,), s, device=device, dtype=dtype)
        t_4d = t_flat.view(1, 1, 1, 1)
        if euler and sigmas is None:
            dt = sigma_next_list[i] - s  # negative
        elif sigmas is not None and i + 1 < len(sigma_list):
            dt = sigma_list[i + 1] - s
        else:
            dt = -s  # last step goes to 0
        sched.append((t_flat, t_4d, dt))
    return sched


def _pred_x0(
    xt: torch.Tensor,
    t: torch.Tensor,
    out: torch.Tensor,
    x0_sampler: bool,
) -> torch.Tensor:
    """Convert transformer output to x̂₀.

    Flow sampler (SiD):      x̂₀ = xt - t * flow
    X0 sampler (SenseFlow):  x̂₀ = out  (transformer already predicts x̂₀)
    """
    return out if x0_sampler else xt - t * out


# ── Denoising step helpers ───────────────────────────────────────────────────
# These encapsulate the difference between SiD (re-noising + x0 prediction)
# and standard Euler ODE stepping (for sd35_base).

def _prepare_latents(
    latents: torch.Tensor,
    dx: torch.Tensor,
    noise: torch.Tensor,
    t_4d: torch.Tensor,
    step_idx: int,
    use_euler: bool,
) -> torch.Tensor:
    """Prepare noisy latents for the current step.

    Euler:  latents are already at the right noise level (Euler ODE state).
    SiD:    reconstruct noisy latents from predicted x0 (dx) + noise.
    """
    if use_euler:
        return latents  # already at correct sigma
    noise_val = latents if step_idx == 0 else noise
    return (1.0 - t_4d) * dx + t_4d * noise_val


def _apply_step(
    latents: torch.Tensor,
    flow: torch.Tensor,
    dx: torch.Tensor,
    t_4d: torch.Tensor,
    dt: float,
    use_euler: bool,
    x0_sampler: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one denoising step.  Returns (updated_latents, updated_dx).

    Euler:  latents_next = latents + dt * flow;  dx = latents - t * flow (for scoring)
    SiD:    dx = pred_x0(latents, t, flow);  latents unchanged (re-noised next iter)
    """
    if use_euler:
        latents_next = latents + dt * flow
        dx_new = latents - t_4d * flow  # x0 estimate for scoring
        return latents_next, dx_new
    else:
        dx_new = _pred_x0(latents, t_4d, flow, x0_sampler)
        return latents, dx_new


def _final_decode_tensor(
    latents: torch.Tensor,
    dx: torch.Tensor,
    use_euler: bool,
) -> torch.Tensor:
    """Return the tensor to pass to VAE decode."""
    return latents if use_euler else dx


def run_baseline(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    seed: int,
    cfg_scale: float = 1.0,
) -> tuple[Image.Image, float]:
    latents = make_latents(ctx, seed, args.height, args.width, emb.cond_text[0].dtype)
    use_euler = getattr(args, "euler_sampler", False)
    sched = step_schedule(ctx.device, latents.dtype, args.steps,
                          getattr(args, "sigmas", None), euler=use_euler)
    if use_euler:
        # Standard Euler ODE: latents starts as pure noise, step forward
        for i, (t_flat, t_4d, dt) in enumerate(sched):
            flow = transformer_step(args, ctx, latents, emb, 0, t_flat, float(cfg_scale))
            latents = latents + dt * flow
        image = decode_to_pil(ctx, latents)
    else:
        # SiD / SenseFlow: re-noise + predict x0
        dx = torch.zeros_like(latents)
        for i, (t_flat, t_4d, dt) in enumerate(sched):
            noise = latents if i == 0 else torch.randn_like(latents)
            latents = (1.0 - t_4d) * dx + t_4d * noise
            flow = transformer_step(args, ctx, latents, emb, 0, t_flat, float(cfg_scale))
            dx = _pred_x0(latents, t_4d, flow, args.x0_sampler)
        image = decode_to_pil(ctx, dx)
    return image, score_image(reward_model, prompt, image)


@torch.no_grad()
def run_baseline_batch(
    args: argparse.Namespace,
    ctx: PipelineContext,
    embs: list[EmbeddingContext],
    reward_model: UnifiedRewardScorer,
    prompts: list[str],
    seeds: list[int],
    cfg_scale: float = 1.0,
) -> list[tuple[Image.Image, float]]:
    """Baseline generation for B prompts batched through the transformer simultaneously."""
    B = len(prompts)
    dtype = embs[0].cond_text[0].dtype
    latents = torch.cat([make_latents(ctx, s, args.height, args.width, dtype) for s in seeds])
    use_euler = getattr(args, "euler_sampler", False)
    sched = step_schedule(ctx.device, latents.dtype, args.steps,
                          getattr(args, "sigmas", None), euler=use_euler)

    enc_hs = torch.cat([emb.cond_text[0] for emb in embs])    # [B, seq, dim]
    pooled = torch.cat([emb.cond_pooled[0] for emb in embs])  # [B, dim]
    if cfg_scale != 1.0 and cfg_scale != 0.0:
        uncond_hs = torch.cat([emb.uncond_text for emb in embs])
        uncond_pooled = torch.cat([emb.uncond_pooled for emb in embs])

    dx = torch.zeros_like(latents)
    for i, (t_flat, t_4d, dt) in enumerate(sched):
        if not use_euler:
            noise = latents if i == 0 else torch.randn_like(latents)
            latents = (1.0 - t_4d) * dx + t_4d * noise
        t_batch = t_flat.expand(B)
        if cfg_scale == 1.0 or cfg_scale == 0.0:
            ctx.nfe += B
            flow = ctx.pipe.transformer(
                hidden_states=latents,
                encoder_hidden_states=enc_hs,
                pooled_projections=pooled,
                timestep=args.time_scale * t_batch,
                return_dict=False,
            )[0]
        else:
            ctx.nfe += 2 * B
            flow_both = ctx.pipe.transformer(
                hidden_states=torch.cat([latents, latents]),
                encoder_hidden_states=torch.cat([uncond_hs, enc_hs]),
                pooled_projections=torch.cat([uncond_pooled, pooled]),
                timestep=args.time_scale * torch.cat([t_batch, t_batch]),
                return_dict=False,
            )[0]
            flow_u, flow_c = flow_both.chunk(2)
            flow = flow_u + cfg_scale * (flow_c - flow_u)
        if use_euler:
            latents = latents + dt * flow
        else:
            dx = _pred_x0(latents, t_4d, flow, args.x0_sampler)

    final = latents if use_euler else dx
    out: list[tuple[Image.Image, float]] = []
    for j in range(B):
        img = decode_to_pil(ctx, final[j : j + 1])
        out.append((img, score_image(reward_model, prompts[j], img)))
    return out


@torch.no_grad()
def run_greedy_batch(
    args: argparse.Namespace,
    ctx: PipelineContext,
    embs: list[EmbeddingContext],
    reward_model: UnifiedRewardScorer,
    prompts: list[str],
    variants_list: list[list[str]],
    seeds: list[int],
) -> list[SearchResult]:
    """Greedy search for B prompts. For each (variant, cfg) action, all B prompts are run
    through the transformer in one batched forward pass."""
    B = len(prompts)
    dtype = embs[0].cond_text[0].dtype
    corr_strengths = list(getattr(args, "correction_strengths", [0.0]))
    actions = [
        (vi, cfg, cs)
        for vi in range(len(embs[0].cond_text))
        for cfg in args.cfg_scales
        for cs in corr_strengths
    ]
    latents = torch.cat([make_latents(ctx, s, args.height, args.width, dtype) for s in seeds])
    dx = torch.zeros_like(latents)
    use_euler = getattr(args, "euler_sampler", False)
    sched = step_schedule(ctx.device, latents.dtype, args.steps,
                          getattr(args, "sigmas", None), euler=use_euler)
    chosen_per: list[list[tuple[int, float, float]]] = [[] for _ in range(B)]

    # Group actions by (variant_idx, cfg) so each transformer call covers all correction variants.
    # correction is applied after the flow step, so the transformer is called once per (vi, cfg) pair.
    vc_actions: list[tuple[int, float]] = list(dict.fromkeys((vi, cfg) for vi, cfg, _ in actions))

    for step_idx, (t_flat, t_4d, dt) in enumerate(sched):
        latents = _prepare_latents(latents, dx, torch.randn_like(latents), t_4d, step_idx, use_euler)

        best_scores = [-float("inf")] * B
        best_actions: list[tuple[int, float, float]] = [actions[0]] * B
        best_dx_list = [dx[j : j + 1].clone() for j in range(B)]
        best_latents_list = [latents[j : j + 1].clone() for j in range(B)]

        for variant_idx, cfg in vc_actions:
            enc_hs = torch.cat([embs[j].cond_text[variant_idx] for j in range(B)])
            pooled = torch.cat([embs[j].cond_pooled[variant_idx] for j in range(B)])
            t_batch = t_flat.expand(B)
            if cfg == 1.0 or cfg == 0.0:
                ctx.nfe += B
                flow = ctx.pipe.transformer(
                    hidden_states=latents,
                    encoder_hidden_states=enc_hs,
                    pooled_projections=pooled,
                    timestep=args.time_scale * t_batch,
                    return_dict=False,
                )[0]
            else:
                ctx.nfe += 2 * B
                uncond_hs = torch.cat([embs[j].uncond_text for j in range(B)])
                uncond_pooled = torch.cat([embs[j].uncond_pooled for j in range(B)])
                flow_both = ctx.pipe.transformer(
                    hidden_states=torch.cat([latents, latents]),
                    encoder_hidden_states=torch.cat([uncond_hs, enc_hs]),
                    pooled_projections=torch.cat([uncond_pooled, pooled]),
                    timestep=args.time_scale * torch.cat([t_batch, t_batch]),
                    return_dict=False,
                )[0]
                flow_u, flow_c = flow_both.chunk(2)
                flow = flow_u + cfg * (flow_c - flow_u)

            _, cand_dx = _apply_step(latents, flow, dx, t_4d, dt, use_euler, args.x0_sampler)
            for cs in corr_strengths:
                for j in range(B):
                    cand_dx_j = cand_dx[j : j + 1].clone()
                    if cs > 0.0:
                        cand_dx_j = apply_reward_correction(ctx, cand_dx_j, prompts[j], reward_model, cs, cfg=cfg)
                    cand_latents_j = (latents[j:j+1] + dt * flow[j:j+1]) if use_euler else latents[j:j+1]
                    cand_img = decode_to_pil(ctx, _final_decode_tensor(cand_latents_j, cand_dx_j, use_euler))
                    cand_score = score_image(reward_model, prompts[j], cand_img)
                    if cand_score > best_scores[j]:
                        best_scores[j] = cand_score
                        best_actions[j] = (variant_idx, cfg, cs)
                        best_dx_list[j] = cand_dx_j
                        if use_euler:
                            best_latents_list[j] = cand_latents_j

        if use_euler:
            latents = torch.cat(best_latents_list)
        dx = torch.cat(best_dx_list)
        for j in range(B):
            chosen_per[j].append(best_actions[j])

    results = []
    final = _final_decode_tensor(latents, dx, use_euler)
    for j in range(B):
        img = decode_to_pil(ctx, final[j : j + 1])
        score = score_image(reward_model, prompts[j], img)
        results.append(SearchResult(image=img, score=score, actions=chosen_per[j]))
    return results


@torch.no_grad()
def run_bon(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    seed: int,
) -> SearchResult:
    """Best of N: generate N independent samples with baseline config, return highest-scoring."""
    n = max(1, int(args.bon_n))
    cfg = float(args.baseline_cfg)
    variant_idx = min(0, len(emb.cond_text) - 1)
    fixed_actions: list[tuple[int, float, float]] = [(variant_idx, cfg, 0.0)] * args.steps

    best_score = -float("inf")
    best_img: Image.Image | None = None

    use_euler = getattr(args, "euler_sampler", False)
    print(f"  bon: n={n} cfg={cfg:.2f} variant={variant_idx} euler={use_euler}")
    for i in range(n):
        s = seed + i
        latents = make_latents(ctx, s, args.height, args.width, emb.cond_text[0].dtype)
        dx = torch.zeros_like(latents)
        sched = step_schedule(ctx.device, latents.dtype, args.steps,
                              getattr(args, "sigmas", None), euler=use_euler)
        for j, (t_flat, t_4d, step_dt) in enumerate(sched):
            latents = _prepare_latents(latents, dx, torch.randn_like(latents), t_4d, j, use_euler)
            flow = transformer_step(args, ctx, latents, emb, variant_idx, t_flat, cfg)
            latents, dx = _apply_step(latents, flow, dx, t_4d, step_dt, use_euler, args.x0_sampler)
        img = decode_to_pil(ctx, _final_decode_tensor(latents, dx, use_euler))
        score = score_image(reward_model, prompt, img)
        mark = ""
        if score > best_score:
            best_score = score
            best_img = img
            mark = " <- best"
        print(f"    sample {i + 1}/{n} seed={s} score={score:.4f}{mark}")

    assert best_img is not None
    return SearchResult(
        image=best_img,
        score=best_score,
        actions=fixed_actions,
        diagnostics={"bon_n": n, "cfg": cfg},
    )


def _resolve_noise_inject_steps(args: argparse.Namespace, total_steps: int) -> list[int]:
    raw = str(getattr(args, "noise_inject_candidate_steps", "")).strip().lower()
    if total_steps <= 0:
        return [0]
    if not raw:
        return [int(total_steps // 2)]
    if raw == "all":
        return list(range(int(total_steps)))
    vals: list[int] = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        if t.isdigit():
            idx = int(t)
            if 0 <= idx < int(total_steps):
                vals.append(int(idx))
    vals = sorted(set(vals))
    if not vals:
        return [int(total_steps // 2)]
    return vals


def _noise_inject_gamma_bank(args: argparse.Namespace) -> list[float]:
    out: list[float] = []
    for v in getattr(args, "noise_inject_gamma_bank", [0.0, 0.25, 0.5]):
        vv = float(v)
        if vv in out:
            continue
        out.append(vv)
    return out if out else [0.0]


def _noise_inject_cfg_bank(args: argparse.Namespace) -> tuple[list[float], bool]:
    override = getattr(args, "noise_inject_cfg", None)
    if override is not None:
        return [float(override)], True

    baseline = float(getattr(args, "baseline_cfg", 1.0))
    out: list[float] = []
    for v in getattr(args, "cfg_scales", []):
        vv = float(v)
        if vv in out:
            continue
        out.append(vv)
    if not out:
        out = [baseline]
    elif baseline not in out:
        out = [baseline] + out
    return out, False


def _build_noise_inject_policies(
    args: argparse.Namespace,
    total_steps: int,
) -> list[tuple[tuple[int, float, int], ...]]:
    mode = str(getattr(args, "noise_inject_mode", "combined")).strip().lower()
    if mode == "seeds_only":
        return [tuple()]

    steps = _resolve_noise_inject_steps(args, int(total_steps))
    gammas = _noise_inject_gamma_bank(args)
    eps_n = max(1, int(getattr(args, "noise_inject_eps_samples", 4)))
    steps_per_rollout = max(1, min(2, int(getattr(args, "noise_inject_steps_per_rollout", 1))))
    include_no = bool(getattr(args, "noise_inject_include_no_inject", True))

    policies: list[tuple[tuple[int, float, int], ...]] = []
    if include_no:
        policies.append(tuple())

    if steps_per_rollout <= 1:
        for step_idx in steps:
            for gamma in gammas:
                if include_no and abs(float(gamma)) <= 1e-12:
                    continue
                for eps_id in range(eps_n):
                    policies.append(((int(step_idx), float(gamma), int(eps_id)),))
    else:
        for s1, s2 in combinations(steps, 2):
            for gamma in gammas:
                if include_no and abs(float(gamma)) <= 1e-12:
                    continue
                for eps_id in range(eps_n):
                    policies.append(
                        (
                            (int(s1), float(gamma), int(eps_id)),
                            (int(s2), float(gamma), int(eps_id)),
                        )
                    )

    # Stable dedup to avoid duplicates when candidate banks overlap.
    seen: set[tuple[tuple[int, float, int], ...]] = set()
    uniq: list[tuple[tuple[int, float, int], ...]] = []
    for policy in policies:
        key = tuple((int(s), float(g), int(e)) for s, g, e in policy)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(key)

    max_policies = max(0, int(getattr(args, "noise_inject_max_policies", 0)))
    if max_policies > 0 and len(uniq) > max_policies:
        uniq = uniq[:max_policies]
    return uniq if uniq else [tuple()]


def _build_step_noise_cache(
    init_latents: torch.Tensor,
    steps: int,
    seed: int,
) -> list[torch.Tensor]:
    out: list[torch.Tensor] = [init_latents.clone()]
    if steps <= 1:
        return out
    gen = torch.Generator(device=init_latents.device).manual_seed(int(seed) + 2048)
    for _ in range(1, int(steps)):
        out.append(
            torch.randn(
                init_latents.shape,
                device=init_latents.device,
                dtype=init_latents.dtype,
                generator=gen,
            )
        )
    return out


def _build_eps_bank_like(
    like: torch.Tensor,
    n_eps: int,
    seed: int,
) -> list[torch.Tensor]:
    gen = torch.Generator(device=like.device).manual_seed(int(seed))
    out: list[torch.Tensor] = []
    for _ in range(max(1, int(n_eps))):
        out.append(
            torch.randn(
                like.shape,
                device=like.device,
                dtype=like.dtype,
                generator=gen,
            )
        )
    return out


@torch.no_grad()
def _run_sparse_noise_rollout(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    init_latents: torch.Tensor,
    step_noise_cache: list[torch.Tensor],
    eps_bank: list[torch.Tensor],
    fixed_variant_idx: int,
    fixed_cfg: float,
    sched: list[tuple[torch.Tensor, torch.Tensor, float]],
    policy: tuple[tuple[int, float, int], ...],
) -> tuple[Image.Image, float]:
    use_euler = bool(getattr(args, "euler_sampler", False))
    latents = init_latents.clone()
    dx = torch.zeros_like(init_latents)

    inject_map: dict[int, tuple[float, int]] = {}
    for step_idx, gamma, eps_id in policy:
        inject_map[int(step_idx)] = (float(gamma), int(eps_id))

    for step_idx, (t_flat, t_4d, dt) in enumerate(sched):
        if not use_euler:
            base_noise = step_noise_cache[step_idx] if step_idx < len(step_noise_cache) else step_noise_cache[-1]
            latents = (1.0 - t_4d) * dx + t_4d * base_noise

        if step_idx in inject_map:
            gamma, eps_id = inject_map[step_idx]
            eps = eps_bank[int(eps_id) % len(eps_bank)]
            latents = latents + float(gamma) * t_4d * eps

        flow = transformer_step(args, ctx, latents, emb, int(fixed_variant_idx), t_flat, float(fixed_cfg))
        latents, dx = _apply_step(latents, flow, dx, t_4d, dt, use_euler, args.x0_sampler)

    final_img = decode_to_pil(ctx, _final_decode_tensor(latents, dx, use_euler))
    final_score = score_image(reward_model, prompt, final_img)
    return final_img, float(final_score)


def run_noise_inject(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    seed: int,
) -> SearchResult:
    mode = str(getattr(args, "noise_inject_mode", "combined")).strip().lower()
    use_euler = bool(getattr(args, "euler_sampler", False))
    seed_budget = max(1, int(getattr(args, "noise_inject_seed_budget", 8)))

    if mode == "reinject_only":
        seed_candidates = [int(seed)]
    else:
        seed_candidates = [int(seed) + i for i in range(seed_budget)]

    # Resolve schedule once using the base seed; dtype/schedule are seed-independent.
    base_latents = make_latents(ctx, int(seed_candidates[0]), args.height, args.width, emb.cond_text[0].dtype)
    sched = step_schedule(
        ctx.device,
        base_latents.dtype,
        int(args.steps),
        getattr(args, "sigmas", None),
        euler=use_euler,
    )
    n_steps = len(sched)

    policies = _build_noise_inject_policies(args, n_steps)
    candidate_steps = _resolve_noise_inject_steps(args, n_steps)
    gamma_bank = _noise_inject_gamma_bank(args)
    eps_samples = max(1, int(getattr(args, "noise_inject_eps_samples", 4)))
    steps_per_rollout = max(1, min(2, int(getattr(args, "noise_inject_steps_per_rollout", 1))))

    fixed_variant_idx = int(getattr(args, "noise_inject_variant_idx", 0))
    fixed_variant_idx = max(0, min(len(emb.cond_text) - 1, fixed_variant_idx))
    cfg_bank, cfg_is_override = _noise_inject_cfg_bank(args)

    print(
        "  noise_inject: "
        f"mode={mode} seeds={len(seed_candidates)} policies_per_seed={len(policies)} "
        f"cfg_candidates={len(cfg_bank)} steps_per_rollout={steps_per_rollout} euler={use_euler}"
    )
    print(
        "    "
        f"candidate_steps={candidate_steps} gamma_bank={[float(g) for g in gamma_bank]} "
        f"eps_samples={eps_samples} cfg_bank={[float(c) for c in cfg_bank]} variant={fixed_variant_idx}"
    )

    nfe_start = int(ctx.nfe)
    corr_start = int(ctx.correction_nfe)
    t_start = time.perf_counter()

    best_score = -float("inf")
    best_img: Image.Image | None = None
    best_seed = int(seed_candidates[0])
    best_policy: tuple[tuple[int, float, int], ...] = tuple()
    best_cfg = float(cfg_bank[0])

    total_rollouts = len(seed_candidates) * len(cfg_bank) * len(policies)
    rollout_idx = 0
    for seed_i, seed_val in enumerate(seed_candidates):
        init_latents = make_latents(ctx, int(seed_val), args.height, args.width, emb.cond_text[0].dtype)
        step_noise_cache = _build_step_noise_cache(init_latents, n_steps, int(seed_val))
        eps_bank = _build_eps_bank_like(init_latents, eps_samples, seed=int(seed_val) + 900001)

        for cfg_val in cfg_bank:
            for policy in policies:
                rollout_idx += 1
                img, score = _run_sparse_noise_rollout(
                    args=args,
                    ctx=ctx,
                    emb=emb,
                    reward_model=reward_model,
                    prompt=prompt,
                    init_latents=init_latents,
                    step_noise_cache=step_noise_cache,
                    eps_bank=eps_bank,
                    fixed_variant_idx=fixed_variant_idx,
                    fixed_cfg=float(cfg_val),
                    sched=sched,
                    policy=policy,
                )
                if score > best_score:
                    best_score = float(score)
                    best_img = img
                    best_seed = int(seed_val)
                    best_policy = tuple(policy)
                    best_cfg = float(cfg_val)
                if rollout_idx == 1 or rollout_idx % 10 == 0 or rollout_idx == total_rollouts:
                    print(f"    rollout {rollout_idx:4d}/{total_rollouts} best={best_score:.4f}")
        if seed_i + 1 < len(seed_candidates):
            del init_latents

    assert best_img is not None
    fixed_actions: list[tuple[int, float, float]] = [(fixed_variant_idx, best_cfg, 0.0)] * int(n_steps)

    search_nfe = int(ctx.nfe) - nfe_start
    search_corr_nfe = int(ctx.correction_nfe) - corr_start
    elapsed = float(time.perf_counter() - t_start)
    diagnostics = {
        "mode": mode,
        "seed_candidates": [int(s) for s in seed_candidates],
        "candidate_steps": [int(s) for s in candidate_steps],
        "gamma_bank": [float(g) for g in gamma_bank],
        "cfg_bank": [float(c) for c in cfg_bank],
        "cfg_source": "override" if cfg_is_override else "cfg_scales",
        "eps_samples": int(eps_samples),
        "steps_per_rollout": int(steps_per_rollout),
        "policies_per_seed": int(len(policies)),
        "cfg_candidates": int(len(cfg_bank)),
        "rollouts_evaluated": int(total_rollouts),
        "equivalent_seed_budget": int(total_rollouts),
        "search_nfe_transformer": int(search_nfe),
        "search_nfe_correction": int(search_corr_nfe),
        "wall_time_sec": float(elapsed),
        "fixed_variant_idx": int(fixed_variant_idx),
        "fixed_cfg": float(best_cfg),
        "chosen_cfg": float(best_cfg),
        "chosen_seed": int(best_seed),
        "chosen_policy": [
            {"step": int(step), "gamma": float(gamma), "eps_id": int(eps_id)}
            for step, gamma, eps_id in best_policy
        ],
        "no_injection_selected": bool(len(best_policy) == 0),
        "reward_per_rollout": float(best_score) / float(max(1, total_rollouts)),
    }
    return SearchResult(
        image=best_img,
        score=float(best_score),
        actions=fixed_actions,
        diagnostics=diagnostics,
    )


@torch.no_grad()
def run_beam(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    variants: list[str],
    seed: int,
) -> SearchResult:
    """Beam search: maintain top-K partial trajectories, expand with full action space per step."""
    corr_strengths = list(getattr(args, "correction_strengths", [0.0]))
    actions = [
        (vi, cfg, cs)
        for vi in range(len(variants))
        for cfg in args.cfg_scales
        for cs in corr_strengths
    ]
    beam_width = max(1, int(args.beam_width))
    dtype = emb.cond_text[0].dtype
    use_euler = getattr(args, "euler_sampler", False)
    sched = step_schedule(ctx.device, dtype, args.steps,
                          getattr(args, "sigmas", None), euler=use_euler)

    init_latents = make_latents(ctx, seed, args.height, args.width, dtype)
    # beams: list of (dx, latents, path)
    beams: list[tuple[torch.Tensor, torch.Tensor, list[tuple[int, float, float]]]] = [
        (torch.zeros_like(init_latents), init_latents, [])
    ]

    print(f"  beam: width={beam_width} actions={len(actions)} steps={args.steps} euler={use_euler}")
    for step_idx, (t_flat, t_4d, dt) in enumerate(sched):
        # candidates: (score, dx, latents, path)
        candidates: list[tuple[float, torch.Tensor, torch.Tensor, list[tuple[int, float, float]]]] = []

        for beam_dx, beam_latents, beam_path in beams:
            latents_in = _prepare_latents(beam_latents, beam_dx, torch.randn_like(beam_latents), t_4d, step_idx, use_euler)
            # Cache transformer output per (vi, cfg) to avoid redundant forward passes
            vc_cache: dict[tuple[int, float], torch.Tensor] = {}
            for vi, cfg, cs in actions:
                vc_key = (vi, cfg)
                if vc_key not in vc_cache:
                    vc_cache[vc_key] = transformer_step(args, ctx, latents_in, emb, vi, t_flat, cfg)
                flow = vc_cache[vc_key]
                _, cand_dx = _apply_step(latents_in, flow, beam_dx, t_4d, dt, use_euler, args.x0_sampler)
                cand_latents = latents_in + dt * flow if use_euler else latents_in
                if cs > 0.0:
                    cand_dx = apply_reward_correction(ctx, cand_dx, prompt, reward_model, cs, cfg=cfg)
                cand_img = decode_to_pil(ctx, _final_decode_tensor(cand_latents, cand_dx, use_euler))
                cand_score = score_image(reward_model, prompt, cand_img)
                candidates.append((cand_score, cand_dx, cand_latents, beam_path + [(vi, cfg, cs)]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        print(
            f"  step {step_idx + 1}/{args.steps}: {len(candidates)} candidates, "
            f"top={candidates[0][0]:.4f}"
        )

        beams = []
        for _, dx, lat, path in candidates[:beam_width]:
            beams.append((dx, lat, path))

    best_dx, best_latents, best_path = beams[0]
    best_img = decode_to_pil(ctx, _final_decode_tensor(best_latents, best_dx, use_euler))
    final_score = score_image(reward_model, prompt, best_img)
    return SearchResult(
        image=best_img,
        score=final_score,
        actions=best_path,
        diagnostics={"beam_width": beam_width, "n_candidates_per_step": len(actions)},
    )


def run_greedy(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    variants: list[str],
    seed: int,
) -> SearchResult:
    corr_strengths = list(getattr(args, "correction_strengths", [0.0]))
    actions = [
        (vi, cfg, cs)
        for vi in range(len(variants))
        for cfg in args.cfg_scales
        for cs in corr_strengths
    ]
    latents = make_latents(ctx, seed, args.height, args.width, emb.cond_text[0].dtype)
    dx = torch.zeros_like(latents)
    use_euler = getattr(args, "euler_sampler", False)
    sched = step_schedule(ctx.device, latents.dtype, args.steps,
                          getattr(args, "sigmas", None), euler=use_euler)
    chosen: list[tuple[int, float, float]] = []
    for step_idx, (t_flat, t_4d, dt) in enumerate(sched):
        latents = _prepare_latents(latents, dx, torch.randn_like(latents), t_4d, step_idx, use_euler)

        best_score = -float("inf")
        best_action = actions[0]
        best_dx = None
        best_latents = None

        # Compute transformer flow once per (variant_idx, cfg) pair, then branch on correction.
        vc_seen: dict[tuple[int, float], torch.Tensor] = {}
        print(f"  step {step_idx + 1}/{args.steps}: {len(actions)} actions")
        for variant_idx, cfg, cs in actions:
            vc_key = (variant_idx, cfg)
            if vc_key not in vc_seen:
                vc_seen[vc_key] = transformer_step(args, ctx, latents, emb, variant_idx, t_flat, cfg)
            flow = vc_seen[vc_key]
            _, cand_dx = _apply_step(latents, flow, dx, t_4d, dt, use_euler, args.x0_sampler)
            if cs > 0.0:
                cand_dx = apply_reward_correction(ctx, cand_dx, prompt, reward_model, cs, cfg=cfg)
            cand_img = decode_to_pil(ctx, cand_dx)
            cand_score = score_image(reward_model, prompt, cand_img)
            marker = ""
            if cand_score > best_score:
                best_score = cand_score
                best_action = (variant_idx, cfg, cs)
                best_dx = cand_dx.clone()
                best_latents = (latents + dt * flow).clone()
                marker = " <- best"
            print(f"    v{variant_idx} cfg={cfg:.2f} corr={cs:.2f} IR={cand_score:.4f}{marker}")

        assert best_dx is not None
        dx = best_dx
        if use_euler:
            latents = best_latents
        chosen.append(best_action)
        preview = variants[best_action[0]][:56]
        print(
            f"  selected v{best_action[0]} cfg={best_action[1]:.2f} corr={best_action[2]:.2f} "
            f"prompt='{preview}' score={best_score:.4f}"
        )

    final_img = decode_to_pil(ctx, _final_decode_tensor(latents, dx, use_euler))
    final_score = score_image(reward_model, prompt, final_img)
    return SearchResult(image=final_img, score=final_score, actions=chosen)


def run_schedule_actions(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    seed: int,
    actions: list[tuple[int, float, float]],
    deterministic_noise: bool = False,
) -> SearchResult:
    if len(actions) != args.steps:
        raise RuntimeError(f"Schedule length {len(actions)} does not match steps={args.steps}")
    if deterministic_noise:
        torch.manual_seed(int(seed))
        if str(ctx.device).startswith("cuda"):
            torch.cuda.manual_seed_all(int(seed))

    latents = make_latents(ctx, seed, args.height, args.width, emb.cond_text[0].dtype)
    dx = torch.zeros_like(latents)
    use_euler = getattr(args, "euler_sampler", False)
    sched = step_schedule(ctx.device, latents.dtype, args.steps,
                          getattr(args, "sigmas", None), euler=use_euler)
    for step_idx, ((t_flat, t_4d, dt), (variant_idx, cfg, cs)) in enumerate(zip(sched, actions)):
        latents = _prepare_latents(latents, dx, torch.randn_like(latents), t_4d, step_idx, use_euler)
        flow = transformer_step(args, ctx, latents, emb, int(variant_idx), t_flat, float(cfg))
        latents, dx = _apply_step(latents, flow, dx, t_4d, dt, use_euler, args.x0_sampler)
        if float(cs) > 0.0:
            dx = apply_reward_correction(ctx, dx, prompt, reward_model, float(cs), cfg=float(cfg))
    image = decode_to_pil(ctx, _final_decode_tensor(latents, dx, use_euler))
    score = score_image(reward_model, prompt, image)
    return SearchResult(image=image, score=score, actions=[(int(v), float(c), float(r)) for v, c, r in actions])


def _batched_flow_for_step_actions(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    latents: torch.Tensor,
    t_flat: torch.Tensor,
    step_actions: list[tuple[int, float, float]],
) -> torch.Tensor:
    """Batch transformer calls grouped by (variant_idx, cfg); correction is ignored here."""
    flow_out = torch.empty_like(latents)
    groups: dict[tuple[int, float], list[int]] = {}
    for idx, (variant_idx, cfg, _cs) in enumerate(step_actions):
        key = (int(variant_idx), float(cfg))
        groups.setdefault(key, []).append(int(idx))

    for (variant_idx, cfg), idxs in groups.items():
        idx_t = torch.tensor(idxs, device=latents.device, dtype=torch.long)
        sub_latents = latents.index_select(0, idx_t)
        sub_t = t_flat.expand(sub_latents.shape[0])
        sub_flow = transformer_step(args, ctx, sub_latents, emb, int(variant_idx), sub_t, float(cfg))
        flow_out.index_copy_(0, idx_t, sub_flow)
    return flow_out


@torch.no_grad()
def score_schedule_actions_batch(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    seed: int,
    actions_batch: list[list[tuple[int, float, float]]],
    deterministic_noise: bool = True,
) -> list[float]:
    if len(actions_batch) == 0:
        return []
    sigmas = getattr(args, "sigmas", None)
    steps = len(sigmas) if sigmas is not None else int(args.steps)
    for actions in actions_batch:
        if len(actions) != steps:
            raise RuntimeError(f"Schedule length {len(actions)} does not match steps={steps}")

    if deterministic_noise:
        torch.manual_seed(int(seed))
        if str(ctx.device).startswith("cuda"):
            torch.cuda.manual_seed_all(int(seed))

    batch_n = len(actions_batch)
    base_latents = make_latents(ctx, seed, args.height, args.width, emb.cond_text[0].dtype)
    latents = base_latents.repeat(batch_n, 1, 1, 1)
    dx = torch.zeros_like(latents)
    use_euler = getattr(args, "euler_sampler", False)
    sched = step_schedule(ctx.device, latents.dtype, steps, sigmas, euler=use_euler)
    shared_noises: list[torch.Tensor] = [base_latents]
    for _ in range(1, steps):
        shared_noises.append(torch.randn_like(base_latents))

    for step_idx, (t_flat, t_4d, dt) in enumerate(sched):
        if not use_euler:
            if step_idx == 0:
                noise = latents
            else:
                noise = shared_noises[step_idx].expand(batch_n, -1, -1, -1)
            latents = (1.0 - t_4d) * dx + t_4d * noise
        step_actions = [actions[step_idx] for actions in actions_batch]
        flow = _batched_flow_for_step_actions(args, ctx, emb, latents, t_flat, step_actions)
        latents, dx = _apply_step(latents, flow, dx, t_4d, dt, use_euler, args.x0_sampler)
        # Apply per-genome correction (uses enable_grad internally, safe under @no_grad).
        for bi, (_vi, _cfg, cs) in enumerate(step_actions):
            if float(cs) > 0.0:
                dx[bi : bi + 1] = apply_reward_correction(ctx, dx[bi : bi + 1], prompt, reward_model, float(cs), cfg=float(_cfg))

    final = _final_decode_tensor(latents, dx, use_euler)
    scores: list[float] = []
    for bi in range(batch_n):
        image = decode_to_pil(ctx, final[bi : bi + 1])
        scores.append(score_image(reward_model, prompt, image))
    return scores


def _decode_genome(
    genome: list[int],
    cfg_scales: list[float],
    corr_strengths: list[float],
    steps: int,
) -> list[tuple[int, float, float]]:
    actions: list[tuple[int, float, float]] = []
    for step in range(steps):
        vi = int(genome[3 * step])
        ci = int(genome[3 * step + 1])
        ri = int(genome[3 * step + 2])
        actions.append((vi, float(cfg_scales[ci]), float(corr_strengths[ri])))
    return actions


def _actions_brief(actions: list[tuple[int, float, float]]) -> str:
    return " ".join(f"s{i+1}:v{vi}/cfg{cfg:.2f}/c{cs:.2f}" for i, (vi, cfg, cs) in enumerate(actions))


def _systematic_resample(weights: torch.Tensor) -> torch.Tensor:
    k = int(weights.shape[0])
    cdf = torch.cumsum(weights, dim=0)
    u = (
        torch.rand(1, device=weights.device, dtype=weights.dtype)
        + torch.arange(k, device=weights.device, dtype=weights.dtype)
    ) / float(k)
    return torch.searchsorted(cdf, u).clamp(0, k - 1)


def _closest_cfg_index(cfg_scales: list[float], target: float) -> int:
    return int(min(range(len(cfg_scales)), key=lambda idx: abs(float(cfg_scales[idx]) - float(target))))


def _phase_variant_choices(steps: int, n_variants: int) -> tuple[list[int], list[int]]:
    if n_variants <= 1:
        return [0], [0]
    split = max(1, n_variants // 2)
    early = list(range(split))
    late = list(range(max(0, n_variants - split), n_variants))
    if not early:
        early = list(range(n_variants))
    if not late:
        late = list(range(n_variants))
    return early, late


def _repair_genome(
    genome: list[int],
    steps: int,
    n_variants: int,
    n_cfg: int,
    n_corr: int,
    phase_constraints: bool,
) -> list[int]:
    out = list(genome[: 3 * steps])
    if len(out) < 3 * steps:
        out.extend([0] * (3 * steps - len(out)))
    early_choices, late_choices = _phase_variant_choices(steps, n_variants)
    for step in range(steps):
        v_pos = 3 * step
        c_pos = v_pos + 1
        r_pos = v_pos + 2
        vi = int(out[v_pos]) % max(n_variants, 1)
        ci = int(out[c_pos]) % max(n_cfg, 1)
        ri = int(out[r_pos]) % max(n_corr, 1)
        if phase_constraints and n_variants > 1:
            choices = early_choices if step < max(1, steps // 2) else late_choices
            if vi not in choices:
                vi = choices[vi % len(choices)]
        out[v_pos] = vi
        out[c_pos] = ci
        out[r_pos] = ri
    return out


def _random_genome(
    steps: int,
    n_variants: int,
    n_cfg: int,
    n_corr: int,
    phase_constraints: bool,
) -> list[int]:
    early_choices, late_choices = _phase_variant_choices(steps, n_variants)
    genome: list[int] = []
    for step in range(steps):
        if phase_constraints and n_variants > 1:
            choices = early_choices if step < max(1, steps // 2) else late_choices
        else:
            choices = list(range(n_variants))
        vi = int(choices[np.random.randint(len(choices))])
        ci = int(np.random.randint(n_cfg))
        ri = int(np.random.randint(n_corr))
        genome.extend([vi, ci, ri])
    return genome


def _select_parent_rank(ranked_genomes: list[list[int]], rank_pressure: float) -> list[int]:
    n = len(ranked_genomes)
    if n == 1:
        return list(ranked_genomes[0])
    rp = max(0.0, float(rank_pressure))
    if rp == 0.0:
        probs = np.full((n,), 1.0 / n, dtype=np.float64)
    else:
        ranks = np.arange(n, dtype=np.float64)
        probs = np.exp(-rp * ranks / max(1.0, float(n - 1)))
        probs /= probs.sum()
    idx = int(np.random.choice(n, p=probs))
    return list(ranked_genomes[idx])


def _select_parent_tournament(ranked_genomes: list[list[int]], k: int) -> list[int]:
    n = len(ranked_genomes)
    if n == 1:
        return list(ranked_genomes[0])
    kk = max(2, min(int(k), n))
    picks = [int(np.random.randint(n)) for _ in range(kk)]
    best_idx = min(picks)
    return list(ranked_genomes[best_idx])


def _crossover(parent_a: list[int], parent_b: list[int], mode: str) -> list[int]:
    if len(parent_a) != len(parent_b):
        raise RuntimeError("Parents have mismatched genome length.")
    if len(parent_a) <= 1:
        return list(parent_a)
    if mode == "one_point":
        cut = int(np.random.randint(1, len(parent_a)))
        return list(parent_a[:cut] + parent_b[cut:])
    child = []
    for i in range(len(parent_a)):
        child.append(parent_a[i] if float(np.random.rand()) < 0.5 else parent_b[i])
    return child


def _mutate_genome(
    genome: list[int],
    steps: int,
    n_variants: int,
    n_cfg: int,
    n_corr: int,
    mutation_prob: float,
    phase_constraints: bool,
) -> list[int]:
    out = list(genome)
    p = max(0.0, min(1.0, float(mutation_prob)))
    early_choices, late_choices = _phase_variant_choices(steps, n_variants)
    for step in range(steps):
        v_pos = 3 * step
        c_pos = v_pos + 1
        r_pos = v_pos + 2
        if float(np.random.rand()) < p:
            if phase_constraints and n_variants > 1:
                choices = early_choices if step < max(1, steps // 2) else late_choices
            else:
                choices = list(range(n_variants))
            out[v_pos] = int(choices[np.random.randint(len(choices))])
        if float(np.random.rand()) < p:
            out[c_pos] = int(np.random.randint(n_cfg))
        if float(np.random.rand()) < p:
            out[r_pos] = int(np.random.randint(n_corr))
    return out


def run_ga(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    variants: list[str],
    seed: int,
    log_dir: str | None = None,
    log_prefix: str = "",
) -> SearchResult:
    n_variants = len(variants)
    n_cfg = len(args.cfg_scales)
    corr_strengths = list(getattr(args, "correction_strengths", [0.0]))
    n_corr = max(1, len(corr_strengths))
    if n_variants <= 0 or n_cfg <= 0:
        raise RuntimeError("GA requires non-empty variants and cfg_scales.")

    pop_size = max(2, int(args.ga_population))
    generations = max(1, int(args.ga_generations))
    elites = max(1, min(int(args.ga_elites), pop_size))
    steps = int(args.steps)
    genome_len = 3 * steps

    baseline_cfg_idx = _closest_cfg_index(list(args.cfg_scales), float(args.baseline_cfg))
    baseline_genome: list[int] = []
    for _ in range(steps):
        baseline_genome.extend([0, baseline_cfg_idx, 0])  # correction index 0 = no correction

    population: list[list[int]] = [_repair_genome(baseline_genome, steps, n_variants, n_cfg, n_corr, args.ga_phase_constraints)]
    while len(population) < pop_size:
        g = _random_genome(steps, n_variants, n_cfg, n_corr, args.ga_phase_constraints)
        population.append(_repair_genome(g, steps, n_variants, n_cfg, n_corr, args.ga_phase_constraints))

    score_cache: dict[tuple[int, ...], float] = {}
    best_score = -float("inf")
    best_genome = list(population[0])
    history: list[dict[str, Any]] = []
    eval_calls = 0
    cache_hits = 0
    cache_misses = 0
    prev_eval_calls_total = 0

    def eval_genomes(genomes: list[list[int]]) -> list[tuple[float, list[int]]]:
        nonlocal eval_calls, cache_hits, cache_misses
        prepared: list[tuple[list[int], tuple[int, ...]]] = []
        queued: dict[tuple[int, ...], list[tuple[int, float, float]]] = {}
        for genome in genomes:
            eval_calls += 1
            repaired = _repair_genome(genome, steps, n_variants, n_cfg, n_corr, args.ga_phase_constraints)
            key = tuple(repaired)
            prepared.append((repaired, key))
            if key in score_cache or key in queued:
                cache_hits += 1
                continue
            cache_misses += 1
            queued[key] = _decode_genome(repaired, list(args.cfg_scales), corr_strengths, steps)

        pending_items = list(queued.items())
        eval_batch = max(1, int(args.ga_eval_batch))
        for start in range(0, len(pending_items), eval_batch):
            chunk = pending_items[start : start + eval_batch]
            chunk_keys = [item[0] for item in chunk]
            chunk_actions = [item[1] for item in chunk]
            chunk_scores = score_schedule_actions_batch(
                args,
                ctx,
                emb,
                reward_model,
                prompt,
                seed,
                chunk_actions,
                deterministic_noise=True,
            )
            for key, score in zip(chunk_keys, chunk_scores):
                score_cache[key] = float(score)

        out: list[tuple[float, list[int]]] = []
        for repaired, key in prepared:
            out.append((float(score_cache[key]), list(repaired)))
        return out

    print(
        f"  ga: pop={pop_size} gens={generations} elites={elites} "
        f"selection={args.ga_selection} crossover={args.ga_crossover} "
        f"eval_batch={int(args.ga_eval_batch)}"
    )
    for gen in range(generations):
        scored = eval_genomes(population)
        scored.sort(key=lambda item: item[0], reverse=True)

        gen_best_score, gen_best_genome = scored[0]
        gen_mean = float(np.mean([item[0] for item in scored])) if scored else 0.0
        if gen_best_score > best_score:
            best_score = float(gen_best_score)
            best_genome = list(gen_best_genome)

        topk = max(1, int(args.ga_log_topk))
        gen_top = scored[:topk]
        eval_calls_total = int(eval_calls)
        eval_calls_generation = int(eval_calls_total - prev_eval_calls_total)
        prev_eval_calls_total = eval_calls_total
        nfe_per_generation = int(eval_calls_generation * steps)
        nfe_total = int(eval_calls_total * steps)
        cache_hit_rate = float(cache_hits) / float(max(1, eval_calls_total))
        print(f"    gen {gen + 1:02d}/{generations} best={gen_best_score:.4f} mean={gen_mean:.4f}")
        for rank, (score, genome) in enumerate(gen_top, start=1):
            actions = _decode_genome(genome, list(args.cfg_scales), corr_strengths, steps)
            print(f"      #{rank} score={score:.4f} {_actions_brief(actions)}")

        history.append(
            {
                "generation": int(gen),
                "best_score": float(gen_best_score),
                "mean_score": float(gen_mean),
                "eval_calls_total": eval_calls_total,
                "eval_calls_generation": eval_calls_generation,
                "nfe_per_generation": nfe_per_generation,
                "nfe_total": nfe_total,
                "cache_entries": int(len(score_cache)),
                "cache_hits_total": int(cache_hits),
                "cache_misses_total": int(cache_misses),
                "cache_hit_rate": float(cache_hit_rate),
                "top": [
                    {
                        "rank": int(rank),
                        "score": float(score),
                        "genome": [int(x) for x in genome],
                        "actions": [[int(v), float(c), float(r)] for v, c, r in _decode_genome(genome, list(args.cfg_scales), corr_strengths, steps)],
                    }
                    for rank, (score, genome) in enumerate(gen_top, start=1)
                ],
            }
        )

        if gen + 1 >= generations:
            break

        ranked_genomes = [list(genome) for _, genome in scored]
        next_population: list[list[int]] = [list(genome) for _, genome in scored[:elites]]
        while len(next_population) < pop_size:
            if args.ga_selection == "tournament":
                parent_a = _select_parent_tournament(ranked_genomes, int(args.ga_tournament_k))
                parent_b = _select_parent_tournament(ranked_genomes, int(args.ga_tournament_k))
            else:
                parent_a = _select_parent_rank(ranked_genomes, float(args.ga_rank_pressure))
                parent_b = _select_parent_rank(ranked_genomes, float(args.ga_rank_pressure))
            child = _crossover(parent_a, parent_b, args.ga_crossover)
            child = _mutate_genome(
                child,
                steps,
                n_variants,
                n_cfg,
                n_corr,
                float(args.ga_mutation_prob),
                args.ga_phase_constraints,
            )
            child = _repair_genome(child, steps, n_variants, n_cfg, n_corr, args.ga_phase_constraints)
            if len(child) != genome_len:
                child = child[:genome_len]
            next_population.append(child)
        population = next_population

    best_actions = _decode_genome(best_genome, list(args.cfg_scales), corr_strengths, steps)
    best_result = run_schedule_actions(
        args,
        ctx,
        emb,
        reward_model,
        prompt,
        seed,
        best_actions,
        deterministic_noise=True,
    )

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        prefix = f"{log_prefix}_" if log_prefix else ""
        history_json = os.path.join(log_dir, f"{prefix}ga_history.json")
        history_txt = os.path.join(log_dir, f"{prefix}ga_history.txt")
        payload = {
            "prompt": prompt,
            "seed": int(seed),
            "ga_population": int(pop_size),
            "ga_generations": int(generations),
            "ga_elites": int(elites),
            "ga_mutation_prob": float(args.ga_mutation_prob),
            "ga_selection": args.ga_selection,
            "ga_crossover": args.ga_crossover,
            "ga_rank_pressure": float(args.ga_rank_pressure),
            "ga_tournament_k": int(args.ga_tournament_k),
            "ga_phase_constraints": bool(args.ga_phase_constraints),
            "best_score": float(best_result.score),
            "best_genome": [int(x) for x in best_genome],
            "best_actions": [[int(v), float(c), float(r)] for v, c, r in best_actions],
            "cache_stats": {
                "eval_calls_total": int(eval_calls),
                "cache_entries": int(len(score_cache)),
                "cache_hits_total": int(cache_hits),
                "cache_misses_total": int(cache_misses),
                "cache_hit_rate": float(cache_hits) / float(max(1, int(eval_calls))),
                "nfe_total": int(eval_calls * steps),
            },
            "history": history,
        }
        with open(history_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        with open(history_txt, "w", encoding="utf-8") as f:
            f.write("gen\tbest\tmean\ttop_actions\n")
            for row in history:
                top_actions = ""
                if row["top"]:
                    top_actions = _actions_brief(
                        [(int(v), float(c), float(r)) for v, c, r in row["top"][0]["actions"]]
                    )
                f.write(
                    f"{row['generation'] + 1}\t{row['best_score']:.6f}\t"
                    f"{row['mean_score']:.6f}\t{top_actions}\n"
                )

    return SearchResult(
        image=best_result.image,
        score=float(best_result.score),
        actions=[(int(v), float(c), float(r)) for v, c, r in best_actions],
    )


class MCTSNode:
    __slots__ = ("step", "dx", "latents", "children", "visits", "action_visits", "action_values")

    def __init__(self, step: int, dx: torch.Tensor, latents: torch.Tensor | None):
        self.step = step
        self.dx = dx
        self.latents = latents
        self.children: dict[tuple[int, float, float], MCTSNode] = {}
        self.visits = 0
        self.action_visits: dict[tuple[int, float, float], int] = {}
        self.action_values: dict[tuple[int, float, float], float] = {}

    def is_leaf(self, max_steps: int) -> bool:
        return self.step >= max_steps

    def untried_actions(self, actions: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
        return [a for a in actions if a not in self.action_visits]

    def ucb(self, action: tuple[int, float, float], c: float) -> float:
        n = self.action_visits.get(action, 0)
        if n == 0:
            return float("inf")
        mean = self.action_values[action] / n
        return mean + c * math.sqrt(math.log(max(self.visits, 1)) / n)

    def best_ucb(self, actions: list[tuple[int, float, float]], c: float) -> tuple[int, float, float]:
        return max(actions, key=lambda action: self.ucb(action, c))

    def best_exploit(self, actions: list[tuple[int, float, float]]) -> tuple[int, float, float] | None:
        best = None
        best_v = -float("inf")
        for action in actions:
            n = self.action_visits.get(action, 0)
            if n <= 0:
                continue
            value = self.action_values[action] / n
            if value > best_v:
                best_v = value
                best = action
        return best


def _parse_key_steps(args: argparse.Namespace) -> list[int] | None:
    """Return sorted list of key step indices, or None to branch at every step."""
    raw = str(getattr(args, "mcts_key_steps", "")).strip()
    count = int(getattr(args, "mcts_key_step_count", 0))
    total = int(args.steps)
    if count > 0:
        # Auto-compute evenly-spaced key steps (always includes 0)
        if count == 1:
            return [0]
        if count >= total:
            return None  # branch at every step
        indices = sorted(set(int(round(i * (total - 1) / (count - 1))) for i in range(count)))
        return indices
    if raw:
        indices = sorted(set(int(x.strip()) for x in raw.split(",") if x.strip().isdigit()))
        if not indices:
            return None
        return [i for i in indices if 0 <= i < total]
    return None  # default: branch at every step


def _resolve_mcts_fresh_noise_steps(
    args: argparse.Namespace,
    total_steps: int,
    key_steps: list[int] | None = None,
) -> set[int]:
    out: set[int] = set()
    raw = str(getattr(args, "mcts_fresh_noise_steps", "")).strip().lower()
    if raw:
        if raw == "all":
            out.update(range(max(0, int(total_steps))))
        else:
            for tok in raw.split(","):
                t = tok.strip()
                if not t:
                    continue
                if t.isdigit():
                    idx = int(t)
                    if 0 <= idx < int(total_steps):
                        out.add(int(idx))
    if bool(getattr(args, "mcts_fresh_noise_key_steps", False)):
        ks = key_steps if key_steps is not None else _parse_key_steps(args)
        if ks is None:
            ks = list(range(max(0, int(total_steps))))
        for k in ks:
            if 0 <= int(k) < int(total_steps):
                out.add(int(k))
    return out


def _mcts_fresh_noise_samples_for_step(
    args: argparse.Namespace,
    step_idx: int,
    enabled_steps: set[int] | None,
) -> int:
    if not enabled_steps:
        return 1
    if int(step_idx) not in enabled_steps:
        return 1
    return max(1, int(getattr(args, "mcts_fresh_noise_samples", 1)))


def _expand_child(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    node: MCTSNode,
    action: tuple[int, float, float],
    sched: list[tuple[torch.Tensor, torch.Tensor, float]],
    reward_model: "UnifiedRewardScorer",
    prompt: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    use_euler = getattr(args, "euler_sampler", False)
    variant_idx, cfg, cs = action
    t_flat, t_4d, dt = sched[node.step]
    flow = transformer_step(args, ctx, node.latents, emb, variant_idx, t_flat, cfg)
    new_latents, new_dx = _apply_step(node.latents, flow, node.dx, t_4d, dt, use_euler, args.x0_sampler)
    if float(cs) > 0.0:
        new_dx = apply_reward_correction(ctx, new_dx, prompt, reward_model, float(cs), cfg=float(cfg))
    next_step = node.step + 1
    if next_step < len(sched) and not use_euler:
        _, next_t_4d, _ = sched[next_step]
        noise = torch.randn_like(new_dx)
        new_latents = (1.0 - next_t_4d) * new_dx + next_t_4d * noise
    elif next_step >= len(sched):
        new_latents = None
    return new_dx, new_latents


def expand_emb_with_interp(
    emb: EmbeddingContext,
    family: str,
    n_steps: int,
) -> EmbeddingContext:
    """
    Expand EmbeddingContext with slerp/nlerp interpolated variants.

    For each adjacent pair of variants (i, i+1), inserts n_steps interpolated
    embeddings at t = k/(n_steps+1) for k in 1..n_steps.  The MCTS action space
    then covers both the original discrete variants and all interpolated points.
    """
    if family == "none" or n_steps <= 0 or len(emb.cond_text) <= 1:
        return emb

    new_text = list(emb.cond_text)
    new_pooled = list(emb.cond_pooled)
    n_orig = len(emb.cond_text)

    for i in range(n_orig - 1):
        a_t, b_t = emb.cond_text[i], emb.cond_text[i + 1]
        a_p, b_p = emb.cond_pooled[i], emb.cond_pooled[i + 1]
        for k in range(1, n_steps + 1):
            t = k / (n_steps + 1)
            if family == "slerp":
                new_text.append(slerp_pair(a_t, b_t, t))
                new_pooled.append(slerp_pair(a_p, b_p, t))
            else:  # nlerp
                w = torch.tensor([1.0 - t, t], device=a_t.device, dtype=a_t.dtype)
                new_text.append(nlerp_all([a_t, b_t], w))
                new_pooled.append(nlerp_all([a_p, b_p], w))

    return EmbeddingContext(
        cond_text=new_text,
        cond_pooled=new_pooled,
        uncond_text=emb.uncond_text,
        uncond_pooled=emb.uncond_pooled,
    )


def _run_segment(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: "UnifiedRewardScorer",
    prompt: str,
    latents: torch.Tensor,
    dx: torch.Tensor,
    action: tuple,
    sched: list[tuple[torch.Tensor, torch.Tensor, float]],
    start_step: int,
    end_step: int,
    noise_explore_steps: set[int] | None = None,
    eps_bank: list[torch.Tensor] | None = None,
    stats_out: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run denoising steps [start_step, end_step) with a fixed action.

    Returns (latents, dx) after the segment.
    """
    use_euler = getattr(args, "euler_sampler", False)
    variant_idx = int(action[0])
    cfg = float(action[1])
    cs = float(action[2])
    noise_event: tuple[float, int] | None = None
    if len(action) >= 5:
        gamma = float(action[3])
        eps_id = int(action[4])
        if abs(gamma) > 1e-12:
            noise_event = (gamma, eps_id)

    seg_cfg_deltas: list[float] = []
    for step_idx in range(start_step, end_step):
        t_flat, t_4d, dt = sched[step_idx]
        explore_n = _mcts_fresh_noise_samples_for_step(args, step_idx, noise_explore_steps)
        if noise_event is not None and eps_bank and step_idx == int(start_step):
            gamma, eps_id = noise_event
            eps = eps_bank[int(eps_id) % len(eps_bank)]
            latents = latents + float(gamma) * t_4d * eps
        if explore_n <= 1:
            step_stats: dict | None = {} if stats_out is not None else None
            flow = transformer_step(args, ctx, latents, emb, variant_idx, t_flat, cfg, stats_out=step_stats)
            if step_stats is not None:
                seg_cfg_deltas.append(float(step_stats.get("cfg_delta_rms", 0.0)))
            latents, dx = _apply_step(latents, flow, dx, t_4d, dt, use_euler, args.x0_sampler)
            if float(cs) > 0.0:
                dx = apply_reward_correction(ctx, dx, prompt, reward_model, float(cs), cfg=float(cfg))
        else:
            noise_scale = float(getattr(args, "mcts_fresh_noise_scale", 1.0))
            best_score = -float("inf")
            best_latents = latents
            best_dx = dx
            best_delta = 0.0
            # Candidate 0 is the unperturbed latent; the remaining are fresh-noise perturbations.
            for k in range(explore_n):
                if k == 0:
                    latents_in = latents
                else:
                    amp = float(noise_scale) * float(max(0.0, float(t_flat[0].item())))
                    latents_in = latents + (amp * torch.randn_like(latents))
                step_stats = {} if stats_out is not None else None
                flow = transformer_step(args, ctx, latents_in, emb, variant_idx, t_flat, cfg, stats_out=step_stats)
                cand_lat, cand_dx = _apply_step(latents_in, flow, dx, t_4d, dt, use_euler, args.x0_sampler)
                if float(cs) > 0.0:
                    cand_dx = apply_reward_correction(ctx, cand_dx, prompt, reward_model, float(cs), cfg=float(cfg))
                cand_img = decode_to_pil(ctx, _final_decode_tensor(cand_lat, cand_dx, use_euler))
                cand_score = score_image(reward_model, prompt, cand_img)
                if cand_score > best_score:
                    best_score = float(cand_score)
                    best_latents = cand_lat
                    best_dx = cand_dx
                    if step_stats is not None:
                        best_delta = float(step_stats.get("cfg_delta_rms", 0.0))
            if stats_out is not None:
                seg_cfg_deltas.append(float(best_delta))
            latents, dx = best_latents, best_dx
        if step_idx + 1 < int(args.steps) and not use_euler:
            _, next_t_4d, _ = sched[step_idx + 1]
            noise = torch.randn_like(dx)
            latents = (1.0 - next_t_4d) * dx + next_t_4d * noise
    if stats_out is not None:
        if len(seg_cfg_deltas) > 0:
            stats_out["cfg_delta_rms"] = float(sum(seg_cfg_deltas) / float(len(seg_cfg_deltas)))
            stats_out["cfg_delta_rms_max"] = float(max(seg_cfg_deltas))
            stats_out["cfg_delta_rms_per_step"] = list(seg_cfg_deltas)
        else:
            stats_out["cfg_delta_rms"] = 0.0
            stats_out["cfg_delta_rms_max"] = 0.0
            stats_out["cfg_delta_rms_per_step"] = []
    return latents, dx


def run_mcts(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    variants: list[str],
    seed: int,
) -> SearchResult:
    del variants
    family = getattr(args, "mcts_interp_family", "none")
    n_interp = getattr(args, "mcts_n_interp", 1)
    if family != "none":
        emb = expand_emb_with_interp(emb, family, n_interp)
        print(f"  mcts: interp={family} n_interp={n_interp} total_variants={len(emb.cond_text)}")

    corr_strengths = list(getattr(args, "correction_strengths", [0.0]))
    base_actions = [
        (vi, cfg, cs)
        for vi in range(len(emb.cond_text))
        for cfg in args.cfg_scales
        for cs in corr_strengths
    ]
    if len(base_actions) <= 0:
        raise RuntimeError("MCTS requires non-empty action space.")

    total_steps = int(args.steps)
    raw_noise_steps = str(getattr(args, "noise_inject_candidate_steps", "")).strip().lower()
    use_integrated_noise_actions = total_steps <= 4
    if use_integrated_noise_actions:
        if raw_noise_steps:
            integrated_noise_steps = _resolve_noise_inject_steps(args, total_steps)
        else:
            integrated_noise_steps = [s for s in [1, 2] if 0 <= s < total_steps]
        integrated_noise_steps = sorted(set(int(s) for s in integrated_noise_steps))
        integrated_noise_steps_set = set(integrated_noise_steps)
        integrated_gamma_bank = _noise_inject_gamma_bank(args)
        integrated_eps_samples = max(1, int(getattr(args, "noise_inject_eps_samples", 4)))
        integrated_include_no = bool(getattr(args, "noise_inject_include_no_inject", True))
    else:
        integrated_noise_steps = []
        integrated_noise_steps_set: set[int] = set()
        integrated_gamma_bank = []
        integrated_eps_samples = 1
        integrated_include_no = True

    def _strip_action(action: tuple) -> tuple[int, float, float]:
        return (int(action[0]), float(action[1]), float(action[2]))

    # 4-step integrated noise fix:
    # At noise steps, use a disjoint union (normal prompt/CFG actions) U (noise-only anchor actions),
    # instead of the full Cartesian product with every (variant, cfg, correction).
    anchor_variant_idx = 0
    anchor_cfg = float(getattr(args, "baseline_cfg", args.cfg_scales[0] if len(args.cfg_scales) > 0 else 1.0))
    anchor_cs = 0.0
    integrated_noise_only_actions: list[tuple[int, float, float, float, int]] = []
    if use_integrated_noise_actions:
        for gamma in integrated_gamma_bank:
            g = float(gamma)
            if abs(g) <= 1e-12:
                continue
            for eps_id in range(int(integrated_eps_samples)):
                integrated_noise_only_actions.append(
                    (int(anchor_variant_idx), float(anchor_cfg), float(anchor_cs), float(g), int(eps_id))
                )

    latents0 = make_latents(ctx, seed, args.height, args.width, emb.cond_text[0].dtype)
    dx0 = torch.zeros_like(latents0)
    use_euler = getattr(args, "euler_sampler", False)
    sched = step_schedule(ctx.device, latents0.dtype, args.steps,
                          getattr(args, "sigmas", None), euler=use_euler)
    _, t0_4d, _ = sched[0]
    start_latents = _prepare_latents(latents0, dx0, latents0, t0_4d, 0, use_euler)

    # --- Key-step branching setup ---
    if use_integrated_noise_actions:
        # 4-step setting: keep per-step branching so noise_t remains a true step action.
        key_steps = list(range(int(args.steps)))
    else:
        key_steps = _parse_key_steps(args)
        if key_steps is None:
            key_steps = list(range(int(args.steps)))
        if 0 not in key_steps:
            key_steps = [0] + key_steps
    key_steps = sorted(set(key_steps))
    n_key = len(key_steps)
    # For each key step k, the segment runs from key_steps[k] to key_steps[k+1] (or args.steps)
    key_segments: list[tuple[int, int]] = []
    for i in range(n_key):
        seg_start = key_steps[i]
        seg_end = key_steps[i + 1] if i + 1 < n_key else int(args.steps)
        key_segments.append((seg_start, seg_end))

    mcts_eps_bank: list[torch.Tensor] | None = None
    if use_integrated_noise_actions:
        mcts_eps_bank = _build_eps_bank_like(
            latents0,
            integrated_eps_samples,
            seed=int(seed) + 900001,
        )

    def _candidate_actions_for_key(key_idx: int) -> list[tuple]:
        if not use_integrated_noise_actions:
            return list(base_actions)
        seg_start, _ = key_segments[int(key_idx)]
        if int(seg_start) not in integrated_noise_steps_set:
            return list(base_actions)
        out: list[tuple] = list(base_actions)
        out.extend(integrated_noise_only_actions)
        if len(out) <= 0:
            out = list(base_actions)
        return out

    def _fallback_action_for_key(key_idx: int) -> tuple:
        candidates = _candidate_actions_for_key(key_idx)
        target_cfg = float(getattr(args, "baseline_cfg", candidates[0][1]))
        for action in candidates:
            vi, cfg, cs = int(action[0]), float(action[1]), float(action[2])
            gamma = float(action[3]) if len(action) >= 5 else 0.0
            if vi == 0 and abs(cfg - target_cfg) <= 1e-6 and abs(cs) <= 1e-12 and abs(gamma) <= 1e-12:
                return action
        return candidates[0]

    fresh_noise_steps = _resolve_mcts_fresh_noise_steps(args, int(args.steps), key_steps=key_steps)
    fresh_noise_samples = max(1, int(getattr(args, "mcts_fresh_noise_samples", 1)))
    if len(fresh_noise_steps) > 0 and fresh_noise_samples > 1:
        print(
            f"  mcts fresh-noise: steps={sorted(int(x) for x in fresh_noise_steps)} "
            f"samples={fresh_noise_samples} scale={float(getattr(args, 'mcts_fresh_noise_scale', 1.0)):.3f}"
        )

    root = MCTSNode(0, dx0, start_latents)

    best_global_score = -float("inf")
    best_global_dx = None
    best_global_path_internal: list[tuple] = []

    n_actions_first = len(_candidate_actions_for_key(0))
    if use_integrated_noise_actions:
        n_actions_noise = max(len(_candidate_actions_for_key(i)) for i in range(n_key))
        print(
            f"  mcts: sims={args.n_sims} base_actions={len(base_actions)} steps={args.steps} "
            f"key_steps={key_steps} ({n_key} branch points) "
            f"noise_steps={integrated_noise_steps} noise_actions(step)=[{n_actions_first},{n_actions_noise}]"
        )
    else:
        print(
            f"  mcts: sims={args.n_sims} actions={n_actions_first} steps={args.steps} "
            f"key_steps={key_steps} ({n_key} branch points)"
        )

    for sim in range(args.n_sims):
        node = root
        path: list[tuple[MCTSNode, tuple]] = []
        action: tuple | None = None

        while not node.is_leaf(n_key):
            node_actions = _candidate_actions_for_key(node.step)
            untried = node.untried_actions(node_actions)
            if untried:
                action = untried[np.random.randint(len(untried))]
                break
            action = node.best_ucb(node_actions, args.ucb_c)
            path.append((node, action))
            node = node.children[action]

        if action is None:
            raise RuntimeError("MCTS failed to pick an action.")

        if not node.is_leaf(n_key):
            if action not in node.children:
                # Expand: run the segment from key_steps[node.step] to key_steps[node.step+1]
                seg_start, seg_end = key_segments[node.step]
                child_lat, child_dx = _run_segment(
                    args, ctx, emb, reward_model, prompt,
                    node.latents, node.dx, action, sched, seg_start, seg_end,
                    noise_explore_steps=fresh_noise_steps,
                    eps_bank=mcts_eps_bank,
                )
                node.children[action] = MCTSNode(node.step + 1, child_dx, child_lat)
            path.append((node, action))
            node = node.children[action]

        # Rollout from node to terminal
        rollout_dx = node.dx
        rollout_latents = node.latents
        rollout_key_idx = node.step
        rollout_actions: list[tuple] = []
        while rollout_key_idx < n_key:
            rollout_candidates = _candidate_actions_for_key(rollout_key_idx)
            rollout_action = rollout_candidates[np.random.randint(len(rollout_candidates))]
            rollout_actions.append(rollout_action)
            seg_start, seg_end = key_segments[rollout_key_idx]
            rollout_latents, rollout_dx = _run_segment(
                args, ctx, emb, reward_model, prompt,
                rollout_latents, rollout_dx, rollout_action, sched, seg_start, seg_end,
                noise_explore_steps=fresh_noise_steps,
                eps_bank=mcts_eps_bank,
            )
            rollout_key_idx += 1

        rollout_img = decode_to_pil(ctx, _final_decode_tensor(rollout_latents, rollout_dx, use_euler))
        rollout_score = score_image(reward_model, prompt, rollout_img)
        if rollout_score > best_global_score:
            best_global_score = rollout_score
            best_global_dx = _final_decode_tensor(rollout_latents, rollout_dx, use_euler).clone()
            best_global_path_internal = [a for _, a in path] + list(rollout_actions)

        for pnode, paction in path:
            pnode.visits += 1
            pnode.action_visits[paction] = pnode.action_visits.get(paction, 0) + 1
            pnode.action_values[paction] = pnode.action_values.get(paction, 0.0) + rollout_score

        if (sim + 1) % 10 == 0 or sim == 0:
            print(f"    sim {sim + 1:3d}/{args.n_sims} best={best_global_score:.4f}")

    # Exploit: extract best action per key step
    exploit_path_internal: list[tuple] = []
    node = root
    for key_idx in range(n_key):
        node_actions = _candidate_actions_for_key(key_idx)
        action = node.best_exploit(node_actions)
        if action is None:
            break
        exploit_path_internal.append(action)
        if action in node.children:
            node = node.children[action]
        else:
            break

    # Replay exploit path through all denoising steps
    replay_lat = start_latents
    replay_dx = dx0
    for key_idx, exploit_action in enumerate(exploit_path_internal):
        seg_start, seg_end = key_segments[key_idx]
        replay_lat, replay_dx = _run_segment(
            args, ctx, emb, reward_model, prompt,
            replay_lat, replay_dx, exploit_action, sched, seg_start, seg_end,
            noise_explore_steps=fresh_noise_steps,
            eps_bank=mcts_eps_bank,
        )
    # If exploit path is shorter than n_key, fill remaining with baseline
    for key_idx in range(len(exploit_path_internal), n_key):
        fallback = _fallback_action_for_key(key_idx)
        seg_start, seg_end = key_segments[key_idx]
        replay_lat, replay_dx = _run_segment(
            args, ctx, emb, reward_model, prompt,
            replay_lat, replay_dx, fallback, sched, seg_start, seg_end,
            noise_explore_steps=fresh_noise_steps,
            eps_bank=mcts_eps_bank,
        )

    exploit_img = decode_to_pil(ctx, _final_decode_tensor(replay_lat, replay_dx, use_euler))
    exploit_score = score_image(reward_model, prompt, exploit_img)

    selected_img: Image.Image
    selected_score: float
    selected_key_path_internal: list[tuple]
    selected_source: str
    if exploit_score >= best_global_score or best_global_dx is None:
        selected_img = exploit_img
        selected_score = float(exploit_score)
        selected_key_path_internal = list(exploit_path_internal)
        selected_source = "exploit"
    else:
        selected_img = decode_to_pil(ctx, best_global_dx)
        selected_score = float(best_global_score)
        selected_key_path_internal = list(best_global_path_internal)
        selected_source = "best_global"

    sparse_refine_diag: dict[str, Any] | None = None
    if (not use_integrated_noise_actions) and bool(getattr(args, "mcts_sparse_noise_refine", True)):
        critical_steps = _resolve_noise_inject_steps(args, total_steps) if raw_noise_steps else [int(total_steps // 2)]
        critical_steps = sorted(set(int(s) for s in critical_steps if 0 <= int(s) < total_steps))
        if len(critical_steps) > 0:
            gamma_bank = _noise_inject_gamma_bank(args)
            eps_samples = max(1, int(getattr(args, "noise_inject_eps_samples", 4)))
            include_no = bool(getattr(args, "noise_inject_include_no_inject", True))
            max_events = max(1, min(2, int(getattr(args, "noise_inject_steps_per_rollout", 1))))
            policies: list[tuple[tuple[int, float, int], ...]] = []
            if include_no:
                policies.append(tuple())
            if max_events <= 1:
                for step_idx in critical_steps:
                    for gamma in gamma_bank:
                        g = float(gamma)
                        if include_no and abs(g) <= 1e-12:
                            continue
                        for eps_id in range(eps_samples):
                            policies.append(((int(step_idx), float(g), int(eps_id)),))
            else:
                for s1, s2 in combinations(critical_steps, 2):
                    for gamma in gamma_bank:
                        g = float(gamma)
                        if include_no and abs(g) <= 1e-12:
                            continue
                        for eps_id in range(eps_samples):
                            policies.append(
                                (
                                    (int(s1), float(g), int(eps_id)),
                                    (int(s2), float(g), int(eps_id)),
                                )
                            )
            seen_policies: set[tuple[tuple[int, float, int], ...]] = set()
            uniq_policies: list[tuple[tuple[int, float, int], ...]] = []
            for policy in policies:
                if policy in seen_policies:
                    continue
                seen_policies.add(policy)
                uniq_policies.append(policy)
            max_policies = max(0, int(getattr(args, "noise_inject_max_policies", 0)))
            if max_policies > 0 and len(uniq_policies) > max_policies:
                uniq_policies = uniq_policies[:max_policies]
            if len(uniq_policies) <= 0:
                uniq_policies = [tuple()]

            def _expand_key_path_to_step_actions(key_path_internal: list[tuple]) -> list[tuple[int, float, float]]:
                step_actions: list[tuple[int, float, float]] = []
                for key_idx, (seg_start, seg_end) in enumerate(key_segments):
                    if key_idx < len(key_path_internal):
                        action = key_path_internal[key_idx]
                    else:
                        action = _fallback_action_for_key(key_idx)
                    base = _strip_action(action)
                    for _ in range(seg_start, seg_end):
                        step_actions.append(base)
                if len(step_actions) != total_steps:
                    raise RuntimeError(f"Expanded schedule has {len(step_actions)} actions for steps={total_steps}.")
                return step_actions

            branch_rows: list[tuple[str, float, list[tuple]]] = [
                ("exploit", float(exploit_score), list(exploit_path_internal)),
                ("best_global", float(best_global_score), list(best_global_path_internal)),
            ]
            branch_rows.sort(key=lambda x: x[1], reverse=True)
            branch_topk = max(1, int(getattr(args, "mcts_sparse_noise_refine_topk_branches", 2)))
            branch_rows = branch_rows[:branch_topk]

            noise_step_cache = _build_step_noise_cache(latents0, total_steps, int(seed))
            eps_bank = _build_eps_bank_like(latents0, eps_samples, seed=int(seed) + 900001)

            best_refine_score = float(selected_score)
            best_refine_img: Image.Image | None = None
            best_refine_key_path_internal: list[tuple] | None = None
            best_refine_policy: tuple[tuple[int, float, int], ...] = tuple()
            best_refine_source = selected_source
            rollouts_refine = 0

            for source_name, _source_score, key_path_internal in branch_rows:
                step_actions = _expand_key_path_to_step_actions(key_path_internal)
                for policy in uniq_policies:
                    rollouts_refine += 1
                    inject_map = {int(s): (float(g), int(e)) for s, g, e in policy}
                    latents = latents0.clone()
                    dx = torch.zeros_like(latents)
                    for step_idx, ((t_flat, t_4d, dt), (vi, cfg, cs)) in enumerate(zip(sched, step_actions)):
                        base_noise = noise_step_cache[step_idx] if step_idx < len(noise_step_cache) else noise_step_cache[-1]
                        latents = _prepare_latents(latents, dx, base_noise, t_4d, step_idx, use_euler)
                        if step_idx in inject_map:
                            gamma, eps_id = inject_map[step_idx]
                            eps = eps_bank[int(eps_id) % len(eps_bank)]
                            latents = latents + float(gamma) * t_4d * eps
                        flow = transformer_step(args, ctx, latents, emb, int(vi), t_flat, float(cfg))
                        latents, dx = _apply_step(latents, flow, dx, t_4d, dt, use_euler, args.x0_sampler)
                        if float(cs) > 0.0:
                            dx = apply_reward_correction(ctx, dx, prompt, reward_model, float(cs), cfg=float(cfg))
                    img = decode_to_pil(ctx, _final_decode_tensor(latents, dx, use_euler))
                    score = float(score_image(reward_model, prompt, img))
                    if score > best_refine_score:
                        best_refine_score = float(score)
                        best_refine_img = img
                        best_refine_key_path_internal = list(key_path_internal)
                        best_refine_policy = tuple(policy)
                        best_refine_source = str(source_name)

            sparse_refine_diag = {
                "enabled": True,
                "critical_steps": [int(s) for s in critical_steps],
                "gamma_bank": [float(g) for g in gamma_bank],
                "eps_samples": int(eps_samples),
                "max_events": int(max_events),
                "policies": int(len(uniq_policies)),
                "branches_evaluated": int(len(branch_rows)),
                "rollouts_evaluated": int(rollouts_refine),
                "selected_source": str(best_refine_source),
                "selected_policy": [
                    {"step": int(s), "gamma": float(g), "eps_id": int(e)}
                    for s, g, e in best_refine_policy
                ],
                "score_before_refine": float(selected_score),
                "score_after_refine": float(best_refine_score),
            }
            if best_refine_img is not None and best_refine_score > float(selected_score):
                selected_img = best_refine_img
                selected_score = float(best_refine_score)
                if best_refine_key_path_internal is not None:
                    selected_key_path_internal = list(best_refine_key_path_internal)
                    selected_source = f"sparse_refine:{best_refine_source}"

    padded_key_path_internal: list[tuple] = []
    for key_idx in range(n_key):
        if key_idx < len(selected_key_path_internal):
            padded_key_path_internal.append(selected_key_path_internal[key_idx])
        else:
            padded_key_path_internal.append(_fallback_action_for_key(key_idx))
    selected_actions = [_strip_action(a) for a in padded_key_path_internal]

    noise_events: list[dict[str, Any]] = []
    for key_idx, action in enumerate(padded_key_path_internal):
        if len(action) >= 5:
            gamma = float(action[3])
            if abs(gamma) > 1e-12:
                eps_id = int(action[4])
                seg_start, _ = key_segments[key_idx]
                noise_events.append(
                    {
                        "key_step_idx": int(key_idx),
                        "step_idx": int(seg_start),
                        "gamma": float(gamma),
                        "eps_id": int(eps_id),
                    }
                )

    diagnostics = {
        "selected_source": str(selected_source),
        "key_steps": [int(k) for k in key_steps],
        "integrated_noise_actions": bool(use_integrated_noise_actions),
        "integrated_noise_action_space": "disjoint_union_anchor" if use_integrated_noise_actions else "none",
        "integrated_noise_steps": [int(s) for s in integrated_noise_steps],
        "integrated_noise_gamma_bank": [float(g) for g in integrated_gamma_bank],
        "integrated_noise_eps_samples": int(integrated_eps_samples),
        "integrated_noise_include_no_inject_flag": bool(integrated_include_no),
        "integrated_noise_anchor": {
            "variant_idx": int(anchor_variant_idx),
            "cfg": float(anchor_cfg),
            "correction_strength": float(anchor_cs),
        },
        "integrated_noise_only_actions_per_step": int(len(integrated_noise_only_actions)),
        "chosen_noise_events": noise_events,
        "sparse_noise_refine": sparse_refine_diag,
    }
    return SearchResult(
        image=selected_img,
        score=float(selected_score),
        actions=[(int(v), float(c), float(r)) for v, c, r in selected_actions],
        diagnostics=diagnostics,
    )


@torch.no_grad()
def _build_smc_expansion_bank(
    args: argparse.Namespace,
    emb: EmbeddingContext,
) -> list[tuple[int, float, float]]:
    variants_arg = list(getattr(args, "smc_expansion_variants", []) or [])
    cfgs_arg = list(getattr(args, "smc_expansion_cfgs", []) or [])
    cs_arg = list(getattr(args, "smc_expansion_cs", []) or [])
    if len(variants_arg) == 0:
        variants_arg = list(range(len(emb.cond_text)))
    if len(cfgs_arg) == 0:
        cfgs_arg = list(getattr(args, "cfg_scales", [float(getattr(args, "smc_cfg_scale", 1.0))]))
    if len(cs_arg) == 0:
        cs_arg = list(getattr(args, "correction_strengths", [0.0]))
    bank: list[tuple[int, float, float]] = []
    seen: set[tuple[int, float, float]] = set()
    for vi in variants_arg:
        for c in cfgs_arg:
            for s in cs_arg:
                key = (int(vi), float(round(float(c), 6)), float(round(float(s), 6)))
                if key in seen:
                    continue
                seen.add(key)
                bank.append(key)
    if len(bank) <= 0:
        bank = [(int(max(0, min(len(emb.cond_text) - 1, int(getattr(args, "smc_variant_idx", 0))))),
                 float(getattr(args, "smc_cfg_scale", 1.0)),
                 float(next(iter(getattr(args, "correction_strengths", [0.0])), 0.0)))]
    return bank


def _stratified_assign_actions(
    bank: list[tuple[int, float, float]],
    k: int,
    rng: np.random.Generator,
) -> list[tuple[int, float, float]]:
    if k <= 0 or len(bank) <= 0:
        return []
    reps = (k + len(bank) - 1) // len(bank)
    pool = [bank[i % len(bank)] for i in range(reps * len(bank))]
    order = rng.permutation(len(pool))[:k]
    return [pool[int(i)] for i in order]


def run_smc(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    variants: list[str],
    seed: int,
) -> SearchResult:
    del variants
    k = max(2, int(args.smc_k))
    use_expansion = bool(getattr(args, "smc_variant_expansion", False))
    default_cfg = float(args.smc_cfg_scale)
    default_variant = int(max(0, min(len(emb.cond_text) - 1, int(args.smc_variant_idx))))
    default_cs = float(next(iter(getattr(args, "correction_strengths", [0.0])), 0.0))
    bank = _build_smc_expansion_bank(args, emb)
    np_rng = np.random.default_rng(int(seed) + 7007)

    if use_expansion:
        particle_actions = _stratified_assign_actions(bank, k, np_rng)
    else:
        particle_actions = [(default_variant, default_cfg, default_cs) for _ in range(k)]

    particle_latents = []
    for pi in range(k):
        particle_latents.append(make_latents(ctx, seed + pi, args.height, args.width, emb.cond_text[0].dtype))
    latents = torch.cat(particle_latents, dim=0)
    dx = torch.zeros_like(latents)
    use_euler = getattr(args, "euler_sampler", False)
    sched = step_schedule(ctx.device, latents.dtype, args.steps,
                          getattr(args, "sigmas", None), euler=use_euler)
    rng = torch.Generator(device=ctx.device).manual_seed(seed + 6001)

    start_idx = int((1.0 - float(args.resample_start_frac)) * int(args.steps))
    start_idx = max(0, min(int(args.steps) - 1, start_idx))
    log_w = torch.zeros(k, device=ctx.device, dtype=torch.float32)
    ess_hist: list[float] = []
    score_hist: list[float] = []
    resample_count = 0
    expansion_events = 0
    action_path: list[list[tuple[int, float, float]]] = []

    expansion_factor_arg = int(getattr(args, "smc_expansion_factor", -1))
    expansion_factor = len(bank) if expansion_factor_arg <= 0 else min(len(bank), int(expansion_factor_arg))
    proposal_mode = str(getattr(args, "smc_expansion_proposal", "uniform"))
    proposal_tau = max(1e-6, float(getattr(args, "smc_expansion_tau", 1.0)))
    use_lookahead = bool(getattr(args, "smc_expansion_lookahead", False))

    print(
        f"  smc(das): K={k} expansion={use_expansion} bank_size={len(bank)} "
        f"M={expansion_factor} prop={proposal_mode} lookahead={use_lookahead} "
        f"gamma={float(args.smc_gamma):.3f} ess_thr={float(args.ess_threshold):.2f} euler={use_euler}"
    )

    def _forward_particle(
        cur_latents: torch.Tensor,
        cur_dx: torch.Tensor,
        action: tuple[int, float, float],
        t_flat: torch.Tensor,
        t_4d: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vi, c, cs = int(action[0]), float(action[1]), float(action[2])
        flow = transformer_step(args, ctx, cur_latents, emb, vi, t_flat, c)
        if use_euler:
            new_lat = cur_latents + dt * flow
            new_dx = cur_latents - t_4d * flow
        else:
            new_lat = cur_latents
            new_dx = _pred_x0(cur_latents, t_4d, flow, args.x0_sampler)
        if cs > 0.0:
            new_dx = apply_reward_correction(ctx, new_dx, prompt, reward_model, cs, cfg=c)
        return new_lat, new_dx

    for step_idx, (t_flat, t_4d, dt) in enumerate(sched):
        if not use_euler:
            if step_idx == 0:
                noise = latents
            else:
                noise = torch.randn(
                    latents.shape,
                    device=latents.device,
                    dtype=latents.dtype,
                    generator=rng,
                )
            latents = (1.0 - t_4d) * dx + t_4d * noise

        next_dx_parts = []
        next_latents_parts = []
        for pi in range(k):
            p_lat, p_dx = _forward_particle(
                latents[pi : pi + 1], dx[pi : pi + 1], particle_actions[pi], t_flat, t_4d, dt,
            )
            next_dx_parts.append(p_dx)
            if use_euler:
                next_latents_parts.append(p_lat)
        dx = torch.cat(next_dx_parts, dim=0)
        if use_euler:
            latents = torch.cat(next_latents_parts, dim=0)

        action_path.append(list(particle_actions))

        if step_idx < start_idx:
            continue

        score_tensor = _final_decode_tensor(latents, dx, use_euler)
        step_images = [decode_to_pil(ctx, score_tensor[pi : pi + 1]) for pi in range(k)]
        step_scores_list = [float(score_image(reward_model, prompt, img)) for img in step_images]
        step_scores = torch.tensor(step_scores_list, device=dx.device, dtype=torch.float32)
        score_hist.append(float(step_scores.mean().item()))

        lam = (1.0 + float(args.smc_gamma)) ** (int(args.steps) - 1 - step_idx) - 1.0
        log_w = log_w + float(lam) * step_scores
        weights = torch.softmax(log_w, dim=0)
        ess = float(1.0 / torch.sum(weights * weights).item())
        ess_hist.append(ess)

        if ess < float(args.ess_threshold) * float(k):
            if not use_expansion or len(bank) <= 1:
                idx = _systematic_resample(weights)
                dx = dx[idx].clone()
                latents = latents[idx].clone()
                particle_actions = [particle_actions[int(i)] for i in idx.tolist()]
                log_w = torch.zeros_like(log_w)
                resample_count += 1
                continue

            # Variant-expansion resampling: fan out each particle into M children
            # with distinct (variant, cfg, cs) actions from the bank, then
            # systematic-resample K out of K*M.
            children_lat: list[torch.Tensor] = []
            children_dx: list[torch.Tensor] = []
            children_actions: list[tuple[int, float, float]] = []
            children_parent: list[int] = []
            children_logw: list[float] = []

            for pi in range(k):
                if proposal_mode == "score_softmax" and len(bank) > 1:
                    # Proposal favors banks members near the bank centroid of this
                    # particle's own score — simple Boltzmann over step scores.
                    base_score = float(step_scores_list[pi])
                    logits = np.full((len(bank),), base_score, dtype=np.float64)
                    probs = np.exp((logits - float(np.max(logits))) / proposal_tau)
                    probs = probs / max(1e-12, float(np.sum(probs)))
                    idxs = np_rng.choice(len(bank), size=expansion_factor, replace=expansion_factor > len(bank), p=probs)
                else:
                    replace = expansion_factor > len(bank)
                    idxs = np_rng.choice(len(bank), size=expansion_factor, replace=replace)

                for bi in idxs.tolist():
                    children_actions.append(bank[int(bi)])
                    children_parent.append(int(pi))
                    children_lat.append(latents[pi : pi + 1])
                    children_dx.append(dx[pi : pi + 1])
                    children_logw.append(float(log_w[pi].item()))

            n_children = len(children_actions)
            if n_children <= 0:
                continue

            child_logw_tensor = torch.tensor(children_logw, device=dx.device, dtype=torch.float32)
            if use_lookahead and step_idx + 1 < int(args.steps):
                next_t_flat, next_t_4d, next_dt = sched[step_idx + 1]
                la_scores: list[float] = []
                new_lat_children: list[torch.Tensor] = []
                new_dx_children: list[torch.Tensor] = []
                for ci, action in enumerate(children_actions):
                    parent_lat = children_lat[ci]
                    parent_dx = children_dx[ci]
                    if not use_euler:
                        noise_la = torch.randn(parent_dx.shape, device=parent_dx.device, dtype=parent_dx.dtype, generator=rng)
                        la_input_lat = (1.0 - next_t_4d) * parent_dx + next_t_4d * noise_la
                    else:
                        la_input_lat = parent_lat
                    la_lat, la_dx = _forward_particle(
                        la_input_lat, parent_dx, action, next_t_flat, next_t_4d, next_dt,
                    )
                    la_tensor = _final_decode_tensor(la_lat, la_dx, use_euler)
                    la_img = decode_to_pil(ctx, la_tensor)
                    la_scores.append(float(score_image(reward_model, prompt, la_img)))
                    new_lat_children.append(la_lat)
                    new_dx_children.append(la_dx)
                la_scores_tensor = torch.tensor(la_scores, device=dx.device, dtype=torch.float32)
                child_logw_tensor = child_logw_tensor + float(lam) * la_scores_tensor

            child_weights = torch.softmax(child_logw_tensor, dim=0)
            idx = _systematic_resample(child_weights)
            chosen = idx.tolist()
            if use_lookahead and step_idx + 1 < int(args.steps):
                # Advance step index logically so we don't re-run the same step
                # on the chosen children. Instead we keep current (latents, dx)
                # as the pre-step state for the next loop iteration.
                latents = torch.cat([children_lat[int(i)] for i in chosen], dim=0)
                dx = torch.cat([children_dx[int(i)] for i in chosen], dim=0)
            else:
                latents = torch.cat([children_lat[int(i)] for i in chosen], dim=0)
                dx = torch.cat([children_dx[int(i)] for i in chosen], dim=0)
            particle_actions = [children_actions[int(i)] for i in chosen]
            log_w = torch.zeros_like(log_w)
            resample_count += 1
            expansion_events += 1

    final_tensor = _final_decode_tensor(latents, dx, use_euler)
    final_images = [decode_to_pil(ctx, final_tensor[pi : pi + 1]) for pi in range(k)]
    final_scores = [float(score_image(reward_model, prompt, img)) for img in final_images]
    best_idx = int(np.argmax(final_scores))
    best_action = particle_actions[best_idx]
    diagnostics = {
        "smc_style": "das_tempered_resampling_variant_expansion" if use_expansion else "das_tempered_resampling",
        "smc_k": int(k),
        "smc_variant_expansion": bool(use_expansion),
        "smc_expansion_bank": [list(a) for a in bank],
        "smc_expansion_bank_size": int(len(bank)),
        "smc_expansion_factor": int(expansion_factor),
        "smc_expansion_proposal": str(proposal_mode),
        "smc_expansion_lookahead": bool(use_lookahead),
        "smc_expansion_events": int(expansion_events),
        "smc_cfg_scale": float(default_cfg),
        "smc_variant_idx": int(default_variant),
        "smc_gamma": float(args.smc_gamma),
        "resample_start_step": int(start_idx),
        "resample_count": int(resample_count),
        "ess_min": float(min(ess_hist)) if ess_hist else 0.0,
        "ess_mean": float(sum(ess_hist) / len(ess_hist)) if ess_hist else 0.0,
        "reward_traj_mean": [float(v) for v in score_hist],
        "final_particle_scores": [float(v) for v in final_scores],
        "final_particle_actions": [list(a) for a in particle_actions],
        "best_action": list(best_action),
    }
    return SearchResult(
        image=final_images[best_idx],
        score=float(final_scores[best_idx]),
        actions=[tuple(best_action) for _ in range(int(args.steps))],
        diagnostics=diagnostics,
    )


def _font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def save_comparison(
    path: str,
    base_img: Image.Image,
    search_img: Image.Image,
    base_score: float,
    search_score: float,
    actions: list[tuple[int, float] | tuple[int, float, float]],
) -> None:
    w, h = base_img.size
    header_h = 54
    comp = Image.new("RGB", (w * 2, h + header_h), (18, 18, 18))
    draw = ImageDraw.Draw(comp)
    comp.paste(base_img, (0, header_h))
    comp.paste(search_img, (w, header_h))
    draw.text((4, 4), f"baseline IR={base_score:.3f}", fill=(200, 200, 200), font=_font(15))
    d = search_score - base_score
    col = (100, 255, 100) if d >= 0 else (255, 100, 100)
    draw.text((w + 4, 4), f"search IR={search_score:.3f}  delta={d:+.3f}", fill=col, font=_font(15))
    act_tokens: list[str] = []
    for i, action in enumerate(actions):
        if len(action) >= 3:
            v, c, r = int(action[0]), float(action[1]), float(action[2])
            act_tokens.append(f"s{i+1}:v{v}/cfg{c:.2f}/cs{r:.2f}")
        elif len(action) == 2:
            v, c = int(action[0]), float(action[1])
            act_tokens.append(f"s{i+1}:v{v}/cfg{c:.2f}")
    acts = " ".join(act_tokens)
    draw.text((w + 4, 28), acts[:96], fill=(255, 220, 50), font=_font(11))
    comp.save(path)


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    prompts = load_prompts(args)
    ctx = load_pipeline(args)
    reward_model = load_reward_model(args, ctx.device)

    rewrite_cache: dict[str, Any] = {}
    if args.rewrites_file and os.path.exists(args.rewrites_file):
        rewrite_cache = json.load(open(args.rewrites_file))
        print(f"Loaded rewrite cache for {len(rewrite_cache)} prompts.")

    summary: list[dict[str, Any]] = []
    for prompt_idx, prompt in enumerate(prompts):
        slug = f"p{prompt_idx:02d}"
        print(f"\n{'='*72}\n[{slug}] {prompt}\n{'='*72}")
        ctx.nfe = 0
        ctx.correction_nfe = 0
        variants = generate_variants(args, prompt, rewrite_cache)
        with open(os.path.join(args.out_dir, f"{slug}_variants.txt"), "w") as f:
            for vi, text in enumerate(variants):
                f.write(f"v{vi}: {text}\n")
        emb = encode_variants(ctx, variants)

        base_img, base_score = run_baseline(
            args,
            ctx,
            emb,
            reward_model,
            prompt,
            args.seed,
            cfg_scale=float(args.baseline_cfg),
        )
        if args.search_method == "greedy":
            search = run_greedy(args, ctx, emb, reward_model, prompt, variants, args.seed)
        elif args.search_method == "mcts":
            search = run_mcts(args, ctx, emb, reward_model, prompt, variants, args.seed)
        elif args.search_method == "smc":
            search = run_smc(args, ctx, emb, reward_model, prompt, variants, args.seed)
        elif args.search_method == "ga":
            search = run_ga(
                args,
                ctx,
                emb,
                reward_model,
                prompt,
                variants,
                args.seed,
                log_dir=args.out_dir,
                log_prefix=slug,
            )
        elif args.search_method == "bon":
            search = run_bon(args, ctx, emb, reward_model, prompt, args.seed)
        elif args.search_method == "beam":
            search = run_beam(args, ctx, emb, reward_model, prompt, variants, args.seed)
        elif args.search_method == "noise_inject":
            search = run_noise_inject(args, ctx, emb, reward_model, prompt, args.seed)
        else:
            raise RuntimeError(f"Unsupported search_method: {args.search_method}")

        base_path = os.path.join(args.out_dir, f"{slug}_baseline.png")
        search_path = os.path.join(args.out_dir, f"{slug}_{args.search_method}.png")
        comp_path = os.path.join(args.out_dir, f"{slug}_comparison.png")
        base_img.save(base_path)
        search.image.save(search_path)
        save_comparison(comp_path, base_img, search.image, base_score, search.score, search.actions)

        total_nfe = ctx.nfe + ctx.correction_nfe
        print(
            f"baseline={base_score:.4f} {args.search_method}={search.score:.4f} "
            f"delta={search.score - base_score:+.4f}  "
            f"nfe={total_nfe} (T:{ctx.nfe}+R:{ctx.correction_nfe})"
        )
        summary.append(
            {
                "slug": slug,
                "prompt": prompt,
                "variants": variants,
                "baseline_cfg": float(args.baseline_cfg),
                "baseline_IR": base_score,
                f"{args.search_method}_IR": search.score,
                "delta_IR": search.score - base_score,
                "actions": [[int(v), float(c), float(cs)] for v, c, cs in search.actions],
                "search_diagnostics": search.diagnostics,
                "nfe_transformer": ctx.nfe,
                "nfe_correction": ctx.correction_nfe,
                "nfe_total": total_nfe,
            }
        )

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    for row in summary:
        print(f"{row['slug']} delta_IR={row['delta_IR']:+.4f}")
    if summary:
        print(f"mean delta={float(np.mean([r['delta_IR'] for r in summary])):+.4f}")
    print(f"summary json: {summary_path}")


def main(argv: list[str] | None = None) -> None:
    run(normalize_paths(parse_args(argv)))


if __name__ == "__main__":
    main()
