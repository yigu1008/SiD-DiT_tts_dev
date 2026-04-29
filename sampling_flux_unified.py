#!/usr/bin/env python3
"""
Unified FLUX test-time search runner.

Supported search modes:
- greedy: per-step greedy over prompt-bank + guidance-scale actions
- mcts  : MCTS over prompt-bank + guidance-scale actions
- ga    : prompt-bank + guidance-scale GA
- smc   : reward-tilted SMC baseline

For each prompt/seed, the script always runs:
1) baseline (balanced prompt, fixed guidance)
2) selected search method

Artifacts can be capped with --save_first_k to avoid disk growth.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from reward_unified import UnifiedRewardScorer


PROMPT_FOCUS = {
    "balanced": "",
    "subject": "Focus on subject identity, face, and outfit fidelity.",
    "prop": "Focus on key props, object interaction, and hand-object alignment.",
    "background": "Focus on scene layout, background structure, and composition.",
    "detail": "Focus on fine details, textures, and local attributes.",
}

_DEFAULT_GUIDANCE_SCALES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
_DEFAULT_BASELINE_GUIDANCE = 1.0

_BACKEND_CONFIGS: dict[str, dict[str, Any]] = {
    "flux": {
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "transformer_id": None,
        "transformer_subfolder": None,
        "sigmas": None,
        "dtype": "bf16",
        "baseline_guidance_scale": _DEFAULT_BASELINE_GUIDANCE,
        "ga_guidance_scales": list(_DEFAULT_GUIDANCE_SCALES),
        "cfg_scales": list(_DEFAULT_GUIDANCE_SCALES),
    },
    "senseflow_flux": {
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "transformer_id": "domiso/SenseFlow",
        "transformer_subfolder": "SenseFlow-FLUX",
        "sigmas": [1.0, 0.75, 0.5, 0.25],
        "dtype": "bf16",
        # SenseFlow transformer outputs flow/velocity (same as standard FLUX).
        # x0 prediction sampler: derive x̂₀ = xt - t*flow, then re-noise.
        # Gives clean x̂₀ at every intermediate step for reward scoring.
        "baseline_guidance_scale": 0.0,
        "ga_guidance_scales": [0.0],
        "cfg_scales": [0.0],
        "x0_sampler": False,
        "euler_sampler": False,
    },
    # TDD-distilled FLUX.1-dev (RED-AIGC). 8-step LoRA fused at scale 0.125;
    # uses real CFG (unlike schnell). Recipe: load FLUX.1-dev → load
    # FLUX.1-dev_tdd_adv_lora_weights.safetensors → fuse_lora(scale=0.125).
    "tdd_flux": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "transformer_id": None,
        "transformer_subfolder": None,
        "sigmas": None,
        "dtype": "bf16",
        "lora_repo": "RED-AIGC/TDD",
        "lora_filename": "FLUX.1-dev_tdd_adv_lora_weights.safetensors",
        "lora_scale": 0.125,
        "max_sequence_length": 256,
        "baseline_guidance_scale": 2.0,
        "ga_guidance_scales": [1.5, 2.0, 2.5, 3.0, 3.5],
        "cfg_scales": [1.5, 2.0, 2.5, 3.0, 3.5],
    },
}


@dataclass
class PromptEmbed:
    label: str
    text: str
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    text_ids: torch.Tensor


@dataclass
class FluxContext:
    pipe: Any
    device: str
    dtype: torch.dtype
    decode_device_request: str
    decode_device: str
    decode_cpu_dtype: torch.dtype
    decode_cpu_if_free_below_gb: float
    empty_cache_after_decode: bool


@dataclass
class SearchResult:
    image: Image.Image
    score: float
    actions: list[tuple[int, float]]
    diagnostics: dict[str, Any] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified FLUX search runner (greedy/mcts/ga/smc).")
    p.add_argument("--search_method", choices=["greedy", "mcts", "ga", "smc", "bon", "beam", "noise_inject"], default="ga")
    p.add_argument(
        "--backend",
        choices=["flux", "senseflow_flux", "tdd_flux"],
        default=None,
        help="Convenience shortcut: sets model/transformer/sigmas/guidance defaults for FLUX variants.",
    )
    p.add_argument("--model_id", default=os.environ.get("MODEL_ID"))
    p.add_argument("--transformer_id", default=os.environ.get("TRANSFORMER_ID"),
                   help="Optional HuggingFace transformer repo override (e.g. domiso/SenseFlow).")
    p.add_argument("--transformer_subfolder", default=None,
                   help="Optional subfolder inside --transformer_id.")
    p.add_argument("--lora_repo", default=os.environ.get("LORA_REPO"),
                   help="Optional HF LoRA repo (e.g. RED-AIGC/TDD). Loaded via hf_hub_download + load_lora_weights + fuse_lora.")
    p.add_argument("--lora_filename", default=os.environ.get("LORA_FILENAME"),
                   help="Filename inside --lora_repo (e.g. FLUX.1-dev_tdd_adv_lora_weights.safetensors).")
    p.add_argument("--lora_path", default=os.environ.get("LORA_PATH"),
                   help="Optional local LoRA file/dir path. When set, overrides --lora_repo/--lora_filename.")
    p.add_argument("--lora_scale", type=float, default=None,
                   help="fuse_lora scale (e.g. 0.125 for TDD-FLUX).")
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt_file", default=None)
    p.add_argument("--n_prompts", type=int, default=-1)
    p.add_argument(
        "--shuffle_prompts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shuffle prompts before taking --n_prompts.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument(
        "--sigmas",
        nargs="+",
        type=float,
        default=None,
        help="Explicit sigma schedule (SenseFlow-style). Overrides linear schedule from --steps.",
    )
    p.add_argument("--n_variants", type=int, default=-1, help="Prompt variants to keep from the bank; <=0 keeps all.")
    p.add_argument(
        "--cfg_scales",
        nargs="+",
        type=float,
        default=None,
        help="Guidance scales used by greedy/mcts. Defaults to --ga_guidance_scales when unset.",
    )
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--max_sequence_length", type=int, default=512)

    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu_id", type=int, default=-1)
    p.add_argument(
        "--auto_select_gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --gpu_id=-1, choose visible GPU with most free memory.",
    )
    p.add_argument(
        "--cuda_alloc_conf",
        default="expandable_segments:True",
        help="Set PYTORCH_CUDA_ALLOC_CONF if not already set.",
    )

    p.add_argument(
        "--reward_backend",
        choices=["auto", "unifiedreward", "unified", "imagereward", "pickscore", "hpsv3", "hpsv2", "blend", "all"],
        default="imagereward",
    )
    p.add_argument(
        "--reward_device",
        default="cpu",
        help="cpu | same | cuda | cuda:N",
    )
    p.add_argument(
        "--reward_model",
        default="CodeGoat24/UnifiedReward-qwen-7b",
        help="Legacy alias for UnifiedReward model id (kept for compatibility).",
    )
    p.add_argument(
        "--unifiedreward_model",
        default=None,
        help="UnifiedReward model id override. Defaults to --reward_model when unset.",
    )
    p.add_argument(
        "--image_reward_model",
        default="ImageReward-v1.0",
        help="ImageReward model id/checkpoint name.",
    )
    p.add_argument(
        "--pickscore_model",
        default="yuvalkirstain/PickScore_v1",
        help="PickScore model id.",
    )
    p.add_argument(
        "--reward_weights",
        nargs=2,
        type=float,
        default=[1.0, 1.0],
        help="Blend backend weights: imagereward hps(v2/v3)",
    )
    p.add_argument("--reward_api_base", default=None, help="Optional OpenAI-compatible API base for UnifiedReward.")
    p.add_argument("--reward_api_key", default="unifiedreward")
    p.add_argument("--reward_api_model", default="UnifiedReward-7b-v1.5")
    p.add_argument("--reward_max_new_tokens", type=int, default=512)
    p.add_argument(
        "--reward_prompt_mode",
        choices=["standard", "strict"],
        default="standard",
        help="UnifiedReward prompt template mode.",
    )

    p.add_argument(
        "--offload_text_encoder_after_encode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--decode_device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="VAE decode device.",
    )
    p.add_argument(
        "--decode_cpu_dtype",
        default="fp32",
        choices=["fp16", "bf16", "fp32"],
        help="VAE dtype used on CPU decode.",
    )
    p.add_argument(
        "--decode_cpu_if_free_below_gb",
        type=float,
        default=16.0,
        help="Auto mode: switch decode to CPU when free GPU memory is below this threshold.",
    )
    p.add_argument(
        "--empty_cache_after_decode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--baseline_guidance_scale", type=float, default=1.0)

    # GA
    p.add_argument("--ga_population", type=int, default=24)
    p.add_argument("--ga_generations", type=int, default=8)
    p.add_argument("--ga_elites", type=int, default=3)
    p.add_argument("--ga_mutation_prob", type=float, default=0.10)
    p.add_argument("--ga_tournament_k", type=int, default=3)
    p.add_argument(
        "--ga_selection",
        choices=["tournament", "rank"],
        default="rank",
        help="Parent selection strategy for GA reproduction.",
    )
    p.add_argument(
        "--ga_rank_pressure",
        type=float,
        default=1.7,
        help="Linear-ranking selection pressure in [1,2]; higher favors top ranks.",
    )
    p.add_argument("--ga_crossover", choices=["uniform", "one_point"], default="uniform")
    p.add_argument(
        "--ga_init_mode",
        choices=["random", "bayes", "hybrid"],
        default="random",
        help="Population init: random, bayes-prior, or hybrid mixture.",
    )
    p.add_argument(
        "--ga_bayes_init_frac",
        type=float,
        default=0.7,
        help="In hybrid mode, fraction of non-anchor population sampled from Bayesian prior.",
    )
    p.add_argument(
        "--ga_prior_strength",
        type=float,
        default=2.0,
        help="Sharpening factor for prior-guided sampling; higher means stronger prior.",
    )
    p.add_argument(
        "--ga_prior_cfg_center",
        type=float,
        default=1.0,
        help="Reference guidance center for prior-guided initialization.",
    )
    p.add_argument("--ga_log_topk", type=int, default=3)
    p.add_argument(
        "--ga_log_evals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reserved for detailed GA trace logging (JSONL).",
    )
    p.add_argument(
        "--ga_phase_constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--ga_guidance_scales",
        nargs="+",
        type=float,
        default=list(_DEFAULT_GUIDANCE_SCALES),
    )
    p.add_argument("--n_sims", type=int, default=50, help="MCTS simulation budget per prompt/seed.")
    p.add_argument("--ucb_c", type=float, default=1.41, help="MCTS UCB exploration constant.")
    p.add_argument(
        "--mcts_fresh_noise_steps",
        default="",
        help="Comma-separated step indices for extra fresh-noise exploration in MCTS (e.g. '0,2'). "
             "'all' enables every step.",
    )
    p.add_argument(
        "--mcts_fresh_noise_samples",
        type=int,
        default=1,
        help="Number of fresh-noise candidates to try at selected MCTS steps. 1 disables extra exploration.",
    )
    p.add_argument(
        "--mcts_fresh_noise_scale",
        type=float,
        default=1.0,
        help="Scale for additive latent perturbation used by fresh-noise exploration.",
    )
    p.add_argument("--bon_n", type=int, default=16,
                   help="Number of independent samples for best-of-N search.")
    p.add_argument(
        "--noise_inject_mode",
        choices=["seeds_only", "reinject_only", "combined"],
        default="combined",
        help="Sparse noise-injection search mode: seeds-only | reinjection-only | combined.",
    )
    p.add_argument(
        "--noise_inject_seed_budget",
        type=int,
        default=8,
        help="Number of root seeds to evaluate for seeds-only or combined mode.",
    )
    p.add_argument(
        "--noise_inject_candidate_steps",
        default="",
        help="Comma-separated candidate reinjection steps (e.g. '1,2'); 'all' uses all steps; empty uses middle step.",
    )
    p.add_argument(
        "--noise_inject_gamma_bank",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.50],
        help="Latent reinjection magnitude bank gamma for x_t <- x_t + gamma*sigma_t*eps.",
    )
    p.add_argument("--noise_inject_eps_samples", type=int, default=4)
    p.add_argument("--noise_inject_steps_per_rollout", type=int, default=1)
    p.add_argument(
        "--noise_inject_include_no_inject",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--noise_inject_max_policies", type=int, default=0)
    p.add_argument("--noise_inject_variant_idx", type=int, default=0)
    p.add_argument(
        "--noise_inject_guidance",
        type=float,
        default=None,
        help="Optional fixed guidance scale for sparse noise-injection search. If unset, search over cfg_scales.",
    )
    p.add_argument("--beam_width", type=int, default=4,
                   help="Number of beams to keep per step in beam search.")

    # SMC
    p.add_argument("--smc_k", type=int, default=12)
    p.add_argument("--smc_gamma", type=float, default=0.10)
    p.add_argument(
        "--smc_potential",
        choices=["tempering", "diff"],
        default="tempering",
        help="SMC potential type. 'tempering' = standard DAS geometric tempering "
             "(weight bump = lam_t * r̂_t with lam_t = (1+gamma)^(T-1-t) - 1). "
             "'diff' = FK-steering difference potential (Singhal et al. 2025): "
             "weight bump = smc_lambda * (r̂_t - r̂_{t-1}); telescopes to smc_lambda * r̂_final.",
    )
    p.add_argument(
        "--smc_lambda",
        type=float,
        default=10.0,
        help="FK-steering inverse temperature lambda used when --smc_potential=diff.",
    )
    p.add_argument("--ess_threshold", type=float, default=0.5)
    p.add_argument("--resample_start_frac", type=float, default=0.3)
    p.add_argument("--smc_guidance_scale", type=float, default=1.25)
    p.add_argument("--smc_chunk", type=int, default=4, help="Transformer batch chunk for particles.")
    p.add_argument(
        "--smc_variant_expansion",
        action="store_true",
        default=False,
        help="Enable (variant, guidance_scale) expansion at SMC resample points.",
    )
    p.add_argument("--smc_expansion_variants", nargs="+", type=int, default=[])
    p.add_argument("--smc_expansion_guidances", nargs="+", type=float, default=[])
    p.add_argument("--smc_expansion_factor", type=int, default=-1)
    p.add_argument(
        "--smc_expansion_proposal", choices=["uniform", "score_softmax"], default="uniform"
    )
    p.add_argument("--smc_expansion_tau", type=float, default=1.0)
    p.add_argument("--smc_expansion_lookahead", action="store_true", default=False)

    p.add_argument(
        "--x0_sampler",
        action="store_true",
        default=False,
        help="Treat transformer output as a direct x̂₀ prediction instead of flow/velocity. "
             "Set automatically for senseflow backends.",
    )
    p.add_argument(
        "--euler_sampler",
        action="store_true",
        default=False,
        help="Use Euler ODE stepping (latents += dt*flow) instead of SiD re-noising. "
             "Set automatically for sd35_base-like backends.",
    )
    p.add_argument("--save_first_k", type=int, default=10)
    p.add_argument("--out_dir", default="./flux_sampling_out")
    return _apply_backend_defaults(p.parse_args(argv))


def _apply_backend_defaults(args: argparse.Namespace) -> argparse.Namespace:
    cfg = _BACKEND_CONFIGS.get(args.backend or "", {})

    # Fill nullable fields first.
    for key in ("model_id", "transformer_id", "transformer_subfolder", "sigmas",
                "lora_repo", "lora_filename", "lora_scale"):
        if getattr(args, key, None) is None and key in cfg:
            setattr(args, key, cfg[key])

    # dtype has a parser default; only force backend dtype when backend explicitly set.
    if args.backend and "dtype" in cfg:
        args.dtype = str(cfg["dtype"])

    # Backend-supplied max_sequence_length overrides parser default (512) only
    # when the user kept that default. TDD-FLUX wants 256.
    if args.backend and "max_sequence_length" in cfg:
        if int(getattr(args, "max_sequence_length", 512)) == 512:
            args.max_sequence_length = int(cfg["max_sequence_length"])

    # Euler sampler from backend config
    if not getattr(args, "euler_sampler", False) and cfg.get("euler_sampler", False):
        args.euler_sampler = True

    # SenseFlow defaults should apply only when caller kept legacy defaults.
    if (args.backend or "").startswith("senseflow"):
        if not getattr(args, "x0_sampler", False):
            args.x0_sampler = bool(cfg.get("x0_sampler", False))
        if list(getattr(args, "ga_guidance_scales", [])) == list(_DEFAULT_GUIDANCE_SCALES):
            args.ga_guidance_scales = list(cfg.get("ga_guidance_scales", [0.0]))
        if getattr(args, "cfg_scales", None) is None:
            args.cfg_scales = list(cfg.get("cfg_scales", args.ga_guidance_scales))
        elif list(args.cfg_scales) == list(_DEFAULT_GUIDANCE_SCALES):
            args.cfg_scales = list(cfg.get("cfg_scales", [0.0]))
        if float(getattr(args, "baseline_guidance_scale", _DEFAULT_BASELINE_GUIDANCE)) == float(
            _DEFAULT_BASELINE_GUIDANCE
        ):
            args.baseline_guidance_scale = float(cfg.get("baseline_guidance_scale", 0.0))

    # TDD-FLUX defaults: real CFG (≈2.0), 8 steps, max_seq_len=256.
    if (args.backend or "") == "tdd_flux":
        if list(getattr(args, "ga_guidance_scales", [])) == list(_DEFAULT_GUIDANCE_SCALES):
            args.ga_guidance_scales = list(cfg.get("ga_guidance_scales", [2.0]))
        if getattr(args, "cfg_scales", None) is None:
            args.cfg_scales = list(cfg.get("cfg_scales", args.ga_guidance_scales))
        elif list(args.cfg_scales) == list(_DEFAULT_GUIDANCE_SCALES):
            args.cfg_scales = list(cfg.get("cfg_scales", [2.0]))
        if float(getattr(args, "baseline_guidance_scale", _DEFAULT_BASELINE_GUIDANCE)) == float(
            _DEFAULT_BASELINE_GUIDANCE
        ):
            args.baseline_guidance_scale = float(cfg.get("baseline_guidance_scale", 2.0))

    if args.model_id is None:
        args.model_id = str(_BACKEND_CONFIGS["flux"]["model_id"])

    if args.sigmas is not None:
        args.sigmas = [float(s) for s in args.sigmas]
        args.steps = len(args.sigmas)

    return args


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def pick_device(args: argparse.Namespace) -> str:
    req = str(args.device).strip().lower()
    if req != "cuda" or not torch.cuda.is_available():
        return "cpu"
    count = torch.cuda.device_count()
    if count <= 0:
        return "cpu"

    if args.gpu_id >= 0:
        if args.gpu_id >= count:
            raise RuntimeError(f"--gpu_id={args.gpu_id} out of range (count={count}).")
        return f"cuda:{args.gpu_id}"

    if not args.auto_select_gpu:
        return "cuda:0"

    best_idx = 0
    best_free = -1
    for idx in range(count):
        free_bytes, _ = torch.cuda.mem_get_info(idx)
        if free_bytes > best_free:
            best_free = free_bytes
            best_idx = idx
    return f"cuda:{best_idx}"


def cuda_free_gb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    try:
        idx = int(device.split(":", 1)[1])
    except Exception:
        idx = 0
    try:
        free_bytes, _ = torch.cuda.mem_get_info(idx)
    except Exception:
        return None
    return float(free_bytes) / (1024 ** 3)


def _hf_offline_mode() -> bool:
    return str(os.environ.get("HF_HUB_OFFLINE", "")).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_flux_lora_source(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Resolve LoRA source to (path_or_repo, weight_name_or_none).

    Returns:
      - local file: ("/abs/path/dir", "file.safetensors")
      - local dir + named file: ("/abs/path/dir", "adapter.safetensors")
      - hf cached download: ("/cache/.../snapshots/HASH", "file.safetensors")
      - no LoRA configured: (None, None)

    File paths are always split into (parent_dir, basename) so that
    diffusers receives an explicit weight_name and skips
    _best_guess_weight_name, which raises in offline mode.
    """
    lora_path = getattr(args, "lora_path", None)
    lora_repo = getattr(args, "lora_repo", None)
    lora_filename = getattr(args, "lora_filename", None)

    if lora_path:
        p = Path(str(lora_path)).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"LoRA path not found: {p}")
        if p.is_dir() and lora_filename:
            return str(p), str(lora_filename)
        if p.is_file():
            # Split file → (dir, filename) so diffusers doesn't run
            # _best_guess_weight_name (which fails in offline mode).
            return str(p.parent), p.name
        return str(p), None

    if not lora_repo:
        return None, None

    # Local-path repo override (directory or file).
    repo_path = Path(str(lora_repo)).expanduser()
    if repo_path.exists():
        if not repo_path.is_absolute():
            repo_path = (Path.cwd() / repo_path).resolve()
        if repo_path.is_dir() and lora_filename:
            return str(repo_path), str(lora_filename)
        if repo_path.is_file():
            return str(repo_path.parent), repo_path.name
        return str(repo_path), None

    # HF repo id path.
    if not lora_filename:
        return str(lora_repo), None

    from huggingface_hub import hf_hub_download

    offline = _hf_offline_mode()
    dl_kwargs: dict[str, Any] = {
        "repo_id": str(lora_repo),
        "filename": str(lora_filename),
        "cache_dir": os.environ.get("HF_HOME"),
        "token": os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        "local_files_only": bool(offline),
    }
    try:
        cached = hf_hub_download(**dl_kwargs)
    except Exception as exc:
        if offline:
            raise RuntimeError(
                "HF_HUB_OFFLINE=1 and TDD LoRA is not available in local cache. "
                f"Missing: repo={lora_repo} file={lora_filename}. "
                "Pre-download it before enabling offline mode, or set --lora_path "
                "to a local LoRA file."
            ) from exc
        raise
    cached_path = Path(str(cached))
    return str(cached_path.parent), cached_path.name


def load_pipeline(args: argparse.Namespace, device: str, dtype: torch.dtype) -> FluxContext:
    print(
        f"Loading FLUX pipeline: {args.model_id} "
        f"(backend={args.backend or 'flux'} "
        f"sigmas={args.sigmas if args.sigmas is not None else f'steps={args.steps}'})"
    )
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    if getattr(args, "transformer_id", None):
        try:
            from diffusers.models.transformers import FluxTransformer2DModel
        except Exception:
            from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        transformer_subfolder = getattr(args, "transformer_subfolder", None)
        if transformer_subfolder:
            kwargs["subfolder"] = transformer_subfolder
        print(
            f"Loading FLUX transformer override: {args.transformer_id} "
            f"subfolder={transformer_subfolder}"
        )
        pipe.transformer = FluxTransformer2DModel.from_pretrained(args.transformer_id, **kwargs).to(device)

    lora_source, lora_weight = _resolve_flux_lora_source(args)
    if lora_source:
        scale = float(getattr(args, "lora_scale", 1.0) or 1.0)
        if lora_weight:
            print(f"Loading FLUX LoRA: source={lora_source} weight={lora_weight} fuse_scale={scale}")
            pipe.load_lora_weights(lora_source, weight_name=lora_weight)
        else:
            print(f"Loading FLUX LoRA: source={lora_source} fuse_scale={scale}")
            pipe.load_lora_weights(lora_source)
        pipe.fuse_lora(lora_scale=scale)
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
    pipe.transformer.eval().requires_grad_(False)
    pipe.vae.eval().requires_grad_(False)
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.eval().requires_grad_(False)
    if getattr(pipe, "text_encoder_2", None) is not None:
        pipe.text_encoder_2.eval().requires_grad_(False)

    decode_cpu_dtype = resolve_dtype(args.decode_cpu_dtype)
    decode_device = args.decode_device
    if decode_device == "auto":
        if not device.startswith("cuda"):
            decode_device = "cpu"
        else:
            free_gb = cuda_free_gb(device)
            if free_gb is not None and free_gb < args.decode_cpu_if_free_below_gb:
                decode_device = "cpu"
            else:
                decode_device = "cuda"

    if decode_device == "cpu":
        pipe.vae.to(device="cpu", dtype=decode_cpu_dtype)

    return FluxContext(
        pipe=pipe,
        device=device,
        dtype=dtype,
        decode_device_request=args.decode_device,
        decode_device=decode_device,
        decode_cpu_dtype=decode_cpu_dtype,
        decode_cpu_if_free_below_gb=float(args.decode_cpu_if_free_below_gb),
        empty_cache_after_decode=bool(args.empty_cache_after_decode),
    )


def resolve_reward_device(args: argparse.Namespace, pipeline_device: str) -> str:
    v = str(args.reward_device).strip().lower()
    if v in {"same", "auto"}:
        return pipeline_device
    if v == "cuda":
        return "cuda:0"
    return str(args.reward_device)


def load_reward(args: argparse.Namespace, pipeline_device: str):
    reward_device = resolve_reward_device(args, pipeline_device)
    unified_model = args.unifiedreward_model if args.unifiedreward_model else args.reward_model
    print(f"Loading reward backend={args.reward_backend} on {reward_device} ...")
    scorer = UnifiedRewardScorer(
        device=reward_device,
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
    return scorer, reward_device


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt is not None:
        return [str(args.prompt).strip()]
    if not args.prompt_file:
        raise RuntimeError("Provide --prompt or --prompt_file.")
    with open(args.prompt_file, encoding="utf-8") as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    if args.shuffle_prompts:
        rng = random.Random(args.seed)
        rng.shuffle(all_prompts)
    if args.n_prompts < 0:
        return all_prompts
    return all_prompts[: int(args.n_prompts)]


def build_prompt_bank(prompt: str) -> list[tuple[str, str]]:
    bank: list[tuple[str, str]] = []
    for label, hint in PROMPT_FOCUS.items():
        if hint:
            bank.append((label, f"{prompt}. {hint}"))
        else:
            bank.append((label, prompt))
    return bank


def select_prompt_bank(prompt: str, n_variants: int) -> list[tuple[str, str]]:
    bank = build_prompt_bank(prompt)
    if n_variants is None or int(n_variants) <= 0:
        return bank
    keep = max(1, min(int(n_variants), len(bank)))
    return bank[:keep]


def guidance_bank_for_search(args: argparse.Namespace) -> list[float]:
    raw = args.cfg_scales if args.cfg_scales else args.ga_guidance_scales
    bank = [float(v) for v in raw]
    if len(bank) == 0:
        bank = [1.0]
    # Keep order while dropping duplicates.
    out: list[float] = []
    seen: set[float] = set()
    for v in bank:
        if v in seen:
            continue
        seen.add(v)
        out.append(float(v))
    return out if out else [1.0]


def _move_text_encoders(pipe: Any, device: str) -> None:
    te = getattr(pipe, "text_encoder", None)
    if te is not None:
        te.to(device)
    te2 = getattr(pipe, "text_encoder_2", None)
    if te2 is not None:
        te2.to(device)


@torch.no_grad()
def encode_prompt_bank(
    args: argparse.Namespace,
    ctx: FluxContext,
    prompt_bank: list[tuple[str, str]],
) -> list[PromptEmbed]:
    _move_text_encoders(ctx.pipe, ctx.device)
    embeds: list[PromptEmbed] = []
    for label, text in prompt_bank:
        pe, ppe, txt_ids = ctx.pipe.encode_prompt(
            prompt=text,
            prompt_2=text,
            device=ctx.device,
            num_images_per_prompt=1,
            max_sequence_length=int(args.max_sequence_length),
        )
        embeds.append(
            PromptEmbed(
                label=label,
                text=text,
                prompt_embeds=pe.detach(),
                pooled_prompt_embeds=ppe.detach(),
                text_ids=txt_ids.detach(),
            )
        )

    if args.offload_text_encoder_after_encode and ctx.device.startswith("cuda"):
        _move_text_encoders(ctx.pipe, "cpu")
        gc.collect()
        torch.cuda.empty_cache()
    return embeds


@torch.no_grad()
def make_initial_latents(
    ctx: FluxContext,
    seed: int,
    height: int,
    width: int,
    batch_size: int,
) -> torch.Tensor:
    exec_device = torch.device(ctx.device) if isinstance(ctx.device, str) else ctx.device
    generator = torch.Generator(device=exec_device).manual_seed(seed)
    num_channels_latents = int(ctx.pipe.transformer.config.in_channels) // 4
    packed, _ = ctx.pipe.prepare_latents(
        batch_size=batch_size,
        num_channels_latents=num_channels_latents,
        height=height,
        width=width,
        dtype=ctx.dtype,
        device=exec_device,
        generator=generator,
    )
    return ctx.pipe._unpack_latents(
        packed,
        height=height,
        width=width,
        vae_scale_factor=ctx.pipe.vae_scale_factor,
    )


def build_t_schedule(steps: int, sigmas: list[float] | None = None) -> list[float]:
    if sigmas is not None:
        out = [float(s) for s in sigmas]
        if len(out) == 0:
            raise RuntimeError("Empty --sigmas is not allowed.")
        return out
    if steps <= 1:
        return [1.0]
    denom = float(steps - 1)
    return [1.0 - float(i) / denom for i in range(steps)]


def _resolve_mcts_fresh_noise_steps(args: argparse.Namespace, total_steps: int) -> set[int]:
    out: set[int] = set()
    raw = str(getattr(args, "mcts_fresh_noise_steps", "")).strip().lower()
    if not raw:
        return out
    if raw == "all":
        return set(range(max(0, int(total_steps))))
    for tok in raw.split(","):
        t = tok.strip()
        if t.isdigit():
            idx = int(t)
            if 0 <= idx < int(total_steps):
                out.add(int(idx))
    return out


def _mcts_fresh_noise_samples_for_step(args: argparse.Namespace, step_idx: int, enabled_steps: set[int]) -> int:
    if int(step_idx) not in enabled_steps:
        return 1
    return max(1, int(getattr(args, "mcts_fresh_noise_samples", 1)))


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


def _noise_inject_guidance_bank(args: argparse.Namespace) -> tuple[list[float], bool]:
    override = getattr(args, "noise_inject_guidance", None)
    if override is not None:
        return [float(override)], True

    baseline = float(getattr(args, "baseline_guidance_scale", 1.0))
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


def _build_flux_step_noise_cache(
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
def flux_transformer_step(
    ctx: FluxContext,
    latents: torch.Tensor,
    embed: PromptEmbed,
    t_val: float,
    guidance_scale: float,
) -> torch.Tensor:
    bsz = int(latents.shape[0])
    packed_latents = ctx.pipe._pack_latents(
        latents,
        batch_size=bsz,
        num_channels_latents=int(latents.shape[1]),
        height=int(latents.shape[2]),
        width=int(latents.shape[3]),
    )
    latent_image_ids = ctx.pipe._prepare_latent_image_ids(
        bsz,
        int(latents.shape[2]) // 2,
        int(latents.shape[3]) // 2,
        latents.device,
        latents.dtype,
    )

    prompt_embeds = embed.prompt_embeds.expand(bsz, -1, -1).to(device=ctx.device)
    pooled = embed.pooled_prompt_embeds.expand(bsz, -1).to(device=ctx.device)
    text_ids = embed.text_ids.to(device=ctx.device)
    t_flat = torch.full((bsz,), float(t_val), device=ctx.device, dtype=latents.dtype)

    # guidance_embeds is True for FLUX.1-dev (guidance distillation) and False for
    # FLUX.1-schnell. Older diffusers versions raise TypeError when guidance is passed
    # to a schnell transformer, so only include it when the config requires it.
    transformer_kwargs: dict = dict(
        hidden_states=packed_latents,
        timestep=t_flat,
        pooled_projections=pooled,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        return_dict=False,
    )
    if getattr(ctx.pipe.transformer.config, "guidance_embeds", False):
        guidance = torch.full((bsz,), float(guidance_scale), device=ctx.device, dtype=latents.dtype)
        transformer_kwargs["guidance"] = guidance

    flow_packed = ctx.pipe.transformer(**transformer_kwargs)[0]

    unpack_h = int(latents.shape[2]) * int(ctx.pipe.vae_scale_factor)
    unpack_w = int(latents.shape[3]) * int(ctx.pipe.vae_scale_factor)
    flow = ctx.pipe._unpack_latents(
        flow_packed,
        height=unpack_h,
        width=unpack_w,
        vae_scale_factor=ctx.pipe.vae_scale_factor,
    )
    return flow


@torch.no_grad()
def flux_transformer_step_chunked(
    ctx: FluxContext,
    latents: torch.Tensor,
    embed: PromptEmbed,
    t_val: float,
    guidance_scale: float,
    chunk: int,
) -> torch.Tensor:
    bsz = int(latents.shape[0])
    if chunk <= 0 or bsz <= chunk:
        return flux_transformer_step(ctx, latents, embed, t_val, guidance_scale)
    outs = []
    for start in range(0, bsz, chunk):
        end = min(start + chunk, bsz)
        outs.append(
            flux_transformer_step(
                ctx=ctx,
                latents=latents[start:end],
                embed=embed,
                t_val=t_val,
                guidance_scale=guidance_scale,
            )
        )
    return torch.cat(outs, dim=0)


def _is_cuda_oom(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "cuda" in msg and "out of memory" in msg


def _ensure_vae_on(ctx: FluxContext, target: str) -> None:
    if target == "cpu":
        ctx.pipe.vae.to(device="cpu", dtype=ctx.decode_cpu_dtype)
        ctx.decode_device = "cpu"
        if ctx.device.startswith("cuda"):
            torch.cuda.empty_cache()
    else:
        ctx.pipe.vae.to(device=ctx.device, dtype=ctx.dtype)
        ctx.decode_device = "cuda"


@torch.no_grad()
def decode_to_pil(ctx: FluxContext, dx: torch.Tensor) -> Image.Image:
    if ctx.decode_device_request == "auto" and ctx.device.startswith("cuda"):
        free_gb = cuda_free_gb(ctx.device)
        if free_gb is not None and free_gb < ctx.decode_cpu_if_free_below_gb:
            _ensure_vae_on(ctx, "cpu")
        else:
            _ensure_vae_on(ctx, "cuda")
    elif ctx.decode_device_request == "cpu":
        _ensure_vae_on(ctx, "cpu")
    elif ctx.decode_device_request == "cuda":
        _ensure_vae_on(ctx, "cuda")

    scale = float(ctx.pipe.vae.config.scaling_factor)
    shift = float(getattr(ctx.pipe.vae.config, "shift_factor", 0.0))
    z = (dx / scale) + shift

    if ctx.decode_device == "cpu":
        z = z.to(device="cpu", dtype=ctx.decode_cpu_dtype)
        image = ctx.pipe.vae.decode(z, return_dict=False)[0]
    else:
        z = z.to(device=ctx.device, dtype=ctx.dtype)
        try:
            image = ctx.pipe.vae.decode(z, return_dict=False)[0]
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            _ensure_vae_on(ctx, "cpu")
            z = z.to(device="cpu", dtype=ctx.decode_cpu_dtype)
            image = ctx.pipe.vae.decode(z, return_dict=False)[0]

    pil = ctx.pipe.image_processor.postprocess(image, output_type="pil")[0]
    if ctx.empty_cache_after_decode and ctx.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return pil


def score_image(reward_model: Any, prompt: str, image: Image.Image) -> float:
    return float(reward_model.score(prompt, image))


def build_action_space(num_variants: int, guidance_bank: list[float]) -> list[tuple[int, float]]:
    return [(int(vi), float(g)) for vi in range(int(num_variants)) for g in guidance_bank]


def _pred_x0(
    xt: torch.Tensor,
    t: torch.Tensor,
    out: torch.Tensor,
    x0_sampler: bool,
) -> torch.Tensor:
    """Convert transformer output to x̂₀.

    Flow sampler (FLUX):     x̂₀ = xt - t * flow
    X0 sampler (SenseFlow):  x̂₀ = out  (transformer already predicts x̂₀)
    """
    return out if x0_sampler else xt - t * out


def _final_decode_tensor(
    latents: torch.Tensor,
    dx: torch.Tensor,
    use_euler: bool,
) -> torch.Tensor:
    """Return the tensor to pass to VAE decode.

    Euler: decode the accumulated latents (ODE solution).
    SiD/SenseFlow re-noising: decode dx (the x̂₀ prediction).
    """
    return latents if use_euler else dx


def _compute_dt(t_values: list[float], step_idx: int) -> float:
    """Compute Euler step size dt = t_{next} - t_{current}."""
    if step_idx + 1 < len(t_values):
        return float(t_values[step_idx + 1]) - float(t_values[step_idx])
    return -float(t_values[step_idx])  # last step goes to 0


@torch.no_grad()
def run_action_sequence(
    args: argparse.Namespace,
    ctx: FluxContext,
    reward_model: Any,
    prompt: str,
    embeds: list[PromptEmbed],
    seed: int,
    actions: list[tuple[int, float]],
) -> SearchResult:
    if len(actions) != int(args.steps):
        raise ValueError(f"Expected {args.steps} actions, got {len(actions)}")

    use_euler = getattr(args, "euler_sampler", False)
    init_latents = make_initial_latents(ctx, seed, args.height, args.width, batch_size=1)
    dx = torch.zeros_like(init_latents)
    latents = init_latents.clone()
    rng = torch.Generator(device=ctx.device).manual_seed(seed + 2048)
    t_values = build_t_schedule(int(args.steps), getattr(args, "sigmas", None))

    for step_idx, (variant_idx, guidance) in enumerate(actions):
        t_val = float(t_values[step_idx])
        t_4d = torch.tensor(t_val, device=ctx.device, dtype=init_latents.dtype).view(1, 1, 1, 1)
        if not use_euler:
            if step_idx == 0:
                noise = init_latents
            else:
                noise = torch.randn(init_latents.shape, device=ctx.device,
                                    dtype=init_latents.dtype, generator=rng)
            latents = (1.0 - t_4d) * dx + t_4d * noise
        flow_pred = flux_transformer_step(ctx, latents, embeds[int(variant_idx)], t_val, float(guidance))
        dx = _pred_x0(latents, t_4d, flow_pred, args.x0_sampler)
        if use_euler:
            dt = _compute_dt(t_values, step_idx)
            latents = latents + dt * flow_pred

    image = decode_to_pil(ctx, _final_decode_tensor(latents, dx, use_euler))
    score = score_image(reward_model, prompt, image)
    return SearchResult(image=image, score=float(score), actions=list(actions), diagnostics=None)


@torch.no_grad()
def run_greedy(
    args: argparse.Namespace,
    ctx: FluxContext,
    reward_model: Any,
    prompt: str,
    embeds: list[PromptEmbed],
    guidance_bank: list[float],
    seed: int,
) -> SearchResult:
    actions = build_action_space(len(embeds), guidance_bank)
    if len(actions) == 0:
        raise RuntimeError("Greedy search requires non-empty action space.")

    use_euler = getattr(args, "euler_sampler", False)
    init_latents = make_initial_latents(ctx, seed, args.height, args.width, batch_size=1)
    dx = torch.zeros_like(init_latents)
    latents = init_latents.clone()
    rng = torch.Generator(device=ctx.device).manual_seed(seed + 2048)
    t_values = build_t_schedule(int(args.steps), getattr(args, "sigmas", None))
    chosen: list[tuple[int, float]] = []
    history: list[dict[str, Any]] = []
    nfe_total = 0

    for step_idx, t_val in enumerate(t_values):
        t_4d = torch.tensor(float(t_val), device=ctx.device, dtype=init_latents.dtype).view(1, 1, 1, 1)
        if not use_euler:
            if step_idx == 0:
                noise = init_latents
            else:
                noise = torch.randn(
                    init_latents.shape,
                    device=ctx.device,
                    dtype=init_latents.dtype,
                    generator=rng,
                )
            latents = (1.0 - t_4d) * dx + t_4d * noise

        best_score = -float("inf")
        best_action = actions[0]
        best_dx = None
        best_latents = None
        print(f"  greedy step {step_idx + 1}/{args.steps} ({len(actions)} actions)")
        for variant_idx, guidance in actions:
            flow_pred = flux_transformer_step(ctx, latents, embeds[int(variant_idx)], float(t_val), float(guidance))
            nfe_total += 1
            cand_dx = _pred_x0(latents, t_4d, flow_pred, args.x0_sampler)
            if use_euler:
                dt = _compute_dt(t_values, step_idx)
                cand_latents = latents + dt * flow_pred
                cand_decode = _final_decode_tensor(cand_latents, cand_dx, use_euler)
            else:
                cand_latents = None
                cand_decode = cand_dx
            cand_img = decode_to_pil(ctx, cand_decode)
            cand_score = score_image(reward_model, prompt, cand_img)
            del cand_img
            if cand_score > best_score:
                best_score = float(cand_score)
                best_action = (int(variant_idx), float(guidance))
                best_dx = cand_dx.clone()
                if use_euler:
                    best_latents = cand_latents.clone()

        assert best_dx is not None
        dx = best_dx
        if use_euler:
            latents = best_latents
        chosen.append(best_action)
        history.append(
            {
                "step": int(step_idx),
                "chosen_variant": int(best_action[0]),
                "chosen_guidance": float(best_action[1]),
                "score": float(best_score),
            }
        )
        print(
            f"    selected step={step_idx + 1} v{best_action[0]} "
            f"g={best_action[1]:.2f} score={best_score:.4f}"
        )

    final_img = decode_to_pil(ctx, _final_decode_tensor(latents, dx, use_euler))
    final_score = score_image(reward_model, prompt, final_img)
    diagnostics = {
        "history": history,
        "steps": int(args.steps),
        "num_actions_per_step": int(len(actions)),
        "nfe_total": int(nfe_total),
        "guidance_bank": [float(v) for v in guidance_bank],
    }
    return SearchResult(
        image=final_img,
        score=float(final_score),
        actions=[(int(v), float(g)) for v, g in chosen],
        diagnostics=diagnostics,
    )


class MCTSNode:
    __slots__ = ("step", "dx", "latents", "children", "visits", "action_visits", "action_values")

    def __init__(self, step: int, dx: torch.Tensor, latents: torch.Tensor | None):
        self.step = int(step)
        self.dx = dx
        self.latents = latents
        self.children: dict[tuple[int, float], MCTSNode] = {}
        self.visits = 0
        self.action_visits: dict[tuple[int, float], int] = {}
        self.action_values: dict[tuple[int, float], float] = {}

    def is_leaf(self, max_steps: int) -> bool:
        return self.step >= int(max_steps)

    def untried_actions(self, actions: list[tuple[int, float]]) -> list[tuple[int, float]]:
        return [a for a in actions if a not in self.action_visits]

    def ucb(self, action: tuple[int, float], c: float) -> float:
        n = self.action_visits.get(action, 0)
        if n <= 0:
            return float("inf")
        mean = self.action_values[action] / float(n)
        return mean + float(c) * math.sqrt(math.log(max(self.visits, 1)) / float(n))

    def best_ucb(self, actions: list[tuple[int, float]], c: float) -> tuple[int, float]:
        return max(actions, key=lambda a: self.ucb(a, c))

    def best_exploit(self, actions: list[tuple[int, float]]) -> tuple[int, float] | None:
        best = None
        best_mean = -float("inf")
        for action in actions:
            n = self.action_visits.get(action, 0)
            if n <= 0:
                continue
            mean = self.action_values[action] / float(n)
            if mean > best_mean:
                best_mean = mean
                best = action
        return best


@torch.no_grad()
def run_bon(
    args: argparse.Namespace,
    ctx: FluxContext,
    reward_model: Any,
    prompt: str,
    embeds: list[PromptEmbed],
    seed: int,
) -> SearchResult:
    """Best of N: generate N independent samples with baseline guidance, return highest-scoring."""
    n = max(1, int(args.bon_n))
    use_euler = getattr(args, "euler_sampler", False)
    guidance = float(args.baseline_guidance_scale)
    variant_idx = 0
    fixed_actions: list[tuple[int, float]] = [(variant_idx, guidance)] * int(args.steps)
    t_values = build_t_schedule(int(args.steps), getattr(args, "sigmas", None))

    best_score = -float("inf")
    best_img: Image.Image | None = None

    print(f"  bon: n={n} guidance={guidance:.2f} variant={variant_idx} euler={use_euler}")
    for i in range(n):
        s = seed + i
        init_latents = make_initial_latents(ctx, s, args.height, args.width, batch_size=1)
        dx = torch.zeros_like(init_latents)
        latents = init_latents.clone()
        rng = torch.Generator(device=ctx.device).manual_seed(s + 2048)
        for step_idx, t_val in enumerate(t_values):
            t_4d = torch.tensor(float(t_val), device=ctx.device, dtype=init_latents.dtype).view(1, 1, 1, 1)
            if not use_euler:
                if step_idx == 0:
                    noise = init_latents
                else:
                    noise = torch.randn(
                        init_latents.shape, device=ctx.device, dtype=init_latents.dtype, generator=rng
                    )
                latents = (1.0 - t_4d) * dx + t_4d * noise
            flow = flux_transformer_step(ctx, latents, embeds[variant_idx], float(t_val), guidance)
            dx = _pred_x0(latents, t_4d, flow, args.x0_sampler)
            if use_euler:
                dt = _compute_dt(t_values, step_idx)
                latents = latents + dt * flow
        img = decode_to_pil(ctx, _final_decode_tensor(latents, dx, use_euler))
        score = score_image(reward_model, prompt, img)
        mark = ""
        if score > best_score:
            best_score = float(score)
            best_img = img
            mark = " <- best"
        print(f"    sample {i + 1}/{n} seed={s} score={score:.4f}{mark}")

    assert best_img is not None
    return SearchResult(
        image=best_img,
        score=best_score,
        actions=fixed_actions,
        diagnostics={"bon_n": n, "guidance": guidance},
    )


@torch.no_grad()
def _run_sparse_noise_rollout(
    args: argparse.Namespace,
    ctx: FluxContext,
    reward_model: Any,
    prompt: str,
    embeds: list[PromptEmbed],
    init_latents: torch.Tensor,
    step_noise_cache: list[torch.Tensor],
    eps_bank: list[torch.Tensor],
    fixed_variant_idx: int,
    fixed_guidance: float,
    t_values: list[float],
    policy: tuple[tuple[int, float, int], ...],
) -> tuple[Image.Image, float]:
    use_euler = bool(getattr(args, "euler_sampler", False))
    latents = init_latents.clone()
    dx = torch.zeros_like(init_latents)

    inject_map: dict[int, tuple[float, int]] = {}
    for step_idx, gamma, eps_id in policy:
        inject_map[int(step_idx)] = (float(gamma), int(eps_id))

    for step_idx, t_val in enumerate(t_values):
        t_4d = torch.tensor(float(t_val), device=ctx.device, dtype=init_latents.dtype).view(1, 1, 1, 1)
        if not use_euler:
            base_noise = step_noise_cache[step_idx] if step_idx < len(step_noise_cache) else step_noise_cache[-1]
            latents = (1.0 - t_4d) * dx + t_4d * base_noise

        if step_idx in inject_map:
            gamma, eps_id = inject_map[step_idx]
            eps = eps_bank[int(eps_id) % len(eps_bank)]
            latents = latents + float(gamma) * t_4d * eps

        flow = flux_transformer_step(
            ctx=ctx,
            latents=latents,
            embed=embeds[int(fixed_variant_idx)],
            t_val=float(t_val),
            guidance_scale=float(fixed_guidance),
        )
        dx = _pred_x0(latents, t_4d, flow, args.x0_sampler)
        if use_euler:
            dt = _compute_dt(t_values, step_idx)
            latents = latents + dt * flow

    final_img = decode_to_pil(ctx, _final_decode_tensor(latents, dx, use_euler))
    final_score = score_image(reward_model, prompt, final_img)
    return final_img, float(final_score)


@torch.no_grad()
def run_noise_inject(
    args: argparse.Namespace,
    ctx: FluxContext,
    reward_model: Any,
    prompt: str,
    embeds: list[PromptEmbed],
    seed: int,
) -> SearchResult:
    mode = str(getattr(args, "noise_inject_mode", "combined")).strip().lower()
    seed_budget = max(1, int(getattr(args, "noise_inject_seed_budget", 8)))
    if mode == "reinject_only":
        seed_candidates = [int(seed)]
    else:
        seed_candidates = [int(seed) + i for i in range(seed_budget)]

    t_values = build_t_schedule(int(args.steps), getattr(args, "sigmas", None))
    n_steps = len(t_values)
    policies = _build_noise_inject_policies(args, n_steps)
    candidate_steps = _resolve_noise_inject_steps(args, n_steps)
    gamma_bank = _noise_inject_gamma_bank(args)
    eps_samples = max(1, int(getattr(args, "noise_inject_eps_samples", 4)))
    steps_per_rollout = max(1, min(2, int(getattr(args, "noise_inject_steps_per_rollout", 1))))

    fixed_variant_idx = int(getattr(args, "noise_inject_variant_idx", 0))
    fixed_variant_idx = max(0, min(len(embeds) - 1, fixed_variant_idx))
    guidance_bank, guidance_is_override = _noise_inject_guidance_bank(args)

    print(
        "  noise_inject: "
        f"mode={mode} seeds={len(seed_candidates)} policies_per_seed={len(policies)} "
        f"guidance_candidates={len(guidance_bank)} steps_per_rollout={steps_per_rollout}"
    )
    print(
        "    "
        f"candidate_steps={candidate_steps} gamma_bank={[float(g) for g in gamma_bank]} "
        f"eps_samples={eps_samples} guidance_bank={[float(g) for g in guidance_bank]} variant={fixed_variant_idx}"
    )

    t_start = time.perf_counter()
    best_score = -float("inf")
    best_img: Image.Image | None = None
    best_seed = int(seed_candidates[0])
    best_policy: tuple[tuple[int, float, int], ...] = tuple()
    best_guidance = float(guidance_bank[0])
    total_rollouts = len(seed_candidates) * len(guidance_bank) * len(policies)
    rollout_idx = 0

    for seed_val in seed_candidates:
        init_latents = make_initial_latents(ctx, int(seed_val), args.height, args.width, batch_size=1)
        step_noise_cache = _build_flux_step_noise_cache(init_latents, n_steps, int(seed_val))
        eps_bank = _build_eps_bank_like(init_latents, eps_samples, seed=int(seed_val) + 900001)

        for guidance_val in guidance_bank:
            for policy in policies:
                rollout_idx += 1
                img, score = _run_sparse_noise_rollout(
                    args=args,
                    ctx=ctx,
                    reward_model=reward_model,
                    prompt=prompt,
                    embeds=embeds,
                    init_latents=init_latents,
                    step_noise_cache=step_noise_cache,
                    eps_bank=eps_bank,
                    fixed_variant_idx=fixed_variant_idx,
                    fixed_guidance=float(guidance_val),
                    t_values=t_values,
                    policy=policy,
                )
                if score > best_score:
                    best_score = float(score)
                    best_img = img
                    best_seed = int(seed_val)
                    best_policy = tuple(policy)
                    best_guidance = float(guidance_val)
                if rollout_idx == 1 or rollout_idx % 10 == 0 or rollout_idx == total_rollouts:
                    print(f"    rollout {rollout_idx:4d}/{total_rollouts} best={best_score:.4f}")

    assert best_img is not None
    fixed_actions: list[tuple[int, float]] = [(fixed_variant_idx, best_guidance)] * int(n_steps)
    nfe_total = int(total_rollouts) * int(n_steps)
    elapsed = float(time.perf_counter() - t_start)
    diagnostics = {
        "mode": mode,
        "seed_candidates": [int(s) for s in seed_candidates],
        "candidate_steps": [int(s) for s in candidate_steps],
        "gamma_bank": [float(g) for g in gamma_bank],
        "guidance_bank": [float(g) for g in guidance_bank],
        "guidance_source": "override" if guidance_is_override else "cfg_scales",
        "eps_samples": int(eps_samples),
        "steps_per_rollout": int(steps_per_rollout),
        "policies_per_seed": int(len(policies)),
        "guidance_candidates": int(len(guidance_bank)),
        "rollouts_evaluated": int(total_rollouts),
        "equivalent_seed_budget": int(total_rollouts),
        "nfe_total": int(nfe_total),
        "wall_time_sec": float(elapsed),
        "fixed_variant_idx": int(fixed_variant_idx),
        "fixed_guidance": float(best_guidance),
        "chosen_guidance": float(best_guidance),
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
        actions=[(int(v), float(g)) for v, g in fixed_actions],
        diagnostics=diagnostics,
    )


@torch.no_grad()
def run_beam(
    args: argparse.Namespace,
    ctx: FluxContext,
    reward_model: Any,
    prompt: str,
    embeds: list[PromptEmbed],
    guidance_bank: list[float],
    seed: int,
) -> SearchResult:
    """Beam search: maintain top-K partial trajectories, expand with full action space per step."""
    actions = build_action_space(len(embeds), guidance_bank)
    if len(actions) == 0:
        raise RuntimeError("Beam search requires non-empty action space.")
    use_euler = getattr(args, "euler_sampler", False)
    beam_width = max(1, int(args.beam_width))
    t_values = build_t_schedule(int(args.steps), getattr(args, "sigmas", None))
    init_latents = make_initial_latents(ctx, seed, args.height, args.width, batch_size=1)
    nfe_total = 0

    # beams: list of (dx, latents, path)
    beams: list[tuple[torch.Tensor, torch.Tensor, list[tuple[int, float]]]] = [
        (torch.zeros_like(init_latents), init_latents, [])
    ]

    print(f"  beam: width={beam_width} actions={len(actions)} steps={args.steps} euler={use_euler}")
    for step_idx, t_val in enumerate(t_values):
        t_4d = torch.tensor(float(t_val), device=ctx.device, dtype=init_latents.dtype).view(1, 1, 1, 1)
        candidates: list[tuple[float, torch.Tensor, torch.Tensor, list[tuple[int, float]]]] = []

        for beam_dx, beam_latents, beam_path in beams:
            if not use_euler:
                if step_idx == 0:
                    noise = beam_latents
                else:
                    noise = torch.randn(beam_latents.shape, device=ctx.device, dtype=beam_latents.dtype)
                latents_in = (1.0 - t_4d) * beam_dx + t_4d * noise
            else:
                latents_in = beam_latents
            for vi, guidance in actions:
                flow = flux_transformer_step(ctx, latents_in, embeds[int(vi)], float(t_val), float(guidance))
                nfe_total += 1
                cand_dx = _pred_x0(latents_in, t_4d, flow, args.x0_sampler)
                if use_euler:
                    dt = _compute_dt(t_values, step_idx)
                    cand_latents = latents_in + dt * flow
                    cand_decode = _final_decode_tensor(cand_latents, cand_dx, use_euler)
                else:
                    cand_latents = latents_in
                    cand_decode = cand_dx
                cand_img = decode_to_pil(ctx, cand_decode)
                cand_score = score_image(reward_model, prompt, cand_img)
                del cand_img
                candidates.append((cand_score, cand_dx, cand_latents, beam_path + [(vi, guidance)]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        print(
            f"  step {step_idx + 1}/{args.steps}: {len(candidates)} candidates, "
            f"top={candidates[0][0]:.4f}"
        )

        if step_idx + 1 < int(args.steps):
            if not use_euler:
                next_t = float(t_values[step_idx + 1])
                next_t_4d = torch.tensor(next_t, device=ctx.device, dtype=init_latents.dtype).view(1, 1, 1, 1)
                beams = []
                for _, dx, _, path in candidates[:beam_width]:
                    noise = torch.randn(dx.shape, device=ctx.device, dtype=dx.dtype)
                    next_latents = (1.0 - next_t_4d) * dx + next_t_4d * noise
                    beams.append((dx, next_latents, path))
            else:
                beams = [(dx, lat, path) for _, dx, lat, path in candidates[:beam_width]]
        else:
            beams = [(dx, lat, path) for _, dx, lat, path in candidates[:beam_width]]

    best_dx, best_latents, best_path = beams[0]
    best_img = decode_to_pil(ctx, _final_decode_tensor(best_latents, best_dx, use_euler))
    final_score = score_image(reward_model, prompt, best_img)
    return SearchResult(
        image=best_img,
        score=float(final_score),
        actions=best_path,
        diagnostics={"beam_width": beam_width, "nfe_total": nfe_total},
    )


@torch.no_grad()
def run_mcts(
    args: argparse.Namespace,
    ctx: FluxContext,
    reward_model: Any,
    prompt: str,
    embeds: list[PromptEmbed],
    guidance_bank: list[float],
    seed: int,
) -> SearchResult:
    actions = build_action_space(len(embeds), guidance_bank)
    if len(actions) == 0:
        raise RuntimeError("MCTS requires non-empty action space.")

    use_euler = getattr(args, "euler_sampler", False)
    n_actions = len(actions)
    t_values = build_t_schedule(int(args.steps), getattr(args, "sigmas", None))
    fresh_noise_steps = _resolve_mcts_fresh_noise_steps(args, int(args.steps))
    fresh_noise_samples = max(1, int(getattr(args, "mcts_fresh_noise_samples", 1)))
    if len(fresh_noise_steps) > 0 and fresh_noise_samples > 1:
        print(
            f"  mcts fresh-noise: steps={sorted(int(x) for x in fresh_noise_steps)} "
            f"samples={fresh_noise_samples} scale={float(getattr(args, 'mcts_fresh_noise_scale', 1.0)):.3f}"
        )
    init_latents = make_initial_latents(ctx, seed, args.height, args.width, batch_size=1)
    dx0 = torch.zeros_like(init_latents)
    if use_euler:
        start_latents = init_latents
    else:
        t0_4d = torch.tensor(float(t_values[0]), device=ctx.device, dtype=init_latents.dtype).view(1, 1, 1, 1)
        start_latents = (1.0 - t0_4d) * dx0 + t0_4d * init_latents
    root = MCTSNode(step=0, dx=dx0, latents=start_latents)

    np_rng = np.random.default_rng(seed + 1337)
    noise_gen = torch.Generator(device=ctx.device).manual_seed(seed + 2048)
    nfe_total = 0
    best_global_score = -float("inf")
    best_global_actions: list[tuple[int, float]] = []
    best_global_image: Image.Image | None = None

    def _step_forward(
        current_latents: torch.Tensor,
        step_idx: int,
        action: tuple[int, float],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        nonlocal nfe_total
        variant_idx, guidance = action
        t_val = float(t_values[step_idx])
        t_4d = torch.tensor(t_val, device=ctx.device, dtype=current_latents.dtype).view(1, 1, 1, 1)
        explore_n = _mcts_fresh_noise_samples_for_step(args, step_idx, fresh_noise_steps)
        if explore_n <= 1:
            flow = flux_transformer_step(ctx, current_latents, embeds[int(variant_idx)], t_val, float(guidance))
            nfe_total += 1
            new_dx = _pred_x0(current_latents, t_4d, flow, args.x0_sampler)
            next_step = int(step_idx) + 1
            if use_euler:
                dt = _compute_dt(t_values, step_idx)
                new_latents = current_latents + dt * flow
                if next_step >= int(args.steps):
                    return new_dx, new_latents  # keep latents for final decode
                return new_dx, new_latents
            if next_step >= int(args.steps):
                return new_dx, None
            next_t = float(t_values[next_step])
            next_t_4d = torch.tensor(next_t, device=ctx.device, dtype=new_dx.dtype).view(1, 1, 1, 1)
            noise = torch.randn(
                new_dx.shape,
                device=ctx.device,
                dtype=new_dx.dtype,
                generator=noise_gen,
            )
            new_latents = (1.0 - next_t_4d) * new_dx + next_t_4d * noise
            return new_dx, new_latents

        noise_scale = float(getattr(args, "mcts_fresh_noise_scale", 1.0))
        best_score = -float("inf")
        best_dx = None
        best_latents = None
        next_step = int(step_idx) + 1
        for k in range(explore_n):
            if k == 0:
                latents_in = current_latents
            else:
                amp = float(noise_scale) * float(max(0.0, t_val))
                latents_in = current_latents + (
                    amp
                    * torch.randn(
                        current_latents.shape,
                        device=current_latents.device,
                        dtype=current_latents.dtype,
                        generator=noise_gen,
                    )
                )
            flow = flux_transformer_step(ctx, latents_in, embeds[int(variant_idx)], t_val, float(guidance))
            nfe_total += 1
            cand_dx = _pred_x0(latents_in, t_4d, flow, args.x0_sampler)
            if use_euler:
                dt = _compute_dt(t_values, step_idx)
                cand_latents = latents_in + dt * flow
            else:
                if next_step >= int(args.steps):
                    cand_latents = None
                else:
                    next_t = float(t_values[next_step])
                    next_t_4d = torch.tensor(next_t, device=ctx.device, dtype=cand_dx.dtype).view(1, 1, 1, 1)
                    noise = torch.randn(
                        cand_dx.shape,
                        device=ctx.device,
                        dtype=cand_dx.dtype,
                        generator=noise_gen,
                    )
                    cand_latents = (1.0 - next_t_4d) * cand_dx + next_t_4d * noise
            decode_tensor = _final_decode_tensor(cand_latents if cand_latents is not None else latents_in, cand_dx, use_euler)
            cand_img = decode_to_pil(ctx, decode_tensor)
            cand_score = score_image(reward_model, prompt, cand_img)
            if cand_score > best_score:
                best_score = float(cand_score)
                best_dx = cand_dx
                best_latents = cand_latents
        assert best_dx is not None
        return best_dx, best_latents

    print(f"  mcts: sims={args.n_sims} actions={n_actions} steps={args.steps} c={args.ucb_c}")
    for sim in range(int(args.n_sims)):
        node = root
        tree_path: list[tuple[MCTSNode, tuple[int, float]]] = []

        while not node.is_leaf(int(args.steps)):
            untried = node.untried_actions(actions)
            if untried:
                action = untried[int(np_rng.integers(0, len(untried)))]
                break
            action = node.best_ucb(actions, float(args.ucb_c))
            tree_path.append((node, action))
            node = node.children[action]

        if not node.is_leaf(int(args.steps)):
            if action not in node.children:
                child_dx, child_latents = _step_forward(node.latents, node.step, action)
                node.children[action] = MCTSNode(step=node.step + 1, dx=child_dx, latents=child_latents)
            tree_path.append((node, action))
            node = node.children[action]

        rollout_dx = node.dx
        rollout_latents = node.latents
        rollout_step = int(node.step)
        rollout_actions = [a for _, a in tree_path]
        while rollout_step < int(args.steps):
            rollout_action = actions[int(np_rng.integers(0, n_actions))]
            rollout_actions.append(rollout_action)
            rollout_dx, rollout_latents = _step_forward(rollout_latents, rollout_step, rollout_action)
            rollout_step += 1

        rollout_img = decode_to_pil(ctx, _final_decode_tensor(rollout_latents, rollout_dx, use_euler))
        rollout_score = score_image(reward_model, prompt, rollout_img)
        if rollout_score > best_global_score:
            best_global_score = float(rollout_score)
            best_global_actions = [(int(v), float(g)) for v, g in rollout_actions[: int(args.steps)]]
            best_global_image = rollout_img
        else:
            del rollout_img

        for pnode, paction in tree_path:
            pnode.visits += 1
            pnode.action_visits[paction] = pnode.action_visits.get(paction, 0) + 1
            pnode.action_values[paction] = pnode.action_values.get(paction, 0.0) + float(rollout_score)

        if (sim + 1) % 10 == 0 or sim == 0:
            print(f"    sim {sim + 1:3d}/{args.n_sims} best={best_global_score:.4f}")

    exploit_actions: list[tuple[int, float]] = []
    node = root
    for _ in range(int(args.steps)):
        action = node.best_exploit(actions)
        if action is None:
            break
        exploit_actions.append((int(action[0]), float(action[1])))
        if action in node.children:
            node = node.children[action]
        else:
            break
    if len(exploit_actions) < int(args.steps):
        fallback = (0, float(guidance_bank[0]))
        exploit_actions.extend([fallback] * (int(args.steps) - len(exploit_actions)))

    exploit_result = run_action_sequence(
        args=args,
        ctx=ctx,
        reward_model=reward_model,
        prompt=prompt,
        embeds=embeds,
        seed=seed,
        actions=exploit_actions,
    )
    nfe_total += int(args.steps)

    if best_global_image is not None and best_global_score >= float(exploit_result.score):
        diagnostics = {
            "nfe_total": int(nfe_total),
            "n_sims": int(args.n_sims),
            "n_actions": int(n_actions),
            "best_source": "simulation",
        }
        return SearchResult(
            image=best_global_image,
            score=float(best_global_score),
            actions=best_global_actions,
            diagnostics=diagnostics,
        )

    diagnostics = {
        "nfe_total": int(nfe_total),
        "n_sims": int(args.n_sims),
        "n_actions": int(n_actions),
        "best_source": "exploit",
    }
    exploit_result.diagnostics = diagnostics
    return exploit_result


def _ga_step_phase(step_idx: int, steps: int) -> str:
    if steps <= 1:
        return "late"
    third = max(1, steps // 3)
    if step_idx < third:
        return "early"
    if step_idx < 2 * third:
        return "middle"
    return "late"


def _ga_allowed_prompt_indices(
    step_idx: int,
    steps: int,
    prompt_bank: list[tuple[str, str]],
    use_constraints: bool,
) -> list[int]:
    all_ids = list(range(len(prompt_bank)))
    if not use_constraints:
        return all_ids

    label_to_idx = {label: idx for idx, (label, _) in enumerate(prompt_bank)}
    phase = _ga_step_phase(step_idx, steps)
    if phase == "early":
        preferred = ["balanced", "background", "subject"]
    elif phase == "middle":
        preferred = ["subject", "prop", "background", "balanced"]
    else:
        preferred = ["detail", "balanced", "prop", "subject"]
    out = [label_to_idx[label] for label in preferred if label in label_to_idx]
    return out if out else all_ids


def _ga_random_genome(
    rng: np.random.Generator,
    steps: int,
    prompt_bank: list[tuple[str, str]],
    guidance_bank: list[float],
    use_constraints: bool,
) -> list[int]:
    genome: list[int] = []
    for step in range(steps):
        allowed = _ga_allowed_prompt_indices(step, steps, prompt_bank, use_constraints)
        p = int(allowed[int(rng.integers(0, len(allowed)))])
        g = int(rng.integers(0, len(guidance_bank)))
        genome.extend([p, g])
    return genome


def _ga_decode_genome(
    genome: list[int],
    args: argparse.Namespace,
    prompt_bank: list[tuple[str, str]],
    guidance_bank: list[float],
) -> tuple[list[int], list[tuple[int, float]]]:
    if len(genome) != 2 * int(args.steps):
        raise ValueError(f"Invalid genome length: {len(genome)}")
    repaired = list(genome)
    actions: list[tuple[int, float]] = []
    for step in range(int(args.steps)):
        p_gene = int(repaired[2 * step])
        g_gene = int(repaired[2 * step + 1])

        allowed = _ga_allowed_prompt_indices(step, int(args.steps), prompt_bank, args.ga_phase_constraints)
        if not allowed:
            allowed = list(range(len(prompt_bank)))
        allowed_set = set(allowed)
        if p_gene in allowed_set:
            p_idx = p_gene
        else:
            p_idx = int(allowed[p_gene % len(allowed)])
        g_idx = int(g_gene % len(guidance_bank))
        repaired[2 * step] = p_idx
        repaired[2 * step + 1] = g_idx
        actions.append((p_idx, float(guidance_bank[g_idx])))
    return repaired, actions


def _ga_actions_to_genome(
    actions: list[tuple[int, float]],
    guidance_bank: list[float],
) -> list[int]:
    genome: list[int] = []
    for vi, g in actions:
        g_idx = min(range(len(guidance_bank)), key=lambda i: abs(float(guidance_bank[i]) - float(g)))
        genome.extend([int(vi), int(g_idx)])
    return genome


def _ga_step_phase3(step_idx: int, steps: int) -> str:
    if steps <= 1:
        return "late"
    pos = float(step_idx) / float(max(1, steps - 1))
    if pos < 0.34:
        return "early"
    if pos < 0.67:
        return "mid"
    return "late"


def _normalize_probs(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    arr = np.clip(arr, 0.0, None)
    s = float(arr.sum())
    if s <= 0.0 or not np.isfinite(s):
        return np.full_like(arr, 1.0 / float(arr.size))
    return arr / s


def _ga_prior_prompt_probs(
    step_idx: int,
    steps: int,
    prompt_bank: list[tuple[str, str]],
    allowed: list[int],
    prior_strength: float,
) -> np.ndarray:
    phase = _ga_step_phase3(step_idx, steps)
    phase_weights: dict[str, dict[str, float]] = {
        "early": {"balanced": 0.45, "background": 0.30, "subject": 0.20, "prop": 0.03, "detail": 0.02},
        "mid": {"balanced": 0.25, "subject": 0.28, "prop": 0.25, "background": 0.17, "detail": 0.05},
        "late": {"detail": 0.35, "balanced": 0.25, "prop": 0.20, "subject": 0.15, "background": 0.05},
    }
    table = phase_weights.get(phase, phase_weights["mid"])
    raw = []
    for idx in allowed:
        label = str(prompt_bank[idx][0])
        raw.append(float(table.get(label, 0.05)))
    probs = _normalize_probs(raw)
    sharp = max(0.0, float(prior_strength))
    if sharp > 0.0 and probs.size > 0:
        probs = _normalize_probs(list(np.power(probs, sharp)))
    return probs


def _ga_prior_guidance_probs(
    step_idx: int,
    steps: int,
    guidance_bank: list[float],
    prior_strength: float,
    center_value: float,
) -> np.ndarray:
    if len(guidance_bank) == 0:
        return np.asarray([], dtype=np.float64)
    lo = float(min(guidance_bank))
    hi = float(max(guidance_bank))
    phase = _ga_step_phase3(step_idx, steps)
    if phase == "early":
        center = float(center_value)
    elif phase == "mid":
        center = float(center_value) + 0.15
    else:
        center = float(center_value) + 0.25
    center = max(lo, min(hi, center))
    spread = max(0.08, (hi - lo) / 3.0)
    raw = [math.exp(-0.5 * ((float(cfg) - center) / spread) ** 2) for cfg in guidance_bank]
    probs = _normalize_probs(raw)
    sharp = max(0.0, float(prior_strength))
    if sharp > 0.0 and probs.size > 0:
        probs = _normalize_probs(list(np.power(probs, sharp)))
    return probs


def _ga_prior_genome(
    rng: np.random.Generator,
    args: argparse.Namespace,
    steps: int,
    prompt_bank: list[tuple[str, str]],
    guidance_bank: list[float],
    use_constraints: bool,
) -> list[int]:
    genome: list[int] = []
    for step in range(steps):
        allowed = _ga_allowed_prompt_indices(step, steps, prompt_bank, use_constraints)
        if not allowed:
            allowed = list(range(len(prompt_bank)))
        p_probs = _ga_prior_prompt_probs(
            step_idx=step,
            steps=steps,
            prompt_bank=prompt_bank,
            allowed=allowed,
            prior_strength=float(args.ga_prior_strength),
        )
        g_probs = _ga_prior_guidance_probs(
            step_idx=step,
            steps=steps,
            guidance_bank=guidance_bank,
            prior_strength=float(args.ga_prior_strength),
            center_value=float(args.ga_prior_cfg_center),
        )
        p_gene = int(rng.choice(np.asarray(allowed, dtype=np.int64), p=p_probs))
        g_gene = int(rng.choice(np.arange(len(guidance_bank), dtype=np.int64), p=g_probs))
        genome.extend([p_gene, g_gene])
    return genome


def _ga_mutate(
    genome: list[int],
    rng: np.random.Generator,
    args: argparse.Namespace,
    prompt_bank: list[tuple[str, str]],
    guidance_bank: list[float],
) -> list[int]:
    out = list(genome)
    for step in range(int(args.steps)):
        p_pos = 2 * step
        g_pos = p_pos + 1
        if rng.random() < float(args.ga_mutation_prob):
            allowed = _ga_allowed_prompt_indices(step, int(args.steps), prompt_bank, args.ga_phase_constraints)
            out[p_pos] = int(allowed[int(rng.integers(0, len(allowed)))]) if allowed else int(
                rng.integers(0, len(prompt_bank))
            )
        if rng.random() < float(args.ga_mutation_prob):
            out[g_pos] = int(rng.integers(0, len(guidance_bank)))
    return out


def _ga_crossover(
    a: list[int],
    b: list[int],
    rng: np.random.Generator,
    mode: str,
) -> tuple[list[int], list[int]]:
    n = len(a)
    if n != len(b):
        raise ValueError("Genome lengths differ for crossover.")
    if n < 2:
        return list(a), list(b)
    if mode == "one_point":
        point = int(rng.integers(1, n))
        return list(a[:point] + b[point:]), list(b[:point] + a[point:])
    c1: list[int] = []
    c2: list[int] = []
    for ga, gb in zip(a, b):
        if rng.random() < 0.5:
            c1.append(int(ga))
            c2.append(int(gb))
        else:
            c1.append(int(gb))
            c2.append(int(ga))
    return c1, c2


def _ga_tournament_select(
    scored: list[dict[str, Any]],
    rng: np.random.Generator,
    k: int,
) -> list[int]:
    if not scored:
        raise RuntimeError("Empty population in tournament selection.")
    picks = [scored[int(rng.integers(0, len(scored)))] for _ in range(max(1, int(k)))]
    best = max(picks, key=lambda row: float(row["score"]))
    return list(best["genome"])


def _ga_rank_select(
    scored: list[dict[str, Any]],
    rng: np.random.Generator,
    rank_pressure: float,
) -> list[int]:
    if not scored:
        raise RuntimeError("Rank selection received empty population.")
    n = len(scored)
    if n == 1:
        return list(scored[0]["genome"])
    s = float(max(1.0, min(2.0, rank_pressure)))
    probs = np.empty(n, dtype=np.float64)
    for idx_desc in range(n):
        rank_worst_first = n - 1 - idx_desc
        probs[idx_desc] = ((2.0 - s) / n) + (2.0 * rank_worst_first * (s - 1.0) / (n * (n - 1)))
    probs = probs / probs.sum()
    chosen = int(rng.choice(np.arange(n, dtype=np.int64), p=probs))
    return list(scored[chosen]["genome"])


@torch.no_grad()
def run_ga(
    args: argparse.Namespace,
    ctx: FluxContext,
    reward_model: Any,
    prompt: str,
    embeds: list[PromptEmbed],
    prompt_bank: list[tuple[str, str]],
    guidance_bank: list[float],
    seed: int,
) -> SearchResult:
    pop_size = max(4, int(args.ga_population))
    elites = min(max(1, int(args.ga_elites)), pop_size)
    rng = np.random.default_rng(seed + 9103)

    baseline_actions = [(0, 1.0) for _ in range(int(args.steps))]
    baseline_genome = _ga_actions_to_genome(baseline_actions, guidance_bank)
    population: list[list[int]] = []
    seen_genomes: set[tuple[int, ...]] = set()

    def _try_add_unique(genome: list[int]) -> bool:
        key = tuple(int(x) for x in genome)
        if key in seen_genomes:
            return False
        seen_genomes.add(key)
        population.append([int(x) for x in genome])
        return True

    _try_add_unique(list(baseline_genome))

    slots = max(0, pop_size - len(population))
    mode = str(args.ga_init_mode)
    frac = max(0.0, min(1.0, float(args.ga_bayes_init_frac)))
    if mode == "random":
        prior_target = 0
    elif mode == "bayes":
        prior_target = slots
    else:
        prior_target = int(round(float(slots) * frac))
    prior_added = 0
    random_added = 0

    prior_attempts = 0
    prior_attempt_limit = max(20, 20 * max(1, prior_target))
    while len(population) < pop_size and prior_added < prior_target and prior_attempts < prior_attempt_limit:
        g = _ga_prior_genome(rng, args, int(args.steps), prompt_bank, guidance_bank, args.ga_phase_constraints)
        prior_attempts += 1
        if _try_add_unique(g):
            prior_added += 1

    random_target = max(0, pop_size - len(population))
    random_attempts = 0
    random_attempt_limit = max(20, 20 * max(1, random_target))
    while len(population) < pop_size and random_attempts < random_attempt_limit:
        g = _ga_random_genome(rng, int(args.steps), prompt_bank, guidance_bank, args.ga_phase_constraints)
        random_attempts += 1
        if _try_add_unique(g):
            random_added += 1

    while len(population) < pop_size:
        if mode != "random" and prior_added < prior_target:
            g = _ga_prior_genome(rng, args, int(args.steps), prompt_bank, guidance_bank, args.ga_phase_constraints)
            prior_added += 1
        else:
            g = _ga_random_genome(rng, int(args.steps), prompt_bank, guidance_bank, args.ga_phase_constraints)
            random_added += 1
        population.append([int(x) for x in g])

    init_stats = {
        "mode": mode,
        "population": int(pop_size),
        "anchor_added": 1,
        "prior_target": int(prior_target),
        "prior_added": int(prior_added),
        "random_added": int(random_added),
        "unique_after_init": int(len({tuple(g) for g in population})),
        "prior_attempts": int(prior_attempts),
        "random_attempts": int(random_attempts),
    }
    print(
        f"    [ga:init] mode={mode} pop={pop_size} "
        f"anchor=1 prior={prior_added}/{prior_target} random={random_added} "
        f"unique={init_stats['unique_after_init']}"
    )
    cache: dict[tuple[int, ...], dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    best_global: dict[str, Any] | None = None
    eval_calls = 0
    cache_hits = 0
    cache_misses = 0
    prev_eval_calls_total = 0

    def _eval(genome: list[int], need_image: bool = False) -> dict[str, Any]:
        nonlocal eval_calls, cache_hits, cache_misses
        eval_calls += 1
        repaired, actions = _ga_decode_genome(genome, args, prompt_bank, guidance_bank)
        key = tuple(repaired)
        cached = cache.get(key)
        if cached is not None and (not need_image or cached.get("image") is not None):
            cache_hits += 1
            return cached
        cache_misses += 1
        result = run_action_sequence(
            args=args,
            ctx=ctx,
            reward_model=reward_model,
            prompt=prompt,
            embeds=embeds,
            seed=seed,
            actions=actions,
        )
        row = {
            "genome": repaired,
            "actions": actions,
            "score": float(result.score),
            "image": result.image if need_image else None,
        }
        cache[key] = row
        return row

    for gen in range(int(args.ga_generations)):
        scored = [_eval(g, need_image=False) for g in population]
        scored.sort(key=lambda row: float(row["score"]), reverse=True)
        best = scored[0]
        if best_global is None or float(best["score"]) > float(best_global["score"]):
            best_global = _eval(best["genome"], need_image=True)

        scores = [float(row["score"]) for row in scored]
        top_payload = []
        for rank, row in enumerate(scored[: max(1, int(args.ga_log_topk))]):
            top_payload.append(
                {
                    "rank": rank,
                    "score": float(row["score"]),
                    "genome": [int(x) for x in row["genome"]],
                    "actions": [[int(v), float(g)] for v, g in row["actions"]],
                }
            )
        eval_calls_total = int(eval_calls)
        eval_calls_generation = int(eval_calls_total - prev_eval_calls_total)
        prev_eval_calls_total = eval_calls_total
        nfe_per_generation = int(eval_calls_generation * int(args.steps))
        nfe_total = int(eval_calls_total * int(args.steps))
        cache_hit_rate = float(cache_hits) / float(max(1, eval_calls_total))
        history.append(
            {
                "generation": gen,
                "best": float(scores[0]),
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "worst": float(scores[-1]),
                "eval_calls_total": eval_calls_total,
                "eval_calls_generation": eval_calls_generation,
                "nfe_per_generation": nfe_per_generation,
                "nfe_total": nfe_total,
                "cache_entries": int(len(cache)),
                "cache_hits_total": int(cache_hits),
                "cache_misses_total": int(cache_misses),
                "cache_hit_rate": float(cache_hit_rate),
                "top": top_payload,
            }
        )
        print(
            f"    [ga] gen={gen + 1:02d}/{args.ga_generations} "
            f"best={scores[0]:.4f} mean={float(np.mean(scores)):.4f}"
        )

        if gen == int(args.ga_generations) - 1:
            break

        next_pop: list[list[int]] = [list(row["genome"]) for row in scored[:elites]]
        use_rank_selection = str(args.ga_selection).lower() == "rank"
        while len(next_pop) < pop_size:
            if use_rank_selection:
                p1 = _ga_rank_select(scored, rng, float(args.ga_rank_pressure))
                p2 = _ga_rank_select(scored, rng, float(args.ga_rank_pressure))
            else:
                p1 = _ga_tournament_select(scored, rng, int(args.ga_tournament_k))
                p2 = _ga_tournament_select(scored, rng, int(args.ga_tournament_k))
            c1, c2 = _ga_crossover(p1, p2, rng, args.ga_crossover)
            next_pop.append(_ga_mutate(c1, rng, args, prompt_bank, guidance_bank))
            if len(next_pop) < pop_size:
                next_pop.append(_ga_mutate(c2, rng, args, prompt_bank, guidance_bank))
        population = next_pop

    assert best_global is not None
    return SearchResult(
        image=best_global["image"],
        score=float(best_global["score"]),
        actions=[(int(v), float(g)) for v, g in best_global["actions"]],
        diagnostics={
            "history": history,
            "best_genome": [int(x) for x in best_global["genome"]],
            "baseline_genome": [int(x) for x in baseline_genome],
            "initialization": init_stats,
            "selection": {
                "mode": str(args.ga_selection),
                "tournament_k": int(args.ga_tournament_k),
                "rank_pressure": float(args.ga_rank_pressure),
            },
            "cache_stats": {
                "eval_calls_total": int(eval_calls),
                "cache_entries": int(len(cache)),
                "cache_hits_total": int(cache_hits),
                "cache_misses_total": int(cache_misses),
                "cache_hit_rate": float(cache_hits) / float(max(1, int(eval_calls))),
                "nfe_total": int(eval_calls * int(args.steps)),
            },
            "guidance_bank": [float(v) for v in guidance_bank],
            "prompt_bank": [{"label": lbl, "text": txt} for lbl, txt in prompt_bank],
        },
    )


def systematic_resample(weights: torch.Tensor) -> torch.Tensor:
    k = int(weights.shape[0])
    cdf = torch.cumsum(weights, dim=0)
    u = (torch.rand(1, device=weights.device) + torch.arange(k, device=weights.device)) / float(k)
    return torch.searchsorted(cdf, u).clamp(0, k - 1)


@torch.no_grad()
def score_dx_batch(
    ctx: FluxContext,
    reward_model: Any,
    prompt: str,
    dx: torch.Tensor,
) -> torch.Tensor:
    scores = []
    for i in range(int(dx.shape[0])):
        img = decode_to_pil(ctx, dx[i : i + 1])
        scores.append(float(score_image(reward_model, prompt, img)))
        del img
    return torch.tensor(scores, device=dx.device, dtype=torch.float32)


@torch.no_grad()
def _build_smc_expansion_bank_flux(
    args: argparse.Namespace,
    n_variants: int,
) -> list[tuple[int, float]]:
    n_variants = max(1, int(n_variants))
    variants_arg = list(getattr(args, "smc_expansion_variants", []) or [])
    guidances_arg = list(getattr(args, "smc_expansion_guidances", []) or [])
    if len(variants_arg) == 0:
        variants_arg = list(range(n_variants))

    valid_variants: list[int] = []
    dropped_variants: list[int] = []
    seen_variants: set[int] = set()
    for vi_raw in variants_arg:
        vi = int(vi_raw)
        if 0 <= vi < n_variants:
            if vi not in seen_variants:
                seen_variants.add(vi)
                valid_variants.append(vi)
        else:
            dropped_variants.append(vi)
    if len(valid_variants) == 0:
        valid_variants = list(range(n_variants))
    if len(dropped_variants) > 0:
        kept_str = ",".join(str(v) for v in valid_variants)
        dropped_unique = sorted(set(int(v) for v in dropped_variants))
        dropped_str = ",".join(str(v) for v in dropped_unique[:8])
        more = "..." if len(dropped_unique) > 8 else ""
        print(
            "[warn] smc_expansion_variants contains out-of-range indices "
            f"for prompt variants (n={n_variants}); dropped=[{dropped_str}{more}] kept=[{kept_str}]"
        )

    if len(guidances_arg) == 0:
        guidances_arg = [float(getattr(args, "smc_guidance_scale", 1.25))]
    bank: list[tuple[int, float]] = []
    seen: set[tuple[int, float]] = set()
    for vi in valid_variants:
        for g in guidances_arg:
            key = (int(vi), float(round(float(g), 6)))
            if key in seen:
                continue
            seen.add(key)
            bank.append(key)
    if len(bank) <= 0:
        bank = [(0, float(getattr(args, "smc_guidance_scale", 1.25)))]
    return bank


def _stratified_assign_actions_flux(
    bank: list[tuple[int, float]],
    k: int,
    rng: np.random.Generator,
) -> list[tuple[int, float]]:
    if k <= 0 or len(bank) <= 0:
        return []
    reps = (k + len(bank) - 1) // len(bank)
    pool = [bank[i % len(bank)] for i in range(reps * len(bank))]
    order = rng.permutation(len(pool))[:k]
    return [pool[int(i)] for i in order]


def run_smc(
    args: argparse.Namespace,
    ctx: FluxContext,
    reward_model: Any,
    prompt: str,
    balanced_embed: PromptEmbed,
    seed: int,
    embeds: list[PromptEmbed] | None = None,
) -> SearchResult:
    k = max(2, int(args.smc_k))
    use_euler = getattr(args, "euler_sampler", False)
    use_expansion = bool(getattr(args, "smc_variant_expansion", False))
    embeds_list: list[PromptEmbed] = list(embeds) if embeds else [balanced_embed]
    if len(embeds_list) == 0:
        embeds_list = [balanced_embed]
    bank = _build_smc_expansion_bank_flux(args, n_variants=len(embeds_list))
    np_rng = np.random.default_rng(int(seed) + 7007)

    default_guidance = float(args.smc_guidance_scale)
    if use_expansion:
        particle_actions = _stratified_assign_actions_flux(bank, k, np_rng)
    else:
        particle_actions = [(0, default_guidance) for _ in range(k)]

    init_latents = make_initial_latents(ctx, seed, args.height, args.width, batch_size=k)
    dx = torch.zeros_like(init_latents)
    latents = init_latents.clone()
    rng = torch.Generator(device=ctx.device).manual_seed(seed + 6001)
    t_values = build_t_schedule(int(args.steps), getattr(args, "sigmas", None))
    start_idx = int((1.0 - float(args.resample_start_frac)) * int(args.steps))
    log_w = torch.zeros(k, device=ctx.device, dtype=torch.float32)
    prev_step_scores = torch.zeros(k, device=ctx.device, dtype=torch.float32)
    ess_hist: list[float] = []
    reward_hist: list[float] = []
    resample_count = 0
    expansion_events = 0

    expansion_factor_arg = int(getattr(args, "smc_expansion_factor", -1))
    expansion_factor = len(bank) if expansion_factor_arg <= 0 else min(len(bank), int(expansion_factor_arg))
    proposal_mode = str(getattr(args, "smc_expansion_proposal", "uniform"))
    proposal_tau = max(1e-6, float(getattr(args, "smc_expansion_tau", 1.0)))
    use_lookahead = bool(getattr(args, "smc_expansion_lookahead", False))
    potential = str(getattr(args, "smc_potential", "tempering"))
    fk_lambda = float(getattr(args, "smc_lambda", 10.0))

    print(
        f"  smc: K={k} expansion={use_expansion} bank_size={len(bank)} M={expansion_factor} "
        f"prop={proposal_mode} lookahead={use_lookahead} potential={potential} "
        f"gamma={float(args.smc_gamma):.3f} lambda={fk_lambda:.3f} "
        f"ess_thr={float(args.ess_threshold):.2f} euler={use_euler}"
    )

    for step_idx, t_val in enumerate(t_values):
        t_4d = torch.tensor(float(t_val), device=ctx.device, dtype=init_latents.dtype).view(1, 1, 1, 1)
        if not use_euler:
            if step_idx == 0:
                noise = init_latents
            else:
                noise = torch.randn(
                    init_latents.shape,
                    device=ctx.device,
                    dtype=init_latents.dtype,
                    generator=rng,
                )
            latents = (1.0 - t_4d) * dx + t_4d * noise

        # Per-particle forward; group by identical action to batch where possible.
        action_groups: dict[tuple[int, float], list[int]] = {}
        for pi, a in enumerate(particle_actions):
            action_groups.setdefault(a, []).append(int(pi))
        new_dx = torch.zeros_like(dx)
        new_latents = latents.clone() if use_euler else latents
        for action, idx_list in action_groups.items():
            sub_latents = latents[idx_list]
            vi, g = int(action[0]), float(action[1])
            emb = embeds_list[int(vi) % len(embeds_list)]
            flow = flux_transformer_step_chunked(
                ctx=ctx,
                latents=sub_latents,
                embed=emb,
                t_val=float(t_val),
                guidance_scale=float(g),
                chunk=int(args.smc_chunk),
            )
            sub_dx = _pred_x0(sub_latents, t_4d, flow, args.x0_sampler)
            for slot, pi in enumerate(idx_list):
                new_dx[pi : pi + 1] = sub_dx[slot : slot + 1]
                if use_euler:
                    dt = _compute_dt(t_values, step_idx)
                    new_latents[pi : pi + 1] = sub_latents[slot : slot + 1] + dt * flow[slot : slot + 1]
        dx = new_dx
        latents = new_latents

        if step_idx < start_idx:
            continue

        step_scores = score_dx_batch(ctx, reward_model, prompt, _final_decode_tensor(latents, dx, use_euler))
        step_scores_list = [float(v) for v in step_scores.detach().cpu().tolist()]
        reward_hist.append(float(step_scores.mean().item()))
        if potential == "diff":
            # FK-steering difference potential: g_t = exp(lam * (r̂_t - r̂_{t-1}));
            # log-weight bump telescopes to lam * r̂_final across steps.
            log_w = log_w + fk_lambda * (step_scores - prev_step_scores)
            lam = fk_lambda  # used by lookahead branch below
        else:
            lam = (1.0 + float(args.smc_gamma)) ** (int(args.steps) - 1 - step_idx) - 1.0
            log_w = log_w + float(lam) * step_scores
        prev_step_scores = step_scores.clone()
        w = torch.softmax(log_w, dim=0)
        ess = float(1.0 / torch.sum(w * w).item())
        ess_hist.append(ess)

        if ess < float(args.ess_threshold) * float(k):
            if not use_expansion or len(bank) <= 1:
                idx = systematic_resample(w)
                dx = dx[idx].clone()
                if use_euler:
                    latents = latents[idx].clone()
                particle_actions = [particle_actions[int(i)] for i in idx.tolist()]
                prev_step_scores = prev_step_scores[idx].clone()
                log_w = torch.zeros_like(log_w)
                resample_count += 1
                continue

            children_lat: list[torch.Tensor] = []
            children_dx: list[torch.Tensor] = []
            children_actions: list[tuple[int, float]] = []
            children_logw: list[float] = []
            children_prev: list[float] = []
            for pi in range(k):
                if proposal_mode == "score_softmax" and len(bank) > 1:
                    base = float(step_scores_list[pi])
                    logits = np.full((len(bank),), base, dtype=np.float64)
                    probs = np.exp((logits - float(np.max(logits))) / proposal_tau)
                    probs = probs / max(1e-12, float(np.sum(probs)))
                    idxs = np_rng.choice(len(bank), size=expansion_factor, replace=expansion_factor > len(bank), p=probs)
                else:
                    idxs = np_rng.choice(len(bank), size=expansion_factor, replace=expansion_factor > len(bank))
                for bi in idxs.tolist():
                    children_actions.append(bank[int(bi)])
                    children_lat.append(latents[pi : pi + 1])
                    children_dx.append(dx[pi : pi + 1])
                    children_logw.append(float(log_w[pi].item()))
                    children_prev.append(float(prev_step_scores[pi].item()))

            child_logw = torch.tensor(children_logw, device=dx.device, dtype=torch.float32)
            children_prev_tensor = torch.tensor(children_prev, device=dx.device, dtype=torch.float32)
            la_t: torch.Tensor | None = None
            if use_lookahead and step_idx + 1 < len(t_values):
                next_t_val = float(t_values[step_idx + 1])
                next_t_4d = torch.tensor(next_t_val, device=ctx.device, dtype=init_latents.dtype).view(1, 1, 1, 1)
                la_scores: list[float] = []
                for ci, action in enumerate(children_actions):
                    parent_dx = children_dx[ci]
                    parent_lat = children_lat[ci]
                    if use_euler:
                        la_in = parent_lat
                    else:
                        noise_la = torch.randn(parent_dx.shape, device=parent_dx.device, dtype=parent_dx.dtype, generator=rng)
                        la_in = (1.0 - next_t_4d) * parent_dx + next_t_4d * noise_la
                    vi, g = int(action[0]), float(action[1])
                    emb = embeds_list[int(vi) % len(embeds_list)]
                    la_flow = flux_transformer_step_chunked(
                        ctx=ctx, latents=la_in, embed=emb,
                        t_val=next_t_val, guidance_scale=float(g), chunk=int(args.smc_chunk),
                    )
                    la_dx = _pred_x0(la_in, next_t_4d, la_flow, args.x0_sampler)
                    if use_euler:
                        dt_la = _compute_dt(t_values, step_idx + 1)
                        la_out = la_in + dt_la * la_flow
                    else:
                        la_out = la_in
                    la_tensor = _final_decode_tensor(la_out, la_dx, use_euler)
                    la_score = score_dx_batch(ctx, reward_model, prompt, la_tensor)
                    la_scores.append(float(la_score[0].item()))
                la_t = torch.tensor(la_scores, device=dx.device, dtype=torch.float32)
                if potential == "diff":
                    child_logw = child_logw + fk_lambda * (la_t - children_prev_tensor)
                else:
                    child_logw = child_logw + float(lam) * la_t

            child_weights = torch.softmax(child_logw, dim=0)
            idx = systematic_resample(child_weights)
            chosen = idx.tolist()
            latents = torch.cat([children_lat[int(i)] for i in chosen], dim=0)
            dx = torch.cat([children_dx[int(i)] for i in chosen], dim=0)
            particle_actions = [children_actions[int(i)] for i in chosen]
            chosen_idx_t = torch.tensor(chosen, device=dx.device, dtype=torch.long)
            if la_t is not None:
                prev_step_scores = la_t.index_select(0, chosen_idx_t).clone()
            else:
                prev_step_scores = children_prev_tensor.index_select(0, chosen_idx_t).clone()
            log_w = torch.zeros_like(log_w)
            resample_count += 1
            expansion_events += 1

    final_decode = _final_decode_tensor(latents, dx, use_euler)
    final_scores = score_dx_batch(ctx, reward_model, prompt, final_decode)
    best_idx = int(torch.argmax(final_scores).item())
    best_img = decode_to_pil(ctx, final_decode[best_idx : best_idx + 1])
    best_action = particle_actions[best_idx]
    if potential == "diff":
        smc_style = "fk_diff_variant_expansion" if use_expansion else "fk_diff"
    else:
        smc_style = "das_tempered_resampling_variant_expansion" if use_expansion else "das_tempered_resampling"
    diagnostics = {
        "smc_style": smc_style,
        "smc_potential": potential,
        "smc_lambda": float(fk_lambda),
        "smc_k": int(k),
        "gamma": float(args.smc_gamma),
        "guidance_scale": float(default_guidance),
        "smc_variant_expansion": bool(use_expansion),
        "smc_expansion_bank": [list(a) for a in bank],
        "smc_expansion_bank_size": int(len(bank)),
        "smc_expansion_factor": int(expansion_factor),
        "smc_expansion_proposal": str(proposal_mode),
        "smc_expansion_lookahead": bool(use_lookahead),
        "smc_expansion_events": int(expansion_events),
        "resample_count": int(resample_count),
        "ess_min": float(min(ess_hist)) if ess_hist else 0.0,
        "ess_mean": float(sum(ess_hist) / len(ess_hist)) if ess_hist else 0.0,
        "reward_traj_mean": reward_hist,
        "reward_all": [float(v) for v in final_scores.detach().cpu().tolist()],
        "final_particle_actions": [list(a) for a in particle_actions],
        "best_action": list(best_action),
    }
    return SearchResult(
        image=best_img,
        score=float(final_scores[best_idx].item()),
        actions=[(int(best_action[0]), float(best_action[1])) for _ in range(int(args.steps))],
        diagnostics=diagnostics,
    )


def _font(size: int = 15):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def save_comparison(
    path: str,
    baseline: Image.Image,
    search: Image.Image,
    baseline_score: float,
    search_score: float,
    actions: list[tuple[int, float]],
) -> None:
    w, h = baseline.size
    hdr = 52
    canvas = Image.new("RGB", (w * 2, h + hdr), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    canvas.paste(baseline, (0, hdr))
    canvas.paste(search, (w, hdr))
    draw.text((6, 6), f"baseline IR={baseline_score:.4f}", fill=(210, 210, 210), font=_font(15))
    delta = search_score - baseline_score
    col = (100, 255, 100) if delta >= 0 else (255, 120, 120)
    draw.text((w + 6, 6), f"search IR={search_score:.4f}  delta={delta:+.4f}", fill=col, font=_font(15))
    action_text = " ".join(f"v{v}/g{g:.2f}" for v, g in actions[:8])
    draw.text((w + 6, 28), action_text, fill=(255, 215, 80), font=_font(11))
    canvas.save(path)


def print_aggregate(summary: list[dict[str, Any]], search_method: str) -> None:
    rows = []
    for entry in summary:
        for sample in entry["samples"]:
            rows.append(sample)
    if not rows:
        return
    b = [float(x["baseline_score"]) for x in rows]
    s = [float(x["search_score"]) for x in rows]
    d = [float(x["delta_score"]) for x in rows]
    print("\nAggregate:")
    print(f"  baseline mean={sum(b)/len(b):.4f}")
    print(f"  {search_method} mean={sum(s)/len(s):.4f}")
    print(f"  delta mean={sum(d)/len(d):+.4f}")


def run(args: argparse.Namespace) -> None:
    if args.cuda_alloc_conf and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf
        print(f"Set PYTORCH_CUDA_ALLOC_CONF={args.cuda_alloc_conf}")

    os.makedirs(args.out_dir, exist_ok=True)
    prompts = load_prompts(args)
    if not prompts:
        raise RuntimeError("No prompts loaded.")

    device = pick_device(args)
    dtype = resolve_dtype(args.dtype)

    if device.startswith("cuda"):
        free_gb = cuda_free_gb(device)
        total_gb = None
        try:
            idx = int(device.split(":", 1)[1])
            _, total = torch.cuda.mem_get_info(idx)
            total_gb = float(total) / (1024 ** 3)
        except Exception:
            pass
        if free_gb is not None and total_gb is not None:
            print(f"Visible GPU memory: {device} free={free_gb:.2f}GB total={total_gb:.2f}GB")

    ctx = load_pipeline(args, device=device, dtype=dtype)
    reward_model, reward_device = load_reward(args, pipeline_device=device)
    print(f"Loaded. device={device} dtype={args.dtype} reward_device={reward_device} decode_device={ctx.decode_device}")

    summary: list[dict[str, Any]] = []

    for p_idx, prompt in enumerate(prompts):
        slug = f"p{p_idx:04d}"
        save_entry_artifacts = args.save_first_k < 0 or p_idx < int(args.save_first_k)
        print(f"\n{'=' * 72}\n[{slug}] {prompt}\n{'=' * 72}")

        prompt_bank = select_prompt_bank(prompt, int(args.n_variants))
        embeds = encode_prompt_bank(args, ctx, prompt_bank)
        if save_entry_artifacts:
            with open(os.path.join(args.out_dir, f"{slug}_variants.txt"), "w", encoding="utf-8") as f:
                for i, (label, text) in enumerate(prompt_bank):
                    f.write(f"v{i}[{label}]: {text}\n")

        prompt_samples: list[dict[str, Any]] = []
        for sample_i in range(int(args.n_samples)):
            seed = int(args.seed) + sample_i
            print(f"  sample {sample_i + 1}/{args.n_samples} seed={seed}")

            baseline_actions = [(0, float(args.baseline_guidance_scale)) for _ in range(int(args.steps))]
            baseline = run_action_sequence(
                args=args,
                ctx=ctx,
                reward_model=reward_model,
                prompt=prompt,
                embeds=embeds,
                seed=seed,
                actions=baseline_actions,
            )

            if args.search_method == "ga":
                search = run_ga(
                    args=args,
                    ctx=ctx,
                    reward_model=reward_model,
                    prompt=prompt,
                    embeds=embeds,
                    prompt_bank=prompt_bank,
                    guidance_bank=[float(v) for v in args.ga_guidance_scales],
                    seed=seed,
                )
            elif args.search_method == "greedy":
                search = run_greedy(
                    args=args,
                    ctx=ctx,
                    reward_model=reward_model,
                    prompt=prompt,
                    embeds=embeds,
                    guidance_bank=guidance_bank_for_search(args),
                    seed=seed,
                )
            elif args.search_method == "mcts":
                search = run_mcts(
                    args=args,
                    ctx=ctx,
                    reward_model=reward_model,
                    prompt=prompt,
                    embeds=embeds,
                    guidance_bank=guidance_bank_for_search(args),
                    seed=seed,
                )
            elif args.search_method == "bon":
                search = run_bon(
                    args=args,
                    ctx=ctx,
                    reward_model=reward_model,
                    prompt=prompt,
                    embeds=embeds,
                    seed=seed,
                )
            elif args.search_method == "beam":
                search = run_beam(
                    args=args,
                    ctx=ctx,
                    reward_model=reward_model,
                    prompt=prompt,
                    embeds=embeds,
                    guidance_bank=guidance_bank_for_search(args),
                    seed=seed,
                )
            elif args.search_method == "noise_inject":
                search = run_noise_inject(
                    args=args,
                    ctx=ctx,
                    reward_model=reward_model,
                    prompt=prompt,
                    embeds=embeds,
                    seed=seed,
                )
            else:
                search = run_smc(
                    args=args,
                    ctx=ctx,
                    reward_model=reward_model,
                    prompt=prompt,
                    balanced_embed=embeds[0],
                    seed=seed,
                    embeds=embeds,
                )

            print(
                f"    baseline={baseline.score:.4f} {args.search_method}={search.score:.4f} "
                f"delta={search.score - baseline.score:+.4f}"
            )

            if save_entry_artifacts:
                base_path = os.path.join(args.out_dir, f"{slug}_s{sample_i}_baseline.png")
                search_path = os.path.join(args.out_dir, f"{slug}_s{sample_i}_{args.search_method}.png")
                comp_path = os.path.join(args.out_dir, f"{slug}_s{sample_i}_comparison.png")
                baseline.image.save(base_path)
                search.image.save(search_path)
                save_comparison(
                    path=comp_path,
                    baseline=baseline.image,
                    search=search.image,
                    baseline_score=baseline.score,
                    search_score=search.score,
                    actions=search.actions,
                )
                if search.diagnostics is not None:
                    diag_path = os.path.join(args.out_dir, f"{slug}_s{sample_i}_{args.search_method}_diag.json")
                    with open(diag_path, "w", encoding="utf-8") as f:
                        json.dump(search.diagnostics, f, indent=2)

            sample_payload = {
                "seed": seed,
                "baseline_score": float(baseline.score),
                "search_score": float(search.score),
                "delta_score": float(search.score - baseline.score),
                "actions": [[int(v), float(g)] for v, g in search.actions],
                "artifacts_saved": bool(save_entry_artifacts),
            }
            if search.diagnostics is not None:
                sample_payload["diagnostics"] = search.diagnostics
            prompt_samples.append(sample_payload)

            del baseline
            del search
            gc.collect()
            if ctx.device.startswith("cuda"):
                torch.cuda.empty_cache()

        summary.append(
            {
                "slug": slug,
                "prompt": prompt,
                "search_method": args.search_method,
                "samples": prompt_samples,
            }
        )

        del embeds
        gc.collect()
        if ctx.device.startswith("cuda"):
            torch.cuda.empty_cache()

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    rows = []
    for entry in summary:
        for sample in entry["samples"]:
            rows.append(sample)
    baseline_vals = [float(r["baseline_score"]) for r in rows]
    search_vals = [float(r["search_score"]) for r in rows]
    delta_vals = [float(r["delta_score"]) for r in rows]
    aggregate = {
        "model_id": args.model_id,
        "search_method": args.search_method,
        "n_prompts": len(prompts),
        "n_samples": int(args.n_samples),
        "save_first_k": int(args.save_first_k),
        "mean_baseline_score": float(sum(baseline_vals) / len(baseline_vals)) if baseline_vals else None,
        "mean_search_score": float(sum(search_vals) / len(search_vals)) if search_vals else None,
        "mean_delta_score": float(sum(delta_vals) / len(delta_vals)) if delta_vals else None,
    }
    aggregate_path = os.path.join(args.out_dir, "aggregate_summary.json")
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print_aggregate(summary, args.search_method)
    print(f"\nSummary saved:   {summary_path}")
    print(f"Aggregate saved: {aggregate_path}")
    print(f"Outputs:         {os.path.abspath(args.out_dir)}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
