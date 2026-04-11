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
from dataclasses import dataclass
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
    parser.add_argument("--search_method", choices=["greedy", "mcts", "ga", "smc", "bon", "beam"], default="greedy")

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

    parser.add_argument("--n_sims", type=int, default=50)
    parser.add_argument("--ucb_c", type=float, default=1.41)
    parser.add_argument("--bon_n", type=int, default=16,
                        help="Number of independent samples for best-of-N search.")
    parser.add_argument("--beam_width", type=int, default=4,
                        help="Number of beams to keep per step in beam search.")
    parser.add_argument("--smc_k", type=int, default=8)
    parser.add_argument("--smc_gamma", type=float, default=0.10)
    parser.add_argument("--ess_threshold", type=float, default=0.5)
    parser.add_argument("--resample_start_frac", type=float, default=0.3)
    parser.add_argument("--smc_cfg_scale", type=float, default=1.25)
    parser.add_argument("--smc_variant_idx", type=int, default=0)
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
        "sigmas": [1.0, 0.75],
        "dtype": "bfloat16",  # SenseFlow official scripts use bfloat16
        "gen_batch_size": 2,
        "cfg_scales": [0.0],  # SenseFlow release uses guidance_scale=0.0
        "baseline_cfg": 0.0,
        "x0_sampler": True,  # SenseFlow transformer predicts x̂₀ directly, not flow
    },
    "senseflow_medium": {
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "transformer_id": "domiso/SenseFlow",
        "transformer_subfolder": "SenseFlow-SD35M/transformer",
        "sigmas": [1.0, 0.9, 0.75, 0.5],
        "dtype": "bfloat16",
        "gen_batch_size": 2,
        "cfg_scales": [0.0],
        "baseline_cfg": 0.0,
        "x0_sampler": True,  # SenseFlow transformer predicts x̂₀ directly, not flow
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

    # Compat shims: constants/functions removed from transformers in newer versions
    # but still imported by older diffusers/ImageReward code.
    try:
        import transformers.utils as _tu
        if not hasattr(_tu, "FLAX_WEIGHTS_NAME"):
            _tu.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
        import transformers.modeling_utils as _tmu
        if not hasattr(_tmu, "apply_chunking_to_forward"):
            def _apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
                if chunk_size > 0:
                    tensor_shape = input_tensors[0].shape[chunk_dim]
                    if tensor_shape % chunk_size != 0:
                        raise ValueError(
                            f"tensor shape {tensor_shape} not divisible by chunk_size {chunk_size}"
                        )
                    num_chunks = tensor_shape // chunk_size
                    return torch.cat(
                        [forward_fn(*[t.narrow(chunk_dim, c * chunk_size, chunk_size) for t in input_tensors])
                         for c in range(num_chunks)],
                        dim=chunk_dim,
                    )
                return forward_fn(*input_tensors)
            _tmu.apply_chunking_to_forward = _apply_chunking_to_forward
        if not hasattr(_tmu, "find_pruneable_heads_and_indices"):
            def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
                mask = torch.ones(n_heads, head_size)
                heads = set(heads) - already_pruned_heads
                for head in sorted(heads):
                    head -= sum(1 for h in sorted(already_pruned_heads) if h < head)
                    mask[head] = 0
                mask = mask.view(-1).eq(1)
                index = torch.arange(len(mask))[mask].long()
                return heads, index
            _tmu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices
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
        #
        # Use the HF repo ID (e.g. "domiso/SenseFlow") with subfolder — NOT a local
        # snapshot path.  from_pretrained resolves from HF cache in offline mode as long
        # as snapshot_download was called before HF_HUB_OFFLINE=1.
        from diffusers.models.transformers import SD3Transformer2DModel
        tf_kwargs: dict = {"torch_dtype": dtype}
        if transformer_subfolder:
            tf_kwargs["subfolder"] = transformer_subfolder
        print(f"Loading transformer from {transformer_id} subfolder={transformer_subfolder}")
        pretrained_kwargs["transformer"] = SD3Transformer2DModel.from_pretrained(
            transformer_id, **tf_kwargs
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


def run_mcts(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    variants: list[str],
    seed: int,
) -> SearchResult:
    family = getattr(args, "mcts_interp_family", "none")
    n_interp = getattr(args, "mcts_n_interp", 1)
    if family != "none":
        emb = expand_emb_with_interp(emb, family, n_interp)
        print(f"  mcts: interp={family} n_interp={n_interp} total_variants={len(emb.cond_text)}")
    corr_strengths = list(getattr(args, "correction_strengths", [0.0]))
    actions = [
        (vi, cfg, cs)
        for vi in range(len(emb.cond_text))
        for cfg in args.cfg_scales
        for cs in corr_strengths
    ]
    n_actions = len(actions)
    latents0 = make_latents(ctx, seed, args.height, args.width, emb.cond_text[0].dtype)
    dx0 = torch.zeros_like(latents0)
    use_euler = getattr(args, "euler_sampler", False)
    sched = step_schedule(ctx.device, latents0.dtype, args.steps,
                          getattr(args, "sigmas", None), euler=use_euler)
    _, t0_4d, _ = sched[0]
    start_latents = _prepare_latents(latents0, dx0, latents0, t0_4d, 0, use_euler)
    root = MCTSNode(0, dx0, start_latents)

    best_global_score = -float("inf")
    best_global_dx = None
    best_global_path: list[tuple[int, float, float]] = []

    print(f"  mcts: sims={args.n_sims} actions={n_actions} steps={args.steps}")
    for sim in range(args.n_sims):
        node = root
        path: list[tuple[MCTSNode, tuple[int, float, float]]] = []

        while not node.is_leaf(args.steps):
            untried = node.untried_actions(actions)
            if untried:
                action = untried[np.random.randint(len(untried))]
                break
            action = node.best_ucb(actions, args.ucb_c)
            path.append((node, action))
            node = node.children[action]

        if not node.is_leaf(args.steps):
            if action not in node.children:
                child_dx, child_lat = _expand_child(args, ctx, emb, node, action, sched, reward_model, prompt)
                node.children[action] = MCTSNode(node.step + 1, child_dx, child_lat)
            path.append((node, action))
            node = node.children[action]

        rollout_dx = node.dx
        rollout_latents = node.latents
        rollout_step = node.step
        while rollout_step < args.steps:
            variant_idx, cfg, cs = actions[np.random.randint(n_actions)]
            t_flat, t_4d, dt = sched[rollout_step]
            flow = transformer_step(args, ctx, rollout_latents, emb, variant_idx, t_flat, cfg)
            rollout_latents, rollout_dx = _apply_step(
                rollout_latents, flow, rollout_dx, t_4d, dt, use_euler, args.x0_sampler)
            if float(cs) > 0.0:
                rollout_dx = apply_reward_correction(ctx, rollout_dx, prompt, reward_model, float(cs), cfg=float(cfg))
            rollout_step += 1
            if rollout_step < args.steps and not use_euler:
                _, next_t_4d, _ = sched[rollout_step]
                noise = torch.randn_like(rollout_dx)
                rollout_latents = (1.0 - next_t_4d) * rollout_dx + next_t_4d * noise

        rollout_img = decode_to_pil(ctx, _final_decode_tensor(rollout_latents, rollout_dx, use_euler))
        rollout_score = score_image(reward_model, prompt, rollout_img)
        if rollout_score > best_global_score:
            best_global_score = rollout_score
            best_global_dx = _final_decode_tensor(rollout_latents, rollout_dx, use_euler).clone()
            best_global_path = [a for _, a in path]

        for pnode, paction in path:
            pnode.visits += 1
            pnode.action_visits[paction] = pnode.action_visits.get(paction, 0) + 1
            pnode.action_values[paction] = pnode.action_values.get(paction, 0.0) + rollout_score

        if (sim + 1) % 10 == 0 or sim == 0:
            print(f"    sim {sim + 1:3d}/{args.n_sims} best={best_global_score:.4f}")

    exploit_path: list[tuple[int, float, float]] = []
    node = root
    for _ in range(args.steps):
        action = node.best_exploit(actions)
        if action is None:
            break
        exploit_path.append(action)
        if action in node.children:
            node = node.children[action]
        else:
            break

    replay_dx = dx0
    replay_lat = start_latents
    for step_idx, (variant_idx, cfg, cs) in enumerate(exploit_path):
        t_flat, t_4d, dt = sched[step_idx]
        flow = transformer_step(args, ctx, replay_lat, emb, variant_idx, t_flat, cfg)
        replay_lat, replay_dx = _apply_step(replay_lat, flow, replay_dx, t_4d, dt, use_euler, args.x0_sampler)
        if float(cs) > 0.0:
            replay_dx = apply_reward_correction(ctx, replay_dx, prompt, reward_model, float(cs), cfg=float(cfg))
        if step_idx + 1 < args.steps and not use_euler:
            _, next_t_4d, _ = sched[step_idx + 1]
            noise = torch.randn_like(replay_dx)
            replay_lat = (1.0 - next_t_4d) * replay_dx + next_t_4d * noise

    exploit_img = decode_to_pil(ctx, _final_decode_tensor(replay_lat, replay_dx, use_euler))
    exploit_score = score_image(reward_model, prompt, exploit_img)

    if exploit_score >= best_global_score:
        return SearchResult(image=exploit_img, score=exploit_score, actions=exploit_path)
    if best_global_dx is None:
        return SearchResult(image=exploit_img, score=exploit_score, actions=exploit_path)
    best_img = decode_to_pil(ctx, best_global_dx)
    return SearchResult(image=best_img, score=best_global_score, actions=best_global_path)


@torch.no_grad()
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
    cfg = float(args.smc_cfg_scale)
    variant_idx = int(max(0, min(len(emb.cond_text) - 1, int(args.smc_variant_idx))))
    corr_strengths = list(getattr(args, "correction_strengths", [0.0]))
    smc_cs = float(corr_strengths[0]) if corr_strengths else 0.0

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

    print(
        f"  smc(das): K={k} cfg={cfg:.2f} variant={variant_idx} "
        f"gamma={float(args.smc_gamma):.3f} ess_thr={float(args.ess_threshold):.2f} euler={use_euler}"
    )
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
            flow = transformer_step(args, ctx, latents[pi : pi + 1], emb, variant_idx, t_flat, cfg)
            if use_euler:
                p_latents = latents[pi : pi + 1] + dt * flow
                p_dx = latents[pi : pi + 1] - t_4d * flow  # x0 estimate for scoring
                next_latents_parts.append(p_latents)
            else:
                p_dx = _pred_x0(latents[pi : pi + 1], t_4d, flow, args.x0_sampler)
            if smc_cs > 0.0:
                p_dx = apply_reward_correction(ctx, p_dx, prompt, reward_model, smc_cs, cfg=cfg)
            next_dx_parts.append(p_dx)
        dx = torch.cat(next_dx_parts, dim=0)
        if use_euler:
            latents = torch.cat(next_latents_parts, dim=0)

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
            idx = _systematic_resample(weights)
            dx = dx[idx].clone()
            latents = latents[idx].clone()
            log_w = torch.zeros_like(log_w)
            resample_count += 1

    final_tensor = _final_decode_tensor(latents, dx, use_euler)
    final_images = [decode_to_pil(ctx, final_tensor[pi : pi + 1]) for pi in range(k)]
    final_scores = [float(score_image(reward_model, prompt, img)) for img in final_images]
    best_idx = int(np.argmax(final_scores))
    diagnostics = {
        "smc_style": "das_tempered_resampling",
        "smc_k": int(k),
        "smc_cfg_scale": float(cfg),
        "smc_variant_idx": int(variant_idx),
        "smc_gamma": float(args.smc_gamma),
        "resample_start_step": int(start_idx),
        "resample_count": int(resample_count),
        "ess_min": float(min(ess_hist)) if ess_hist else 0.0,
        "ess_mean": float(sum(ess_hist) / len(ess_hist)) if ess_hist else 0.0,
        "reward_traj_mean": [float(v) for v in score_hist],
        "final_particle_scores": [float(v) for v in final_scores],
    }
    return SearchResult(
        image=final_images[best_idx],
        score=float(final_scores[best_idx]),
        actions=[(variant_idx, cfg, smc_cs) for _ in range(int(args.steps))],
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
