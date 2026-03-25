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
    parser.add_argument("--search_method", choices=["greedy", "mcts", "ga", "smc"], default="greedy")

    parser.add_argument("--model_id", default="YGu1998/SiD-DiT-SD3.5-large")
    parser.add_argument("--ckpt", default=None)
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
        "--cfg_scales",
        nargs="+",
        type=float,
        default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
    )
    parser.add_argument("--baseline_cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--time_scale", type=float, default=1000.0)
    parser.add_argument("--out_dir", default="./imagereward_sd35_out")

    parser.add_argument("--n_sims", type=int, default=50)
    parser.add_argument("--ucb_c", type=float, default=1.41)
    parser.add_argument("--smc_k", type=int, default=8)
    parser.add_argument("--smc_gamma", type=float, default=0.10)
    parser.add_argument("--ess_threshold", type=float, default=0.5)
    parser.add_argument("--resample_start_frac", type=float, default=0.3)
    parser.add_argument("--smc_cfg_scale", type=float, default=1.25)
    parser.add_argument("--smc_variant_idx", type=int, default=0)
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
    return parser.parse_args(argv)


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
    actions: list[tuple[int, float]]
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

    from sid import SiDSD3Pipeline

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
    dtype = torch.float16
    dev_count = int(torch.cuda.device_count()) if cuda_available else 0
    print(
        f"Loading SD3.5 pipeline: {args.model_id} "
        f"(device={device} cuda_available={cuda_available} device_count={dev_count} "
        f"local_rank={local_rank} cvd={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})"
    )
    pipe = SiDSD3Pipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
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


def generate_variants(args: argparse.Namespace, prompt: str, cache: dict[str, list[str]]) -> list[str]:
    if prompt in cache:
        cached = [sanitize_rewrite_text(v, prompt) for v in cache[prompt][: args.n_variants + 1]]
        dedup: list[str] = []
        for v in cached:
            if v not in dedup:
                dedup.append(v)
        return dedup if dedup else [prompt]
    if args.n_variants <= 0 or args.no_qwen:
        return [prompt]
    variants = [prompt]
    styles = (REWRITE_STYLES * ((args.n_variants // len(REWRITE_STYLES)) + 1))[: args.n_variants]
    for style in styles:
        rewritten = sanitize_rewrite_text(qwen_rewrite(args, prompt, style), prompt)
        if rewritten not in variants:
            variants.append(rewritten)
    return variants


def encode_variants(ctx: PipelineContext, variants: list[str], max_sequence_length: int = 256) -> EmbeddingContext:
    cond_text: list[torch.Tensor] = []
    cond_pooled: list[torch.Tensor] = []
    for variant in variants:
        pe, pp = ctx.pipe.encode_prompt(
            prompt=variant,
            prompt_2=variant,
            prompt_3=variant,
            device=ctx.device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        cond_text.append(pe.detach())
        cond_pooled.append(pp.detach())

    ue, up = ctx.pipe.encode_prompt(
        prompt="",
        prompt_2="",
        prompt_3="",
        device=ctx.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )
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

    if cfg == 1.0:
        velocity = ctx.pipe.transformer(
            hidden_states=latents,
            encoder_hidden_states=pe,
            pooled_projections=pp,
            timestep=args.time_scale * t_flat,
            return_dict=False,
        )[0]
        return velocity

    flow = ctx.pipe.transformer(
        hidden_states=torch.cat([latents, latents]),
        encoder_hidden_states=torch.cat([emb.uncond_text, pe]),
        pooled_projections=torch.cat([emb.uncond_pooled, pp]),
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


def score_image(reward_model: UnifiedRewardScorer, prompt: str, image: Image.Image) -> float:
    return float(reward_model.score(prompt, image))


def step_schedule(device: str, dtype: torch.dtype, steps: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    sched: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(steps):
        scalar_t = 999.0 * (1.0 - float(i) / float(steps))
        t_flat = torch.full((1,), scalar_t / 999.0, device=device, dtype=dtype)
        t_4d = t_flat.view(1, 1, 1, 1)
        sched.append((t_flat, t_4d))
    return sched


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
    dx = torch.zeros_like(latents)
    sched = step_schedule(ctx.device, latents.dtype, args.steps)
    for i, (t_flat, t_4d) in enumerate(sched):
        noise = latents if i == 0 else torch.randn_like(latents)
        latents = (1.0 - t_4d) * dx + t_4d * noise
        flow = transformer_step(args, ctx, latents, emb, 0, t_flat, float(cfg_scale))
        dx = latents - t_4d * flow
    image = decode_to_pil(ctx, dx)
    return image, score_image(reward_model, prompt, image)


def run_greedy(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    variants: list[str],
    seed: int,
) -> SearchResult:
    actions = [(vi, cfg) for vi in range(len(variants)) for cfg in args.cfg_scales]
    latents = make_latents(ctx, seed, args.height, args.width, emb.cond_text[0].dtype)
    dx = torch.zeros_like(latents)
    sched = step_schedule(ctx.device, latents.dtype, args.steps)
    chosen: list[tuple[int, float]] = []
    for step_idx, (t_flat, t_4d) in enumerate(sched):
        noise = latents if step_idx == 0 else torch.randn_like(latents)
        latents = (1.0 - t_4d) * dx + t_4d * noise

        best_score = -float("inf")
        best_action = actions[0]
        best_dx = None

        print(f"  step {step_idx + 1}/{args.steps}: {len(actions)} actions")
        for variant_idx, cfg in actions:
            flow = transformer_step(args, ctx, latents, emb, variant_idx, t_flat, cfg)
            cand_dx = latents - t_4d * flow
            cand_img = decode_to_pil(ctx, cand_dx)
            cand_score = score_image(reward_model, prompt, cand_img)
            marker = ""
            if cand_score > best_score:
                best_score = cand_score
                best_action = (variant_idx, cfg)
                best_dx = cand_dx.clone()
                marker = " <- best"
            print(f"    v{variant_idx} cfg={cfg:.2f} IR={cand_score:.4f}{marker}")

        assert best_dx is not None
        dx = best_dx
        chosen.append(best_action)
        preview = variants[best_action[0]][:56]
        print(
            f"  selected v{best_action[0]} cfg={best_action[1]:.2f} "
            f"prompt='{preview}' score={best_score:.4f}"
        )

    final_img = decode_to_pil(ctx, dx)
    final_score = score_image(reward_model, prompt, final_img)
    return SearchResult(image=final_img, score=final_score, actions=chosen)


def run_schedule_actions(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    seed: int,
    actions: list[tuple[int, float]],
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
    sched = step_schedule(ctx.device, latents.dtype, args.steps)
    for step_idx, ((t_flat, t_4d), (variant_idx, cfg)) in enumerate(zip(sched, actions)):
        noise = latents if step_idx == 0 else torch.randn_like(latents)
        latents = (1.0 - t_4d) * dx + t_4d * noise
        flow = transformer_step(args, ctx, latents, emb, int(variant_idx), t_flat, float(cfg))
        dx = latents - t_4d * flow
    image = decode_to_pil(ctx, dx)
    score = score_image(reward_model, prompt, image)
    return SearchResult(image=image, score=score, actions=[(int(v), float(c)) for v, c in actions])


def _batched_flow_for_step_actions(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    latents: torch.Tensor,
    t_flat: torch.Tensor,
    step_actions: list[tuple[int, float]],
) -> torch.Tensor:
    flow_out = torch.empty_like(latents)
    groups: dict[tuple[int, float], list[int]] = {}
    for idx, (variant_idx, cfg) in enumerate(step_actions):
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
    actions_batch: list[list[tuple[int, float]]],
    deterministic_noise: bool = True,
) -> list[float]:
    if len(actions_batch) == 0:
        return []
    steps = int(args.steps)
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
    sched = step_schedule(ctx.device, latents.dtype, steps)
    shared_noises: list[torch.Tensor] = [base_latents]
    for _ in range(1, steps):
        shared_noises.append(torch.randn_like(base_latents))

    for step_idx, (t_flat, t_4d) in enumerate(sched):
        if step_idx == 0:
            noise = latents
        else:
            noise = shared_noises[step_idx].expand(batch_n, -1, -1, -1)
        latents = (1.0 - t_4d) * dx + t_4d * noise
        step_actions = [actions[step_idx] for actions in actions_batch]
        flow = _batched_flow_for_step_actions(args, ctx, emb, latents, t_flat, step_actions)
        dx = latents - t_4d * flow

    scores: list[float] = []
    for bi in range(batch_n):
        image = decode_to_pil(ctx, dx[bi : bi + 1])
        scores.append(score_image(reward_model, prompt, image))
    return scores


def _decode_genome(genome: list[int], cfg_scales: list[float], steps: int) -> list[tuple[int, float]]:
    actions: list[tuple[int, float]] = []
    for step in range(steps):
        vi = int(genome[2 * step])
        ci = int(genome[2 * step + 1])
        actions.append((vi, float(cfg_scales[ci])))
    return actions


def _actions_brief(actions: list[tuple[int, float]]) -> str:
    return " ".join(f"s{i+1}:v{vi}/cfg{cfg:.2f}" for i, (vi, cfg) in enumerate(actions))


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
    phase_constraints: bool,
) -> list[int]:
    out = list(genome[: 2 * steps])
    if len(out) < 2 * steps:
        out.extend([0] * (2 * steps - len(out)))
    early_choices, late_choices = _phase_variant_choices(steps, n_variants)
    for step in range(steps):
        v_pos = 2 * step
        c_pos = v_pos + 1
        vi = int(out[v_pos]) % max(n_variants, 1)
        ci = int(out[c_pos]) % max(n_cfg, 1)
        if phase_constraints and n_variants > 1:
            choices = early_choices if step < max(1, steps // 2) else late_choices
            if vi not in choices:
                vi = choices[vi % len(choices)]
        out[v_pos] = vi
        out[c_pos] = ci
    return out


def _random_genome(
    steps: int,
    n_variants: int,
    n_cfg: int,
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
        genome.extend([vi, ci])
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
    mutation_prob: float,
    phase_constraints: bool,
) -> list[int]:
    out = list(genome)
    p = max(0.0, min(1.0, float(mutation_prob)))
    early_choices, late_choices = _phase_variant_choices(steps, n_variants)
    for step in range(steps):
        v_pos = 2 * step
        c_pos = v_pos + 1
        if float(np.random.rand()) < p:
            if phase_constraints and n_variants > 1:
                choices = early_choices if step < max(1, steps // 2) else late_choices
            else:
                choices = list(range(n_variants))
            out[v_pos] = int(choices[np.random.randint(len(choices))])
        if float(np.random.rand()) < p:
            out[c_pos] = int(np.random.randint(n_cfg))
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
    if n_variants <= 0 or n_cfg <= 0:
        raise RuntimeError("GA requires non-empty variants and cfg_scales.")

    pop_size = max(2, int(args.ga_population))
    generations = max(1, int(args.ga_generations))
    elites = max(1, min(int(args.ga_elites), pop_size))
    steps = int(args.steps)
    genome_len = 2 * steps

    baseline_cfg_idx = _closest_cfg_index(list(args.cfg_scales), float(args.baseline_cfg))
    baseline_genome: list[int] = []
    for _ in range(steps):
        baseline_genome.extend([0, baseline_cfg_idx])

    population: list[list[int]] = [_repair_genome(baseline_genome, steps, n_variants, n_cfg, args.ga_phase_constraints)]
    while len(population) < pop_size:
        g = _random_genome(steps, n_variants, n_cfg, args.ga_phase_constraints)
        population.append(_repair_genome(g, steps, n_variants, n_cfg, args.ga_phase_constraints))

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
        queued: dict[tuple[int, ...], list[tuple[int, float]]] = {}
        for genome in genomes:
            eval_calls += 1
            repaired = _repair_genome(genome, steps, n_variants, n_cfg, args.ga_phase_constraints)
            key = tuple(repaired)
            prepared.append((repaired, key))
            if key in score_cache or key in queued:
                cache_hits += 1
                continue
            cache_misses += 1
            queued[key] = _decode_genome(repaired, list(args.cfg_scales), steps)

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
            actions = _decode_genome(genome, list(args.cfg_scales), steps)
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
                        "actions": [[int(v), float(c)] for v, c in _decode_genome(genome, list(args.cfg_scales), steps)],
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
                float(args.ga_mutation_prob),
                args.ga_phase_constraints,
            )
            child = _repair_genome(child, steps, n_variants, n_cfg, args.ga_phase_constraints)
            if len(child) != genome_len:
                child = child[:genome_len]
            next_population.append(child)
        population = next_population

    best_actions = _decode_genome(best_genome, list(args.cfg_scales), steps)
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
            "best_actions": [[int(v), float(c)] for v, c in best_actions],
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
            f.write(
                "gen\tbest\tmean\ttop_actions\n"
            )
            for row in history:
                top_actions = ""
                if row["top"]:
                    top_actions = _actions_brief(
                        [(int(v), float(c)) for v, c in row["top"][0]["actions"]]
                    )
                f.write(
                    f"{row['generation'] + 1}\t{row['best_score']:.6f}\t"
                    f"{row['mean_score']:.6f}\t{top_actions}\n"
                )

    return SearchResult(
        image=best_result.image,
        score=float(best_result.score),
        actions=[(int(v), float(c)) for v, c in best_actions],
    )


class MCTSNode:
    __slots__ = ("step", "dx", "latents", "children", "visits", "action_visits", "action_values")

    def __init__(self, step: int, dx: torch.Tensor, latents: torch.Tensor | None):
        self.step = step
        self.dx = dx
        self.latents = latents
        self.children: dict[tuple[int, float], MCTSNode] = {}
        self.visits = 0
        self.action_visits: dict[tuple[int, float], int] = {}
        self.action_values: dict[tuple[int, float], float] = {}

    def is_leaf(self, max_steps: int) -> bool:
        return self.step >= max_steps

    def untried_actions(self, actions: list[tuple[int, float]]) -> list[tuple[int, float]]:
        return [a for a in actions if a not in self.action_visits]

    def ucb(self, action: tuple[int, float], c: float) -> float:
        n = self.action_visits.get(action, 0)
        if n == 0:
            return float("inf")
        mean = self.action_values[action] / n
        return mean + c * math.sqrt(math.log(max(self.visits, 1)) / n)

    def best_ucb(self, actions: list[tuple[int, float]], c: float) -> tuple[int, float]:
        return max(actions, key=lambda action: self.ucb(action, c))

    def best_exploit(self, actions: list[tuple[int, float]]) -> tuple[int, float] | None:
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
    action: tuple[int, float],
    sched: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    variant_idx, cfg = action
    t_flat, t_4d = sched[node.step]
    flow = transformer_step(args, ctx, node.latents, emb, variant_idx, t_flat, cfg)
    new_dx = node.latents - t_4d * flow
    next_step = node.step + 1
    if next_step < len(sched):
        _, next_t_4d = sched[next_step]
        noise = torch.randn_like(new_dx)
        new_latents = (1.0 - next_t_4d) * new_dx + next_t_4d * noise
    else:
        new_latents = None
    return new_dx, new_latents


def run_mcts(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    variants: list[str],
    seed: int,
) -> SearchResult:
    actions = [(vi, cfg) for vi in range(len(variants)) for cfg in args.cfg_scales]
    n_actions = len(actions)
    latents0 = make_latents(ctx, seed, args.height, args.width, emb.cond_text[0].dtype)
    dx0 = torch.zeros_like(latents0)
    sched = step_schedule(ctx.device, latents0.dtype, args.steps)
    _, t0_4d = sched[0]
    start_latents = (1.0 - t0_4d) * dx0 + t0_4d * latents0
    root = MCTSNode(0, dx0, start_latents)

    best_global_score = -float("inf")
    best_global_dx = None
    best_global_path: list[tuple[int, float]] = []

    print(f"  mcts: sims={args.n_sims} actions={n_actions} steps={args.steps}")
    for sim in range(args.n_sims):
        node = root
        path: list[tuple[MCTSNode, tuple[int, float]]] = []

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
                child_dx, child_lat = _expand_child(args, ctx, emb, node, action, sched)
                node.children[action] = MCTSNode(node.step + 1, child_dx, child_lat)
            path.append((node, action))
            node = node.children[action]

        rollout_dx = node.dx
        rollout_latents = node.latents
        rollout_step = node.step
        while rollout_step < args.steps:
            variant_idx, cfg = actions[np.random.randint(n_actions)]
            t_flat, t_4d = sched[rollout_step]
            flow = transformer_step(args, ctx, rollout_latents, emb, variant_idx, t_flat, cfg)
            rollout_dx = rollout_latents - t_4d * flow
            rollout_step += 1
            if rollout_step < args.steps:
                _, next_t_4d = sched[rollout_step]
                noise = torch.randn_like(rollout_dx)
                rollout_latents = (1.0 - next_t_4d) * rollout_dx + next_t_4d * noise

        rollout_img = decode_to_pil(ctx, rollout_dx)
        rollout_score = score_image(reward_model, prompt, rollout_img)
        if rollout_score > best_global_score:
            best_global_score = rollout_score
            best_global_dx = rollout_dx.clone()
            best_global_path = [a for _, a in path]

        for pnode, paction in path:
            pnode.visits += 1
            pnode.action_visits[paction] = pnode.action_visits.get(paction, 0) + 1
            pnode.action_values[paction] = pnode.action_values.get(paction, 0.0) + rollout_score

        if (sim + 1) % 10 == 0 or sim == 0:
            print(f"    sim {sim + 1:3d}/{args.n_sims} best={best_global_score:.4f}")

    exploit_path: list[tuple[int, float]] = []
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
    for step_idx, (variant_idx, cfg) in enumerate(exploit_path):
        t_flat, t_4d = sched[step_idx]
        flow = transformer_step(args, ctx, replay_lat, emb, variant_idx, t_flat, cfg)
        replay_dx = replay_lat - t_4d * flow
        if step_idx + 1 < args.steps:
            _, next_t_4d = sched[step_idx + 1]
            noise = torch.randn_like(replay_dx)
            replay_lat = (1.0 - next_t_4d) * replay_dx + next_t_4d * noise

    exploit_img = decode_to_pil(ctx, replay_dx)
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

    particle_latents = []
    for pi in range(k):
        particle_latents.append(make_latents(ctx, seed + pi, args.height, args.width, emb.cond_text[0].dtype))
    latents = torch.cat(particle_latents, dim=0)
    dx = torch.zeros_like(latents)
    sched = step_schedule(ctx.device, latents.dtype, args.steps)
    rng = torch.Generator(device=ctx.device).manual_seed(seed + 6001)

    start_idx = int((1.0 - float(args.resample_start_frac)) * int(args.steps))
    start_idx = max(0, min(int(args.steps) - 1, start_idx))
    log_w = torch.zeros(k, device=ctx.device, dtype=torch.float32)
    ess_hist: list[float] = []
    score_hist: list[float] = []
    resample_count = 0

    print(
        f"  smc: K={k} cfg={cfg:.2f} variant={variant_idx} "
        f"gamma={float(args.smc_gamma):.3f} ess_thr={float(args.ess_threshold):.2f}"
    )
    for step_idx, (t_flat, t_4d) in enumerate(sched):
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
        for pi in range(k):
            flow = transformer_step(args, ctx, latents[pi : pi + 1], emb, variant_idx, t_flat, cfg)
            next_dx_parts.append(latents[pi : pi + 1] - t_4d * flow)
        dx = torch.cat(next_dx_parts, dim=0)

        if step_idx < start_idx:
            continue

        step_images = [decode_to_pil(ctx, dx[pi : pi + 1]) for pi in range(k)]
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

    final_images = [decode_to_pil(ctx, dx[pi : pi + 1]) for pi in range(k)]
    final_scores = [float(score_image(reward_model, prompt, img)) for img in final_images]
    best_idx = int(np.argmax(final_scores))
    diagnostics = {
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
        actions=[(variant_idx, cfg) for _ in range(int(args.steps))],
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
    actions: list[tuple[int, float]],
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
    acts = " ".join(f"s{i+1}:v{v}/cfg{c:.2f}" for i, (v, c) in enumerate(actions))
    draw.text((w + 4, 28), acts[:96], fill=(255, 220, 50), font=_font(11))
    comp.save(path)


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    prompts = load_prompts(args)
    ctx = load_pipeline(args)
    reward_model = load_reward_model(args, ctx.device)

    rewrite_cache: dict[str, list[str]] = {}
    if args.rewrites_file and os.path.exists(args.rewrites_file):
        rewrite_cache = json.load(open(args.rewrites_file))
        print(f"Loaded rewrite cache for {len(rewrite_cache)} prompts.")

    summary: list[dict[str, Any]] = []
    for prompt_idx, prompt in enumerate(prompts):
        slug = f"p{prompt_idx:02d}"
        print(f"\n{'='*72}\n[{slug}] {prompt}\n{'='*72}")
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
        else:
            raise RuntimeError(f"Unsupported search_method: {args.search_method}")

        base_path = os.path.join(args.out_dir, f"{slug}_baseline.png")
        search_path = os.path.join(args.out_dir, f"{slug}_{args.search_method}.png")
        comp_path = os.path.join(args.out_dir, f"{slug}_comparison.png")
        base_img.save(base_path)
        search.image.save(search_path)
        save_comparison(comp_path, base_img, search.image, base_score, search.score, search.actions)

        print(
            f"baseline={base_score:.4f} {args.search_method}={search.score:.4f} "
            f"delta={search.score - base_score:+.4f}"
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
                "actions": [[int(v), float(c)] for v, c in search.actions],
                "search_diagnostics": search.diagnostics,
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
