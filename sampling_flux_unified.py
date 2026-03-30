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
from dataclasses import dataclass
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
    p.add_argument("--search_method", choices=["greedy", "mcts", "ga", "smc"], default="ga")
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-schnell")
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
        choices=["auto", "unifiedreward", "unified", "imagereward", "pickscore", "hpsv3", "hpsv2", "blend"],
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
        default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
    )
    p.add_argument("--n_sims", type=int, default=50, help="MCTS simulation budget per prompt/seed.")
    p.add_argument("--ucb_c", type=float, default=1.41, help="MCTS UCB exploration constant.")

    # SMC
    p.add_argument("--smc_k", type=int, default=12)
    p.add_argument("--smc_gamma", type=float, default=0.10)
    p.add_argument("--ess_threshold", type=float, default=0.5)
    p.add_argument("--resample_start_frac", type=float, default=0.3)
    p.add_argument("--smc_guidance_scale", type=float, default=1.25)
    p.add_argument("--smc_chunk", type=int, default=4, help="Transformer batch chunk for particles.")

    p.add_argument("--save_first_k", type=int, default=10)
    p.add_argument("--out_dir", default="./flux_sampling_out")
    return p.parse_args(argv)


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


def load_pipeline(args: argparse.Namespace, device: str, dtype: torch.dtype) -> FluxContext:
    from sid import SiDFluxPipeline

    print(f"Loading FLUX pipeline: {args.model_id}")
    pipe = SiDFluxPipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
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


def build_t_schedule(steps: int) -> list[float]:
    if steps <= 1:
        return [1.0]
    denom = float(steps - 1)
    return [1.0 - float(i) / denom for i in range(steps)]


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

    init_latents = make_initial_latents(ctx, seed, args.height, args.width, batch_size=1)
    dx = torch.zeros_like(init_latents)
    rng = torch.Generator(device=ctx.device).manual_seed(seed + 2048)
    t_values = build_t_schedule(int(args.steps))

    for step_idx, (variant_idx, guidance) in enumerate(actions):
        t_val = float(t_values[step_idx])
        t_4d = torch.tensor(t_val, device=ctx.device, dtype=init_latents.dtype).view(1, 1, 1, 1)
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
        flow_pred = flux_transformer_step(ctx, latents, embeds[int(variant_idx)], t_val, float(guidance))
        dx = latents - t_4d * flow_pred

    image = decode_to_pil(ctx, dx)
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

    init_latents = make_initial_latents(ctx, seed, args.height, args.width, batch_size=1)
    dx = torch.zeros_like(init_latents)
    rng = torch.Generator(device=ctx.device).manual_seed(seed + 2048)
    t_values = build_t_schedule(int(args.steps))
    chosen: list[tuple[int, float]] = []
    history: list[dict[str, Any]] = []
    nfe_total = 0

    for step_idx, t_val in enumerate(t_values):
        t_4d = torch.tensor(float(t_val), device=ctx.device, dtype=init_latents.dtype).view(1, 1, 1, 1)
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
        print(f"  greedy step {step_idx + 1}/{args.steps} ({len(actions)} actions)")
        for variant_idx, guidance in actions:
            flow_pred = flux_transformer_step(ctx, latents, embeds[int(variant_idx)], float(t_val), float(guidance))
            nfe_total += 1
            cand_dx = latents - t_4d * flow_pred
            cand_img = decode_to_pil(ctx, cand_dx)
            cand_score = score_image(reward_model, prompt, cand_img)
            del cand_img
            if cand_score > best_score:
                best_score = float(cand_score)
                best_action = (int(variant_idx), float(guidance))
                best_dx = cand_dx.clone()

        assert best_dx is not None
        dx = best_dx
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

    final_img = decode_to_pil(ctx, dx)
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

    n_actions = len(actions)
    t_values = build_t_schedule(int(args.steps))
    init_latents = make_initial_latents(ctx, seed, args.height, args.width, batch_size=1)
    dx0 = torch.zeros_like(init_latents)
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
        flow = flux_transformer_step(ctx, current_latents, embeds[int(variant_idx)], t_val, float(guidance))
        nfe_total += 1
        new_dx = current_latents - t_4d * flow
        next_step = int(step_idx) + 1
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

        rollout_img = decode_to_pil(ctx, rollout_dx)
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
def run_smc(
    args: argparse.Namespace,
    ctx: FluxContext,
    reward_model: Any,
    prompt: str,
    balanced_embed: PromptEmbed,
    seed: int,
) -> SearchResult:
    k = max(2, int(args.smc_k))
    init_latents = make_initial_latents(ctx, seed, args.height, args.width, batch_size=k)
    dx = torch.zeros_like(init_latents)
    rng = torch.Generator(device=ctx.device).manual_seed(seed + 6001)
    t_values = build_t_schedule(int(args.steps))
    start_idx = int((1.0 - float(args.resample_start_frac)) * int(args.steps))
    log_w = torch.zeros(k, device=ctx.device, dtype=torch.float32)
    ess_hist: list[float] = []
    reward_hist: list[float] = []
    resample_count = 0

    for step_idx, t_val in enumerate(t_values):
        t_4d = torch.tensor(float(t_val), device=ctx.device, dtype=init_latents.dtype).view(1, 1, 1, 1)
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
        flow_pred = flux_transformer_step_chunked(
            ctx=ctx,
            latents=latents,
            embed=balanced_embed,
            t_val=float(t_val),
            guidance_scale=float(args.smc_guidance_scale),
            chunk=int(args.smc_chunk),
        )
        dx = latents - t_4d * flow_pred

        if step_idx < start_idx:
            continue

        step_scores = score_dx_batch(ctx, reward_model, prompt, dx)
        reward_hist.append(float(step_scores.mean().item()))
        lam = (1.0 + float(args.smc_gamma)) ** (int(args.steps) - 1 - step_idx) - 1.0
        log_w = log_w + float(lam) * step_scores
        w = torch.softmax(log_w, dim=0)
        ess = float(1.0 / torch.sum(w * w).item())
        ess_hist.append(ess)
        if ess < float(args.ess_threshold) * float(k):
            idx = systematic_resample(w)
            dx = dx[idx].clone()
            log_w = torch.zeros_like(log_w)
            resample_count += 1

    final_scores = score_dx_batch(ctx, reward_model, prompt, dx)
    best_idx = int(torch.argmax(final_scores).item())
    best_img = decode_to_pil(ctx, dx[best_idx : best_idx + 1])
    diagnostics = {
        "smc_k": int(k),
        "gamma": float(args.smc_gamma),
        "guidance_scale": float(args.smc_guidance_scale),
        "resample_count": int(resample_count),
        "ess_min": float(min(ess_hist)) if ess_hist else 0.0,
        "ess_mean": float(sum(ess_hist) / len(ess_hist)) if ess_hist else 0.0,
        "reward_traj_mean": reward_hist,
        "reward_all": [float(v) for v in final_scores.detach().cpu().tolist()],
    }
    return SearchResult(
        image=best_img,
        score=float(final_scores[best_idx].item()),
        actions=[(0, float(args.smc_guidance_scale)) for _ in range(int(args.steps))],
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
            else:
                search = run_smc(
                    args=args,
                    ctx=ctx,
                    reward_model=reward_model,
                    prompt=prompt,
                    balanced_embed=embeds[0],
                    seed=seed,
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
