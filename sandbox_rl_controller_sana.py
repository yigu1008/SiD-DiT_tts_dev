"""
Sandbox: RL-style closed-loop conditioning controller for SiD-SANA.

This script upgrades one-shot Qwen rewrites into a per-step reactive controller:

    a_t = pi_theta(s_t)

where:
  - state s_t summarizes rollout progress + reward feedback
  - action a_t chooses a prompt-conditioning blend + CFG
  - theta are controller parameters optimized by GA

Baselines included per prompt:
  1) original prompt only
  2) one-shot rewrite (mid-only)
  3) fixed coarse->mid->fine schedule
  4) open-loop GA over fixed action sequences
  5) closed-loop GA over controller parameters

Notes:
  - No anchors
  - No continuous alpha search
  - No sigma/scheduler search
  - Uses actual timestep-derived progress (not raw step id)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import sampling_unified as su


HIERARCHY_SYSTEM_PROMPT = (
    "You are an image-prompt refiner for diffusion. "
    "Given one prompt, output ONLY JSON with keys coarse, mid, fine. "
    "All three must describe the same scene, composition, and subject. "
    "coarse: global scene + layout only. "
    "mid: add main subject attributes and objects. "
    "fine: add high-frequency details and stylistic refinements."
)


@dataclass
class ActionSpec:
    name: str
    prompt_name: str
    w_coarse: float
    w_mid: float
    w_fine: float
    cfg: float


@dataclass
class HierarchyPrompts:
    original: str
    coarse: str
    mid: str
    fine: str


@dataclass
class HierarchyEmbeddingCache:
    prompts: HierarchyPrompts
    action_bank: list[ActionSpec]
    # action_id -> (prompt_embed, prompt_mask)
    action_embeds: dict[int, tuple[torch.Tensor, torch.Tensor]]
    # action_id -> action name
    action_names: dict[int, str]
    ue: torch.Tensor
    um: torch.Tensor
    orig_pe: torch.Tensor
    orig_pm: torch.Tensor
    orig_ue: torch.Tensor
    orig_um: torch.Tensor


@dataclass
class StepRecord:
    step: int
    progress: float
    sigma: float
    prev_action_id: int
    prev_reward: float
    prev_delta_reward: float
    action_id: int
    action_name: str
    preview_reward: float
    delta_reward: float
    sat_proxy: float
    clip_proxy: float


@dataclass
class RolloutTrace:
    final_score: float
    actions: list[int]
    action_names: list[str]
    step_records: list[StepRecord]
    preview_rewards: list[float]
    final_image_path: str | None = None


@dataclass
class PolicyParams:
    tau_early: float
    tau_late: float
    delta_up: float
    delta_down: float
    bias_coarse: float
    bias_mid: float
    bias_fine: float
    cfg_bias: float
    cfg_trend: float


PROMPT_BLEND_BANK: list[ActionSpec] = [
    ActionSpec("coarse_only", "coarse_only", 1.0, 0.0, 0.0, 1.0),
    ActionSpec("mid_only", "mid_only", 0.0, 1.0, 0.0, 1.0),
    ActionSpec("fine_only", "fine_only", 0.0, 0.0, 1.0, 1.0),
    ActionSpec("coarse_mid_50", "coarse_mid_50", 0.5, 0.5, 0.0, 1.0),
    ActionSpec("mid_fine_50", "mid_fine_50", 0.0, 0.5, 0.5, 1.0),
]
PROMPT_ACTION_NAMES = {a.prompt_name for a in PROMPT_BLEND_BANK}
ACTION_BANK: list[ActionSpec] = []
ACTION_NAME_TO_ID: dict[str, int] = {}
ACTION_KEY_TO_ID: dict[tuple[str, float], int] = {}
ACTION_CFG_SCALES: list[float] = [1.0]


def _cfg_key(cfg: float) -> float:
    return round(float(cfg), 6)


def _format_cfg(cfg: float) -> str:
    return f"{float(cfg):.2f}".rstrip("0").rstrip(".")


def _unique_cfg_scales(scales: list[float] | None, fallback: float) -> list[float]:
    raw = scales if scales is not None and len(scales) > 0 else [fallback]
    out: list[float] = []
    seen: set[float] = set()
    for x in raw:
        xf = float(x)
        k = _cfg_key(xf)
        if k in seen:
            continue
        seen.add(k)
        out.append(xf)
    return out if out else [float(fallback)]


def init_action_bank(cfg_scales: list[float]) -> None:
    global ACTION_BANK, ACTION_NAME_TO_ID, ACTION_KEY_TO_ID, ACTION_CFG_SCALES
    ACTION_BANK = []
    ACTION_NAME_TO_ID = {}
    ACTION_KEY_TO_ID = {}
    ACTION_CFG_SCALES = [float(x) for x in cfg_scales]

    for base in PROMPT_BLEND_BANK:
        for cfg in ACTION_CFG_SCALES:
            name = f"{base.prompt_name}/cfg{_format_cfg(cfg)}"
            spec = ActionSpec(
                name=name,
                prompt_name=base.prompt_name,
                w_coarse=float(base.w_coarse),
                w_mid=float(base.w_mid),
                w_fine=float(base.w_fine),
                cfg=float(cfg),
            )
            action_id = len(ACTION_BANK)
            ACTION_BANK.append(spec)
            ACTION_NAME_TO_ID[name] = action_id
            ACTION_KEY_TO_ID[(spec.prompt_name, _cfg_key(spec.cfg))] = action_id


def action_id_for(prompt_name: str, cfg: float) -> int:
    if not ACTION_BANK:
        raise RuntimeError("ACTION_BANK is not initialized.")
    if prompt_name not in PROMPT_ACTION_NAMES:
        raise KeyError(f"Unknown prompt action name: {prompt_name}")
    nearest_cfg = min(ACTION_CFG_SCALES, key=lambda x: abs(float(x) - float(cfg)))
    key = (prompt_name, _cfg_key(nearest_cfg))
    action_id = ACTION_KEY_TO_ID.get(key)
    if action_id is None:
        raise RuntimeError(f"No action id for ({prompt_name}, cfg={nearest_cfg}).")
    return int(action_id)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sandbox RL-style conditioning controller (SANA).")

    # I/O
    p.add_argument("--prompt", type=str, default="a studio portrait of an elderly woman smiling, soft window light, 85mm lens, photorealistic")
    p.add_argument("--prompt_file", type=str, default=None)
    p.add_argument("--max_prompts", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="./sandbox_rl_controller_out")
    p.add_argument("--hierarchy_cache_json", type=str, default=None)
    p.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True)

    # Model
    p.add_argument("--model_id", type=str, default="YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--neg_embed", type=str, default=None)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument("--gpu_id", type=int, default=-1)
    p.add_argument("--auto_select_gpu", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min_free_gb", type=float, default=12.0)

    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--resolution_binning", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--time_scale", type=float, default=1000.0)
    p.add_argument("--guidance_scale", type=float, default=1.0, help="Legacy single CFG fallback when --cfg_scales is omitted.")
    p.add_argument("--cfg_scales", nargs="+", type=float, default=None, help="CFG values searched with each prompt action.")
    p.add_argument("--baseline_cfg", type=float, default=1.0, help="CFG used by baseline/fixed schedules.")
    p.add_argument("--seed", type=int, default=42)

    # Memory/runtime
    p.add_argument("--cuda_alloc_conf", type=str, default="expandable_segments:True")
    p.add_argument("--vae_slicing", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--vae_tiling", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--empty_cache_after_decode", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--offload_text_encoder_after_encode", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sana_no_fp32_attn", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--decode_device", type=str, default="auto")
    p.add_argument("--decode_cpu_dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
    p.add_argument("--decode_cpu_if_free_below_gb", type=float, default=20.0)

    # Reward
    p.add_argument(
        "--reward_type",
        choices=["imagereward", "auto", "unifiedreward", "unified", "hpsv2", "blend"],
        default="imagereward",
    )
    p.add_argument("--reward_device", type=str, default="cpu")
    p.add_argument("--reward_model", type=str, default="CodeGoat24/UnifiedReward-2.0-qwen3vl-4b")
    p.add_argument("--unifiedreward_model", type=str, default=None)
    p.add_argument("--image_reward_model", type=str, default="ImageReward-v1.0")
    p.add_argument("--reward_weights", nargs=2, type=float, default=[1.0, 1.0])
    p.add_argument("--reward_api_base", type=str, default=None)
    p.add_argument("--reward_api_key", type=str, default="unifiedreward")
    p.add_argument("--reward_api_model", type=str, default="UnifiedReward-7b-v1.5")
    p.add_argument("--reward_max_new_tokens", type=int, default=512)
    p.add_argument("--reward_prompt_mode", choices=["standard", "strict"], default="standard")

    # Qwen hierarchy
    p.add_argument("--no_qwen", action="store_true")
    p.add_argument("--qwen_id", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--qwen_python", type=str, default="python3")
    p.add_argument("--qwen_dtype", choices=["float16", "bfloat16"], default="bfloat16")

    # Rollout state/preview
    p.add_argument("--preview_every", type=int, default=1, help="Decode+score every N steps (1 = every step).")
    p.add_argument("--trace_topk_per_gen", type=int, default=2)
    p.add_argument(
        "--log_option_rewards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Post-hoc log per-step rewards for every action option on selected trajectories.",
    )

    # Open-loop GA
    p.add_argument("--openloop_population", type=int, default=24)
    p.add_argument("--openloop_generations", type=int, default=12)
    p.add_argument("--openloop_elites", type=int, default=3)
    p.add_argument("--openloop_mutation_prob", type=float, default=0.15)
    p.add_argument("--openloop_selection", choices=["rank", "tournament"], default="rank")
    p.add_argument("--openloop_rank_pressure", type=float, default=1.7)
    p.add_argument("--openloop_tournament_k", type=int, default=3)

    # Closed-loop GA (policy search)
    p.add_argument("--controller_population", type=int, default=24)
    p.add_argument("--controller_generations", type=int, default=12)
    p.add_argument("--controller_elites", type=int, default=3)
    p.add_argument("--controller_mutation_prob", type=float, default=0.20)
    p.add_argument("--controller_selection", choices=["rank", "tournament"], default="rank")
    p.add_argument("--controller_rank_pressure", type=float, default=1.7)
    p.add_argument("--controller_tournament_k", type=int, default=3)

    # Unused placeholders for load_reward(geneval path)
    p.add_argument("--reward_url", type=str, default=None)
    p.add_argument("--geneval_python", type=str, default=None)
    p.add_argument("--geneval_repo", type=str, default=None)
    p.add_argument("--detector_path", type=str, default=None)
    return p.parse_args(argv)


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file:
        prompts = [line.strip() for line in open(args.prompt_file, encoding="utf-8") if line.strip()]
    else:
        prompts = [args.prompt]
    if args.max_prompts > 0:
        prompts = prompts[: int(args.max_prompts)]
    if not prompts:
        raise RuntimeError("No prompts found.")
    return prompts


def heuristic_hierarchy(prompt: str) -> HierarchyPrompts:
    return HierarchyPrompts(
        original=prompt,
        coarse=f"{prompt} Focus on global scene layout and composition only.",
        mid=f"{prompt} Focus on subject and main objects with medium detail.",
        fine=f"{prompt} Focus on fine textures, tiny details, and precise attributes.",
    )


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def qwen_hierarchy_once(args: argparse.Namespace, prompt: str) -> HierarchyPrompts | None:
    dtype_literal = "torch.bfloat16" if args.qwen_dtype == "bfloat16" else "torch.float16"
    script = f"""
import json, re, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained({repr(args.qwen_id)})
mdl = AutoModelForCausalLM.from_pretrained(
    {repr(args.qwen_id)},
    torch_dtype={dtype_literal},
    device_map="auto")
mdl.eval()
messages = [
    {{"role":"system","content":{repr(HIERARCHY_SYSTEM_PROMPT)}}},
    {{"role":"user","content":"Prompt: " + sys.argv[1] + "\\nReturn strict JSON only."}}
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok([text], return_tensors="pt").to(mdl.device)
with torch.no_grad():
    out = mdl.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.4,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tok.eos_token_id,
    )
decoded = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()
print(decoded)
"""
    proc = subprocess.run(
        [args.qwen_python, "-c", script, prompt],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    obj = _parse_json_object(proc.stdout)
    if obj is None:
        return None
    coarse = str(obj.get("coarse", "")).strip()
    mid = str(obj.get("mid", "")).strip()
    fine = str(obj.get("fine", "")).strip()
    if not coarse or not mid or not fine:
        return None
    return HierarchyPrompts(original=prompt, coarse=coarse, mid=mid, fine=fine)


def qwen_hierarchy_fallback(args: argparse.Namespace, prompt: str) -> HierarchyPrompts:
    coarse = su.qwen_rewrite(
        args,
        prompt,
        "Rewrite this prompt into COARSE level: preserve scene/subject/composition; keep only global layout and major objects.",
    )
    mid = su.qwen_rewrite(
        args,
        prompt,
        "Rewrite this prompt into MID level: preserve scene/subject/composition; include main subject attributes and key objects.",
    )
    fine = su.qwen_rewrite(
        args,
        prompt,
        "Rewrite this prompt into FINE level: preserve scene/subject/composition; include rich detail, texture, and precise attributes.",
    )
    return HierarchyPrompts(
        original=prompt,
        coarse=coarse.strip() or prompt,
        mid=mid.strip() or prompt,
        fine=fine.strip() or prompt,
    )


def build_hierarchy_prompt(
    args: argparse.Namespace,
    prompt: str,
    cache: dict[str, dict[str, str]],
) -> HierarchyPrompts:
    cached = cache.get(prompt)
    if cached is not None:
        c = str(cached.get("coarse", "")).strip()
        m = str(cached.get("mid", "")).strip()
        f = str(cached.get("fine", "")).strip()
        if c and m and f:
            return HierarchyPrompts(original=prompt, coarse=c, mid=m, fine=f)

    if args.no_qwen:
        hp = heuristic_hierarchy(prompt)
    else:
        hp = qwen_hierarchy_once(args, prompt)
        if hp is None:
            hp = qwen_hierarchy_fallback(args, prompt)

    cache[prompt] = {"coarse": hp.coarse, "mid": hp.mid, "fine": hp.fine}
    return hp


def blend_mask(mask_c: torch.Tensor, mask_m: torch.Tensor, mask_f: torch.Tensor, w: ActionSpec) -> torch.Tensor:
    active = []
    if w.w_coarse > 0:
        active.append(mask_c)
    if w.w_mid > 0:
        active.append(mask_m)
    if w.w_fine > 0:
        active.append(mask_f)
    if not active:
        return mask_m
    merged = active[0]
    for m in active[1:]:
        merged = torch.maximum(merged, m)
    return merged


def encode_hierarchy(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    prompts: HierarchyPrompts,
    neg_embeds: torch.Tensor | None,
    neg_mask: torch.Tensor | None,
) -> HierarchyEmbeddingCache:
    hierarchy_variants = [prompts.coarse, prompts.mid, prompts.fine]
    emb = su.encode_variants(args, ctx, hierarchy_variants, neg_embeds, neg_mask)
    emb_orig = su.encode_variants(args, ctx, [prompts.original], neg_embeds, neg_mask)

    (pe_c, pm_c), (pe_m, pm_m), (pe_f, pm_f) = emb.pe_list
    action_embeds: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    action_names: dict[int, str] = {}
    for action_id, spec in enumerate(ACTION_BANK):
        pe = (
            spec.w_coarse * pe_c
            + spec.w_mid * pe_m
            + spec.w_fine * pe_f
        )
        pm = blend_mask(pm_c, pm_m, pm_f, spec)
        action_embeds[action_id] = (pe, pm)
        action_names[action_id] = spec.name

    return HierarchyEmbeddingCache(
        prompts=prompts,
        action_bank=list(ACTION_BANK),
        action_embeds=action_embeds,
        action_names=action_names,
        ue=emb.ue,
        um=emb.um,
        orig_pe=emb_orig.pe_list[0][0],
        orig_pm=emb_orig.pe_list[0][1],
        orig_ue=emb_orig.ue,
        orig_um=emb_orig.um,
    )


def image_quality_proxies(img: Image.Image) -> tuple[float, float]:
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return 0.0, 0.0
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    sat = float(np.mean(mx - mn))
    clip = float(np.mean((arr < 0.02) | (arr > 0.98)))
    return sat, clip


def should_preview(step_idx: int, steps: int, every: int) -> bool:
    if every <= 0:
        return step_idx == steps - 1
    return ((step_idx + 1) % every == 0) or (step_idx == steps - 1)


def analyze_option_rewards_along_path(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    reward_ctx: su.RewardContext,
    prompt: str,
    metadata: dict[str, Any] | None,
    seed: int,
    h: int,
    w: int,
    orig_h: int,
    orig_w: int,
    cache: HierarchyEmbeddingCache,
    action_sequence: list[int],
    tag: str,
) -> list[dict[str, Any]]:
    if len(action_sequence) != int(args.steps):
        raise RuntimeError(f"Expected action_sequence length={args.steps}, got {len(action_sequence)}.")

    latents = su.make_latents(ctx, seed, h, w, cache.orig_pe.dtype)
    sched = su._step_tensors(ctx, args.steps, latents.dtype)
    dx = torch.zeros_like(latents)
    rng = torch.Generator(device=ctx.device).manual_seed(seed + 2048)
    rows: list[dict[str, Any]] = []
    prev_chosen_reward = 0.0
    prev_best_reward = 0.0

    for step_idx, (t_flat, t_4d) in enumerate(sched):
        noise = latents if step_idx == 0 else torch.randn(
            latents.shape, device=latents.device, dtype=latents.dtype, generator=rng
        )
        latents = (1.0 - t_4d) * dx + t_4d * noise

        option_scores: dict[str, float] = {}
        option_dx: dict[int, torch.Tensor] = {}
        for action_id, spec in enumerate(ACTION_BANK):
            pe, pm = cache.action_embeds[action_id]
            velocity = su.transformer_step(
                args, ctx, latents, pe, pm, cache.ue, cache.um, t_flat, float(spec.cfg)
            )
            cand_dx = latents - t_4d * velocity
            option_dx[action_id] = cand_dx
            cand_img = su.decode_to_pil(ctx, cand_dx, orig_h, orig_w, tag=f"{tag}_opt")
            score = float(reward_ctx.score_images(prompt, [cand_img], metadata)[0])
            option_scores[spec.name] = score

        chosen_id = int(action_sequence[step_idx])
        chosen_spec = ACTION_BANK[chosen_id]
        chosen_name = chosen_spec.name
        dx = option_dx[chosen_id]

        ranked = sorted(option_scores.items(), key=lambda kv: kv[1], reverse=True)
        option_deltas = {k: float(v - prev_chosen_reward) for k, v in option_scores.items()}
        chosen_reward = float(option_scores[chosen_name])
        best_reward = float(ranked[0][1])
        chosen_rank = 1 + next(i for i, (name, _) in enumerate(ranked) if name == chosen_name)
        rows.append(
            {
                "step": int(step_idx),
                "progress": float(1.0 - t_flat[0].item()),
                "sigma": float(t_flat[0].item()),
                "chosen_action_id": int(chosen_id),
                "chosen_action_name": chosen_name,
                "chosen_prompt_action": str(chosen_spec.prompt_name),
                "chosen_cfg": float(chosen_spec.cfg),
                "prev_chosen_reward": float(prev_chosen_reward),
                "chosen_reward": chosen_reward,
                "chosen_delta_from_prev_step": float(chosen_reward - prev_chosen_reward),
                "chosen_rank": int(chosen_rank),
                "best_action_name": str(ranked[0][0]),
                "best_reward": best_reward,
                "best_delta_from_prev_step": float(best_reward - prev_chosen_reward),
                "best_delta_from_prev_best": float(best_reward - prev_best_reward),
                "chosen_vs_best_gap": float(chosen_reward - best_reward),
                "option_rewards": {k: float(v) for k, v in option_scores.items()},
                "option_reward_deltas_from_prev_step": option_deltas,
            }
        )
        prev_chosen_reward = chosen_reward
        prev_best_reward = best_reward
    return rows


def run_rollout_action_sequence(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    reward_ctx: su.RewardContext,
    prompt: str,
    metadata: dict[str, Any] | None,
    seed: int,
    h: int,
    w: int,
    orig_h: int,
    orig_w: int,
    cache: HierarchyEmbeddingCache,
    action_sequence: list[int],
    save_path: str | None,
    tag: str,
) -> RolloutTrace:
    if len(action_sequence) != args.steps:
        raise RuntimeError(f"Expected action sequence length={args.steps}, got {len(action_sequence)}.")

    latents = su.make_latents(ctx, seed, h, w, cache.orig_pe.dtype)
    sched = su._step_tensors(ctx, args.steps, latents.dtype)
    dx = torch.zeros_like(latents)
    rng = torch.Generator(device=ctx.device).manual_seed(seed + 2048)

    step_records: list[StepRecord] = []
    preview_rewards: list[float] = []
    prev_reward = 0.0
    prev_delta = 0.0
    prev_action = -1

    for step_idx, (t_flat, t_4d) in enumerate(sched):
        noise = latents if step_idx == 0 else torch.randn(
            latents.shape, device=latents.device, dtype=latents.dtype, generator=rng
        )
        latents = (1.0 - t_4d) * dx + t_4d * noise

        action_id = int(action_sequence[step_idx])
        spec = ACTION_BANK[action_id]
        pe, pm = cache.action_embeds[action_id]
        velocity = su.transformer_step(
            args, ctx, latents, pe, pm, cache.ue, cache.um, t_flat, float(spec.cfg)
        )
        dx = latents - t_4d * velocity

        cur_reward = prev_reward
        delta = 0.0
        sat = 0.0
        clip = 0.0
        if should_preview(step_idx, args.steps, int(args.preview_every)):
            preview = su.decode_to_pil(ctx, dx, orig_h, orig_w, tag=f"{tag}_preview")
            cur_reward = float(reward_ctx.score_images(prompt, [preview], metadata)[0])
            delta = cur_reward - prev_reward
            sat, clip = image_quality_proxies(preview)
            preview_rewards.append(cur_reward)
        step_records.append(
            StepRecord(
                step=step_idx,
                progress=float(1.0 - t_flat[0].item()),
                sigma=float(t_flat[0].item()),
                prev_action_id=prev_action,
                prev_reward=float(prev_reward),
                prev_delta_reward=float(prev_delta),
                action_id=action_id,
                action_name=cache.action_names[action_id],
                preview_reward=float(cur_reward),
                delta_reward=float(delta),
                sat_proxy=float(sat),
                clip_proxy=float(clip),
            )
        )
        prev_action = action_id
        prev_delta = delta
        prev_reward = cur_reward

    final_image = su.decode_to_pil(ctx, dx, orig_h, orig_w, tag=f"{tag}_final")
    final_score = float(reward_ctx.score_images(prompt, [final_image], metadata)[0])
    if save_path is not None and args.save_images:
        final_image.save(save_path)

    return RolloutTrace(
        final_score=final_score,
        actions=[int(x) for x in action_sequence],
        action_names=[cache.action_names[int(x)] for x in action_sequence],
        step_records=step_records,
        preview_rewards=preview_rewards,
        final_image_path=save_path,
    )


def run_rollout_original_prompt(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    reward_ctx: su.RewardContext,
    prompt: str,
    metadata: dict[str, Any] | None,
    seed: int,
    h: int,
    w: int,
    orig_h: int,
    orig_w: int,
    cache: HierarchyEmbeddingCache,
    save_path: str | None,
    tag: str,
) -> RolloutTrace:
    latents = su.make_latents(ctx, seed, h, w, cache.orig_pe.dtype)
    sched = su._step_tensors(ctx, args.steps, latents.dtype)
    dx = torch.zeros_like(latents)
    rng = torch.Generator(device=ctx.device).manual_seed(seed + 2048)
    preview_rewards: list[float] = []
    step_records: list[StepRecord] = []
    prev_reward = 0.0
    prev_delta = 0.0

    for step_idx, (t_flat, t_4d) in enumerate(sched):
        noise = latents if step_idx == 0 else torch.randn(
            latents.shape, device=latents.device, dtype=latents.dtype, generator=rng
        )
        latents = (1.0 - t_4d) * dx + t_4d * noise
        velocity = su.transformer_step(
            args,
            ctx,
            latents,
            cache.orig_pe,
            cache.orig_pm,
            cache.orig_ue,
            cache.orig_um,
            t_flat,
            float(args.baseline_cfg),
        )
        dx = latents - t_4d * velocity

        cur_reward = prev_reward
        delta = 0.0
        sat = 0.0
        clip = 0.0
        if should_preview(step_idx, args.steps, int(args.preview_every)):
            preview = su.decode_to_pil(ctx, dx, orig_h, orig_w, tag=f"{tag}_preview")
            cur_reward = float(reward_ctx.score_images(prompt, [preview], metadata)[0])
            delta = cur_reward - prev_reward
            sat, clip = image_quality_proxies(preview)
            preview_rewards.append(cur_reward)

        step_records.append(
            StepRecord(
                step=step_idx,
                progress=float(1.0 - t_flat[0].item()),
                sigma=float(t_flat[0].item()),
                prev_action_id=-1,
                prev_reward=float(prev_reward),
                prev_delta_reward=float(prev_delta),
                action_id=-1,
                action_name="original_only",
                preview_reward=float(cur_reward),
                delta_reward=float(delta),
                sat_proxy=float(sat),
                clip_proxy=float(clip),
            )
        )
        prev_delta = delta
        prev_reward = cur_reward

    final_image = su.decode_to_pil(ctx, dx, orig_h, orig_w, tag=f"{tag}_final")
    final_score = float(reward_ctx.score_images(prompt, [final_image], metadata)[0])
    if save_path is not None and args.save_images:
        final_image.save(save_path)
    return RolloutTrace(
        final_score=final_score,
        actions=[-1 for _ in range(args.steps)],
        action_names=["original_only" for _ in range(args.steps)],
        step_records=step_records,
        preview_rewards=preview_rewards,
        final_image_path=save_path,
    )


def default_fixed_schedule(steps: int, cfg: float) -> list[int]:
    # early -> coarse_mid_50, middle -> mid_only, late -> mid_fine_50/fine_only
    ids: list[int] = []
    for i in range(steps):
        p = float(i) / float(max(1, steps - 1))
        if p < 0.34:
            ids.append(action_id_for("coarse_mid_50", cfg))
        elif p < 0.67:
            ids.append(action_id_for("mid_only", cfg))
        elif p < 0.90:
            ids.append(action_id_for("mid_fine_50", cfg))
        else:
            ids.append(action_id_for("fine_only", cfg))
    return ids


def tournament_select(scored: list[tuple[float, list[int]]], k: int, rng: np.random.Generator) -> list[int]:
    n = len(scored)
    if n == 0:
        raise RuntimeError("Tournament selection received empty population.")
    kk = max(1, min(k, n))
    picks = [scored[int(rng.integers(0, n))] for _ in range(kk)]
    return list(max(picks, key=lambda x: x[0])[1])


def tournament_select_float(scored: list[tuple[float, list[float]]], k: int, rng: np.random.Generator) -> list[float]:
    n = len(scored)
    if n == 0:
        raise RuntimeError("Tournament selection received empty population.")
    kk = max(1, min(k, n))
    picks = [scored[int(rng.integers(0, n))] for _ in range(kk)]
    return list(max(picks, key=lambda x: x[0])[1])


def rank_select(scored: list[tuple[float, list[int]]], rng: np.random.Generator, rank_pressure: float) -> list[int]:
    n = len(scored)
    if n == 0:
        raise RuntimeError("Rank selection received empty population.")
    if n == 1:
        return list(scored[0][1])

    # Linear ranking (Baker): pressure s in [1, 2].
    # scored is sorted best->worst, while formula uses worst->best rank i.
    s = float(max(1.0, min(2.0, rank_pressure)))
    probs = np.empty(n, dtype=np.float64)
    for idx_desc in range(n):
        rank_worst_first = n - 1 - idx_desc
        probs[idx_desc] = ((2.0 - s) / n) + (2.0 * rank_worst_first * (s - 1.0) / (n * (n - 1)))
    probs = probs / probs.sum()
    chosen = int(rng.choice(np.arange(n, dtype=np.int64), p=probs))
    return list(scored[chosen][1])


def rank_select_float(scored: list[tuple[float, list[float]]], rng: np.random.Generator, rank_pressure: float) -> list[float]:
    n = len(scored)
    if n == 0:
        raise RuntimeError("Rank selection received empty population.")
    if n == 1:
        return list(scored[0][1])

    # Linear ranking (Baker): pressure s in [1, 2].
    # scored is sorted best->worst, while formula uses worst->best rank i.
    s = float(max(1.0, min(2.0, rank_pressure)))
    probs = np.empty(n, dtype=np.float64)
    for idx_desc in range(n):
        rank_worst_first = n - 1 - idx_desc
        probs[idx_desc] = ((2.0 - s) / n) + (2.0 * rank_worst_first * (s - 1.0) / (n * (n - 1)))
    probs = probs / probs.sum()
    chosen = int(rng.choice(np.arange(n, dtype=np.int64), p=probs))
    return list(scored[chosen][1])


def crossover_int(a: list[int], b: list[int], rng: np.random.Generator) -> list[int]:
    out = []
    for ga, gb in zip(a, b):
        out.append(int(ga if rng.random() < 0.5 else gb))
    return out


def mutate_int(genome: list[int], n_actions: int, p: float, rng: np.random.Generator) -> list[int]:
    out = list(genome)
    for i in range(len(out)):
        if rng.random() < p:
            out[i] = int(rng.integers(0, n_actions))
    return out


def run_openloop_ga(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    reward_ctx: su.RewardContext,
    prompt: str,
    metadata: dict[str, Any] | None,
    seed: int,
    h: int,
    w: int,
    orig_h: int,
    orig_w: int,
    cache: HierarchyEmbeddingCache,
    prompt_dir: str,
) -> tuple[RolloutTrace, dict[str, Any]]:
    rng = np.random.default_rng(seed + 1001)
    pop = max(4, int(args.openloop_population))
    elites = max(1, min(int(args.openloop_elites), pop))
    n_actions = len(ACTION_BANK)
    steps = int(args.steps)
    fixed = default_fixed_schedule(steps, float(args.baseline_cfg))

    population: list[list[int]] = [list(fixed)]
    while len(population) < pop:
        population.append([int(rng.integers(0, n_actions)) for _ in range(steps)])

    eval_calls = 0
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_genome = list(fixed)
    best_trace: RolloutTrace | None = None
    ga_start = time.perf_counter()

    def evaluate(genome: list[int], save_path: str | None = None, tag: str = "openloop_ga_eval") -> RolloutTrace:
        nonlocal eval_calls
        eval_calls += 1
        return run_rollout_action_sequence(
            args, ctx, reward_ctx, prompt, metadata, seed, h, w, orig_h, orig_w, cache, genome, save_path, tag
        )

    for gen in range(int(args.openloop_generations)):
        gen_start = time.perf_counter()
        scored: list[tuple[float, list[int], RolloutTrace]] = []
        for genome in population:
            trace = evaluate(genome, save_path=None, tag=f"openloop_ga_g{gen}")
            scored.append((float(trace.final_score), list(genome), trace))
        scored.sort(key=lambda x: x[0], reverse=True)
        topk = max(1, int(args.trace_topk_per_gen))
        top_rows = []
        for rank, (score, genome, trace) in enumerate(scored[:topk], start=1):
            top_rows.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "genome": [int(x) for x in genome],
                    "actions": [ACTION_BANK[int(a)].name for a in genome],
                    "prompt_actions": [ACTION_BANK[int(a)].prompt_name for a in genome],
                    "cfg_scales": [float(ACTION_BANK[int(a)].cfg) for a in genome],
                    "preview_rewards": [float(x) for x in trace.preview_rewards],
                }
            )
        gen_best_score, gen_best_genome, _ = scored[0]
        if gen_best_score > best_score:
            best_score = float(gen_best_score)
            best_genome = list(gen_best_genome)
            best_trace = evaluate(
                best_genome,
                save_path=os.path.join(prompt_dir, "openloop_ga_best.png"),
                tag="openloop_ga_best",
            )
        gen_elapsed = time.perf_counter() - gen_start
        history.append(
            {
                "generation": int(gen),
                "best_score": float(gen_best_score),
                "mean_score": float(np.mean([s for s, _, _ in scored])),
                "eval_calls_total": int(eval_calls),
                "nfe_per_generation": int(len(scored) * steps),
                "nfe_total": int(eval_calls * steps),
                "wallclock_sec": float(gen_elapsed),
                "top": top_rows,
            }
        )

        if gen + 1 >= int(args.openloop_generations):
            break

        use_rank_selection = str(args.openloop_selection).lower() == "rank"
        next_population: list[list[int]] = [list(g) for _, g, _ in scored[:elites]]
        while len(next_population) < pop:
            scored_pairs = [(s, g) for s, g, _ in scored]
            if use_rank_selection:
                pa = rank_select(scored_pairs, rng, float(args.openloop_rank_pressure))
                pb = rank_select(scored_pairs, rng, float(args.openloop_rank_pressure))
            else:
                pa = tournament_select(scored_pairs, int(args.openloop_tournament_k), rng)
                pb = tournament_select(scored_pairs, int(args.openloop_tournament_k), rng)
            child = crossover_int(pa, pb, rng)
            child = mutate_int(child, n_actions, float(args.openloop_mutation_prob), rng)
            next_population.append(child)
        population = next_population

    assert best_trace is not None
    payload = {
        "best_score": float(best_trace.final_score),
        "best_genome": [int(x) for x in best_genome],
        "best_actions": [ACTION_BANK[int(a)].name for a in best_genome],
        "best_prompt_actions": [ACTION_BANK[int(a)].prompt_name for a in best_genome],
        "best_cfg_scales": [float(ACTION_BANK[int(a)].cfg) for a in best_genome],
        "selection": {
            "mode": str(args.openloop_selection),
            "rank_pressure": float(args.openloop_rank_pressure),
            "tournament_k": int(args.openloop_tournament_k),
        },
        "eval_calls_total": int(eval_calls),
        "nfe_total": int(eval_calls * steps),
        "wallclock_total_sec": float(time.perf_counter() - ga_start),
        "history": history,
    }
    return best_trace, payload


def decode_policy_genome(genome: list[float]) -> PolicyParams:
    tau_early = float(np.clip(genome[0], 0.05, 0.70))
    tau_late = float(np.clip(genome[1], 0.20, 0.98))
    if tau_late <= tau_early + 0.05:
        tau_late = min(0.98, tau_early + 0.05)
    delta_up = float(np.clip(genome[2], -0.10, 0.10))
    delta_down = float(np.clip(genome[3], -0.30, 0.05))
    bias_coarse = float(np.clip(genome[4], -1.0, 1.0))
    bias_mid = float(np.clip(genome[5], -1.0, 1.0))
    bias_fine = float(np.clip(genome[6], -1.0, 1.0))
    cfg_bias = float(np.clip(genome[7], -1.0, 1.0))
    cfg_trend = float(np.clip(genome[8], -1.0, 1.0))
    return PolicyParams(
        tau_early=tau_early,
        tau_late=tau_late,
        delta_up=delta_up,
        delta_down=delta_down,
        bias_coarse=bias_coarse,
        bias_mid=bias_mid,
        bias_fine=bias_fine,
        cfg_bias=cfg_bias,
        cfg_trend=cfg_trend,
    )


def random_policy_genome(rng: np.random.Generator) -> list[float]:
    g = [
        float(rng.uniform(0.10, 0.50)),
        float(rng.uniform(0.55, 0.95)),
        float(rng.uniform(-0.03, 0.06)),
        float(rng.uniform(-0.15, 0.02)),
        float(rng.uniform(-0.6, 0.6)),
        float(rng.uniform(-0.6, 0.6)),
        float(rng.uniform(-0.6, 0.6)),
        float(rng.uniform(-0.6, 0.6)),
        float(rng.uniform(-0.6, 0.6)),
    ]
    return g


def crossover_float(a: list[float], b: list[float], rng: np.random.Generator) -> list[float]:
    return [float(aa if rng.random() < 0.5 else bb) for aa, bb in zip(a, b)]


def mutate_float(genome: list[float], p: float, rng: np.random.Generator) -> list[float]:
    sigmas = [0.06, 0.06, 0.02, 0.03, 0.15, 0.15, 0.15, 0.15, 0.15]
    out = list(genome)
    for i in range(len(out)):
        if rng.random() < p:
            out[i] = float(out[i] + rng.normal(0.0, sigmas[i]))
    return out


def policy_select_action(params: PolicyParams, progress: float, prev_delta: float, prev_action: int) -> int:
    cfg_min = min(spec.cfg for spec in ACTION_BANK)
    cfg_max = max(spec.cfg for spec in ACTION_BANK)
    cfg_mid = 0.5 * (cfg_min + cfg_max)
    cfg_span = max(1e-6, cfg_max - cfg_min)

    scores = np.zeros((len(ACTION_BANK),), dtype=np.float64)
    for i, spec in enumerate(ACTION_BANK):
        s = 0.0
        # Phase priors
        if progress < params.tau_early:
            phase_pref = {
                "coarse_only": 0.70,
                "coarse_mid_50": 1.00,
                "mid_only": 0.40,
                "mid_fine_50": 0.10,
                "fine_only": 0.00,
            }
        elif progress > params.tau_late:
            phase_pref = {
                "coarse_only": 0.00,
                "coarse_mid_50": 0.10,
                "mid_only": 0.50,
                "mid_fine_50": 0.90,
                "fine_only": 1.00,
            }
        else:
            phase_pref = {
                "coarse_only": 0.20,
                "coarse_mid_50": 0.60,
                "mid_only": 1.00,
                "mid_fine_50": 0.60,
                "fine_only": 0.20,
            }
        s += phase_pref[spec.prompt_name]

        # Reactive rules from reward delta.
        if prev_delta < params.delta_down:
            if spec.prompt_name == "mid_only":
                s += 0.35
            if spec.prompt_name in {"fine_only", "mid_fine_50"}:
                s -= 0.20
        elif prev_delta < params.delta_up:
            if spec.prompt_name in {"mid_fine_50", "fine_only"}:
                s += 0.25

        # Blend-bias terms.
        s += params.bias_coarse * spec.w_coarse
        s += params.bias_mid * spec.w_mid
        s += params.bias_fine * spec.w_fine

        # CFG preference terms.
        cfg_centered = (spec.cfg - cfg_mid) / cfg_span
        s += params.cfg_bias * cfg_centered
        s += params.cfg_trend * (progress - 0.5) * cfg_centered

        # Small action inertia.
        if prev_action == i:
            s += 0.05
        scores[i] = s
    return int(np.argmax(scores))


def run_rollout_controller(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    reward_ctx: su.RewardContext,
    prompt: str,
    metadata: dict[str, Any] | None,
    seed: int,
    h: int,
    w: int,
    orig_h: int,
    orig_w: int,
    cache: HierarchyEmbeddingCache,
    params: PolicyParams,
    save_path: str | None,
    tag: str,
) -> RolloutTrace:
    latents = su.make_latents(ctx, seed, h, w, cache.orig_pe.dtype)
    sched = su._step_tensors(ctx, args.steps, latents.dtype)
    dx = torch.zeros_like(latents)
    rng = torch.Generator(device=ctx.device).manual_seed(seed + 2048)

    actions: list[int] = []
    action_names: list[str] = []
    step_records: list[StepRecord] = []
    preview_rewards: list[float] = []
    prev_reward = 0.0
    prev_delta = 0.0
    prev_action = -1

    for step_idx, (t_flat, t_4d) in enumerate(sched):
        progress = float(1.0 - t_flat[0].item())
        action_id = policy_select_action(params, progress, prev_delta, prev_action)
        actions.append(action_id)
        action_names.append(cache.action_names[action_id])

        noise = latents if step_idx == 0 else torch.randn(
            latents.shape, device=latents.device, dtype=latents.dtype, generator=rng
        )
        latents = (1.0 - t_4d) * dx + t_4d * noise

        pe, pm = cache.action_embeds[action_id]
        spec = ACTION_BANK[action_id]
        velocity = su.transformer_step(
            args, ctx, latents, pe, pm, cache.ue, cache.um, t_flat, float(spec.cfg)
        )
        dx = latents - t_4d * velocity

        cur_reward = prev_reward
        delta = 0.0
        sat = 0.0
        clip = 0.0
        if should_preview(step_idx, args.steps, int(args.preview_every)):
            preview = su.decode_to_pil(ctx, dx, orig_h, orig_w, tag=f"{tag}_preview")
            cur_reward = float(reward_ctx.score_images(prompt, [preview], metadata)[0])
            delta = cur_reward - prev_reward
            sat, clip = image_quality_proxies(preview)
            preview_rewards.append(cur_reward)
        step_records.append(
            StepRecord(
                step=step_idx,
                progress=progress,
                sigma=float(t_flat[0].item()),
                prev_action_id=prev_action,
                prev_reward=float(prev_reward),
                prev_delta_reward=float(prev_delta),
                action_id=action_id,
                action_name=cache.action_names[action_id],
                preview_reward=float(cur_reward),
                delta_reward=float(delta),
                sat_proxy=float(sat),
                clip_proxy=float(clip),
            )
        )
        prev_action = action_id
        prev_delta = delta
        prev_reward = cur_reward

    final_image = su.decode_to_pil(ctx, dx, orig_h, orig_w, tag=f"{tag}_final")
    final_score = float(reward_ctx.score_images(prompt, [final_image], metadata)[0])
    if save_path is not None and args.save_images:
        final_image.save(save_path)
    return RolloutTrace(
        final_score=final_score,
        actions=actions,
        action_names=action_names,
        step_records=step_records,
        preview_rewards=preview_rewards,
        final_image_path=save_path,
    )


def run_controller_ga(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    reward_ctx: su.RewardContext,
    prompt: str,
    metadata: dict[str, Any] | None,
    seed: int,
    h: int,
    w: int,
    orig_h: int,
    orig_w: int,
    cache: HierarchyEmbeddingCache,
    prompt_dir: str,
) -> tuple[RolloutTrace, dict[str, Any]]:
    rng = np.random.default_rng(seed + 2001)
    pop = max(4, int(args.controller_population))
    elites = max(1, min(int(args.controller_elites), pop))

    default_params = PolicyParams(
        tau_early=0.33,
        tau_late=0.75,
        delta_up=0.0,
        delta_down=-0.06,
        bias_coarse=0.0,
        bias_mid=0.0,
        bias_fine=0.0,
        cfg_bias=0.0,
        cfg_trend=0.0,
    )
    default_genome = [
        default_params.tau_early,
        default_params.tau_late,
        default_params.delta_up,
        default_params.delta_down,
        default_params.bias_coarse,
        default_params.bias_mid,
        default_params.bias_fine,
        default_params.cfg_bias,
        default_params.cfg_trend,
    ]
    population: list[list[float]] = [list(default_genome)]
    while len(population) < pop:
        population.append(random_policy_genome(rng))

    eval_calls = 0
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_genome = list(default_genome)
    best_trace: RolloutTrace | None = None
    ga_start = time.perf_counter()

    def evaluate(genome: list[float], save_path: str | None = None, tag: str = "controller_ga_eval") -> tuple[float, RolloutTrace, PolicyParams]:
        nonlocal eval_calls
        eval_calls += 1
        params = decode_policy_genome(genome)
        trace = run_rollout_controller(
            args, ctx, reward_ctx, prompt, metadata, seed, h, w, orig_h, orig_w, cache, params, save_path, tag
        )
        return float(trace.final_score), trace, params

    for gen in range(int(args.controller_generations)):
        gen_start = time.perf_counter()
        scored: list[tuple[float, list[float], RolloutTrace, PolicyParams]] = []
        for genome in population:
            score, trace, params = evaluate(genome, save_path=None, tag=f"controller_ga_g{gen}")
            scored.append((score, list(genome), trace, params))
        scored.sort(key=lambda x: x[0], reverse=True)
        topk = max(1, int(args.trace_topk_per_gen))
        top_rows = []
        for rank, (score, genome, trace, params) in enumerate(scored[:topk], start=1):
            top_rows.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "genome": [float(x) for x in genome],
                    "params": asdict(params),
                    "actions": [str(x) for x in trace.action_names],
                    "prompt_actions": [ACTION_BANK[int(a)].prompt_name for a in trace.actions],
                    "cfg_scales": [float(ACTION_BANK[int(a)].cfg) for a in trace.actions],
                    "preview_rewards": [float(x) for x in trace.preview_rewards],
                }
            )
        gen_best_score, gen_best_genome, _, _ = scored[0]
        if gen_best_score > best_score:
            best_score = float(gen_best_score)
            best_genome = list(gen_best_genome)
            _, best_trace, _ = evaluate(
                best_genome,
                save_path=os.path.join(prompt_dir, "controller_ga_best.png"),
                tag="controller_ga_best",
            )
        gen_elapsed = time.perf_counter() - gen_start
        history.append(
            {
                "generation": int(gen),
                "best_score": float(gen_best_score),
                "mean_score": float(np.mean([s for s, _, _, _ in scored])),
                "eval_calls_total": int(eval_calls),
                "nfe_per_generation": int(len(scored) * args.steps),
                "nfe_total": int(eval_calls * args.steps),
                "wallclock_sec": float(gen_elapsed),
                "top": top_rows,
            }
        )
        if gen + 1 >= int(args.controller_generations):
            break

        use_rank_selection = str(args.controller_selection).lower() == "rank"
        next_population: list[list[float]] = [list(g) for _, g, _, _ in scored[:elites]]
        while len(next_population) < pop:
            scored_pairs = [(float(s), list(g)) for s, g, _, _ in scored]
            if use_rank_selection:
                ga = rank_select_float(scored_pairs, rng, float(args.controller_rank_pressure))
                gb = rank_select_float(scored_pairs, rng, float(args.controller_rank_pressure))
            else:
                ga = tournament_select_float(
                    scored_pairs,
                    int(args.controller_tournament_k),
                    rng,
                )
                gb = tournament_select_float(
                    scored_pairs,
                    int(args.controller_tournament_k),
                    rng,
                )
            child = crossover_float(ga, gb, rng)
            child = mutate_float(child, float(args.controller_mutation_prob), rng)
            next_population.append(child)
        population = next_population

    assert best_trace is not None
    payload = {
        "best_score": float(best_trace.final_score),
        "best_genome": [float(x) for x in best_genome],
        "best_params": asdict(decode_policy_genome(best_genome)),
        "best_actions": [str(x) for x in best_trace.action_names],
        "best_prompt_actions": [ACTION_BANK[int(a)].prompt_name for a in best_trace.actions],
        "best_cfg_scales": [float(ACTION_BANK[int(a)].cfg) for a in best_trace.actions],
        "selection": {
            "mode": str(args.controller_selection),
            "rank_pressure": float(args.controller_rank_pressure),
            "tournament_k": int(args.controller_tournament_k),
        },
        "eval_calls_total": int(eval_calls),
        "nfe_total": int(eval_calls * args.steps),
        "wallclock_total_sec": float(time.perf_counter() - ga_start),
        "history": history,
    }
    return best_trace, payload


def save_trace_json(path: str, trace: RolloutTrace) -> None:
    payload = {
        "final_score": float(trace.final_score),
        "actions": [int(a) for a in trace.actions],
        "action_names": [str(a) for a in trace.action_names],
        "preview_rewards": [float(x) for x in trace.preview_rewards],
        "preview_reward_changes": [float(x.delta_reward) for x in trace.step_records],
        "step_records": [asdict(x) for x in trace.step_records],
        "final_image_path": trace.final_image_path,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def save_score_board(path: str, rows: list[tuple[str, float]]) -> None:
    width = 1024
    row_h = 34
    header_h = 46
    img = Image.new("RGB", (width, header_h + row_h * max(1, len(rows))), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    draw.text((14, 10), "Method", fill=(220, 220, 220), font=_font(18))
    draw.text((480, 10), "Reward", fill=(220, 220, 220), font=_font(18))
    y = header_h
    for name, score in rows:
        draw.text((14, y + 6), name, fill=(190, 190, 190), font=_font(15))
        col = (120, 255, 120)
        draw.text((480, y + 6), f"{score:.4f}", fill=col, font=_font(15))
        y += row_h
    img.save(path)


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    if args.cuda_alloc_conf and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf
        print(f"Set PYTORCH_CUDA_ALLOC_CONF={args.cuda_alloc_conf}")

    cfg_scales = _unique_cfg_scales(args.cfg_scales, args.guidance_scale)
    init_action_bank(cfg_scales)
    print(
        f"Action space: {len(PROMPT_BLEND_BANK)} prompt actions x {len(cfg_scales)} cfg = {len(ACTION_BANK)}"
    )
    print(f"  cfg_scales={cfg_scales} baseline_cfg={float(args.baseline_cfg):.3f}")

    prompts = load_prompts(args)
    cache_path = args.hierarchy_cache_json if args.hierarchy_cache_json else os.path.join(args.out_dir, "hierarchy_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            hierarchy_cache = json.load(f)
    else:
        hierarchy_cache = {}

    ctx = su.load_pipeline(args)
    reward_ctx = su.load_reward(args, ctx)
    setattr(ctx, "pre_decode_hook", reward_ctx.before_decode)
    neg_embeds, neg_mask = su.load_neg_embed(args, ctx)

    all_summary: list[dict[str, Any]] = []
    for p_idx, prompt in enumerate(prompts):
        slug = f"p{p_idx:04d}"
        prompt_dir = os.path.join(args.out_dir, slug)
        os.makedirs(prompt_dir, exist_ok=True)
        print(f"\n{'=' * 72}\n[{slug}] {prompt}\n{'=' * 72}")

        hierarchy = build_hierarchy_prompt(args, prompt, hierarchy_cache)
        with open(os.path.join(prompt_dir, "hierarchy_prompts.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(hierarchy), f, indent=2, ensure_ascii=False)

        emb_cache = encode_hierarchy(args, ctx, hierarchy, neg_embeds, neg_mask)
        orig_h, orig_w = int(args.height), int(args.width)
        h, w = su.maybe_resize_to_bin(ctx, orig_h, orig_w, args.resolution_binning)
        print(f"  resolution requested={orig_h}x{orig_w}, effective={h}x{w}")

        # 1) original prompt only
        trace_original = run_rollout_original_prompt(
            args,
            ctx,
            reward_ctx,
            prompt,
            None,
            int(args.seed),
            h,
            w,
            orig_h,
            orig_w,
            emb_cache,
            save_path=os.path.join(prompt_dir, "baseline_original.png"),
            tag="baseline_original",
        )
        save_trace_json(os.path.join(prompt_dir, "trace_baseline_original.json"), trace_original)

        # 2) one-shot rewrite (mid only)
        one_shot_actions = [action_id_for("mid_only", float(args.baseline_cfg)) for _ in range(args.steps)]
        trace_one_shot = run_rollout_action_sequence(
            args,
            ctx,
            reward_ctx,
            prompt,
            None,
            int(args.seed),
            h,
            w,
            orig_h,
            orig_w,
            emb_cache,
            action_sequence=one_shot_actions,
            save_path=os.path.join(prompt_dir, "oneshot_mid.png"),
            tag="oneshot_mid",
        )
        save_trace_json(os.path.join(prompt_dir, "trace_oneshot_mid.json"), trace_one_shot)
        option_rewards_paths: dict[str, str] = {}
        if args.log_option_rewards:
            option_rows = analyze_option_rewards_along_path(
                args,
                ctx,
                reward_ctx,
                prompt,
                None,
                int(args.seed),
                h,
                w,
                orig_h,
                orig_w,
                emb_cache,
                one_shot_actions,
                tag="oneshot_mid_options",
            )
            op_path = os.path.join(prompt_dir, "option_rewards_oneshot_mid.json")
            with open(op_path, "w", encoding="utf-8") as f:
                json.dump(option_rows, f, indent=2)
            option_rewards_paths["oneshot_mid"] = op_path

        # 3) fixed coarse->mid->fine
        fixed_actions = default_fixed_schedule(args.steps, float(args.baseline_cfg))
        trace_fixed = run_rollout_action_sequence(
            args,
            ctx,
            reward_ctx,
            prompt,
            None,
            int(args.seed),
            h,
            w,
            orig_h,
            orig_w,
            emb_cache,
            action_sequence=fixed_actions,
            save_path=os.path.join(prompt_dir, "fixed_coarse_mid_fine.png"),
            tag="fixed_schedule",
        )
        save_trace_json(os.path.join(prompt_dir, "trace_fixed_schedule.json"), trace_fixed)
        if args.log_option_rewards:
            option_rows = analyze_option_rewards_along_path(
                args,
                ctx,
                reward_ctx,
                prompt,
                None,
                int(args.seed),
                h,
                w,
                orig_h,
                orig_w,
                emb_cache,
                fixed_actions,
                tag="fixed_schedule_options",
            )
            op_path = os.path.join(prompt_dir, "option_rewards_fixed_schedule.json")
            with open(op_path, "w", encoding="utf-8") as f:
                json.dump(option_rows, f, indent=2)
            option_rewards_paths["fixed_coarse_mid_fine"] = op_path

        # 4) open-loop GA over action sequence
        trace_openloop, openloop_payload = run_openloop_ga(
            args,
            ctx,
            reward_ctx,
            prompt,
            None,
            int(args.seed),
            h,
            w,
            orig_h,
            orig_w,
            emb_cache,
            prompt_dir,
        )
        save_trace_json(os.path.join(prompt_dir, "trace_openloop_ga_best.json"), trace_openloop)
        with open(os.path.join(prompt_dir, "openloop_ga_history.json"), "w", encoding="utf-8") as f:
            json.dump(openloop_payload, f, indent=2)
        if args.log_option_rewards:
            option_rows = analyze_option_rewards_along_path(
                args,
                ctx,
                reward_ctx,
                prompt,
                None,
                int(args.seed),
                h,
                w,
                orig_h,
                orig_w,
                emb_cache,
                [int(x) for x in trace_openloop.actions],
                tag="openloop_ga_options",
            )
            op_path = os.path.join(prompt_dir, "option_rewards_openloop_ga_best.json")
            with open(op_path, "w", encoding="utf-8") as f:
                json.dump(option_rows, f, indent=2)
            option_rewards_paths["openloop_ga_best"] = op_path

        # 5) closed-loop GA over controller parameters
        trace_controller, controller_payload = run_controller_ga(
            args,
            ctx,
            reward_ctx,
            prompt,
            None,
            int(args.seed),
            h,
            w,
            orig_h,
            orig_w,
            emb_cache,
            prompt_dir,
        )
        save_trace_json(os.path.join(prompt_dir, "trace_controller_ga_best.json"), trace_controller)
        with open(os.path.join(prompt_dir, "controller_ga_history.json"), "w", encoding="utf-8") as f:
            json.dump(controller_payload, f, indent=2)
        if args.log_option_rewards:
            option_rows = analyze_option_rewards_along_path(
                args,
                ctx,
                reward_ctx,
                prompt,
                None,
                int(args.seed),
                h,
                w,
                orig_h,
                orig_w,
                emb_cache,
                [int(x) for x in trace_controller.actions],
                tag="controller_ga_options",
            )
            op_path = os.path.join(prompt_dir, "option_rewards_controller_ga_best.json")
            with open(op_path, "w", encoding="utf-8") as f:
                json.dump(option_rows, f, indent=2)
            option_rewards_paths["controller_ga_best"] = op_path

        score_rows = [
            ("baseline_original", float(trace_original.final_score)),
            ("oneshot_mid", float(trace_one_shot.final_score)),
            ("fixed_coarse_mid_fine", float(trace_fixed.final_score)),
            ("openloop_ga_best", float(trace_openloop.final_score)),
            ("controller_ga_best", float(trace_controller.final_score)),
        ]
        save_score_board(os.path.join(prompt_dir, "score_board.png"), score_rows)
        baseline_score = float(trace_original.final_score)

        prompt_summary = {
            "slug": slug,
            "prompt": prompt,
            "hierarchy": asdict(hierarchy),
            "search_space": {
                "cfg_scales": [float(x) for x in cfg_scales],
                "n_prompt_actions": int(len(PROMPT_BLEND_BANK)),
                "n_total_actions": int(len(ACTION_BANK)),
                "baseline_cfg": float(args.baseline_cfg),
            },
            "scores": {name: score for name, score in score_rows},
            "score_deltas_vs_baseline": {name: float(score - baseline_score) for name, score in score_rows},
            "option_reward_logs": option_rewards_paths,
            "openloop_ga": openloop_payload,
            "controller_ga": controller_payload,
        }
        with open(os.path.join(prompt_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(prompt_summary, f, indent=2)
        all_summary.append(prompt_summary)
        print(
            "  scores: "
            f"orig={trace_original.final_score:.4f} "
            f"oneshot={trace_one_shot.final_score:.4f} "
            f"fixed={trace_fixed.final_score:.4f} "
            f"openloop={trace_openloop.final_score:.4f} "
            f"controller={trace_controller.final_score:.4f}"
        )

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(hierarchy_cache, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "sandbox_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2)

    print(f"\nDone. Summary: {os.path.join(args.out_dir, 'sandbox_summary.json')}")


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
