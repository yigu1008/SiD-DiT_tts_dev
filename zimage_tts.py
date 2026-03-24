"""
Test-time scaling pipeline for ZImage Turbo with phase-structured actions.

Main search space (CFG deprecated by default):
  - per-step prompt variant schedule
  - token-group weighting presets (selected phases)
  - scheduler micro-actions (early phase)
  - small latent kicks (middle phase)

Optional ablation:
  - tiny CFG set, disabled by default
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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

TOKEN_GROUP_ORDER = [
    "subject_face",
    "garment_embroidery",
    "prop_lightning_hand",
    "background_pagoda_lights",
]

TOKEN_GROUP_KEYWORDS: Dict[str, List[str]] = {
    "subject_face": [
        "woman",
        "girl",
        "lady",
        "female",
        "face",
        "portrait",
        "hanfu",
        "headdress",
        "hair",
        "bun",
    ],
    "garment_embroidery": [
        "garment",
        "dress",
        "robe",
        "hanfu",
        "embroidery",
        "fabric",
        "clothing",
        "makeup",
        "forehead",
        "pattern",
    ],
    "prop_lightning_hand": [
        "prop",
        "fan",
        "folding",
        "lightning",
        "bolt",
        "lamp",
        "hand",
        "palm",
        "pose",
    ],
    "background_pagoda_lights": [
        "background",
        "pagoda",
        "night",
        "city",
        "lights",
        "outdoor",
        "scene",
        "sky",
    ],
}

TOKEN_WEIGHT_PRESETS: List[Tuple[str, Tuple[float, float, float, float]]] = [
    ("neutral", (1.0, 1.0, 1.0, 1.0)),
    ("subject_boost", (1.2, 1.0, 0.95, 0.9)),
    ("prop_boost", (1.0, 0.95, 1.2, 0.9)),
    ("bg_boost", (1.0, 0.95, 0.95, 1.2)),
    ("detail_boost", (1.1, 1.2, 1.0, 0.9)),
]

PROMPT_VARIANT_LABELS = ["balanced", "subject", "prop", "background", "detail"]
PROMPT_FOCUS_SUFFIX: Dict[str, str] = {
    "balanced": "Keep the full composition balanced across subject, prop, and background.",
    "subject": "Focus priority: woman identity, face, hanfu, headdress.",
    "prop": "Focus priority: lightning-bolt prop, hand pose, and folding fan details.",
    "background": "Focus priority: pagoda silhouette, night atmosphere, and distant city lights.",
    "detail": "Focus priority: embroidery, makeup, and forehead floral pattern fidelity.",
}


@dataclass(frozen=True)
class StepAction:
    prompt_variant_id: int
    token_weight_preset_id: int
    schedule_action: str
    kick_id: int
    cfg_scale: float


@dataclass
class CandidateEval:
    image: Image.Image
    score: float
    intermediate_records: List[Dict[str, Any]]
    intermediate_images: List[Image.Image]
    expanded_schedule: List[StepAction]


@dataclass
class EmbeddingBank:
    base_cond_embeds: List[List[torch.Tensor]]
    neg_embeds: List[torch.Tensor]
    variant_texts: List[str]
    variant_labels: List[str]
    token_masks: Dict[int, Dict[str, torch.BoolTensor]]
    weighted_cache: Dict[Tuple[int, int], List[torch.Tensor]]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZImage Turbo test-time scaling with phase-structured actions.")
    parser.add_argument("--model", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--prompt", type=str, default="a cinematic portrait, soft rim light, 85mm, ultra detailed")
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--outdir", type=str, default="./zimage_tts_out")

    parser.add_argument("--search_method", choices=["base", "greedy", "mcts"], default="greedy")
    parser.add_argument("--n_sims", type=int, default=30)
    parser.add_argument("--ucb_c", type=float, default=1.41)

    parser.add_argument("--n_variants", type=int, default=3, help="Deprecated for Turbo structured variants.")
    parser.add_argument("--use_qwen_variants", action="store_true", help="Optional extra ablation variants from Qwen rewrites.")
    parser.add_argument("--no_qwen", action="store_true")
    parser.add_argument("--qwen_id", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--qwen_python", type=str, default="python3")
    parser.add_argument("--qwen_dtype", type=str, choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--rewrites_file", type=str, default=None)

    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8, help="Logical action steps (before microstep expansion).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--attention", type=str, default="", choices=["", "flash", "_flash_3"])
    parser.add_argument("--compile_transformer", action="store_true")

    parser.add_argument("--baseline_cfg", type=float, default=0.0, help="Turbo default baseline CFG.")
    parser.add_argument(
        "--cfg_scales",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="Deprecated for Turbo action space; kept for compatibility.",
    )
    parser.add_argument("--enable_cfg_ablation", action="store_true", help="Optional tiny CFG ablation in search.")
    parser.add_argument(
        "--cfg_ablation_scales",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="CFG ablation choices when --enable_cfg_ablation is set.",
    )

    parser.add_argument("--kick_eps", type=float, default=0.015, help="Latent kick magnitude.")
    parser.add_argument("--num_kick_dirs", type=int, default=2, help="Number of pre-sampled kick directions.")
    parser.add_argument("--allow_skip", action="store_true", help="Enable skip action as ablation.")
    parser.add_argument("--allow_repeat", action="store_true", help="Enable repeat action as ablation.")

    parser.add_argument("--reward_model", type=str, default="CodeGoat24/UnifiedReward-qwen-7b")
    parser.add_argument("--unifiedreward_model", type=str, default=None)
    parser.add_argument("--image_reward_model", type=str, default="ImageReward-v1.0")
    parser.add_argument("--pickscore_model", type=str, default="yuvalkirstain/PickScore_v1")
    parser.add_argument(
        "--reward_backend",
        type=str,
        choices=["auto", "unifiedreward", "unified", "imagereward", "pickscore", "hpsv2", "blend"],
        default="unifiedreward",
        help="Reward backend selector.",
    )
    parser.add_argument(
        "--reward_weights",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        help="Blend backend weights: imagereward hpsv2",
    )
    parser.add_argument("--reward_api_base", type=str, default=None, help="Optional OpenAI-compatible API base for UnifiedReward.")
    parser.add_argument("--reward_api_key", type=str, default="unifiedreward")
    parser.add_argument("--reward_api_model", type=str, default="UnifiedReward-7b-v1.5")
    parser.add_argument("--reward_max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--reward_prompt_mode",
        type=str,
        choices=["standard", "strict"],
        default="standard",
        help="UnifiedReward prompt template mode.",
    )

    parser.add_argument("--log_final_intermediates", action="store_true")
    parser.add_argument("--save_final_intermediate_images", action="store_true")
    parser.add_argument("--score_final_intermediates", action="store_true")
    return parser.parse_args(argv)


def get_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    return torch.float32


def load_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompt_file:
        prompts = [line.strip() for line in open(args.prompt_file, encoding="utf-8") if line.strip()]
    else:
        prompts = [args.prompt]
    if not prompts:
        raise RuntimeError("No prompts found.")
    return prompts


def sanitize_rewrite_text(candidate: str, fallback: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", str(candidate), flags=re.DOTALL).strip()
    if not text:
        return fallback
    text = text.strip("`\"' ")
    lower = text.lower()
    if lower in _REWRITE_BAD_TOKENS:
        return fallback
    if _REWRITE_PLACEHOLDER_RE.fullmatch(text):
        return fallback
    if "<" in text and ">" in text and len(text) < 24:
        return fallback
    if len(text) < 4:
        return fallback
    return text


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
    result = subprocess.run(
        [args.qwen_python, "-c", script, instruction, prompt],
        capture_output=True,
        text=True,
    )
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


def build_structured_variants(args: argparse.Namespace, prompt: str, cache: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    labels = list(PROMPT_VARIANT_LABELS)
    variants = [f"{prompt} {PROMPT_FOCUS_SUFFIX[label]}" if label != "balanced" else prompt for label in labels]

    if prompt in cache:
        qwen_variants = [sanitize_rewrite_text(v, prompt) for v in cache[prompt][: args.n_variants]]
    else:
        if not args.use_qwen_variants or args.no_qwen or args.n_variants <= 0:
            return labels, variants
        styles = (REWRITE_STYLES * ((args.n_variants // len(REWRITE_STYLES)) + 1))[: args.n_variants]
        qwen_variants = [sanitize_rewrite_text(qwen_rewrite(args, prompt, style), prompt) for style in styles]

    for i, v in enumerate(qwen_variants):
        if not v or v in variants:
            continue
        labels.append(f"qwen_{i}")
        variants.append(v)
    return labels, variants


def score_image(reward_scorer: UnifiedRewardScorer, prompt: str, image: Image.Image) -> float:
    return float(reward_scorer.score(prompt, image))


def decode_latents_to_pil(pipe: Any, latents: torch.Tensor) -> Image.Image:
    with torch.inference_mode():
        vae_param = next(pipe.vae.parameters())
        latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=-1.0)
        latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)
        scaling = getattr(pipe.vae.config, "scaling_factor", 1.0)
        shift = getattr(pipe.vae.config, "shift_factor", 0.0)
        decoded = pipe.vae.decode((latents / scaling) + shift, return_dict=False)[0]
        image = pipe.image_processor.postprocess(decoded, output_type="pil")
    return image[0]


def _font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _kick_name(kick_id: int) -> str:
    if kick_id == 0:
        return "none"
    idx = ((abs(kick_id) - 1) // 2) + 1
    sign = "+" if kick_id % 2 == 1 else "-"
    return f"{sign}u{idx}"


def action_brief(
    action: StepAction,
    variant_labels: Sequence[str],
    preset_names: Sequence[str],
    include_cfg: bool,
) -> str:
    label = variant_labels[action.prompt_variant_id]
    preset = preset_names[action.token_weight_preset_id]
    cfg_part = f"/cfg{action.cfg_scale:.2f}" if include_cfg else ""
    return f"{label}/{preset}/{action.schedule_action}/k={_kick_name(action.kick_id)}{cfg_part}"


def save_comparison(
    path: str,
    baseline_img: Image.Image,
    search_img: Image.Image,
    baseline_score: float,
    search_score: float,
    actions: List[StepAction],
    variant_labels: Sequence[str],
    preset_names: Sequence[str],
    include_cfg: bool,
) -> None:
    w, h = baseline_img.size
    hdr = 64
    comp = Image.new("RGB", (w * 2, h + hdr), (18, 18, 18))
    draw = ImageDraw.Draw(comp)
    comp.paste(baseline_img, (0, hdr))
    comp.paste(search_img, (w, hdr))
    draw.text((4, 4), f"baseline R={baseline_score:.3f}", fill=(200, 200, 200), font=_font(15))
    delta = search_score - baseline_score
    color = (100, 255, 100) if delta >= 0 else (255, 100, 100)
    draw.text((w + 4, 4), f"search R={search_score:.3f} delta={delta:+.3f}", fill=color, font=_font(15))
    acts = " ".join(f"s{i+1}:{action_brief(a, variant_labels, preset_names, include_cfg)}" for i, a in enumerate(actions))
    draw.text((w + 4, 30), acts[:140], fill=(255, 220, 50), font=_font(11))
    comp.save(path)


def make_grid(images: List[Image.Image], cols: int = 3) -> Image.Image:
    w, h = images[0].size
    rows = math.ceil(len(images) / cols)
    canvas = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, image in enumerate(images):
        canvas.paste(image, ((i % cols) * w, (i // cols) * h))
    return canvas


def _normalize_token(token: str) -> str:
    token = token.lower()
    token = token.replace("##", "").replace("▁", "").replace("Ġ", "").replace("</w>", "")
    token = re.sub(r"[^a-z0-9]+", "", token)
    return token


def _extract_token_masks(tokenizer: Any, text: str, max_sequence_length: int) -> Dict[str, torch.BoolTensor]:
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
        return_tensors="pt",
    )
    ids = enc["input_ids"][0].tolist()
    try:
        toks = tokenizer.convert_ids_to_tokens(ids)
    except Exception:
        toks = [tokenizer.decode([tid], skip_special_tokens=False) for tid in ids]
    seq = len(toks)
    masks: Dict[str, torch.BoolTensor] = {name: torch.zeros(seq, dtype=torch.bool) for name in TOKEN_GROUP_ORDER}
    for i, tok in enumerate(toks):
        norm_tok = _normalize_token(tok)
        if len(norm_tok) < 3:
            continue
        for group in TOKEN_GROUP_ORDER:
            if group not in TOKEN_GROUP_KEYWORDS:
                continue
            for kw in TOKEN_GROUP_KEYWORDS[group]:
                norm_kw = _normalize_token(kw)
                if len(norm_kw) < 3:
                    continue
                if norm_tok in norm_kw or norm_kw in norm_tok:
                    masks[group][i] = True
                    break
    return masks


def _get_primary_tokenizer(pipe: Any) -> Any:
    for name in ("tokenizer", "tokenizer_2", "tokenizer_3"):
        tok = getattr(pipe, name, None)
        if tok is not None:
            return tok
    return None


def _apply_token_weight_preset(
    base_embeds: List[torch.Tensor],
    masks: Dict[str, torch.BoolTensor],
    preset_id: int,
) -> List[torch.Tensor]:
    preset_name, weights = TOKEN_WEIGHT_PRESETS[preset_id]
    if preset_name == "neutral":
        return base_embeds
    out: List[torch.Tensor] = []
    for tensor in base_embeds:
        if not isinstance(tensor, torch.Tensor):
            out.append(tensor)
            continue
        x = tensor.clone()
        if x.ndim < 3:
            out.append(x)
            continue
        seq_len = x.shape[1]
        for gi, group in enumerate(TOKEN_GROUP_ORDER):
            mask = masks.get(group)
            if mask is None or int(mask.numel()) != int(seq_len):
                continue
            w = float(weights[gi])
            if abs(w - 1.0) < 1e-6:
                continue
            mask_dev = mask.to(device=x.device)
            if mask_dev.any():
                x[:, mask_dev, :] *= w
        out.append(x)
    return out


def build_embedding_bank(pipe: Any, variants: List[str], variant_labels: List[str], negative_prompt: str, max_sequence_length: int) -> EmbeddingBank:
    base_cond_embeds: List[List[torch.Tensor]] = []
    neg_embeds: Optional[List[torch.Tensor]] = None
    token_masks: Dict[int, Dict[str, torch.BoolTensor]] = {}
    tokenizer = _get_primary_tokenizer(pipe)

    for idx, variant in enumerate(variants):
        pe, ne = pipe.encode_prompt(
            prompt=variant,
            device=pipe._execution_device,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            max_sequence_length=max_sequence_length,
        )
        base_cond_embeds.append(pe)
        if neg_embeds is None:
            neg_embeds = ne
        if tokenizer is not None:
            try:
                token_masks[idx] = _extract_token_masks(tokenizer, variant, max_sequence_length)
            except Exception:
                token_masks[idx] = {}
        else:
            token_masks[idx] = {}
    assert neg_embeds is not None
    return EmbeddingBank(
        base_cond_embeds=base_cond_embeds,
        neg_embeds=neg_embeds,
        variant_texts=list(variants),
        variant_labels=list(variant_labels),
        token_masks=token_masks,
        weighted_cache={},
    )


def get_cond_embeds(bank: EmbeddingBank, variant_id: int, preset_id: int) -> List[torch.Tensor]:
    key = (variant_id, preset_id)
    if key in bank.weighted_cache:
        return bank.weighted_cache[key]
    base = bank.base_cond_embeds[variant_id]
    masks = bank.token_masks.get(variant_id, {})
    weighted = _apply_token_weight_preset(base, masks, preset_id)
    bank.weighted_cache[key] = weighted
    return weighted


def build_kick_bank(args: argparse.Namespace, pipe: Any, device: str, dtype: torch.dtype) -> List[torch.Tensor]:
    if args.num_kick_dirs <= 0:
        return []
    in_channels = int(getattr(pipe.transformer.config, "in_channels", 16))
    vae_scale = int(getattr(pipe, "vae_scale_factor", 8))
    lh = max(1, args.height // vae_scale)
    lw = max(1, args.width // vae_scale)
    sh = max(1, lh // 4)
    sw = max(1, lw // 4)
    gen = torch.Generator(device).manual_seed(args.seed + 12345)
    out: List[torch.Tensor] = []
    for _ in range(args.num_kick_dirs):
        low = torch.randn((1, in_channels, sh, sw), generator=gen, device=device, dtype=dtype)
        up = F.interpolate(low, size=(lh, lw), mode="bilinear", align_corners=False)
        up = up / (up.square().mean().sqrt() + 1e-6)
        out.append(up)
    return out


def apply_latent_kick(latents: torch.Tensor, kick_id: int, kick_bank: Sequence[torch.Tensor], eps: float) -> torch.Tensor:
    if kick_id == 0 or not kick_bank:
        return latents
    idx = (abs(kick_id) - 1) // 2
    if idx < 0 or idx >= len(kick_bank):
        return latents
    sign = 1.0 if (kick_id % 2 == 1) else -1.0
    direction = kick_bank[idx].to(device=latents.device, dtype=latents.dtype)
    if direction.shape[0] != latents.shape[0]:
        direction = direction.expand(latents.shape[0], -1, -1, -1)
    return latents + sign * float(eps) * direction


def normalize_action(action: StepAction) -> StepAction:
    return StepAction(
        prompt_variant_id=action.prompt_variant_id,
        token_weight_preset_id=action.token_weight_preset_id,
        schedule_action="normal",
        kick_id=action.kick_id,
        cfg_scale=float(action.cfg_scale),
    )


def expand_schedule(schedule: List[StepAction]) -> List[StepAction]:
    expanded: List[StepAction] = []
    for action in schedule:
        if action.schedule_action == "skip":
            continue
        norm = normalize_action(action)
        expanded.append(norm)
        if action.schedule_action in {"microstep", "repeat"}:
            expanded.append(StepAction(norm.prompt_variant_id, norm.token_weight_preset_id, "normal", 0, norm.cfg_scale))
    if not expanded and schedule:
        expanded.append(normalize_action(schedule[0]))
    return expanded


def schedule_key(schedule: List[StepAction]) -> Tuple[Tuple[int, int, str, int, float], ...]:
    return tuple(
        (
            int(a.prompt_variant_id),
            int(a.token_weight_preset_id),
            str(a.schedule_action),
            int(a.kick_id),
            float(round(a.cfg_scale, 6)),
        )
        for a in schedule
    )


def run_with_schedule(
    args: argparse.Namespace,
    pipe: Any,
    reward_scorer: UnifiedRewardScorer,
    prompt_for_reward: str,
    bank: EmbeddingBank,
    logical_schedule: List[StepAction],
    kick_bank: Sequence[torch.Tensor],
    seed: int,
    capture_intermediates: bool = False,
    save_intermediate_dir: Optional[str] = None,
    score_intermediates: bool = False,
) -> CandidateEval:
    if not logical_schedule:
        raise ValueError("schedule must be non-empty")
    schedule = expand_schedule(logical_schedule)
    if not schedule:
        raise ValueError("expanded schedule is empty")

    first = schedule[0]
    generator = torch.Generator("cuda").manual_seed(seed)
    step_records: List[Dict[str, Any]] = []
    step_images: List[Image.Image] = []

    if save_intermediate_dir:
        os.makedirs(save_intermediate_dir, exist_ok=True)

    def _on_step_end(_pipe, step_idx: int, timestep, callback_kwargs):
        if capture_intermediates and "latents" in callback_kwargs:
            image = decode_latents_to_pil(pipe, callback_kwargs["latents"])
            step_reward = None
            if score_intermediates:
                step_reward = score_image(reward_scorer, prompt_for_reward, image)
            record = {
                "step_idx": int(step_idx),
                "timestep": float(timestep) if hasattr(timestep, "__float__") else str(timestep),
                "reward": step_reward,
            }
            step_records.append(record)
            label = "n/a" if step_reward is None else f"{step_reward:.4f}"
            canvas = Image.new("RGB", (image.size[0], image.size[1] + 36), (255, 255, 255))
            canvas.paste(image, (0, 36))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 10), f"step={step_idx} t={record['timestep']} R={label}", fill=(0, 0, 0), font=_font(14))
            step_images.append(canvas)
            if save_intermediate_dir:
                image.save(os.path.join(save_intermediate_dir, f"step_{int(step_idx):03d}.png"))

        next_idx = step_idx + 1
        if next_idx < len(schedule):
            nxt = schedule[next_idx]
            _pipe._guidance_scale = float(nxt.cfg_scale)
            callback_kwargs["prompt_embeds"] = get_cond_embeds(bank, nxt.prompt_variant_id, nxt.token_weight_preset_id)
            callback_kwargs["negative_prompt_embeds"] = bank.neg_embeds
            if nxt.kick_id != 0 and "latents" in callback_kwargs:
                callback_kwargs["latents"] = apply_latent_kick(
                    callback_kwargs["latents"],
                    nxt.kick_id,
                    kick_bank,
                    args.kick_eps,
                )
        return callback_kwargs

    kwargs: Dict[str, Any] = {
        "prompt": None,
        "prompt_embeds": get_cond_embeds(bank, first.prompt_variant_id, first.token_weight_preset_id),
        "negative_prompt_embeds": bank.neg_embeds,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": len(schedule),
        "guidance_scale": float(first.cfg_scale),
        "generator": generator,
        "max_sequence_length": args.max_sequence_length,
        "output_type": "pil",
        "callback_on_step_end": _on_step_end,
        "callback_on_step_end_tensor_inputs": ["latents", "prompt_embeds"],
    }
    with torch.inference_mode():
        output = pipe(**kwargs)
    final_image = output.images[0]
    final_score = score_image(reward_scorer, prompt_for_reward, final_image)
    return CandidateEval(
        image=final_image,
        score=final_score,
        intermediate_records=step_records,
        intermediate_images=step_images,
        expanded_schedule=schedule,
    )


def phase_for_step(step_idx: int, total_steps: int) -> str:
    early_end = max(1, total_steps // 3)
    late_start = max(early_end + 1, (2 * total_steps) // 3)
    if step_idx < early_end:
        return "early"
    if step_idx < late_start:
        return "middle"
    return "late"


def default_preset_for_label(label: str) -> int:
    by_name = {name: idx for idx, (name, _) in enumerate(TOKEN_WEIGHT_PRESETS)}
    if label == "subject":
        return by_name["subject_boost"]
    if label == "prop":
        return by_name["prop_boost"]
    if label == "background":
        return by_name["bg_boost"]
    if label == "detail":
        return by_name["detail_boost"]
    return by_name["neutral"]


def _prompt_ids_for_phase(phase: str, variant_id_by_label: Dict[str, int]) -> List[int]:
    if phase == "early":
        names = ["balanced", "subject", "background"]
    elif phase == "middle":
        names = ["subject", "prop", "background"]
    else:
        names = ["detail", "balanced", "prop"]
    return [variant_id_by_label[name] for name in names if name in variant_id_by_label]


def _kick_choices_for_phase(phase: str, num_kick_dirs: int) -> List[int]:
    if phase != "middle" or num_kick_dirs <= 0:
        return [0]
    out = [0]
    for i in range(num_kick_dirs):
        out.extend([2 * i + 1, 2 * i + 2])  # +ui, -ui
    return out


def build_step_action_candidates(
    args: argparse.Namespace,
    step_idx: int,
    total_steps: int,
    variant_labels: Sequence[str],
) -> List[StepAction]:
    phase = phase_for_step(step_idx, total_steps)
    variant_id_by_label = {label: i for i, label in enumerate(variant_labels)}
    prompt_ids = _prompt_ids_for_phase(phase, variant_id_by_label)
    if not prompt_ids:
        prompt_ids = [0]

    cfg_choices = [float(args.baseline_cfg)]
    if args.enable_cfg_ablation:
        cfg_choices = [float(v) for v in args.cfg_ablation_scales]

    by_name = {name: idx for idx, (name, _) in enumerate(TOKEN_WEIGHT_PRESETS)}
    neutral = by_name["neutral"]
    candidates: List[StepAction] = []

    if phase == "early":
        schedule_choices = ["normal", "microstep"]
        if args.allow_repeat:
            schedule_choices.append("repeat")
        if args.allow_skip:
            schedule_choices.append("skip")
        for pid in prompt_ids:
            for sch in schedule_choices:
                for cfg in cfg_choices:
                    candidates.append(StepAction(pid, neutral, sch, 0, cfg))
    elif phase == "middle":
        kick_choices = _kick_choices_for_phase(phase, args.num_kick_dirs)
        for pid in prompt_ids:
            label = variant_labels[pid]
            boosted = default_preset_for_label(label)
            for cfg in cfg_choices:
                for kick in kick_choices:
                    candidates.append(StepAction(pid, neutral, "normal", kick, cfg))
                if boosted != neutral:
                    candidates.append(StepAction(pid, boosted, "normal", 0, cfg))
    else:  # late
        for pid in prompt_ids:
            label = variant_labels[pid]
            boosted = default_preset_for_label(label)
            for cfg in cfg_choices:
                candidates.append(StepAction(pid, neutral, "normal", 0, cfg))
                if boosted != neutral:
                    candidates.append(StepAction(pid, boosted, "normal", 0, cfg))

    seen = set()
    deduped: List[StepAction] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def build_action_space(args: argparse.Namespace, variant_labels: Sequence[str]) -> List[List[StepAction]]:
    return [build_step_action_candidates(args, s, args.steps, variant_labels) for s in range(args.steps)]


def greedy_search(
    args: argparse.Namespace,
    pipe: Any,
    reward_scorer: UnifiedRewardScorer,
    prompt: str,
    bank: EmbeddingBank,
    action_space: List[List[StepAction]],
    kick_bank: Sequence[torch.Tensor],
) -> List[StepAction]:
    cache: Dict[Tuple[int, Tuple[Tuple[int, int, str, int, float], ...]], float] = {}
    chosen: List[StepAction] = []
    preset_names = [name for name, _ in TOKEN_WEIGHT_PRESETS]
    include_cfg = args.enable_cfg_ablation

    def eval_prefix(prefix: List[StepAction]) -> float:
        key = (len(prefix), schedule_key(prefix))
        if key in cache:
            return cache[key]
        result = run_with_schedule(
            args=args,
            pipe=pipe,
            reward_scorer=reward_scorer,
            prompt_for_reward=prompt,
            bank=bank,
            logical_schedule=prefix,
            kick_bank=kick_bank,
            seed=args.seed,
            capture_intermediates=False,
            save_intermediate_dir=None,
            score_intermediates=False,
        )
        cache[key] = result.score
        return result.score

    for step_idx in range(args.steps):
        candidates = action_space[step_idx]
        print(f"  greedy step {step_idx + 1}/{args.steps} ({len(candidates)} actions)")
        best_action = candidates[0]
        best_score = -float("inf")
        for action in candidates:
            score = eval_prefix(chosen + [action])
            marker = ""
            if score > best_score:
                best_score = score
                best_action = action
                marker = " <- best"
            label = action_brief(action, bank.variant_labels, preset_names, include_cfg)
            print(f"    {label} R={score:.4f}{marker}")
        chosen.append(best_action)
        picked = action_brief(best_action, bank.variant_labels, preset_names, include_cfg)
        print(f"  selected step {step_idx + 1}: {picked} score={best_score:.4f}")
    return chosen


class MCTSNode:
    __slots__ = ("depth", "children", "n", "action_n", "action_q")

    def __init__(self, depth: int):
        self.depth = depth
        self.children: Dict[StepAction, "MCTSNode"] = {}
        self.n = 0
        self.action_n: Dict[StepAction, int] = {}
        self.action_q: Dict[StepAction, float] = {}

    def untried(self, actions: List[StepAction]) -> List[StepAction]:
        return [a for a in actions if a not in self.action_n]

    def ucb(self, action: StepAction, c: float) -> float:
        n = self.action_n.get(action, 0)
        if n == 0:
            return float("inf")
        mean = self.action_q[action] / n
        return mean + c * math.sqrt(math.log(max(self.n, 1)) / n)

    def best_ucb(self, actions: List[StepAction], c: float) -> StepAction:
        return max(actions, key=lambda action: self.ucb(action, c))

    def best_exploit(self, actions: List[StepAction]) -> Optional[StepAction]:
        best_action = None
        best_value = -float("inf")
        for action in actions:
            n = self.action_n.get(action, 0)
            if n <= 0:
                continue
            value = self.action_q[action] / n
            if value > best_value:
                best_value = value
                best_action = action
        return best_action


def mcts_search(
    args: argparse.Namespace,
    pipe: Any,
    reward_scorer: UnifiedRewardScorer,
    prompt: str,
    bank: EmbeddingBank,
    action_space: List[List[StepAction]],
    kick_bank: Sequence[torch.Tensor],
) -> List[StepAction]:
    root = MCTSNode(depth=0)
    score_cache: Dict[Tuple[Tuple[int, int, str, int, float], ...], float] = {}

    def eval_schedule(schedule: List[StepAction]) -> float:
        key = schedule_key(schedule)
        if key in score_cache:
            return score_cache[key]
        result = run_with_schedule(
            args=args,
            pipe=pipe,
            reward_scorer=reward_scorer,
            prompt_for_reward=prompt,
            bank=bank,
            logical_schedule=schedule,
            kick_bank=kick_bank,
            seed=args.seed,
            capture_intermediates=False,
            save_intermediate_dir=None,
            score_intermediates=False,
        )
        score_cache[key] = result.score
        return result.score

    best_score = -float("inf")
    best_schedule: List[StepAction] = []
    print(f"  mcts sims={args.n_sims} steps={args.steps}")
    for sim in range(args.n_sims):
        node = root
        path: List[Tuple[MCTSNode, StepAction]] = []
        schedule: List[StepAction] = []

        while node.depth < args.steps:
            actions = action_space[node.depth]
            untried = node.untried(actions)
            if untried:
                action = untried[np.random.randint(len(untried))]
                break
            action = node.best_ucb(actions, args.ucb_c)
            path.append((node, action))
            schedule.append(action)
            node = node.children[action]

        if node.depth < args.steps:
            if action not in node.children:
                node.children[action] = MCTSNode(depth=node.depth + 1)
            path.append((node, action))
            schedule.append(action)
            node = node.children[action]

        while len(schedule) < args.steps:
            actions = action_space[len(schedule)]
            schedule.append(actions[np.random.randint(len(actions))])

        score = eval_schedule(schedule)
        if score > best_score:
            best_score = score
            best_schedule = list(schedule)

        for pnode, paction in path:
            pnode.n += 1
            pnode.action_n[paction] = pnode.action_n.get(paction, 0) + 1
            pnode.action_q[paction] = pnode.action_q.get(paction, 0.0) + score

        if (sim + 1) % 5 == 0 or sim == 0:
            print(f"    sim {sim + 1:3d}/{args.n_sims} best_R={best_score:.4f}")

    exploit: List[StepAction] = []
    node = root
    for depth in range(args.steps):
        action = node.best_exploit(action_space[depth])
        if action is None:
            break
        exploit.append(action)
        if action in node.children:
            node = node.children[action]
        else:
            break
    if len(exploit) < args.steps:
        exploit.extend(best_schedule[len(exploit) :])
    return exploit if exploit else best_schedule


def write_intermediate_logs(outdir: str, schedule_tag: str, records: List[Dict[str, Any]], images: List[Image.Image]) -> None:
    if records:
        stats_path = os.path.join(outdir, f"{schedule_tag}_intermediate_stats.txt")
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write("step_idx\ttimestep\treward\n")
            for rec in records:
                rv = rec["reward"]
                rv_text = f"{rv:.8f}" if rv is not None else "nan"
                f.write(f"{rec['step_idx']}\t{rec['timestep']}\t{rv_text}\n")
    if images:
        grid = make_grid(images, cols=min(3, len(images)))
        grid.save(os.path.join(outdir, f"{schedule_tag}_intermediate_grid.png"))


def action_to_json(action: StepAction, variant_labels: Sequence[str], preset_names: Sequence[str]) -> Dict[str, Any]:
    return {
        "prompt_variant_id": int(action.prompt_variant_id),
        "prompt_variant_label": variant_labels[action.prompt_variant_id],
        "token_weight_preset_id": int(action.token_weight_preset_id),
        "token_weight_preset_label": preset_names[action.token_weight_preset_id],
        "schedule_action": str(action.schedule_action),
        "kick_id": int(action.kick_id),
        "kick_label": _kick_name(int(action.kick_id)),
        "cfg_scale": float(action.cfg_scale),
    }


def load_zimage_pipeline(model_id: str, dtype: torch.dtype, device: str) -> Tuple[Any, str]:
    loader_errors: List[str] = []

    try:
        from diffusers import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        ).to(device)
        return pipe, "ZImagePipeline"
    except Exception as exc:
        loader_errors.append(f"ZImagePipeline: {exc}")

    try:
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        ).to(device)
        return pipe, "DiffusionPipeline(trust_remote_code=True)"
    except Exception as exc:
        loader_errors.append(f"DiffusionPipeline: {exc}")

    try:
        from diffusers import AutoPipelineForText2Image

        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        ).to(device)
        return pipe, "AutoPipelineForText2Image(trust_remote_code=True)"
    except Exception as exc:
        loader_errors.append(f"AutoPipelineForText2Image: {exc}")

    detail = "\n  - ".join(loader_errors)
    raise RuntimeError(
        "Unable to load Z-Image pipeline with current diffusers build.\n"
        "Tried:\n"
        f"  - {detail}\n"
        "Use a separate env for latest diffusers or Tongyi-MAI/Z-Image native inference."
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda"
    dtype = get_dtype(args.dtype)
    print(f"Loading ZImage pipeline: {args.model}")
    pipe, loader_name = load_zimage_pipeline(args.model, dtype, device)
    print(f"Pipeline loader: {loader_name}")
    if args.attention:
        pipe.transformer.set_attention_backend(args.attention)
    if args.compile_transformer:
        pipe.transformer.compile()

    unified_model = args.unifiedreward_model if args.unifiedreward_model else args.reward_model
    reward_scorer = UnifiedRewardScorer(
        device=device,
        backend=args.reward_backend,
        image_reward_model=args.image_reward_model,
        pickscore_model=args.pickscore_model,
        unifiedreward_model=unified_model,
        unified_weights=(float(args.reward_weights[0]), float(args.reward_weights[1])),
        unifiedreward_api_base=args.reward_api_base,
        unifiedreward_api_key=args.reward_api_key,
        unifiedreward_api_model=args.reward_api_model,
        max_new_tokens=int(args.reward_max_new_tokens),
        unifiedreward_prompt_mode=args.reward_prompt_mode,
    )
    print(f"Reward: {reward_scorer.describe()}")

    if not args.enable_cfg_ablation:
        print("CFG is deprecated for Turbo search; using fixed baseline_cfg unless --enable_cfg_ablation is set.")

    prompts = load_prompts(args)
    rewrite_cache: Dict[str, List[str]] = {}
    if args.rewrites_file and os.path.exists(args.rewrites_file):
        rewrite_cache = json.load(open(args.rewrites_file, encoding="utf-8"))

    preset_names = [name for name, _ in TOKEN_WEIGHT_PRESETS]
    summary: List[Dict[str, Any]] = []

    for pidx, prompt in enumerate(prompts):
        slug = f"p{pidx:02d}"
        print(f"\n{'='*72}\n[{slug}] {prompt}\n{'='*72}")

        variant_labels, variants = build_structured_variants(args, prompt, rewrite_cache)
        with open(os.path.join(args.outdir, f"{slug}_variants.txt"), "w", encoding="utf-8") as f:
            for vi, text in enumerate(variants):
                f.write(f"v{vi}[{variant_labels[vi]}]: {text}\n")

        bank = build_embedding_bank(
            pipe=pipe,
            variants=variants,
            variant_labels=variant_labels,
            negative_prompt=args.negative_prompt,
            max_sequence_length=args.max_sequence_length,
        )
        action_space = build_action_space(args, variant_labels)
        kick_bank = build_kick_bank(args, pipe, device=device, dtype=dtype)

        variant_id_by_label = {label: i for i, label in enumerate(variant_labels)}
        balanced_id = variant_id_by_label.get("balanced", 0)
        neutral_id = 0
        baseline_schedule = [
            StepAction(
                prompt_variant_id=balanced_id,
                token_weight_preset_id=neutral_id,
                schedule_action="normal",
                kick_id=0,
                cfg_scale=float(args.baseline_cfg),
            )
            for _ in range(args.steps)
        ]

        baseline_result = run_with_schedule(
            args=args,
            pipe=pipe,
            reward_scorer=reward_scorer,
            prompt_for_reward=prompt,
            bank=bank,
            logical_schedule=baseline_schedule,
            kick_bank=kick_bank,
            seed=args.seed,
            capture_intermediates=False,
            save_intermediate_dir=None,
            score_intermediates=False,
        )

        if args.search_method == "base":
            chosen_schedule = list(baseline_schedule)
        elif args.search_method == "greedy":
            chosen_schedule = greedy_search(args, pipe, reward_scorer, prompt, bank, action_space, kick_bank)
        else:
            chosen_schedule = mcts_search(args, pipe, reward_scorer, prompt, bank, action_space, kick_bank)

        inter_dir = os.path.join(args.outdir, f"{slug}_{args.search_method}_steps") if args.save_final_intermediate_images else None
        search_result = run_with_schedule(
            args=args,
            pipe=pipe,
            reward_scorer=reward_scorer,
            prompt_for_reward=prompt,
            bank=bank,
            logical_schedule=chosen_schedule,
            kick_bank=kick_bank,
            seed=args.seed,
            capture_intermediates=args.log_final_intermediates,
            save_intermediate_dir=inter_dir,
            score_intermediates=args.score_final_intermediates,
        )

        baseline_path = os.path.join(args.outdir, f"{slug}_baseline.png")
        search_path = os.path.join(args.outdir, f"{slug}_{args.search_method}.png")
        comp_path = os.path.join(args.outdir, f"{slug}_comparison.png")
        baseline_result.image.save(baseline_path)
        search_result.image.save(search_path)
        save_comparison(
            comp_path,
            baseline_result.image,
            search_result.image,
            baseline_result.score,
            search_result.score,
            chosen_schedule,
            variant_labels=variant_labels,
            preset_names=preset_names,
            include_cfg=args.enable_cfg_ablation,
        )

        if args.log_final_intermediates:
            write_intermediate_logs(
                outdir=args.outdir,
                schedule_tag=f"{slug}_{args.search_method}",
                records=search_result.intermediate_records,
                images=search_result.intermediate_images,
            )

        print(
            f"baseline={baseline_result.score:.4f} "
            f"{args.search_method}={search_result.score:.4f} "
            f"delta={search_result.score - baseline_result.score:+.4f}"
        )

        summary.append(
            {
                "slug": slug,
                "prompt": prompt,
                "variants": [{"label": variant_labels[i], "text": variants[i]} for i in range(len(variants))],
                "baseline_reward": float(baseline_result.score),
                f"{args.search_method}_reward": float(search_result.score),
                "delta_reward": float(search_result.score - baseline_result.score),
                "baseline_schedule": [action_to_json(a, variant_labels, preset_names) for a in baseline_schedule],
                "chosen_schedule": [action_to_json(a, variant_labels, preset_names) for a in chosen_schedule],
                "chosen_expanded_schedule": [action_to_json(a, variant_labels, preset_names) for a in search_result.expanded_schedule],
            }
        )

    summary_path = os.path.join(args.outdir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    for row in summary:
        print(f"{row['slug']} delta_reward={row['delta_reward']:+.4f}")
    if summary:
        print(f"mean delta={float(np.mean([row['delta_reward'] for row in summary])):+.4f}")
    print(f"summary json: {summary_path}")


if __name__ == "__main__":
    main()
