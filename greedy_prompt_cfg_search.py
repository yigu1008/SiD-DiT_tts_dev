"""
greedy_prompt_cfg_search.py
===========================
At each SiD denoising step, greedily pick the best (prompt_variant, cfg_scale)
pair by scoring the one-step x̂₀ prediction with ImageReward.

Prompt variants are minimal-change rewrites generated once by Qwen3-4B
(lighting, lens, mood tweaks) — same variants reused across all steps.
Only the selected (variant, cfg_scale) action changes per step.

Key improvements over naïve sequential loop
--------------------------------------------
• All n_v × n_cfg candidates are evaluated in a SINGLE batched transformer
  forward at each step (batch = total candidates × 2 for CFG), not N loops.
• VAE decode is batched over all candidates at once.
• Uncond shared across all candidates; only cond changes per variant.
• timestep_scale read from transformer.config (not hardcoded to 1000).
• --max_batch cap prevents OOM (splits into sub-batches transparently).
• --no_qwen flag for cfg-scale-only ablation without loading Qwen3.

Action space per step (defaults):
    variants   : [original] + 3 Qwen3 rewrites  →  4 total
    cfg_scales : [1.0 1.25 1.5 1.75 2.0 2.25 2.5] → 7 total
    candidates : 4 × 7 = 28  (1 batched forward per step)

Usage
-----
python greedy_prompt_cfg_search.py \\
    --prompt "a studio portrait of an elderly woman, soft window light" \\
    --neg_embed neg_embed_ckpts/neg_embed_best.pt

python greedy_prompt_cfg_search.py \\
    --prompt_file prompts.txt \\
    --n_variants 3 --steps 4 \\
    --cfg_scales 1.0 1.5 2.0 2.5 \\
    --out_dir ./greedy_out

# cfg-scale search only (no Qwen3)
python greedy_prompt_cfg_search.py \\
    --prompt_file prompts.txt --no_qwen \\
    --cfg_scales 1.0 1.5 2.0 3.0 4.5
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_id",    default="YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow")
parser.add_argument("--ckpt",        default=None,
                    help="Path to .pt_G checkpoint. If set, overwrites the "
                         "transformer weights loaded from --model_id.")
parser.add_argument("--qwen_id",     default="Qwen/Qwen3-4B")
parser.add_argument("--prompt",      default="a studio portrait of an elderly woman smiling, "
                                              "soft window light, 85mm lens, photorealistic")
parser.add_argument("--prompt_file", default=None)
parser.add_argument("--neg_embed",   default=None)
parser.add_argument("--n_variants",  type=int,   default=3)
parser.add_argument("--steps",       type=int,   default=4)
parser.add_argument("--cfg_scales",  nargs="+",  type=float,
                    default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5])
parser.add_argument("--seed",        type=int,   default=42)
parser.add_argument("--width",       type=int,   default=512)
parser.add_argument("--height",      type=int,   default=512)
parser.add_argument("--out_dir",     default="./greedy_prompt_out_cfg")
parser.add_argument("--qwen_device", default="auto")
parser.add_argument("--qwen_python", default="python3",
                    help="Python executable for Qwen3 subprocess. "
                         "E.g. /home/ygu/envs/qwen3_env/bin/python")
parser.add_argument("--qwen_dtype",  default="bfloat16",
                    choices=["float16", "bfloat16"])
parser.add_argument("--max_batch",   type=int,   default=28,
                    help="Max candidates per batched forward. Reduce if OOM.")
parser.add_argument("--rewrites_file", default=None,
                    help="JSON file mapping prompt -> [variants] from pre-generated rewrites.")
parser.add_argument("--no_qwen",     action="store_true",
                    help="Skip Qwen3 rewrites; search over cfg_scales only.")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
sid_device = "cuda" if torch.cuda.is_available() else "cpu"
sid_dtype  = torch.float16
qwen_dtype = torch.bfloat16 if args.qwen_dtype == "bfloat16" else torch.float16


# ── SiD pipeline ───────────────────────────────────────────────────────────────
repo_root = Path(__file__).parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from sid import SiDSanaPipeline
except ImportError as e:
    sys.exit(f"Cannot import SiDSanaPipeline: {e}\n"
             "Run from the cloned SiD-DiT Space repo.")

from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
    ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN)
try:
    from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import (
        ASPECT_RATIO_2048_BIN)
except ImportError:
    ASPECT_RATIO_2048_BIN = ASPECT_RATIO_1024_BIN

ASPECT_RATIO_BINS = {16: ASPECT_RATIO_512_BIN,
                     32: ASPECT_RATIO_1024_BIN,
                     64: ASPECT_RATIO_2048_BIN}

print("Loading SiD pipeline ...")
pipe = SiDSanaPipeline.from_pretrained(
    args.model_id, torch_dtype=sid_dtype).to(sid_device)

if args.ckpt:
    print(f"Loading weights from {args.ckpt} ...")
    raw = torch.load(args.ckpt, map_location=sid_device, weights_only=False)

    def _unwrap(d, depth=0):
        """Recursively unwrap known wrapper keys until we hit a real state dict."""
        if not isinstance(d, dict):
            return d
        dotted = sum(1 for k in d if "." in str(k))
        if dotted / max(len(d), 1) > 0.5:
            return d
        if depth > 4:
            return d
        for key in ("ema", "ema_model", "model_ema", "model",
                    "state_dict", "generator", "G_state"):
            if key in d and isinstance(d[key], dict):
                print(f"  {'  '*depth}unwrapping '{key}' ({len(d[key])} keys)")
                return _unwrap(d[key], depth + 1)
        return d

    sd = _unwrap(raw)
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = pipe.transformer.load_state_dict(sd, strict=False)
    if missing:
        print(f"  missing  {len(missing)} keys (first 3): {missing[:3]}")
    if unexpected:
        print(f"  unexpected {len(unexpected)} keys (first 3): {unexpected[:3]}")
    print(f"  ✓ loaded {len(sd) - len(unexpected)}/{len(sd)} params")

pipe.transformer.eval()
TIME_SCALE = 1000.0   # SANA base timestep embedding scale
LATENT_C   = pipe.transformer.config.in_channels
OUT_C      = pipe.transformer.config.out_channels
VARIANCE_SPLIT = (OUT_C // 2 == LATENT_C)
print(f"✓ SiD pipeline  TIME_SCALE={TIME_SCALE}  variance_split={VARIANCE_SPLIT}")


# ── ImageReward ───────────────────────────────────────────────────────────────
print("Loading ImageReward ...")
import ImageReward as RM
reward_model = RM.load("ImageReward-v1.0", device=sid_device)
reward_model.eval()
print("✓ ImageReward")


# ── Qwen3-4B ──────────────────────────────────────────────────────────────────
# Qwen3 is called via subprocess in qwen_rewrite(), so we do NOT load it
# into this process. Just set a flag for the gate check in generate_variants.
qwen_available = not args.no_qwen


# ── Prompt rewrites ────────────────────────────────────────────────────────────
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


def qwen_rewrite(prompt: str, instruction: str) -> str:
    """Call Qwen3 via subprocess to avoid transformers version conflicts."""
    import subprocess, json, sys
    qwen_python = args.qwen_python  # path to python in qwen3 venv
    script = f"""
import sys, re, json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained({repr(args.qwen_id)})
model = AutoModelForCausalLM.from_pretrained(
    {repr(args.qwen_id)}, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()
SYSTEM = {repr(REWRITE_SYSTEM)}
user_msg = sys.argv[1] + "\n\nOriginal prompt: " + sys.argv[2] + " /no_think"
messages = [{{"role":"system","content":SYSTEM}},
            {{"role":"user","content":user_msg}}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=120,
        temperature=0.6, top_p=0.9, do_sample=True,
        pad_token_id=tokenizer.eos_token_id)
new_toks = out[0][inputs.input_ids.shape[1]:]
result = tokenizer.decode(new_toks, skip_special_tokens=True).strip()
result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
for line in result.splitlines():
    line = line.strip()
    if line:
        print(line); sys.exit(0)
print(sys.argv[2])
"""
    r = subprocess.run(
        [qwen_python, "-c", script, instruction, prompt],
        capture_output=True, text=True)
    result = r.stdout.strip()
    return result if result else prompt


# Load pre-generated rewrites if provided
_rewrites_cache: dict = {}
if not args.no_qwen and getattr(args, "rewrites_file", None):
    import json as _json
    _rewrites_cache = _json.load(open(args.rewrites_file))
    print(f"Loaded rewrites for {len(_rewrites_cache)} prompts from {args.rewrites_file}")

def generate_variants(prompt: str, n: int) -> list[str]:
    # Use pre-generated rewrites if available
    if prompt in _rewrites_cache:
        variants = _rewrites_cache[prompt][:n+1]
        for i, v in enumerate(variants):
            print(f"  variant {i}: {v}")
        return variants
    if args.no_qwen or not qwen_available:
        return [prompt]
    variants = [prompt]
    styles   = (REWRITE_STYLES * ((n // len(REWRITE_STYLES)) + 1))[:n]
    for instr in styles:
        v = qwen_rewrite(prompt, instr)
        variants.append(v)
        print(f"  variant {len(variants)-1}: {v}")
    return variants


# ── Neg embed loading ──────────────────────────────────────────────────────────
pretrained_neg_embeds = None
pretrained_neg_mask   = None

if args.neg_embed:
    ckpt = torch.load(args.neg_embed, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "neg_embeds" in ckpt:
        pretrained_neg_embeds = ckpt["neg_embeds"].to(sid_device, sid_dtype)
        pretrained_neg_mask   = ckpt["neg_mask"].to(sid_device)
    else:
        pretrained_neg_embeds = ckpt.to(sid_device, sid_dtype)
        pretrained_neg_mask   = torch.ones(
            pretrained_neg_embeds.shape[:2], device=sid_device, dtype=torch.long)
    print(f"✓ Neg embed: {pretrained_neg_embeds.shape}")


# ── Encode variants ────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_variants(variants: list[str], max_seq: int = 256):
    """
    Returns:
        pe_list : [(embeds [1,L,D], mask [1,L]), ...]
        ue, um  : uncond embeds + mask (null-text or pretrained neg embed)
    """
    pe_list = []
    ue = um = None

    for i, v in enumerate(variants):
        pe, pm, _ne, _nm = pipe.encode_prompt(
            prompt=v,
            do_classifier_free_guidance=True,
            negative_prompt="",
            device=sid_device,
            num_images_per_prompt=1,
            max_sequence_length=max_seq,
        )
        pe_list.append((pe.detach(), pm.detach()))
        if i == 0:
            ue, um = _ne.detach(), _nm.detach()

    # Override uncond with pretrained neg embed
    if pretrained_neg_embeds is not None:
        L_c  = pe_list[0][0].shape[1]
        L_n  = pretrained_neg_embeds.shape[1]
        pne  = pretrained_neg_embeds
        pnm  = pretrained_neg_mask
        if L_n < L_c:
            pne = torch.cat([pne, torch.zeros(
                1, L_c-L_n, pne.shape[2], device=sid_device, dtype=pne.dtype)], 1)
            pnm = torch.cat([pnm, torch.zeros(
                1, L_c-L_n, device=sid_device, dtype=pnm.dtype)], 1)
        elif L_n > L_c:
            pne, pnm = pne[:, :L_c], pnm[:, :L_c]
        ue = pne.to(pe_list[0][0].dtype)
        um = pnm

    return pe_list, ue, um


# ── Batched CFG forward ────────────────────────────────────────────────────────
@torch.no_grad()
def batched_cfg_forward(latents, pe_list, ue, um, cfg_scales, t_flat, max_batch):
    """
    Evaluate all (variant × cfg_scale) candidates in batched transformer calls.

    latents : [1, C, H, W]
    Returns : [n_variants * n_cfg, C, H, W]  one x̂₀ per candidate
    """
    n_v   = len(pe_list)
    n_cfg = len(cfg_scales)
    total = n_v * n_cfg
    C = latents.shape[1]

    # Build flat arrays of cond embeddings and cfg values
    # Index order: variant-major, cfg-minor
    # flat_idx = v_idx * n_cfg + cfg_idx
    all_ce  = torch.cat([pe[0].expand(n_cfg, -1, -1)
                          for pe, _ in pe_list], dim=0)   # [total, L, D]
    all_cm  = torch.cat([pm[0].expand(n_cfg, -1)
                          for _, pm in pe_list], dim=0)   # [total, L]
    cfg_arr = torch.tensor(
        [c for _ in range(n_v) for c in cfg_scales],
        device=sid_device, dtype=torch.float32)           # [total]

    ue_exp = ue[0].unsqueeze(0).expand(total, -1, -1)    # [total, L, D]
    um_exp = um[0].unsqueeze(0).expand(total, -1)        # [total, L]
    lat_exp = latents.expand(total, -1, -1, -1)          # [total, C, H, W]

    flow_out = torch.zeros_like(lat_exp)

    for s in range(0, total, max_batch):
        e    = min(s + max_batch, total)
        idx  = slice(s, e)
        n_ch = e - s
        cfg_ch = cfg_arr[idx]

        # ── cfg == 1: cond-only forward ───────────────────────────────
        m1 = (cfg_ch == 1.0)
        if m1.any():
            sel = m1.nonzero(as_tuple=True)[0]
            mdtype = pipe.transformer.dtype
            v = pipe.transformer(
                hidden_states          = lat_exp[s:e][sel].to(mdtype),
                encoder_hidden_states  = all_ce[s:e][sel].to(mdtype),
                encoder_attention_mask = all_cm[s:e][sel],
                timestep               = TIME_SCALE * t_flat.expand(len(sel)),
                return_dict=False)[0]
            if VARIANCE_SPLIT:
                v = v.chunk(2, dim=1)[0]
            flow_out[s + sel] = v.float()

        # ── cfg > 1: separate uncond/cond forwards ────────────────────
        # FIX: run uncond and cond as separate batch=1 forwards to avoid
        # CUDA 32-bit index overflow in depthwise conv at high resolution.
        mN = ~m1
        if mN.any():
            sel  = mN.nonzero(as_tuple=True)[0]
            n_s  = len(sel)
            mdtype = pipe.transformer.dtype

            v_u_list = []
            v_c_list = []
            for j in range(n_s):
                idx_j = sel[j]
                lat_j = lat_exp[s:e][idx_j:idx_j+1].to(mdtype)
                ts_j  = TIME_SCALE * t_flat

                u_j = pipe.transformer(
                    hidden_states=lat_j,
                    encoder_hidden_states=ue_exp[s:e][idx_j:idx_j+1].to(mdtype),
                    encoder_attention_mask=um_exp[s:e][idx_j:idx_j+1],
                    timestep=ts_j, return_dict=False)[0]
                if VARIANCE_SPLIT:
                    u_j = u_j.chunk(2, dim=1)[0]
                v_u_list.append(u_j.float())

                c_j = pipe.transformer(
                    hidden_states=lat_j,
                    encoder_hidden_states=all_ce[s:e][idx_j:idx_j+1].to(mdtype),
                    encoder_attention_mask=all_cm[s:e][idx_j:idx_j+1],
                    timestep=ts_j, return_dict=False)[0]
                if VARIANCE_SPLIT:
                    c_j = c_j.chunk(2, dim=1)[0]
                v_c_list.append(c_j.float())

            v_u = torch.cat(v_u_list, dim=0)
            v_c = torch.cat(v_c_list, dim=0)
            g   = cfg_ch[sel].view(-1, 1, 1, 1)
            flow_out[s + sel] = v_u + g * (v_c - v_u)

    return flow_out   # [total, C, H, W]


# ── Latent init ────────────────────────────────────────────────────────────────
def make_latents(seed, h, w, dtype, device):
    try:
        g = torch.Generator(device=device).manual_seed(seed)
        return pipe.prepare_latents(1, LATENT_C, h, w, dtype, device, g)
    except Exception:
        torch.manual_seed(seed)
        return torch.randn(1, LATENT_C, h, w, device=device, dtype=dtype)


# ── Decode ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def transformer_step(latents, pe, pm, ue, um, t_flat, cfg):
    """Single CFG transformer forward for one (variant, cfg) candidate."""
    if cfg == 1.0:
        v = pipe.transformer(
            hidden_states=latents,
            encoder_hidden_states=pe,
            encoder_attention_mask=pm,
            timestep=TIME_SCALE * t_flat,
            return_dict=False)[0]
        if VARIANCE_SPLIT:
            v = v.chunk(2, dim=1)[0]
        return v
    else:
        latent_in = torch.cat([latents, latents])
        embeds_in = torch.cat([ue, pe])
        attn_in   = torch.cat([um, pm])
        t_in      = torch.cat([t_flat, t_flat])
        flow_both = pipe.transformer(
            hidden_states=latent_in,
            encoder_hidden_states=embeds_in,
            encoder_attention_mask=attn_in,
            timestep=TIME_SCALE * t_in,
            return_dict=False)[0]
        if VARIANCE_SPLIT:
            flow_both = flow_both.chunk(2, dim=1)[0]
        flow_uncond, flow_cond = flow_both.chunk(2)
        return flow_uncond + cfg * (flow_cond - flow_uncond)


@torch.no_grad()
def decode_to_pil(D_x, orig_h, orig_w):
    """Decode a single latent [1,C,H,W] to PIL."""
    image = pipe.vae.decode(
        D_x / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = pipe.image_processor.resize_and_crop_tensor(image, orig_h, orig_w)
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


@torch.no_grad()
def decode_to_pil_batch(D_x_batch, orig_h, orig_w):
    return [decode_to_pil(D_x_batch[i:i+1], orig_h, orig_w)
            for i in range(D_x_batch.shape[0])]


def score_batch(prompt, imgs):
    return [float(reward_model.score(prompt, img)) for img in imgs]


# ── Baseline ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_baseline(prompt, seed, h, w, orig_h, orig_w):
    pe_list, ue, um = encode_variants([prompt])
    pe, pm  = pe_list[0]
    latents = make_latents(seed, h, w, pe.dtype, sid_device)
    D_x     = torch.zeros_like(latents)

    for i in range(args.steps):
        scalar_t = 999.0 * (1.0 - i / args.steps)
        t_flat   = torch.full((1,), scalar_t / 999.0,
                               device=sid_device, dtype=pe.dtype)
        t_4d     = t_flat.view(1, 1, 1, 1)
        noise    = latents if i == 0 else torch.randn_like(latents)
        latents  = (1.0 - t_4d) * D_x + t_4d * noise
        v = pipe.transformer(
            hidden_states=latents,
            encoder_hidden_states=pe,
            encoder_attention_mask=pm,
            timestep=TIME_SCALE * t_flat, return_dict=False)[0]
        if VARIANCE_SPLIT:
            v = v.chunk(2, dim=1)[0]
        D_x = latents - t_4d * v

    imgs  = decode_to_pil_batch(D_x, orig_h, orig_w)
    score = score_batch(prompt, imgs)[0]
    return imgs[0], score


# ── Greedy search ──────────────────────────────────────────────────────────────
@torch.no_grad()
def run_greedy(prompt, variants, seed, h, w, orig_h, orig_w):
    pe_list, ue, um = encode_variants(variants)
    n_v   = len(variants)
    n_cfg = len(args.cfg_scales)
    total = n_v * n_cfg

    latents = make_latents(seed, h, w, pe_list[0][0].dtype, sid_device)
    D_x     = torch.zeros_like(latents)
    chosen  = []   # [(v_idx, cfg, score)]

    print(f"  {n_v} variants × {n_cfg} cfg = {total} candidates/step")

    for i in range(args.steps):
        scalar_t = 999.0 * (1.0 - i / args.steps)
        t_flat   = torch.full((1,), scalar_t / 999.0,
                               device=sid_device, dtype=latents.dtype)
        t_4d     = t_flat.view(1, 1, 1, 1)
        noise    = latents if i == 0 else torch.randn_like(latents)
        latents  = (1.0 - t_4d) * D_x + t_4d * noise

        print(f"\n  step {i+1}/{args.steps}  t={scalar_t/999:.3f}")

        best_score = -float("inf")
        best_D_x   = None
        v_best     = 0
        cfg_best   = args.cfg_scales[0]

        for v_idx, (pe, pm) in enumerate(pe_list):
            for cfg in args.cfg_scales:
                flow_pred = transformer_step(
                    latents, pe, pm, ue, um, t_flat, cfg)
                cand_D_x = latents - t_4d * flow_pred
                cand_img = decode_to_pil(cand_D_x, orig_h, orig_w)
                sc       = score_batch(prompt, [cand_img])[0]
                tag = f"v{v_idx}('{variants[v_idx][:30]}') γ={cfg:.2f}"
                marker = ""
                if sc > best_score:
                    best_score = sc
                    best_D_x   = cand_D_x.clone()
                    v_best     = v_idx
                    cfg_best   = cfg
                    marker     = " ← BEST"
                print(f"    {tag:<52}  IR={sc:.4f}{marker}")

        D_x = best_D_x
        chosen.append((v_best, cfg_best, best_score))
        print(f"  → BEST v{v_best} γ={cfg_best:.2f}  IR={best_score:.4f}")
        print(f"           '{variants[v_best][:70]}'")

    torch.cuda.empty_cache()  # single cleanup before final decode
    imgs_final  = decode_to_pil_batch(D_x, orig_h, orig_w)
    final_score = score_batch(prompt, imgs_final)[0]
    return imgs_final[0], final_score, chosen


# ── Grid ───────────────────────────────────────────────────────────────────────
def _font(size=16):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_comparison(base_img, greedy_img, base_score,
                    greedy_score, chosen, variants):
    W, H_img = base_img.size
    HDR = 54
    comp = Image.new("RGB", (W * 2, H_img + HDR), (18, 18, 18))
    draw = ImageDraw.Draw(comp)
    comp.paste(base_img,   (0, HDR))
    comp.paste(greedy_img, (W, HDR))
    draw.text((4,   4), f"baseline  IR={base_score:.3f}",
              fill=(200, 200, 200), font=_font(15))
    col = (100, 255, 100) if greedy_score >= base_score else (255, 100, 100)
    draw.text((W+4, 4),
              f"greedy  IR={greedy_score:.3f}  Δ={greedy_score-base_score:+.3f}",
              fill=col, font=_font(15))
    acts = " → ".join(f"v{vi}/γ{c:.2f}" for vi, c, _ in chosen)
    draw.text((W+4, 28), acts[:90], fill=(255, 220, 50), font=_font(11))
    return comp


# ── Prompt list ────────────────────────────────────────────────────────────────
if args.prompt_file:
    prompts = [l.strip() for l in open(args.prompt_file) if l.strip()]
else:
    prompts = [args.prompt]

# ── Phase 1: generate all variants upfront (uses Qwen3) ──────────────────────
all_variants = {}
for pidx, prompt in enumerate(prompts):
    slug = f"p{pidx:02d}"
    print(f"\n[{slug}] Generating {args.n_variants} variants ...")
    print(f"  v0 (original): {prompt}")
    variants = generate_variants(prompt, args.n_variants)
    all_variants[pidx] = variants
    with open(os.path.join(args.out_dir, f"{slug}_variants.txt"), "w") as f:
        for vi, v in enumerate(variants):
            f.write(f"v{vi}: {v}\n")

# ── Phase 2: run search for all prompts ──────────────────────────────────────
summary = []

for pidx, prompt in enumerate(prompts):
    slug = f"p{pidx:02d}"
    variants = all_variants[pidx]
    print(f"\n{'='*72}")
    print(f"[{slug}] {prompt}")
    print(f"{'='*72}")

    sample_size = pipe.transformer.config.sample_size
    orig_h, orig_w = args.height, args.width
    h, w = orig_h, orig_w
    if sample_size in ASPECT_RATIO_BINS:
        h, w = pipe.image_processor.classify_height_width_bin(
            h, w, ratios=ASPECT_RATIO_BINS[sample_size])

    # Baseline
    print("\nBaseline ...")
    base_img, base_sc = run_baseline(prompt, args.seed, h, w, orig_h, orig_w)
    base_img.save(os.path.join(args.out_dir, f"{slug}_baseline.png"))
    print(f"Baseline IR={base_sc:.4f}")

    # Greedy search
    print("\nGreedy search ...")
    greedy_img, greedy_sc, chosen = run_greedy(
        prompt, variants, args.seed, h, w, orig_h, orig_w)
    greedy_img.save(os.path.join(args.out_dir, f"{slug}_greedy.png"))

    print(f"\nGreedy IR={greedy_sc:.4f}  Δ={greedy_sc - base_sc:+.4f}")
    for si, (vi, cfg, sc) in enumerate(chosen):
        print(f"  step {si+1}: v{vi} γ={cfg:.2f}  IR={sc:.4f}"
              f"  '{variants[vi][:60]}'")

    # Comparison grid
    comp = make_comparison(base_img, greedy_img, base_sc,
                           greedy_sc, chosen, variants)
    comp.save(os.path.join(args.out_dir, f"{slug}_comparison.png"))

    summary.append({
        "slug": slug, "prompt": prompt, "variants": variants,
        "base_IR": base_sc, "greedy_IR": greedy_sc,
        "delta_IR": greedy_sc - base_sc,
        "actions": [(vi, cfg, sc) for vi, cfg, sc in chosen],
    })

# ── Summary ────────────────────────────────────────────────────────────────────
log_path  = os.path.join(args.out_dir, "greedy_summary.txt")
json_path = os.path.join(args.out_dir, "greedy_summary.json")

with open(log_path, "w") as f:
    f.write(f"{'slug':<6} {'baseline':>10} {'greedy':>10} {'delta':>8}  actions\n")
    f.write("-" * 80 + "\n")
    for row in summary:
        acts = " ".join(f"v{vi}/γ{c:.2f}" for vi, c, _ in row["actions"])
        f.write(f"{row['slug']:<6} {row['base_IR']:>10.4f} "
                f"{row['greedy_IR']:>10.4f} {row['delta_IR']:>+8.4f}  {acts}\n")
        f.write(f"  prompt: {row['prompt']}\n")
        for vi, v in enumerate(row["variants"]):
            f.write(f"  v{vi}: {v}\n")
        f.write("\n")

with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
print(f"{'slug':<6} {'baseline':>10} {'greedy':>10} {'delta':>8}  actions")
print("-" * 60)
for row in summary:
    acts = " ".join(f"v{vi}/γ{c:.2f}" for vi, c, _ in row["actions"])
    print(f"{row['slug']:<6} {row['base_IR']:>10.4f} "
          f"{row['greedy_IR']:>10.4f} {row['delta_IR']:>+8.4f}  {acts}")

if summary:
    mean_d = sum(r["delta_IR"] for r in summary) / len(summary)
    print(f"\n  mean Δ IR over {len(summary)} prompt(s): {mean_d:+.4f}")

print(f"\nLog  → {log_path}")
print(f"JSON → {json_path}")
print(f"Out  → {os.path.abspath(args.out_dir)}/")