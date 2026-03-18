"""
greedy_cfg_search.py
====================
At each SiD denoising step, greedily pick the (cfg_scale, use_neg_embed)
action that maximises ImageReward evaluated on the one-step x̂₀ prediction.

The search space at every step:
  cfg_scales   : [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
  use_neg_embed: [False, True]   (True only if --neg_embed is provided)
  → up to 14 candidates per step, 4 steps = 56 transformer calls total

After the greedy search the best action sequence is printed and the final
image is saved alongside a baseline (fixed cfg=1, no neg embed).

Usage
-----
  python greedy_cfg_search.py [OPTIONS]

Options
-------
  --model_id      HF repo id            (default: YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow)
  --prompt        Text prompt
  --neg_embed     Path to neg_embed.pt  (optional)
  --steps         SiD denoising steps   (default: 4)
  --cfg_scales    Search space          (default: 1.0 1.25 1.5 1.75 2.0 2.25 2.5)
  --seed          RNG seed              (default: 42)
  --width / --height                    (default: 512)
  --time_scale                          (default: 1000.0)
  --out_dir                             (default: ./greedy_out)
  --prompt_file   Text file, one prompt per line (overrides --prompt)
"""

import argparse, os, sys
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_id",   default="YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow")
parser.add_argument("--prompt",     default="a studio portrait of an elderly woman smiling, "
                                             "soft window light, 85mm lens, photorealistic")
parser.add_argument("--prompt_file", default=None)
parser.add_argument("--neg_embed",  default=None)
parser.add_argument("--steps",      type=int,   default=4)
parser.add_argument("--cfg_scales", nargs="+",  type=float,
                    default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5])
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--width",      type=int,   default=512)
parser.add_argument("--height",     type=int,   default=512)
parser.add_argument("--time_scale", type=float, default=1000.0)
parser.add_argument("--out_dir",    default="./greedy_out")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16

# ── Imports ───────────────────────────────────────────────────────────────────
repo_root = Path(__file__).parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from sid import SiDSanaPipeline
except ImportError as e:
    sys.exit(f"Cannot import SiDSanaPipeline: {e}\n"
             "Run from inside the cloned Space repo.")

from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
    ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN)
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN

# ── Load models ───────────────────────────────────────────────────────────────
print(f"Loading pipeline ...")
pipe = SiDSanaPipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
print("✓ Pipeline loaded")

print("Loading ImageReward ...")
import ImageReward as RM
reward_model = RM.load("ImageReward-v1.0", device=device)
reward_model.eval()
print("✓ ImageReward loaded")

# ── Negative embedding ────────────────────────────────────────────────────────
pretrained_neg_embeds = None
pretrained_neg_mask   = None

if args.neg_embed:
    ckpt = torch.load(args.neg_embed, map_location="cpu")
    if isinstance(ckpt, dict):
        pretrained_neg_embeds = ckpt["neg_embeds"].to(device=device, dtype=dtype)
        pretrained_neg_mask   = ckpt["neg_mask"].to(device=device)
    else:
        pretrained_neg_embeds = ckpt.to(device=device, dtype=dtype)
        pretrained_neg_mask   = torch.ones(ckpt.shape[:2], device=device, dtype=torch.long)
    print(f"✓ Loaded neg embedding: {pretrained_neg_embeds.shape}\n")
else:
    print("No --neg_embed; search space = cfg_scales only\n")

# ── Prompt list ───────────────────────────────────────────────────────────────
if args.prompt_file:
    prompts = [l.strip() for l in open(args.prompt_file) if l.strip()]
else:
    prompts = [args.prompt]

# ── Helpers ───────────────────────────────────────────────────────────────────
ASPECT_RATIO_BINS = {16: ASPECT_RATIO_512_BIN,
                     32: ASPECT_RATIO_1024_BIN,
                     64: ASPECT_RATIO_2048_BIN}

@torch.no_grad()
def get_x0_hat(D_x, latents, t_4d):
    """Reconstruct current best x̂₀ from D_x (same as final decode formula)."""
    return D_x  # In SiD, D_x IS the running x̂₀ estimate after each step


@torch.no_grad()
def decode_to_pil(D_x, orig_h, orig_w):
    image = pipe.vae.decode(
        D_x / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = pipe.image_processor.resize_and_crop_tensor(image, orig_h, orig_w)
    image = pipe.image_processor.postprocess(image, output_type="pil")
    return image[0]


@torch.no_grad()
def score_pil(prompt, img):
    return float(reward_model.score(prompt, img))


@torch.no_grad()
def transformer_step(latents, prompt_embeds, prompt_attn_mask,
                     neg_embeds, neg_attn_mask,
                     t_flat, guidance_scale, use_neg, time_scale):
    """One transformer forward (CFG-blended) → returns flow_pred."""
    do_cfg = (guidance_scale != 1.0)

    if do_cfg:
        # pick uncond source
        if use_neg and pretrained_neg_embeds is not None:
            ue, um = neg_embeds, neg_attn_mask
        else:
            ue, um = neg_embeds, neg_attn_mask   # text-null embeds passed in

        latent_in    = torch.cat([latents, latents])
        embeds_in    = torch.cat([ue,      prompt_embeds])
        attn_in      = torch.cat([um,      prompt_attn_mask])
        t_in         = torch.cat([t_flat,  t_flat])

        flow_both = pipe.transformer(
            hidden_states=latent_in,
            encoder_hidden_states=embeds_in,
            encoder_attention_mask=attn_in,
            timestep=time_scale * t_in,
            return_dict=False,
        )[0]
        flow_uncond, flow_cond = flow_both.chunk(2)
        return flow_uncond + guidance_scale * (flow_cond - flow_uncond)
    else:
        return pipe.transformer(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attn_mask,
            timestep=time_scale * t_flat,
            return_dict=False,
        )[0]


def prepare_embeds(prompt, exec_device, max_sequence_length=256):
    """Encode prompt + null-text negative (always). Returns all four tensors."""
    pe, pm, ne, nm = pipe.encode_prompt(
        prompt=prompt,
        do_classifier_free_guidance=True,
        negative_prompt="",
        device=exec_device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )

    # Optionally build pre-trained neg tensors aligned to cond seq length
    pne, pnm = None, None
    if pretrained_neg_embeds is not None:
        B, L_cond = pe.shape[0], pe.shape[1]
        pne = pretrained_neg_embeds.expand(B, -1, -1).to(dtype=pe.dtype, device=exec_device)
        pnm = pretrained_neg_mask.expand(B, -1).to(device=exec_device)
        L_neg = pne.shape[1]
        if L_neg < L_cond:
            pne = torch.cat([pne, torch.zeros(B, L_cond-L_neg, pne.shape[2],
                             device=exec_device, dtype=pne.dtype)], dim=1)
            pnm = torch.cat([pnm, torch.zeros(B, L_cond-L_neg,
                             device=exec_device, dtype=pnm.dtype)], dim=1)
        elif L_neg > L_cond:
            pne, pnm = pne[:, :L_cond], pnm[:, :L_cond]

    return pe, pm, ne, nm, pne, pnm


# ── Build action space ────────────────────────────────────────────────────────
# Each action = (cfg_scale, use_neg_embed)
# use_neg_embed=True only makes sense if pretrained_neg_embeds is loaded
actions = []
for cfg in args.cfg_scales:
    actions.append((cfg, False))
    if pretrained_neg_embeds is not None:
        actions.append((cfg, True))

print(f"Action space: {len(actions)} actions × {args.steps} steps")
print(f"  {actions}\n")


# ── Greedy search ─────────────────────────────────────────────────────────────
@torch.no_grad()
def greedy_search(prompt, seed):
    exec_device = pipe._execution_device

    # Encode
    pe, pm, ne, nm, pne, pnm = prepare_embeds(prompt, exec_device)

    # Resolution binning
    orig_h, orig_w = args.height, args.width
    sample_size = pipe.transformer.config.sample_size
    h, w = orig_h, orig_w
    if sample_size in ASPECT_RATIO_BINS:
        h, w = pipe.image_processor.classify_height_width_bin(
            h, w, ratios=ASPECT_RATIO_BINS[sample_size])

    # Initial latents
    g = torch.Generator(device=exec_device).manual_seed(seed)
    latents = pipe.prepare_latents(
        1, pipe.transformer.config.in_channels, h, w,
        pe.dtype, exec_device, g)

    D_x            = torch.zeros_like(latents)
    initial_latents = latents.clone()
    chosen_actions  = []   # record best action at each step

    for i in range(args.steps):
        # Timestep
        scalar_t = 999.0 * (1.0 - float(i) / float(args.steps))
        t_flat   = torch.full((latents.shape[0],), scalar_t / 999.0,
                              device=exec_device, dtype=latents.dtype)
        t_4d     = t_flat.view(-1, 1, 1, 1)

        noise   = latents if i == 0 else torch.randn_like(latents)
        latents = (1.0 - t_4d) * D_x + t_4d * noise

        # ── Greedy action selection ───────────────────────────────────────
        best_score  = -float("inf")
        best_action = actions[0]
        best_D_x    = None

        print(f"  step {i+1}/{args.steps}  evaluating {len(actions)} actions ...")
        for cfg, use_neg in actions:
            # pick the right neg embeds
            if use_neg and pne is not None:
                cur_ne, cur_nm = pne, pnm
            else:
                cur_ne, cur_nm = ne, nm

            flow_pred = transformer_step(
                latents, pe, pm, cur_ne, cur_nm,
                t_flat, cfg, use_neg, args.time_scale)

            candidate_D_x = latents - t_4d * flow_pred

            # Score candidate x̂₀ via ImageReward
            pil_img = decode_to_pil(candidate_D_x, orig_h, orig_w)
            score   = score_pil(prompt, pil_img)

            print(f"    cfg={cfg:.2f} neg={'Y' if use_neg else 'N'}  IR={score:.4f}")

            if score > best_score:
                best_score  = score
                best_action = (cfg, use_neg)
                best_D_x    = candidate_D_x.clone()

        chosen_actions.append((best_action, best_score))
        D_x = best_D_x
        print(f"  → chose cfg={best_action[0]:.2f} neg={'Y' if best_action[1] else 'N'}"
              f"  IR={best_score:.4f}\n")

    # Final decode
    final_img = decode_to_pil(D_x, orig_h, orig_w)
    final_score = score_pil(prompt, final_img)
    return final_img, final_score, chosen_actions


@torch.no_grad()
def baseline(prompt, seed):
    """Fixed cfg=1.0, no neg embed — standard SiD inference."""
    exec_device = pipe._execution_device
    pe, pm, _, _, _, _ = prepare_embeds(prompt, exec_device)

    orig_h, orig_w = args.height, args.width
    sample_size = pipe.transformer.config.sample_size
    h, w = orig_h, orig_w
    if sample_size in ASPECT_RATIO_BINS:
        h, w = pipe.image_processor.classify_height_width_bin(
            h, w, ratios=ASPECT_RATIO_BINS[sample_size])

    g = torch.Generator(device=exec_device).manual_seed(seed)
    latents = pipe.prepare_latents(
        1, pipe.transformer.config.in_channels, h, w,
        pe.dtype, exec_device, g)

    D_x = torch.zeros_like(latents)
    for i in range(args.steps):
        scalar_t = 999.0 * (1.0 - float(i) / float(args.steps))
        t_flat   = torch.full((latents.shape[0],), scalar_t / 999.0,
                              device=exec_device, dtype=latents.dtype)
        t_4d     = t_flat.view(-1, 1, 1, 1)
        noise   = latents if i == 0 else torch.randn_like(latents)
        latents = (1.0 - t_4d) * D_x + t_4d * noise
        flow_pred = pipe.transformer(
            hidden_states=latents,
            encoder_hidden_states=pe,
            encoder_attention_mask=pm,
            timestep=args.time_scale * t_flat,
            return_dict=False,
        )[0]
        D_x = latents - t_4d * flow_pred

    img = decode_to_pil(D_x, orig_h, orig_w)
    return img, score_pil(prompt, img)


# ── Font helper ───────────────────────────────────────────────────────────────
def _font(size=16):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


# ── Main loop over prompts ────────────────────────────────────────────────────
summary_rows = []

for prompt_idx, prompt in enumerate(prompts):
    slug = f"p{prompt_idx:02d}"
    print(f"\n{'='*70}")
    print(f"Prompt {prompt_idx}: {prompt}")
    print(f"{'='*70}\n")

    # Baseline
    print("Running baseline (cfg=1.0, no neg) ...")
    base_img, base_score = baseline(prompt, args.seed)
    base_path = os.path.join(args.out_dir, f"{slug}_baseline.png")
    base_img.save(base_path)
    print(f"Baseline IR={base_score:.4f}  saved → {base_path}\n")

    # Greedy
    print("Running greedy search ...")
    greedy_img, greedy_score, chosen = greedy_search(prompt, args.seed)
    greedy_path = os.path.join(args.out_dir, f"{slug}_greedy.png")
    greedy_img.save(greedy_path)

    print(f"\nGreedy IR={greedy_score:.4f}  saved → {greedy_path}")
    print(f"Delta IR = {greedy_score - base_score:+.4f}")
    print("Chosen action sequence:")
    for step_i, ((cfg, use_neg), score) in enumerate(chosen):
        print(f"  step {step_i+1}: cfg={cfg:.2f}  neg={'Y' if use_neg else 'N'}"
              f"  IR={score:.4f}")

    # Comparison image: baseline | greedy
    W, H = base_img.size
    LABEL_H = 40
    comp = Image.new("RGB", (W * 2, H + LABEL_H), (18, 18, 18))
    draw = ImageDraw.Draw(comp)
    font = _font(16)
    comp.paste(base_img,   (0, LABEL_H))
    comp.paste(greedy_img, (W, LABEL_H))
    draw.text((4,   4), f"baseline  IR={base_score:.3f}",   fill=(200, 200, 200), font=font)
    draw.text((W+4, 4), f"greedy    IR={greedy_score:.3f}", fill=(100, 255, 100), font=font)

    # Annotate chosen actions on the greedy side
    action_str = " → ".join(
        f"s{i+1}:cfg{a[0]:.2f}{'N' if a[1] else ''}"
        for i, (a, _) in enumerate(chosen))
    draw.text((W+4, 22), action_str[:80], fill=(255, 220, 50), font=_font(11))

    comp_path = os.path.join(args.out_dir, f"{slug}_comparison.png")
    comp.save(comp_path)
    print(f"Comparison → {comp_path}")

    summary_rows.append({
        "prompt_idx":    prompt_idx,
        "prompt":        prompt[:60],
        "baseline_IR":   base_score,
        "greedy_IR":     greedy_score,
        "delta_IR":      greedy_score - base_score,
        "actions":       [(a, s) for a, s in chosen],
    })

# ── Summary ───────────────────────────────────────────────────────────────────
log_path = os.path.join(args.out_dir, "greedy_summary.txt")
with open(log_path, "w") as f:
    f.write(f"{'idx':<5} {'baseline':>10} {'greedy':>10} {'delta':>8}  actions\n")
    f.write("-" * 80 + "\n")
    for row in summary_rows:
        action_str = " ".join(
            f"cfg{a[0]:.2f}{'N' if a[1] else '.'}"
            for (a, _) in row["actions"])
        f.write(f"{row['prompt_idx']:<5} {row['baseline_IR']:>10.4f} "
                f"{row['greedy_IR']:>10.4f} {row['delta_IR']:>+8.4f}  {action_str}\n")
        f.write(f"      {row['prompt']}\n\n")

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"{'idx':<5} {'baseline':>10} {'greedy':>10} {'delta':>8}  actions")
print("-" * 70)
for row in summary_rows:
    action_str = " ".join(
        f"cfg{a[0]:.2f}{'N' if a[1] else '.'}"
        for (a, _) in row["actions"])
    print(f"{row['prompt_idx']:<5} {row['baseline_IR']:>10.4f} "
          f"{row['greedy_IR']:>10.4f} {row['delta_IR']:>+8.4f}  {action_str}")

print(f"\nFull log → {log_path}")
print(f"Outputs  → {os.path.abspath(args.out_dir)}/")