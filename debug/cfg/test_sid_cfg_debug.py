"""
test_sid_cfg.py  –  CFG sweep: with vs without pre-trained neg embedding + ImageReward
=======================================================================================
Run from inside the cloned Space repo (needs the sid/ module):

    git clone https://huggingface.co/spaces/YGu1998/SiD-DiT-SANA-0.6B-RF
    cd SiD-DiT-SANA-0.6B-RF
    python test_sid_cfg.py [OPTIONS]

Outputs
-------
  cfg_sweep/
    grid_no_neg.png          — all CFG scales, text null as uncond
    grid_with_neg.png        — all CFG scales, pre-trained neg embed as uncond
    grid_comparison.png      — side-by-side: top row = no_neg, bottom = with_neg
    reward_log.txt           — ImageReward scores for every image
    cfg_X.XX_no_neg.png      — individual images
    cfg_X.XX_with_neg.png

Options
-------
  --model_id      HF repo id                (default: YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow)
  --ckpt          Path to .pt_G checkpoint  (overrides transformer weights)
  --prompt        Text prompt
  --neg_prompt    Negative prompt text      (default: "")
  --neg_embed     Path to neg_embed.pt      REQUIRED for the with-neg row
  --steps         Denoising steps           (default: 4)
  --cfg_scales    Space-separated floats    (default: 1.0 1.5 2.0 3.0 4.5 7.0)
  --seed          RNG seed                  (default: 42)
  --width         Image width               (default: 512)
  --height        Image height              (default: 512)
  --time_scale    SiD time_scale param      (default: 1000.0)
  --out_dir       Output directory          (default: ./cfg_sweep)
"""

import argparse, os, sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_id",   default="YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow")
parser.add_argument("--ckpt",       default=None,
                    help="Path to .pt_G checkpoint. If set, overwrites the "
                         "transformer weights loaded from --model_id.")
parser.add_argument("--prompt",     default="a studio portrait of an elderly woman smiling, soft window light, 85mm lens, photorealistic")
parser.add_argument("--neg_prompt", default="")
parser.add_argument("--neg_embed",  default=None,
                    help="Path to neg_embed.pt (keys: neg_embeds [1,L,D], neg_mask [1,L])")
parser.add_argument("--steps",      type=int,   default=4)
parser.add_argument("--cfg_scales", nargs="+",  type=float,
                    default=[1.0, 1.5, 2.0, 3.0, 4.5, 7.0])
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--width",      type=int,   default=512)
parser.add_argument("--height",     type=int,   default=512)
parser.add_argument("--time_scale", type=float, default=1000.0)
parser.add_argument("--out_dir",    default="./cfg_sweep_long")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16

# ── Import SiDSanaPipeline ───────────────────────────────────────────────────
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from sid import SiDSanaPipeline
    print("✓ Imported SiDSanaPipeline")
except ImportError as e:
    sys.exit(f"Cannot import SiDSanaPipeline: {e}\n"
             "Run this script from inside the cloned Space repo.")

# ── Load pipeline ─────────────────────────────────────────────────────────────
print(f"Loading {args.model_id} ...")
pipe = SiDSanaPipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
print("✓ Pipeline loaded")

# ── Load custom checkpoint ────────────────────────────────────────────────────
if args.ckpt:
    print(f"Loading weights from {args.ckpt} ...")
    raw = torch.load(args.ckpt, map_location=device, weights_only=False)

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

# ── Load ImageReward ──────────────────────────────────────────────────────────
print("Loading ImageReward ...")
try:
    import ImageReward as RM
    reward_model = RM.load("ImageReward-v1.0", device=device)
    reward_model.eval()
    print("✓ ImageReward loaded\n")
    HAS_REWARD = True
except Exception as e:
    print(f"  ImageReward not available ({e}) — scores will be skipped\n")
    HAS_REWARD = False

def score_image(prompt: str, img: Image.Image) -> float:
    if not HAS_REWARD:
        return float("nan")
    with torch.no_grad():
        return float(reward_model.score(prompt, img))

# ── Load pre-trained negative embedding ──────────────────────────────────────
pretrained_neg_embeds = None
pretrained_neg_mask   = None

if args.neg_embed is not None:
    ckpt = torch.load(args.neg_embed, map_location="cpu")
    if isinstance(ckpt, dict):
        if "neg_embeds" not in ckpt:
            raise KeyError(f"neg_embed.pt has keys {list(ckpt.keys())}, "
                           f"expected 'neg_embeds' and 'neg_mask'")
        pretrained_neg_embeds = ckpt["neg_embeds"].to(device=device, dtype=dtype)
        pretrained_neg_mask   = ckpt["neg_mask"].to(device=device)
    elif isinstance(ckpt, torch.Tensor):
        pretrained_neg_embeds = ckpt.to(device=device, dtype=dtype)
        pretrained_neg_mask   = torch.ones(ckpt.shape[:2], device=device, dtype=torch.long)
    else:
        raise ValueError(f"Unrecognised checkpoint type: {type(ckpt)}")
    print(f"✓ Loaded pre-trained neg embedding: {pretrained_neg_embeds.shape}")
else:
    print("No --neg_embed provided — will only produce the no-neg row\n")


# ── Core generate function ────────────────────────────────────────────────────
@torch.no_grad()
def generate(
    prompt,
    guidance_scale=1.0,
    negative_prompt="",
    use_pretrained_neg=False,
    num_inference_steps=4,
    width=512,
    height=512,
    generator=None,
    time_scale=1000.0,
    max_sequence_length=256,
    noise_type="fresh",
):
    do_cfg     = (guidance_scale != 1.0)
    exec_device = pipe._execution_device

    prompt_embeds, prompt_attn_mask, neg_embeds, neg_attn_mask = pipe.encode_prompt(
        prompt=prompt,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt if do_cfg else "",
        device=exec_device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )

    if do_cfg and use_pretrained_neg and pretrained_neg_embeds is not None:
        B  = prompt_embeds.shape[0]
        ne = pretrained_neg_embeds.expand(B, -1, -1).to(
                 dtype=prompt_embeds.dtype, device=exec_device)
        nm = pretrained_neg_mask.expand(B, -1).to(device=exec_device)
        L_neg, L_cond = ne.shape[1], prompt_embeds.shape[1]
        if L_neg < L_cond:
            ne = torch.cat([ne,
                torch.zeros(B, L_cond - L_neg, ne.shape[2],
                            device=exec_device, dtype=ne.dtype)], dim=1)
            nm = torch.cat([nm,
                torch.zeros(B, L_cond - L_neg,
                            device=exec_device, dtype=nm.dtype)], dim=1)
        elif L_neg > L_cond:
            ne = ne[:, :L_cond, :]
            nm = nm[:, :L_cond]
        neg_embeds    = ne
        neg_attn_mask = nm

    from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
        ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN)
    from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
    sample_size = pipe.transformer.config.sample_size
    aspect_ratio_bins = {16: ASPECT_RATIO_512_BIN,
                         32: ASPECT_RATIO_1024_BIN,
                         64: ASPECT_RATIO_2048_BIN}
    orig_h, orig_w = height, width
    if sample_size in aspect_ratio_bins:
        height, width = pipe.image_processor.classify_height_width_bin(
            height, width, ratios=aspect_ratio_bins[sample_size])

    latents = pipe.prepare_latents(
        1, pipe.transformer.config.in_channels, height, width,
        prompt_embeds.dtype, exec_device, generator,
    )

    D_x             = torch.zeros_like(latents)
    initial_latents = latents.clone()

    for i in range(num_inference_steps):
        if noise_type == "fresh":
            noise = latents if i == 0 else torch.randn_like(latents)
        elif noise_type == "fixed":
            noise = initial_latents
        else:
            noise = latents if i == 0 else ((latents - (1.0 - t_4d) * D_x) / t_4d).detach()

        scalar_t = 999.0 * (1.0 - float(i) / float(num_inference_steps))
        t_flat   = torch.full((latents.shape[0],), scalar_t / 999.0,
                              device=exec_device, dtype=latents.dtype)
        t_4d     = t_flat.view(-1, 1, 1, 1)
        latents  = (1.0 - t_4d) * D_x + t_4d * noise

        if do_cfg:
            latent_in    = torch.cat([latents,       latents])
            embeds_in    = torch.cat([neg_embeds,    prompt_embeds])
            attn_mask_in = torch.cat([neg_attn_mask, prompt_attn_mask])
            t_in         = torch.cat([t_flat,        t_flat])

            flow_both = pipe.transformer(
                hidden_states=latent_in,
                encoder_hidden_states=embeds_in,
                encoder_attention_mask=attn_mask_in,
                timestep=time_scale * t_in,
                return_dict=False,
            )[0]
            flow_uncond, flow_cond = flow_both.chunk(2)
            flow_pred = flow_uncond + guidance_scale * (flow_cond - flow_uncond)
        else:
            flow_pred = pipe.transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attn_mask,
                timestep=time_scale * t_flat,
                return_dict=False,
            )[0]

        D_x = latents - t_4d * flow_pred

    image = pipe.vae.decode(D_x / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = pipe.image_processor.resize_and_crop_tensor(image, orig_h, orig_w)
    image = pipe.image_processor.postprocess(image, output_type="pil")
    return image[0]


# ── Grid builder ──────────────────────────────────────────────────────────────
LABEL_H = 34

def _load_font(size=18):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()

def make_row_grid(images_with_labels, title=""):
    n    = len(images_with_labels)
    W, H = images_with_labels[0][0].size
    title_h = LABEL_H if title else 0
    grid    = Image.new("RGB", (W * n, H + LABEL_H + title_h), (18, 18, 18))
    draw    = ImageDraw.Draw(grid)
    font_sm = _load_font(16)
    font_lg = _load_font(20)
    if title:
        draw.text((6, 4), title, fill=(200, 200, 200), font=font_lg)
    for i, (img, label) in enumerate(images_with_labels):
        grid.paste(img, (i * W, LABEL_H + title_h))
        draw.text((i * W + 4, title_h + 4), label, fill=(255, 220, 50), font=font_sm)
    return grid

def make_comparison_grid(rows):
    n_rows = len(rows)
    n_cols = len(rows[0][1])
    W, H   = rows[0][1][0][0].size
    ROW_TITLE_H = LABEL_H
    COL_LABEL_H = LABEL_H
    total_h = n_rows * (H + COL_LABEL_H) + ROW_TITLE_H
    total_w = n_cols * W + LABEL_H
    MARGIN  = LABEL_H
    grid  = Image.new("RGB", (total_w, total_h), (18, 18, 18))
    draw  = ImageDraw.Draw(grid)
    font_sm = _load_font(14)
    for j, (img, label) in enumerate(rows[0][1]):
        draw.text((MARGIN + j * W + 4, 4), label.split("\n")[0],
                  fill=(180, 180, 180), font=font_sm)
    for r, (row_title, images_with_labels) in enumerate(rows):
        y_off = ROW_TITLE_H + r * (H + COL_LABEL_H)
        draw.text((2, y_off + H // 2), row_title[:10],
                  fill=(200, 200, 200), font=font_sm)
        for c, (img, label) in enumerate(images_with_labels):
            x_off = MARGIN + c * W
            grid.paste(img, (x_off, y_off + COL_LABEL_H))
            score_line = label.split("\n")[1] if "\n" in label else ""
            if score_line:
                draw.text((x_off + 4, y_off + COL_LABEL_H + H - 20),
                          score_line, fill=(100, 255, 100), font=font_sm)
    return grid


# ── Main sweep ────────────────────────────────────────────────────────────────
modes = [("no_neg", False)]
if pretrained_neg_embeds is not None:
    modes.append(("with_neg", True))

ckpt_label = os.path.basename(args.ckpt) if args.ckpt else "default"
print(f"Prompt  : {args.prompt}")
print(f"Ckpt    : {ckpt_label}")
print(f"Neg src : {'pre-trained: ' + args.neg_embed if pretrained_neg_embeds is not None else 'text null'}")
print(f"Steps   : {args.steps}  |  seed: {args.seed}  |  "
      f"{args.width}×{args.height}  |  time_scale={args.time_scale}")
print(f"Scales  : {args.cfg_scales}")
print(f"Modes   : {[m for m, _ in modes]}\n")

results = {name: {} for name, _ in modes}
reward_lines = []

for mode_name, use_pretrained in modes:
    print(f"── {mode_name} ──────────────────────────────")
    for cfg in args.cfg_scales:
        print(f"  cfg={cfg:<5.2f} ...", end=" ", flush=True)
        g = torch.Generator(device=device).manual_seed(args.seed)
        img = generate(
            prompt=args.prompt,
            guidance_scale=cfg,
            negative_prompt=args.neg_prompt,
            use_pretrained_neg=use_pretrained,
            num_inference_steps=args.steps,
            width=args.width,
            height=args.height,
            generator=g,
            time_scale=args.time_scale,
        )
        reward = score_image(args.prompt, img)
        results[mode_name][cfg] = (img, reward)
        fname = os.path.join(args.out_dir, f"cfg_{cfg:.2f}_{mode_name}.png")
        img.save(fname)
        reward_str = f"{reward:.4f}" if not np.isnan(reward) else "n/a"
        print(f"IR={reward_str}  saved → {fname}")
        reward_lines.append(f"{mode_name}  cfg={cfg:.2f}  reward={reward_str}")
    print()

# ── Save reward log ───────────────────────────────────────────────────────────
log_path = os.path.join(args.out_dir, "reward_log.txt")
with open(log_path, "w") as f:
    f.write(f"prompt: {args.prompt}\n")
    f.write(f"ckpt: {ckpt_label}\n")
    f.write(f"neg_embed: {args.neg_embed or 'none'}\n\n")
    for line in reward_lines:
        f.write(line + "\n")
print(f"✓ Reward log → {log_path}\n")

# ── Per-mode row grids ────────────────────────────────────────────────────────
for mode_name, _ in modes:
    row_items = []
    for cfg in sorted(results[mode_name]):
        img, reward = results[mode_name][cfg]
        r_str = f"{reward:.3f}" if not np.isnan(reward) else "n/a"
        label = f"cfg={cfg:.1f}  IR={r_str}"
        row_items.append((img, label))
    grid = make_row_grid(row_items, title=f"{mode_name}  ({ckpt_label})")
    gpath = os.path.join(args.out_dir, f"grid_{mode_name}.png")
    grid.save(gpath)
    print(f"✓ Row grid  → {gpath}")

# ── Comparison grid ───────────────────────────────────────────────────────────
if len(modes) > 1:
    rows = []
    for mode_name, _ in modes:
        row_items = []
        for cfg in sorted(results[mode_name]):
            img, reward = results[mode_name][cfg]
            r_str = f"{reward:.3f}" if not np.isnan(reward) else "n/a"
            label = f"cfg={cfg:.1f}\nIR={r_str}"
            row_items.append((img, label))
        rows.append((mode_name, row_items))
    comp_grid = make_comparison_grid(rows)
    comp_path = os.path.join(args.out_dir, "grid_comparison.png")
    comp_grid.save(comp_path)
    print(f"✓ Comparison grid → {comp_path}")

# ── Console summary ───────────────────────────────────────────────────────────
print(f"\n{'cfg':<8}", end="")
for mode_name, _ in modes:
    print(f"{mode_name:>16}", end="")
print()
print("-" * (8 + 16 * len(modes)))
for cfg in sorted(args.cfg_scales):
    print(f"{cfg:<8.2f}", end="")
    for mode_name, _ in modes:
        _, reward = results[mode_name][cfg]
        r_str = f"{reward:.4f}" if not np.isnan(reward) else "   n/a"
        print(f"{r_str:>16}", end="")
    print()

print(f"\nDone. All outputs in: {os.path.abspath(args.out_dir)}/")
