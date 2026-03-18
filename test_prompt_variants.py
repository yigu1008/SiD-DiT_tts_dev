"""
test_prompt_variants.py
=======================
Generate one image per prompt variant (fixed seed, cfg=1.0) and save a
comparison grid with ImageReward scores.

Usage
-----
  python test_prompt_variants.py [OPTIONS]

Options
-------
  --model_id      HF repo id        (default: YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow)
  --qwen_id       Qwen model id     (default: Qwen/Qwen3-4B)
  --prompt        Text prompt
  --prompt_file   One prompt per line (overrides --prompt)
  --n_variants    Number of rewrites (default: 3)
  --cfg           Fixed CFG scale   (default: 1.0)
  --seed          Fixed RNG seed    (default: 42)
  --steps         SiD steps         (default: 4)
  --width/--height                  (default: 512)
  --time_scale                      (default: 1000.0)
  --out_dir                         (default: ./variant_test)
  --qwen_device   cpu|cuda|auto     (default: auto)
  --qwen_dtype    float16|bfloat16  (default: bfloat16)
"""

import argparse, os, re, sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_id",    default="YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow")
parser.add_argument("--qwen_id",     default="Qwen/Qwen3-4B")
parser.add_argument("--prompt",      default="a studio portrait of an elderly woman smiling, "
                                              "soft window light, 85mm lens, photorealistic")
parser.add_argument("--prompt_file", default=None)
parser.add_argument("--n_variants",  type=int,   default=3)
parser.add_argument("--cfg",         type=float, default=1.0)
parser.add_argument("--seed",        type=int,   default=42)
parser.add_argument("--steps",       type=int,   default=4)
parser.add_argument("--width",       type=int,   default=512)
parser.add_argument("--height",      type=int,   default=512)
parser.add_argument("--time_scale",  type=float, default=1000.0)
parser.add_argument("--out_dir",     default="./variant_test")
parser.add_argument("--qwen_device", default="auto")
parser.add_argument("--qwen_dtype",  default="bfloat16",
                    choices=["float16", "bfloat16"])
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device     = "cuda" if torch.cuda.is_available() else "cpu"
sid_dtype  = torch.float16
qwen_dtype = torch.bfloat16 if args.qwen_dtype == "bfloat16" else torch.float16

# ── Load SiD pipeline ────────────────────────────────────────────────────────
repo_root = Path(__file__).parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from sid import SiDSanaPipeline
except ImportError as e:
    sys.exit(f"Cannot import SiDSanaPipeline: {e}\nRun from inside the cloned Space repo.")

from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
    ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN)
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
ASPECT_RATIO_BINS = {16: ASPECT_RATIO_512_BIN, 32: ASPECT_RATIO_1024_BIN,
                     64: ASPECT_RATIO_2048_BIN}

print("Loading SiD pipeline ...")
pipe = SiDSanaPipeline.from_pretrained(args.model_id, torch_dtype=sid_dtype).to(device)
print("✓ SiD pipeline loaded")

# ── Load ImageReward ──────────────────────────────────────────────────────────
print("Loading ImageReward ...")
import ImageReward as RM
reward_model = RM.load("ImageReward-v1.0", device=device)
reward_model.eval()
print("✓ ImageReward loaded")

# ── Load Qwen3-4B ─────────────────────────────────────────────────────────────
print(f"Loading {args.qwen_id} ...")
from transformers import AutoTokenizer, AutoModelForCausalLM
qwen_tok = AutoTokenizer.from_pretrained(args.qwen_id)
qwen_mdl = AutoModelForCausalLM.from_pretrained(
    args.qwen_id, torch_dtype=qwen_dtype, device_map=args.qwen_device)
qwen_mdl.eval()
print("✓ Qwen3-4B loaded\n")

# ── Prompt rewriting ──────────────────────────────────────────────────────────
SYSTEM = (
    "You are a concise image prompt editor. "
    "Given a text-to-image prompt, produce a single minimally-changed rewrite. "
    "Keep the subject and composition identical. "
    "You may only change: lighting descriptors, camera/lens terms, mood adjectives, "
    "or add/remove one small detail. "
    "Output ONLY the rewritten prompt, no explanation, no quotes."
)
STYLES = [
    "Adjust the lighting or time of day slightly.",
    "Swap or add a camera/lens detail.",
    "Change one mood or atmosphere word.",
]

def rewrite(prompt: str, instruction: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"{instruction}\n\nOriginal: {prompt} /no_think"},
    ]
    text   = qwen_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_tok([text], return_tensors="pt").to(qwen_mdl.device)
    with torch.no_grad():
        out = qwen_mdl.generate(
            **inputs, max_new_tokens=120, temperature=0.6, top_p=0.9,
            do_sample=True, pad_token_id=qwen_tok.eos_token_id)
    result = qwen_tok.decode(out[0][inputs.input_ids.shape[1]:],
                             skip_special_tokens=True).strip()
    result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
    for line in result.splitlines():
        if line.strip():
            return line.strip()
    return prompt

def make_variants(prompt: str, n: int) -> list[str]:
    variants = [prompt]
    styles   = (STYLES * ((n // len(STYLES)) + 1))[:n]
    for instr in styles:
        variants.append(rewrite(prompt, instr))
    return variants

# ── SiD inference (fixed seed, no CFG search) ────────────────────────────────
@torch.no_grad()
def generate(prompt: str) -> Image.Image:
    exec_device = pipe._execution_device

    pe, pm, _, _ = pipe.encode_prompt(
        prompt=prompt, do_classifier_free_guidance=False,
        negative_prompt="", device=exec_device,
        num_images_per_prompt=1, max_sequence_length=256)

    sample_size = pipe.transformer.config.sample_size
    orig_h, orig_w = args.height, args.width
    h, w = orig_h, orig_w
    if sample_size in ASPECT_RATIO_BINS:
        h, w = pipe.image_processor.classify_height_width_bin(
            h, w, ratios=ASPECT_RATIO_BINS[sample_size])

    g = torch.Generator(device=exec_device).manual_seed(args.seed)
    latents = pipe.prepare_latents(
        1, pipe.transformer.config.in_channels, h, w,
        pe.dtype, exec_device, g)

    D_x = torch.zeros_like(latents)
    for i in range(args.steps):
        t_val  = (999.0 * (1.0 - i / args.steps)) / 999.0
        t_flat = torch.full((1,), t_val, device=exec_device, dtype=pe.dtype)
        t_4d   = t_flat.view(1, 1, 1, 1)
        noise  = latents if i == 0 else torch.randn_like(latents)
        latents = (1 - t_4d) * D_x + t_4d * noise
        flow   = pipe.transformer(
            hidden_states=latents, encoder_hidden_states=pe,
            encoder_attention_mask=pm,
            timestep=args.time_scale * t_flat, return_dict=False)[0]
        D_x = latents - t_4d * flow

    img = pipe.vae.decode(D_x / pipe.vae.config.scaling_factor, return_dict=False)[0]
    img = pipe.image_processor.resize_and_crop_tensor(img, orig_h, orig_w)
    return pipe.image_processor.postprocess(img, output_type="pil")[0]

# ── Grid builder ──────────────────────────────────────────────────────────────
def _font(size=14):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()

def make_grid(items):
    """items: list of (img, label_top, label_bottom)"""
    W, H    = items[0][0].size
    TOP_H   = 20
    BOT_H   = 30
    n       = len(items)
    grid    = Image.new("RGB", (W * n, H + TOP_H + BOT_H), (18, 18, 18))
    draw    = ImageDraw.Draw(grid)
    for i, (img, top, bot) in enumerate(items):
        grid.paste(img, (i * W, TOP_H))
        draw.text((i * W + 4, 2),          top[:40], fill=(200, 200, 200), font=_font(13))
        draw.text((i * W + 4, TOP_H + H + 4), bot,   fill=(100, 255, 100), font=_font(13))
    return grid

# ── Prompt list ───────────────────────────────────────────────────────────────
if args.prompt_file:
    prompts = [l.strip() for l in open(args.prompt_file) if l.strip()]
else:
    prompts = [args.prompt]

# ── Main loop ─────────────────────────────────────────────────────────────────
for pidx, prompt in enumerate(prompts):
    slug = f"p{pidx:02d}"
    print(f"\n{'='*60}")
    print(f"[{slug}] {prompt}")
    print(f"{'='*60}")

    # Generate variants
    print(f"Generating {args.n_variants} variants ...")
    variants = make_variants(prompt, args.n_variants)
    for vi, v in enumerate(variants):
        label = "original" if vi == 0 else f"variant {vi}"
        print(f"  {label}: {v}")

    # Save variants text
    with open(os.path.join(args.out_dir, f"{slug}_variants.txt"), "w") as f:
        for vi, v in enumerate(variants):
            f.write(f"v{vi}: {v}\n")

    # Generate + score each variant
    print(f"\nGenerating images (seed={args.seed}, cfg={args.cfg}) ...")
    items = []
    for vi, v in enumerate(variants):
        label = "original" if vi == 0 else f"v{vi}"
        print(f"  {label} ...", end=" ", flush=True)
        img   = generate(v)
        score = float(reward_model.score(prompt, img))
        print(f"IR={score:.4f}")
        img.save(os.path.join(args.out_dir, f"{slug}_v{vi}.png"))
        items.append((img, label, f"IR={score:.4f}"))

    # Save grid
    grid      = make_grid(items)
    grid_path = os.path.join(args.out_dir, f"{slug}_grid.png")
    grid.save(grid_path)
    print(f"Grid → {grid_path}")

print(f"\nDone. Outputs in: {os.path.abspath(args.out_dir)}/")