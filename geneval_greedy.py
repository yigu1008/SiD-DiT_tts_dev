"""
greedy_geneval_search.py
========================
Greedy cfg_scale search using GenEval v1 as the reward.

Two scoring modes:
  --reward_url   : HTTP reward server (Flow-GRPO style, faster)
  --geneval_python + --geneval_repo + --detector_path
                 : Subprocess fallback (no server setup needed)

If --reward_url is unreachable or not specified, falls back to subprocess.

Subprocess mode
---------------
Calls a self-contained scorer script in the GenEval conda env.
The scorer loads Mask2Former, runs detection on candidate images,
and prints JSON scores to stdout.  Slower (reloads model per call)
but requires zero extra setup beyond a working geneval env.

Usage
-----
# Subprocess mode (no server needed):
python greedy_geneval_search.py \
    --geneval_prompts /home/ygu/geneval/prompts/evaluation_metadata.jsonl \
    --geneval_python  /home/ygu/miniconda3/envs/geneval/bin/python \
    --geneval_repo    /home/ygu/geneval \
    --detector_path   /home/ygu/geneval/dectect/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth \
    --cfg_scales 1.0 1.5 2.0 2.5 3.0 4.0 \
    --steps 4 --n_samples 4 \
    --start_index 0 --end_index 10

# HTTP server mode (faster, if you set up reward-server):
python greedy_geneval_search.py \
    --geneval_prompts /home/ygu/geneval/prompts/evaluation_metadata.jsonl \
    --reward_url http://127.0.0.1:5000 \
    --steps 4 --n_samples 4
"""

import argparse, base64, glob, io, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Greedy cfg search with GenEval reward")
parser.add_argument("--model_id",    default="YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow")
parser.add_argument("--ckpt",        default=None)

# prompts
parser.add_argument("--geneval_prompts", required=True,
                    help="evaluation_metadata.jsonl  ({tag, include, prompt} per line)")
parser.add_argument("--start_index", type=int, default=0)
parser.add_argument("--end_index",   type=int, default=-1)

# scoring: HTTP mode
parser.add_argument("--reward_url",  default=None,
                    help="GenEval reward server URL (optional)")

# scoring: subprocess mode
parser.add_argument("--geneval_python", default=None,
                    help="Python binary in geneval conda env, "
                         "e.g. /home/ygu/miniconda3/envs/geneval/bin/python")
parser.add_argument("--geneval_repo",   default=None,
                    help="Path to cloned geneval repo root")
parser.add_argument("--detector_path",  default=None,
                    help="Path to Mask2Former .pth or folder with config+weights")

# model
parser.add_argument("--neg_embed",   default=None)
parser.add_argument("--steps",       type=int,   default=4)
parser.add_argument("--cfg_scales",  nargs="+",  type=float,
                    default=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
parser.add_argument("--n_samples",   type=int, default=4)
parser.add_argument("--seed",        type=int,   default=42)
parser.add_argument("--width",       type=int,   default=512)
parser.add_argument("--height",      type=int,   default=512)
parser.add_argument("--out_dir",     default="./greedy_geneval_out")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
sid_device = "cuda" if torch.cuda.is_available() else "cpu"
sid_dtype  = torch.float16


# ── Determine scoring mode ────────────────────────────────────────────────────
USE_HTTP = False
USE_SUBPROCESS = False

if args.reward_url:
    import requests
    try:
        requests.get(f"{args.reward_url}/", timeout=3)
        USE_HTTP = True
        print(f"✓ Reward server at {args.reward_url}")
    except Exception:
        print(f"⚠ Reward server unreachable at {args.reward_url}, trying subprocess mode")

if not USE_HTTP:
    if args.geneval_python and args.geneval_repo and args.detector_path:
        USE_SUBPROCESS = True
        print(f"✓ Subprocess scoring mode")
        print(f"  python:   {args.geneval_python}")
        print(f"  repo:     {args.geneval_repo}")
        print(f"  detector: {args.detector_path}")
    else:
        print("ERROR: No scoring method available.")
        print("  Either provide --reward_url (HTTP server)")
        print("  or all of: --geneval_python, --geneval_repo, --detector_path")
        sys.exit(1)


# ── SiD pipeline ───────────────────────────────────────────────────────────────
repo_root = Path(__file__).parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from sid import SiDSanaPipeline
except ImportError as e:
    sys.exit(f"Cannot import SiDSanaPipeline: {e}")

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
    print(f"Loading checkpoint {args.ckpt} ...")
    raw = torch.load(args.ckpt, map_location=sid_device, weights_only=False)
    def _unwrap(d, depth=0):
        if not isinstance(d, dict): return d
        dotted = sum(1 for k in d if "." in str(k))
        if dotted / max(len(d), 1) > 0.5: return d
        if depth > 4: return d
        for key in ("ema","ema_model","model_ema","model","state_dict","generator","G_state"):
            if key in d and isinstance(d[key], dict):
                return _unwrap(d[key], depth+1)
        return d
    sd = _unwrap(raw)
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    m, u = pipe.transformer.load_state_dict(sd, strict=False)
    print(f"  loaded {len(sd)-len(u)}/{len(sd)} params  missing={len(m)} unexpected={len(u)}")

pipe.transformer.eval()
TIME_SCALE = 1000.0
LATENT_C   = pipe.transformer.config.in_channels
OUT_C      = pipe.transformer.config.out_channels
VARIANCE_SPLIT = (OUT_C // 2 == LATENT_C)
print(f"✓ SiD  variance_split={VARIANCE_SPLIT}")

# ── Neg embed ──────────────────────────────────────────────────────────────────
pretrained_neg_embeds = pretrained_neg_mask = None
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

# ── Load prompts ───────────────────────────────────────────────────────────────
def load_prompts(path, start=0, end=-1):
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip(): entries.append(json.loads(line))
    if end == -1: end = len(entries)
    return entries[start:end]

prompt_entries = load_prompts(args.geneval_prompts, args.start_index, args.end_index)
print(f"✓ {len(prompt_entries)} prompts loaded")


# ══════════════════════════════════════════════════════════════════════════════
# SCORING: two backends, same interface
# ══════════════════════════════════════════════════════════════════════════════

# ── Subprocess scorer script (written to out_dir, runs in geneval env) ────────
SCORER_SCRIPT = r'''
"""
_geneval_scorer.py — runs in geneval conda env via subprocess.
Loads Mask2Former, scores candidate images, prints JSON to stdout.
"""
import argparse, glob, json, os, sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", required=True)
parser.add_argument("--metadata_json", required=True)
parser.add_argument("--detector_path", required=True)
parser.add_argument("--geneval_repo", required=True)
args = parser.parse_args()

sys.path.insert(0, args.geneval_repo)
sys.path.insert(0, os.path.join(args.geneval_repo, "evaluation"))

metadata = json.loads(args.metadata_json)
include_spec = metadata["include"]
if isinstance(include_spec, str):
    include_spec = eval(include_spec)
tag = metadata.get("tag", "unknown")

# ── Find config + checkpoint ──────────────────────────────────────────────
det_path = args.detector_path

# If detector_path is a .pth file, look for .py config next to it or in mmdet
if det_path.endswith(".pth"):
    ckpt_file = det_path
    # Try to find config in same directory
    det_dir = os.path.dirname(det_path)
    configs = glob.glob(os.path.join(det_dir, "*.py"))
    if configs:
        config_file = configs[0]
    else:
        # Try mmdetection configs
        try:
            import mmdet
            mmdet_root = os.path.dirname(os.path.dirname(mmdet.__file__))
            config_file = os.path.join(
                mmdet_root, "configs", "mask2former",
                "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py")
            if not os.path.exists(config_file):
                # panoptic variant
                config_file = os.path.join(
                    mmdet_root, "configs", "mask2former",
                    "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco-panoptic.py")
        except:
            config_file = None
else:
    # detector_path is a directory
    configs = glob.glob(os.path.join(det_path, "*.py"))
    ckpts   = glob.glob(os.path.join(det_path, "*.pth"))
    config_file = configs[0] if configs else None
    ckpt_file   = ckpts[0]  if ckpts  else None

if not config_file or not os.path.exists(config_file):
    print(json.dumps([{"file":"error","score":0.0,"correct":False,
        "error":f"config not found for {det_path}"}]))
    sys.exit(0)
if not ckpt_file or not os.path.exists(ckpt_file):
    print(json.dumps([{"file":"error","score":0.0,"correct":False,
        "error":f"ckpt not found for {det_path}"}]))
    sys.exit(0)

# ── Load detector ─────────────────────────────────────────────────────────
from mmdet.apis import init_detector, inference_detector
model = init_detector(config_file, ckpt_file, device="cuda:0")
CLASSES = model.CLASSES if hasattr(model, 'CLASSES') else []
class_to_idx = {name.lower(): idx for idx, name in enumerate(CLASSES)}

# ── Score one image ───────────────────────────────────────────────────────
def score_image(image_path):
    result = inference_detector(model, image_path)
    bbox_results = result[0] if isinstance(result, tuple) else result

    scores_per_obj = []
    is_correct = True
    detected_info = {}

    for obj_spec in include_spec:
        cls_name = obj_spec["class"].lower()
        required = obj_spec.get("count", 1)

        cls_idx = class_to_idx.get(cls_name, -1)
        if cls_idx == -1:
            # partial match
            for cn, ci in class_to_idx.items():
                if cls_name in cn or cn in cls_name:
                    cls_idx = ci; break

        if cls_idx == -1 or cls_idx >= len(bbox_results):
            scores_per_obj.append(0.0)
            is_correct = False
            continue

        dets = bbox_results[cls_idx]
        if len(dets) > 0:
            dets = dets[dets[:, 4] > 0.3]
        n_det = len(dets)
        max_conf = float(dets[:, 4].max()) if n_det > 0 else 0.0

        # Flow-GRPO style: counting reward = 1 - |Ngen - Nref| / Nref
        count_score = max(0.0, 1.0 - abs(n_det - required) / required)
        obj_score = count_score * max(max_conf, 0.5 if n_det > 0 else 0.0)
        scores_per_obj.append(obj_score)

        if n_det != required and tag == "counting":
            is_correct = False
        elif n_det < required:
            is_correct = False

        detected_info[cls_name] = n_det

    # Position check
    if tag == "position" and len(include_spec) >= 2:
        position = metadata.get("position", None)
        if position and len(include_spec) >= 2:
            cls_a = include_spec[0]["class"].lower()
            cls_b = include_spec[1]["class"].lower()
            idx_a = class_to_idx.get(cls_a, -1)
            idx_b = class_to_idx.get(cls_b, -1)
            if idx_a >= 0 and idx_b >= 0:
                dets_a = bbox_results[idx_a]
                dets_b = bbox_results[idx_b]
                if len(dets_a) > 0: dets_a = dets_a[dets_a[:,4] > 0.3]
                if len(dets_b) > 0: dets_b = dets_b[dets_b[:,4] > 0.3]
                if len(dets_a) > 0 and len(dets_b) > 0:
                    ca = (dets_a[0,:2] + dets_a[0,2:4]) / 2
                    cb = (dets_b[0,:2] + dets_b[0,2:4]) / 2
                    ok = False
                    if position == "left of"  and ca[0] < cb[0]: ok = True
                    if position == "right of" and ca[0] > cb[0]: ok = True
                    if position == "above"    and ca[1] < cb[1]: ok = True
                    if position == "below"    and ca[1] > cb[1]: ok = True
                    scores_per_obj.append(1.0 if ok else 0.0)
                    if not ok: is_correct = False
                else:
                    scores_per_obj.append(0.0)
                    is_correct = False

    soft = float(np.mean(scores_per_obj)) if scores_per_obj else 0.0
    return {"score": soft, "correct": is_correct, "detected": detected_info}

# ── Score all candidates ──────────────────────────────────────────────────
image_files = sorted(glob.glob(os.path.join(args.image_dir, "candidate_*.png")))
results = []
for img_path in image_files:
    info = score_image(img_path)
    results.append({
        "file": os.path.basename(img_path),
        "score": info["score"],
        "correct": info["correct"],
        "detected": info["detected"],
    })

print(json.dumps(results))
'''

SCORER_PATH = os.path.join(args.out_dir, "_geneval_scorer.py")
if USE_SUBPROCESS:
    with open(SCORER_PATH, "w") as f:
        f.write(SCORER_SCRIPT)
    print(f"✓ Scorer script written to {SCORER_PATH}")


# ── Subprocess scoring function ────────────────────────────────────────────────
def score_via_subprocess(images: list[Image.Image], metadata: dict) -> list[float]:
    """Save images to tmpdir, call scorer in geneval env, parse scores."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, img in enumerate(images):
            img.save(os.path.join(tmpdir, f"candidate_{i:03d}.png"))

        meta_json = json.dumps(metadata)
        cmd = [
            args.geneval_python, SCORER_PATH,
            "--image_dir",      tmpdir,
            "--metadata_json",  meta_json,
            "--detector_path",  args.detector_path,
            "--geneval_repo",   args.geneval_repo,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if r.returncode != 0:
            print(f"  ⚠ Scorer error (rc={r.returncode}): {r.stderr[:300]}")
            return [0.0] * len(images)

        # Parse JSON from stdout (skip warnings)
        for line in r.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("["):
                try:
                    results = json.loads(line)
                    return [x["score"] for x in results]
                except json.JSONDecodeError:
                    continue

        print(f"  ⚠ Could not parse scorer output: {r.stdout[:200]}")
        return [0.0] * len(images)


# ── HTTP scoring function ─────────────────────────────────────────────────────
def score_via_http(images: list[Image.Image], metadata: dict) -> list[float]:
    """POST images to reward server, get scores back."""
    import requests
    payload = {
        "images": [pil_to_b64(img) for img in images],
        "metadata": metadata,
    }
    try:
        r = requests.post(f"{args.reward_url}/geneval", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["scores"]
    except Exception as e:
        print(f"  ⚠ HTTP error: {e}")
        return [0.0] * len(images)

def pil_to_b64(img):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Unified scoring interface ──────────────────────────────────────────────────
def score_batch(images: list[Image.Image], metadata: dict) -> list[float]:
    if USE_HTTP:
        return score_via_http(images, metadata)
    else:
        return score_via_subprocess(images, metadata)

def score_single(img: Image.Image, metadata: dict) -> tuple[float, bool]:
    scores = score_batch([img], metadata)
    # For binary: run subprocess returns score; threshold at 1.0 for "correct"
    return scores[0], scores[0] >= 0.99


# ══════════════════════════════════════════════════════════════════════════════
# SiD generation helpers
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_prompt(prompt, max_seq=256):
    pe, pm, ne, nm = pipe.encode_prompt(
        prompt=prompt, do_classifier_free_guidance=True,
        negative_prompt="", device=sid_device,
        num_images_per_prompt=1, max_sequence_length=max_seq)
    ue, um = ne.detach(), nm.detach()
    if pretrained_neg_embeds is not None:
        L_c = pe.shape[1]; L_n = pretrained_neg_embeds.shape[1]
        pne, pnm = pretrained_neg_embeds, pretrained_neg_mask
        if L_n < L_c:
            pne = torch.cat([pne, torch.zeros(1,L_c-L_n,pne.shape[2],device=sid_device,dtype=pne.dtype)],1)
            pnm = torch.cat([pnm, torch.zeros(1,L_c-L_n,device=sid_device,dtype=pnm.dtype)],1)
        elif L_n > L_c:
            pne, pnm = pne[:,:L_c], pnm[:,:L_c]
        ue, um = pne.to(pe.dtype), pnm
    return pe.detach(), pm.detach(), ue, um

@torch.no_grad()
def transformer_step(latents, pe, pm, ue, um, t_flat, cfg):
    if cfg == 1.0:
        v = pipe.transformer(hidden_states=latents, encoder_hidden_states=pe,
            encoder_attention_mask=pm, timestep=TIME_SCALE*t_flat, return_dict=False)[0]
        if VARIANCE_SPLIT: v = v.chunk(2,dim=1)[0]
        return v
    latent_in = torch.cat([latents,latents])
    embeds_in = torch.cat([ue,pe]); attn_in = torch.cat([um,pm])
    t_in = torch.cat([t_flat,t_flat])
    flow_both = pipe.transformer(hidden_states=latent_in, encoder_hidden_states=embeds_in,
        encoder_attention_mask=attn_in, timestep=TIME_SCALE*t_in, return_dict=False)[0]
    if VARIANCE_SPLIT: flow_both = flow_both.chunk(2,dim=1)[0]
    fu, fc = flow_both.chunk(2)
    return fu + cfg*(fc - fu)

def make_latents(seed, h, w, dtype, device):
    try:
        g = torch.Generator(device=device).manual_seed(seed)
        return pipe.prepare_latents(1, LATENT_C, h, w, dtype, device, g)
    except:
        torch.manual_seed(seed)
        return torch.randn(1,LATENT_C,h,w,device=device,dtype=dtype)

@torch.no_grad()
def decode_to_pil(D_x, oh, ow):
    img = pipe.vae.decode(D_x / pipe.vae.config.scaling_factor, return_dict=False)[0]
    img = pipe.image_processor.resize_and_crop_tensor(img, oh, ow)
    return pipe.image_processor.postprocess(img, output_type="pil")[0]

@torch.no_grad()
def run_baseline(prompt, seed, h, w, oh, ow):
    pe,pm,ue,um = encode_prompt(prompt)
    lat = make_latents(seed, h, w, pe.dtype, sid_device)
    D_x = torch.zeros_like(lat)
    for i in range(args.steps):
        st = 999.0*(1.0-i/args.steps)
        tf = torch.full((1,), st/999.0, device=sid_device, dtype=pe.dtype)
        t4 = tf.view(1,1,1,1)
        noise = lat if i==0 else torch.randn_like(lat)
        lat = (1.0-t4)*D_x + t4*noise
        v = pipe.transformer(hidden_states=lat, encoder_hidden_states=pe,
            encoder_attention_mask=pm, timestep=TIME_SCALE*tf, return_dict=False)[0]
        if VARIANCE_SPLIT: v = v.chunk(2,dim=1)[0]
        D_x = lat - t4*v
    return decode_to_pil(D_x, oh, ow)

@torch.no_grad()
def run_greedy(prompt, metadata, seed, h, w, oh, ow):
    pe,pm,ue,um = encode_prompt(prompt)
    lat = make_latents(seed, h, w, pe.dtype, sid_device)
    D_x = torch.zeros_like(lat)
    chosen = []
    for i in range(args.steps):
        st = 999.0*(1.0-i/args.steps)
        tf = torch.full((1,), st/999.0, device=sid_device, dtype=lat.dtype)
        t4 = tf.view(1,1,1,1)
        noise = lat if i==0 else torch.randn_like(lat)
        lat = (1.0-t4)*D_x + t4*noise
        print(f"    step {i+1}/{args.steps}  t={st/999:.3f}")

        cand_Dx, cand_imgs = [], []
        for cfg in args.cfg_scales:
            fp = transformer_step(lat, pe, pm, ue, um, tf, cfg)
            dx = lat - t4*fp; cand_Dx.append(dx)
            cand_imgs.append(decode_to_pil(dx, oh, ow))

        scores = score_batch(cand_imgs, metadata)
        bi = int(np.argmax(scores))
        for ci,(cfg,sc) in enumerate(zip(args.cfg_scales, scores)):
            print(f"      γ={cfg:.2f}  GE={sc:.4f}{'  ← BEST' if ci==bi else ''}")
        D_x = cand_Dx[bi].clone()
        chosen.append((args.cfg_scales[bi], scores[bi]))
    torch.cuda.empty_cache()
    return decode_to_pil(D_x, oh, ow), chosen


# ── GenEval folder layout ─────────────────────────────────────────────────────
def save_geneval_fmt(img, meta, idx, root, si=0):
    d = os.path.join(root, f"{idx:05d}", "samples"); os.makedirs(d, exist_ok=True)
    with open(os.path.join(root, f"{idx:05d}", "metadata.jsonl"), "w") as f:
        json.dump(meta, f)
    img.save(os.path.join(d, f"{si:04d}.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════
summary = []
ge_dir = os.path.join(args.out_dir, "geneval_images")
os.makedirs(ge_dir, exist_ok=True)

for pidx, entry in enumerate(prompt_entries):
    gi = args.start_index + pidx
    prompt, tag = entry["prompt"], entry.get("tag","unknown")
    slug = f"p{gi:05d}"
    print(f"\n{'='*72}\n[{slug}] tag={tag}  {prompt}\n{'='*72}")

    ss = pipe.transformer.config.sample_size
    oh, ow = args.height, args.width; h, w = oh, ow
    if ss in ASPECT_RATIO_BINS:
        h, w = pipe.image_processor.classify_height_width_bin(h, w, ratios=ASPECT_RATIO_BINS[ss])

    sr = []
    for si in range(args.n_samples):
        sd = args.seed + si
        print(f"\n  sample {si+1}/{args.n_samples}  seed={sd}")

        # Baseline
        base_img = run_baseline(prompt, sd, h, w, oh, ow)
        base_soft, base_pass = score_single(base_img, entry)
        print(f"  Baseline: GE={base_soft:.4f} pass={'✓' if base_pass else '✗'}")

        # Greedy search
        print(f"  Greedy search ({len(args.cfg_scales)} cfg × {args.steps} steps):")
        greedy_img, chosen = run_greedy(prompt, entry, sd, h, w, oh, ow)
        greedy_soft, greedy_pass = score_single(greedy_img, entry)
        print(f"  Greedy:   GE={greedy_soft:.4f} pass={'✓' if greedy_pass else '✗'}"
              f"  Δ={greedy_soft - base_soft:+.4f}")

        base_img.save(os.path.join(args.out_dir, f"{slug}_s{si}_baseline.png"))
        greedy_img.save(os.path.join(args.out_dir, f"{slug}_s{si}_greedy.png"))
        save_geneval_fmt(greedy_img, entry, gi, ge_dir, si)

        sr.append({"seed":sd,
                    "base_soft":base_soft, "base_pass":base_pass,
                    "greedy_soft":greedy_soft, "greedy_pass":greedy_pass,
                    "actions":chosen})

    bp = np.mean([s["base_pass"]   for s in sr])
    gp = np.mean([s["greedy_pass"] for s in sr])
    summary.append({"slug":slug,"index":gi,"tag":tag,"prompt":prompt,
                     "base_pass_rate":float(bp),"greedy_pass_rate":float(gp),
                     "samples":sr})
    print(f"\n  PROMPT: base pass={bp:.2f}  greedy pass={gp:.2f}")


# ── Summary ────────────────────────────────────────────────────────────────────
jp = os.path.join(args.out_dir, "summary.json")
with open(jp,"w") as f: json.dump(summary, f, indent=2)

print(f"\n{'='*72}\nRESULTS\n{'='*72}")
tags = sorted(set(r["tag"] for r in summary))
print(f"\n{'task':<20} {'N':>4} {'base':>8} {'greedy':>8} {'Δ':>8}")
print("-"*52)
tb=tg=0
for tag in tags:
    rows = [r for r in summary if r["tag"]==tag]
    b = np.mean([r["base_pass_rate"] for r in rows])
    g = np.mean([r["greedy_pass_rate"] for r in rows])
    print(f"{tag:<20} {len(rows):>4} {b:>8.3f} {g:>8.3f} {g-b:>+8.3f}")
    tb+=b*len(rows); tg+=g*len(rows)
if summary:
    n=len(summary)
    print(f"{'OVERALL':<20} {n:>4} {tb/n:>8.3f} {tg/n:>8.3f} {(tg-tb)/n:>+8.3f}")

print(f"\n→ {jp}")
print(f"→ {os.path.abspath(ge_dir)}/")
print(f"\nOfficial eval:")
print(f"  python evaluate_images.py {os.path.abspath(ge_dir)} \\")
print(f"      --outfile results.jsonl --model-path <DETECTOR>")
print(f"  python summary_scores.py results.jsonl")