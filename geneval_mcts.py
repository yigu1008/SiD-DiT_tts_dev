"""
mcts_geneval_search.py
======================
MCTS (Monte Carlo Tree Search) over cfg_scale actions at each SiD denoising
step, using GenEval v1 as the reward.

Scoring: subprocess (geneval env) or HTTP reward server (Flow-GRPO style).

Tree structure:
  - Depth = number of denoising steps (default 4)
  - Each node = denoising state (latents, D_x) at a step
  - Each edge = action (cfg_scale)
  - Action space: n_cfg (default 6)

Each MCTS simulation:
  1. SELECT  — walk tree via UCB1 until reaching an unexplored action
  2. EXPAND  — apply transformer_step to get the child state
  3. ROLLOUT — random actions to the leaf, then decode + GenEval score
  4. BACKUP  — propagate the score up the path

Usage
-----
python mcts_geneval_search.py \
    --geneval_prompts /path/to/evaluation_metadata.jsonl \
    --geneval_python  /path/to/envs/geneval/bin/python \
    --geneval_repo    /path/to/geneval \
    --detector_path   /path/to/mask2former.pth \
    --n_sims 30 --ucb_c 1.41 \
    --cfg_scales 1.0 1.5 2.0 2.5 3.0 4.0 \
    --start_index 0 --end_index 10
"""

import argparse, base64, glob, io, json, math, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="MCTS cfg search with GenEval reward")
parser.add_argument("--model_id",    default="YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow")
parser.add_argument("--ckpt",        default=None)

# prompts
parser.add_argument("--geneval_prompts", required=True)
parser.add_argument("--start_index", type=int, default=0)
parser.add_argument("--end_index",   type=int, default=-1)

# scoring: HTTP mode
parser.add_argument("--reward_url",  default=None)

# scoring: subprocess mode
parser.add_argument("--geneval_python", default=None)
parser.add_argument("--geneval_repo",   default=None)
parser.add_argument("--detector_path",  default=None)

# model
parser.add_argument("--neg_embed",   default=None)
parser.add_argument("--steps",       type=int,   default=4)
parser.add_argument("--cfg_scales",  nargs="+",  type=float,
                    default=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
parser.add_argument("--n_samples",   type=int, default=4)
parser.add_argument("--seed",        type=int,   default=42)
parser.add_argument("--width",       type=int,   default=512)
parser.add_argument("--height",      type=int,   default=512)
parser.add_argument("--out_dir",     default="./mcts_geneval_out")

# MCTS
parser.add_argument("--n_sims",      type=int,   default=30,
                    help="MCTS simulations per prompt. Lower than ImageReward "
                         "version since GenEval scoring is heavier.")
parser.add_argument("--ucb_c",       type=float, default=1.41)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
sid_device = "cuda" if torch.cuda.is_available() else "cpu"
sid_dtype  = torch.float16


# ── Determine scoring mode ────────────────────────────────────────────────────
USE_HTTP = False
USE_SUBPROCESS = False

if args.reward_url:
    import requests as _req
    try:
        _req.get(f"{args.reward_url}/", timeout=3)
        USE_HTTP = True
        print(f"✓ Reward server at {args.reward_url}")
    except Exception:
        print(f"⚠ Reward server unreachable, trying subprocess mode")

if not USE_HTTP:
    if args.geneval_python and args.geneval_repo and args.detector_path:
        USE_SUBPROCESS = True
        print(f"✓ Subprocess scoring mode")
        print(f"  python:   {args.geneval_python}")
        print(f"  repo:     {args.geneval_repo}")
        print(f"  detector: {args.detector_path}")
    else:
        print("ERROR: No scoring method available.")
        print("  Provide --reward_url  OR  --geneval_python + --geneval_repo + --detector_path")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# SiD pipeline
# ═══════════════════════════════════════════════════════════════════════════════
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
    from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
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


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING (identical to greedy_geneval_search.py)
# ═══════════════════════════════════════════════════════════════════════════════

SCORER_SCRIPT = r'''
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

det_path = args.detector_path
if det_path.endswith(".pth"):
    ckpt_file = det_path
    det_dir = os.path.dirname(det_path)
    configs = glob.glob(os.path.join(det_dir, "*.py"))
    if configs:
        config_file = configs[0]
    else:
        try:
            import mmdet
            mmdet_root = os.path.dirname(os.path.dirname(mmdet.__file__))
            config_file = os.path.join(mmdet_root, "configs", "mask2former",
                "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py")
            if not os.path.exists(config_file):
                config_file = os.path.join(mmdet_root, "configs", "mask2former",
                    "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco-panoptic.py")
        except:
            config_file = None
else:
    configs = glob.glob(os.path.join(det_path, "*.py"))
    ckpts   = glob.glob(os.path.join(det_path, "*.pth"))
    config_file = configs[0] if configs else None
    ckpt_file   = ckpts[0]  if ckpts  else None

if not config_file or not os.path.exists(str(config_file)):
    print(json.dumps([{"file":"error","score":0.0,"correct":False,
        "error":f"config not found for {det_path}"}]))
    sys.exit(0)
if not ckpt_file or not os.path.exists(str(ckpt_file)):
    print(json.dumps([{"file":"error","score":0.0,"correct":False,
        "error":f"ckpt not found for {det_path}"}]))
    sys.exit(0)

from mmdet.apis import init_detector, inference_detector
model = init_detector(config_file, ckpt_file, device="cuda:0")
CLASSES = model.CLASSES if hasattr(model, 'CLASSES') else []
class_to_idx = {name.lower(): idx for idx, name in enumerate(CLASSES)}

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
            for cn, ci in class_to_idx.items():
                if cls_name in cn or cn in cls_name:
                    cls_idx = ci; break
        if cls_idx == -1 or cls_idx >= len(bbox_results):
            scores_per_obj.append(0.0); is_correct = False; continue
        dets = bbox_results[cls_idx]
        if len(dets) > 0: dets = dets[dets[:, 4] > 0.3]
        n_det = len(dets)
        max_conf = float(dets[:, 4].max()) if n_det > 0 else 0.0
        count_score = max(0.0, 1.0 - abs(n_det - required) / required)
        obj_score = count_score * max(max_conf, 0.5 if n_det > 0 else 0.0)
        scores_per_obj.append(obj_score)
        if n_det != required and tag == "counting": is_correct = False
        elif n_det < required: is_correct = False
        detected_info[cls_name] = n_det
    if tag == "position" and len(include_spec) >= 2:
        position = metadata.get("position", None)
        if position:
            cls_a, cls_b = include_spec[0]["class"].lower(), include_spec[1]["class"].lower()
            idx_a, idx_b = class_to_idx.get(cls_a,-1), class_to_idx.get(cls_b,-1)
            if idx_a >= 0 and idx_b >= 0:
                da, db = bbox_results[idx_a], bbox_results[idx_b]
                if len(da)>0: da = da[da[:,4]>0.3]
                if len(db)>0: db = db[db[:,4]>0.3]
                if len(da)>0 and len(db)>0:
                    ca = (da[0,:2]+da[0,2:4])/2; cb = (db[0,:2]+db[0,2:4])/2
                    ok = False
                    if position=="left of" and ca[0]<cb[0]: ok=True
                    if position=="right of" and ca[0]>cb[0]: ok=True
                    if position=="above" and ca[1]<cb[1]: ok=True
                    if position=="below" and ca[1]>cb[1]: ok=True
                    scores_per_obj.append(1.0 if ok else 0.0)
                    if not ok: is_correct = False
                else: scores_per_obj.append(0.0); is_correct = False
    soft = float(np.mean(scores_per_obj)) if scores_per_obj else 0.0
    return {"score": soft, "correct": is_correct, "detected": detected_info}

image_files = sorted(glob.glob(os.path.join(args.image_dir, "candidate_*.png")))
results = []
for img_path in image_files:
    info = score_image(img_path)
    results.append({"file": os.path.basename(img_path), "score": info["score"],
        "correct": info["correct"], "detected": info["detected"]})
print(json.dumps(results))
'''

SCORER_PATH = os.path.join(args.out_dir, "_geneval_scorer.py")
if USE_SUBPROCESS:
    with open(SCORER_PATH, "w") as f:
        f.write(SCORER_SCRIPT)
    print(f"✓ Scorer script: {SCORER_PATH}")


def score_via_subprocess(images, metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, img in enumerate(images):
            img.save(os.path.join(tmpdir, f"candidate_{i:03d}.png"))
        r = subprocess.run(
            [args.geneval_python, SCORER_PATH,
             "--image_dir", tmpdir,
             "--metadata_json", json.dumps(metadata),
             "--detector_path", args.detector_path,
             "--geneval_repo", args.geneval_repo],
            capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            print(f"  ⚠ Scorer error: {r.stderr[:200]}")
            return [0.0] * len(images)
        for line in r.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("["):
                try:
                    return [x["score"] for x in json.loads(line)]
                except: continue
        print(f"  ⚠ Parse error: {r.stdout[:200]}")
        return [0.0] * len(images)


def score_via_http(images, metadata):
    import requests
    def _b64(img):
        buf = io.BytesIO(); img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    try:
        r = requests.post(f"{args.reward_url}/geneval",
            json={"images": [_b64(i) for i in images], "metadata": metadata},
            timeout=120)
        r.raise_for_status()
        return r.json()["scores"]
    except Exception as e:
        print(f"  ⚠ HTTP error: {e}")
        return [0.0] * len(images)


def score_batch(images, metadata):
    if USE_HTTP:
        return score_via_http(images, metadata)
    return score_via_subprocess(images, metadata)


def score_single_image(img, metadata):
    """Score one image. Returns (soft_score, is_correct)."""
    scores = score_batch([img], metadata)
    return scores[0], scores[0] >= 0.99


# ═══════════════════════════════════════════════════════════════════════════════
# SiD generation helpers
# ═══════════════════════════════════════════════════════════════════════════════

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
        if VARIANCE_SPLIT: v = v.chunk(2, dim=1)[0]
        return v
    flow_both = pipe.transformer(
        hidden_states=torch.cat([latents, latents]),
        encoder_hidden_states=torch.cat([ue, pe]),
        encoder_attention_mask=torch.cat([um, pm]),
        timestep=TIME_SCALE * torch.cat([t_flat, t_flat]),
        return_dict=False)[0]
    if VARIANCE_SPLIT: flow_both = flow_both.chunk(2, dim=1)[0]
    fu, fc = flow_both.chunk(2)
    return fu + cfg * (fc - fu)


def make_latents(seed, h, w, dtype, device):
    try:
        g = torch.Generator(device=device).manual_seed(seed)
        return pipe.prepare_latents(1, LATENT_C, h, w, dtype, device, g)
    except:
        torch.manual_seed(seed)
        return torch.randn(1, LATENT_C, h, w, device=device, dtype=dtype)


@torch.no_grad()
def decode_to_pil(D_x, oh, ow):
    img = pipe.vae.decode(D_x / pipe.vae.config.scaling_factor, return_dict=False)[0]
    img = pipe.image_processor.resize_and_crop_tensor(img, oh, ow)
    return pipe.image_processor.postprocess(img, output_type="pil")[0]


# ═══════════════════════════════════════════════════════════════════════════════
# MCTS
# ═══════════════════════════════════════════════════════════════════════════════

class MCTSNode:
    __slots__ = ("step", "D_x", "latents", "children", "N",
                 "action_N", "action_Q", "parent", "parent_action")

    def __init__(self, step, D_x, latents, parent=None, parent_action=None):
        self.step    = step
        self.D_x     = D_x
        self.latents = latents
        self.parent  = parent
        self.parent_action = parent_action
        self.children = {}
        self.action_N = {}
        self.action_Q = {}
        self.N = 0

    def is_leaf(self, n_steps):
        return self.step >= n_steps

    def untried_actions(self, actions):
        return [a for a in actions if a not in self.action_N]

    def ucb1(self, action, c):
        n = self.action_N.get(action, 0)
        if n == 0: return float("inf")
        return self.action_Q[action] / n + c * math.sqrt(math.log(self.N) / n)

    def best_action_ucb(self, actions, c):
        return max(actions, key=lambda a: self.ucb1(a, c))

    def best_action_exploit(self, actions):
        best_a, best_q = None, -float("inf")
        for a in actions:
            n = self.action_N.get(a, 0)
            if n > 0:
                q = self.action_Q[a] / n
                if q > best_q:
                    best_q = q; best_a = a
        return best_a


@torch.no_grad()
def run_mcts(prompt, metadata, seed, h, w, oh, ow):
    pe, pm, ue, um = encode_prompt(prompt)
    actions = list(args.cfg_scales)  # action = cfg_scale (float)
    n_actions = len(actions)

    # Step schedule
    latents_init = make_latents(seed, h, w, pe.dtype, sid_device)
    step_schedule = []
    for i in range(args.steps):
        scalar_t = 999.0 * (1.0 - i / args.steps)
        t_flat = torch.full((1,), scalar_t / 999.0,
                             device=sid_device, dtype=latents_init.dtype)
        t_4d = t_flat.view(1, 1, 1, 1)
        step_schedule.append((t_flat, t_4d))

    # Initial state
    D_x_init = torch.zeros_like(latents_init)
    t0_val, t0_4d = step_schedule[0]
    latents_0 = (1.0 - t0_4d) * D_x_init + t0_4d * latents_init

    root = MCTSNode(step=0, D_x=D_x_init, latents=latents_0)

    best_global_score = -float("inf")
    best_global_D_x   = None
    best_global_path  = []

    print(f"    MCTS: {args.n_sims} sims, {n_actions} actions/step, "
          f"{args.steps} steps, c={args.ucb_c}")

    for sim in range(args.n_sims):
        # ── SELECT ──
        node = root
        path = []

        while not node.is_leaf(args.steps):
            untried = node.untried_actions(actions)
            if untried:
                action = untried[np.random.randint(len(untried))]
                break
            else:
                action = node.best_action_ucb(actions, args.ucb_c)
                path.append((node, action))
                node = node.children[action]

        # ── EXPAND ──
        if not node.is_leaf(args.steps):
            if action not in node.children:
                cfg = action
                t_val, t_4d = step_schedule[node.step]
                flow_pred = transformer_step(
                    node.latents, pe, pm, ue, um, t_val, cfg)
                new_D_x = node.latents - t_4d * flow_pred

                next_step = node.step + 1
                if next_step < len(step_schedule):
                    t_next_val, t_next_4d = step_schedule[next_step]
                    noise = torch.randn_like(new_D_x)
                    new_latents = (1.0 - t_next_4d) * new_D_x + t_next_4d * noise
                else:
                    new_latents = None

                child = MCTSNode(step=next_step, D_x=new_D_x,
                                 latents=new_latents,
                                 parent=node, parent_action=action)
                node.children[action] = child

            path.append((node, action))
            node = node.children[action]

        # ── ROLLOUT (random to leaf) ──
        rollout_D_x = node.D_x
        rollout_latents = node.latents
        rollout_step = node.step

        while rollout_step < args.steps:
            cfg = actions[np.random.randint(n_actions)]
            t_val, t_4d = step_schedule[rollout_step]
            flow_pred = transformer_step(
                rollout_latents, pe, pm, ue, um, t_val, cfg)
            rollout_D_x = rollout_latents - t_4d * flow_pred
            rollout_step += 1
            if rollout_step < args.steps:
                t_next_val, t_next_4d = step_schedule[rollout_step]
                noise = torch.randn_like(rollout_D_x)
                rollout_latents = (1.0 - t_next_4d) * rollout_D_x + t_next_4d * noise

        # ── EVALUATE ──
        img = decode_to_pil(rollout_D_x, oh, ow)
        score = score_batch([img], metadata)[0]

        if score > best_global_score:
            best_global_score = score
            best_global_D_x = rollout_D_x.clone()
            best_global_path = [a for _, a in path]

        # ── BACKUP ──
        for (p_node, p_action) in path:
            p_node.N += 1
            p_node.action_N[p_action] = p_node.action_N.get(p_action, 0) + 1
            p_node.action_Q[p_action] = p_node.action_Q.get(p_action, 0.0) + score

        if (sim + 1) % 5 == 0 or sim == 0:
            print(f"      sim {sim+1:3d}/{args.n_sims}  "
                  f"best_GE={best_global_score:.4f}  root_visits={root.N}")

    # ── Extract best trajectory by exploitation ──
    best_path_exploit = []
    node = root
    for step_i in range(args.steps):
        best_a = node.best_action_exploit(actions)
        if best_a is None: break
        best_path_exploit.append(best_a)
        if best_a in node.children:
            node = node.children[best_a]
        else: break

    # Replay exploit path
    replay_D_x = D_x_init
    replay_latents = latents_0
    for step_i, cfg in enumerate(best_path_exploit):
        t_val, t_4d = step_schedule[step_i]
        flow_pred = transformer_step(replay_latents, pe, pm, ue, um, t_val, cfg)
        replay_D_x = replay_latents - t_4d * flow_pred
        if step_i + 1 < args.steps:
            t_next_val, t_next_4d = step_schedule[step_i + 1]
            noise = torch.randn_like(replay_D_x)
            replay_latents = (1.0 - t_next_4d) * replay_D_x + t_next_4d * noise

    exploit_img = decode_to_pil(replay_D_x, oh, ow)
    exploit_score = score_batch([exploit_img], metadata)[0]

    if exploit_score >= best_global_score:
        final_img, final_score, final_path = exploit_img, exploit_score, best_path_exploit
        print(f"    → exploit path GE={exploit_score:.4f}")
    else:
        final_img = decode_to_pil(best_global_D_x, oh, ow)
        final_score, final_path = best_global_score, best_global_path
        print(f"    → best sim GE={best_global_score:.4f}")

    for si, cfg in enumerate(final_path):
        print(f"      step {si+1}: γ={cfg:.2f}")

    return final_img, final_score, final_path


# ── Baseline ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_baseline(prompt, seed, h, w, oh, ow):
    pe, pm, ue, um = encode_prompt(prompt)
    lat = make_latents(seed, h, w, pe.dtype, sid_device)
    D_x = torch.zeros_like(lat)
    for i in range(args.steps):
        st = 999.0 * (1.0 - i / args.steps)
        tf = torch.full((1,), st/999.0, device=sid_device, dtype=pe.dtype)
        t4 = tf.view(1,1,1,1)
        noise = lat if i == 0 else torch.randn_like(lat)
        lat = (1.0 - t4) * D_x + t4 * noise
        v = pipe.transformer(hidden_states=lat, encoder_hidden_states=pe,
            encoder_attention_mask=pm, timestep=TIME_SCALE*tf, return_dict=False)[0]
        if VARIANCE_SPLIT: v = v.chunk(2, dim=1)[0]
        D_x = lat - t4 * v
    return decode_to_pil(D_x, oh, ow)


# ── GenEval folder layout ─────────────────────────────────────────────────────
def save_geneval_fmt(img, meta, idx, root, si=0):
    d = os.path.join(root, f"{idx:05d}", "samples"); os.makedirs(d, exist_ok=True)
    with open(os.path.join(root, f"{idx:05d}", "metadata.jsonl"), "w") as f:
        json.dump(meta, f)
    img.save(os.path.join(d, f"{si:04d}.png"))


# ═══════════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════════
summary = []
ge_dir = os.path.join(args.out_dir, "geneval_images")
os.makedirs(ge_dir, exist_ok=True)

for pidx, entry in enumerate(prompt_entries):
    gi = args.start_index + pidx
    prompt, tag = entry["prompt"], entry.get("tag", "unknown")
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
        base_soft, base_pass = score_single_image(base_img, entry)
        print(f"  Baseline: GE={base_soft:.4f} pass={'✓' if base_pass else '✗'}")

        # MCTS
        print(f"  MCTS search:")
        mcts_img, mcts_score, mcts_path = run_mcts(
            prompt, entry, sd, h, w, oh, ow)
        mcts_soft, mcts_pass = score_single_image(mcts_img, entry)
        print(f"  MCTS:     GE={mcts_soft:.4f} pass={'✓' if mcts_pass else '✗'}"
              f"  Δ={mcts_soft - base_soft:+.4f}")

        base_img.save(os.path.join(args.out_dir, f"{slug}_s{si}_baseline.png"))
        mcts_img.save(os.path.join(args.out_dir, f"{slug}_s{si}_mcts.png"))
        save_geneval_fmt(mcts_img, entry, gi, ge_dir, si)

        sr.append({"seed": sd,
                    "base_soft": base_soft, "base_pass": base_pass,
                    "mcts_soft": mcts_soft, "mcts_pass": mcts_pass,
                    "path": mcts_path, "actions": mcts_path})

    bp = np.mean([s["base_pass"] for s in sr])
    gp = np.mean([s["mcts_pass"] for s in sr])
    summary.append({"slug": slug, "index": gi, "tag": tag, "prompt": prompt,
                     "base_pass_rate": float(bp), "mcts_pass_rate": float(gp),
                     "n_sims": args.n_sims, "samples": sr})
    print(f"\n  PROMPT: base={bp:.2f}  mcts={gp:.2f}")


# ── Summary ────────────────────────────────────────────────────────────────────
jp = os.path.join(args.out_dir, "summary.json")
with open(jp, "w") as f: json.dump(summary, f, indent=2)

print(f"\n{'='*72}\nRESULTS  ({args.n_sims} sims/prompt)\n{'='*72}")
tags = sorted(set(r["tag"] for r in summary))
print(f"\n{'task':<20} {'N':>4} {'base':>8} {'mcts':>8} {'Δ':>8}")
print("-" * 52)
tb = tg = 0
for tag in tags:
    rows = [r for r in summary if r["tag"] == tag]
    b = np.mean([r["base_pass_rate"] for r in rows])
    g = np.mean([r["mcts_pass_rate"] for r in rows])
    print(f"{tag:<20} {len(rows):>4} {b:>8.3f} {g:>8.3f} {g-b:>+8.3f}")
    tb += b * len(rows); tg += g * len(rows)
if summary:
    n = len(summary)
    print(f"{'OVERALL':<20} {n:>4} {tb/n:>8.3f} {tg/n:>8.3f} {(tg-tb)/n:>+8.3f}")

print(f"\n→ {jp}")
print(f"→ {os.path.abspath(ge_dir)}/")
print(f"\nOfficial eval:")
print(f"  python evaluate_images.py {os.path.abspath(ge_dir)} \\")
print(f"      --outfile results.jsonl --model-path <DETECTOR>")
print(f"  python summary_scores.py results.jsonl")