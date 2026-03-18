"""
mcts_prompt_cfg_search.py
=========================
MCTS (Monte Carlo Tree Search) over (prompt_variant, cfg_scale) actions
at each SiD denoising step. Uses UCT (UCB1 for trees) to balance
exploration vs exploitation.

Tree structure:
  - Depth = number of denoising steps (default 4)
  - Each node = denoising state (latents, D_x) at a step
  - Each edge = action (variant_idx, cfg_scale)
  - Action space: n_variants × n_cfg (default 4 × 7 = 28)

Each MCTS simulation:
  1. SELECT  — walk tree via UCB1 until reaching an unexplored action
  2. EXPAND  — apply transformer_step to get the child state
  3. ROLLOUT — random actions to the leaf, then decode + ImageReward
  4. BACKUP  — propagate the score up the path

Budget: ~50 simulations (configurable via --n_sims).
Cached states avoid redundant transformer calls on revisited paths.

Usage
-----
python mcts_prompt_cfg_search.py \\
    --prompt "a studio portrait of an elderly woman, soft window light" \\
    --n_sims 50 --ucb_c 1.41

python mcts_prompt_cfg_search.py \\
    --prompt_file prompts.txt --no_qwen \\
    --cfg_scales 1.0 1.5 2.0 2.5 --n_sims 80
"""

import argparse
import json
import math
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
parser.add_argument("--ckpt",        default=None)
parser.add_argument("--qwen_id",     default="Qwen/Qwen3-4B")
parser.add_argument("--prompt",      default="a studio portrait of an elderly woman smiling, "
                                              "soft window light, 85mm lens, photorealistic")
parser.add_argument("--prompt_file", default=None)
parser.add_argument("--neg_embed",   default=None)
parser.add_argument("--n_variants",  type=int,   default=3)
parser.add_argument("--steps",       type=int,   default=4)
parser.add_argument("--cfg_scales",  nargs="+",  type=float,
                    default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0,4.25,4.5,4.75,5.0])
parser.add_argument("--seed",        type=int,   default=42)
parser.add_argument("--width",       type=int,   default=512)
parser.add_argument("--height",      type=int,   default=512)
parser.add_argument("--out_dir",     default="./mcts_prompt_out_cfg_augmented_search space")
parser.add_argument("--qwen_python", default="python3")
parser.add_argument("--qwen_dtype",  default="bfloat16", choices=["float16", "bfloat16"])
parser.add_argument("--rewrites_file", default=None)
parser.add_argument("--no_qwen",     action="store_true")
# MCTS-specific
parser.add_argument("--n_sims",      type=int,   default=50,
                    help="Number of MCTS simulations per prompt.")
parser.add_argument("--ucb_c",       type=float, default=1.41,
                    help="UCB1 exploration constant (sqrt(2) ≈ 1.41).")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
sid_device = "cuda" if torch.cuda.is_available() else "cpu"
sid_dtype  = torch.float16
qwen_dtype = torch.bfloat16 if args.qwen_dtype == "bfloat16" else torch.float16


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline, models, embeddings — identical to greedy script
# ═══════════════════════════════════════════════════════════════════════════════
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
    print(f"Loading weights from {args.ckpt} ...")
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
    print(f"  ✓ loaded {len(sd)-len(u)}/{len(sd)} params")

pipe.transformer.eval()
TIME_SCALE = 1000.0
LATENT_C   = pipe.transformer.config.in_channels
OUT_C      = pipe.transformer.config.out_channels
VARIANCE_SPLIT = (OUT_C // 2 == LATENT_C)
print(f"✓ SiD pipeline  TIME_SCALE={TIME_SCALE}  variance_split={VARIANCE_SPLIT}")

print("Loading ImageReward ...")
import ImageReward as RM
reward_model = RM.load("ImageReward-v1.0", device=sid_device)
reward_model.eval()
print("✓ ImageReward")

# Qwen3 — subprocess only, never loaded in-process
qwen_available = not args.no_qwen

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

def qwen_rewrite(prompt, instruction):
    import subprocess
    script = f"""
import sys, re; from transformers import AutoModelForCausalLM, AutoTokenizer; import torch
tok = AutoTokenizer.from_pretrained({repr(args.qwen_id)})
mdl = AutoModelForCausalLM.from_pretrained({repr(args.qwen_id)}, torch_dtype=torch.bfloat16, device_map="auto"); mdl.eval()
msgs = [{{"role":"system","content":{repr(REWRITE_SYSTEM)}}},{{"role":"user","content":sys.argv[1]+"\\n\\nOriginal prompt: "+sys.argv[2]+" /no_think"}}]
txt = tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
inp = tok([txt],return_tensors="pt").to(mdl.device)
with torch.no_grad(): out=mdl.generate(**inp,max_new_tokens=120,temperature=0.6,top_p=0.9,do_sample=True,pad_token_id=tok.eos_token_id)
r=tok.decode(out[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip()
r=re.sub(r"<think>.*?</think>","",r,flags=re.DOTALL).strip()
for l in r.splitlines():
    l=l.strip()
    if l: print(l); sys.exit(0)
print(sys.argv[2])
"""
    r = subprocess.run([args.qwen_python, "-c", script, instruction, prompt],
                       capture_output=True, text=True)
    return r.stdout.strip() or prompt

_rewrites_cache = {}
if not args.no_qwen and args.rewrites_file:
    _rewrites_cache = json.load(open(args.rewrites_file))

def generate_variants(prompt, n):
    if prompt in _rewrites_cache:
        return _rewrites_cache[prompt][:n+1]
    if args.no_qwen or not qwen_available:
        return [prompt]
    variants = [prompt]
    styles = (REWRITE_STYLES * ((n // len(REWRITE_STYLES)) + 1))[:n]
    for instr in styles:
        variants.append(qwen_rewrite(prompt, instr))
    return variants

# Neg embed
pretrained_neg_embeds = pretrained_neg_mask = None
if args.neg_embed:
    ckpt = torch.load(args.neg_embed, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "neg_embeds" in ckpt:
        pretrained_neg_embeds = ckpt["neg_embeds"].to(sid_device, sid_dtype)
        pretrained_neg_mask   = ckpt["neg_mask"].to(sid_device)
    else:
        pretrained_neg_embeds = ckpt.to(sid_device, sid_dtype)
        pretrained_neg_mask   = torch.ones(ckpt.shape[:2], device=sid_device, dtype=torch.long)

@torch.no_grad()
def encode_variants(variants, max_seq=256):
    pe_list, ue, um = [], None, None
    for i, v in enumerate(variants):
        pe, pm, _ne, _nm = pipe.encode_prompt(
            prompt=v, do_classifier_free_guidance=True, negative_prompt="",
            device=sid_device, num_images_per_prompt=1, max_sequence_length=max_seq)
        pe_list.append((pe.detach(), pm.detach()))
        if i == 0: ue, um = _ne.detach(), _nm.detach()
    if pretrained_neg_embeds is not None:
        L_c, L_n = pe_list[0][0].shape[1], pretrained_neg_embeds.shape[1]
        pne, pnm = pretrained_neg_embeds, pretrained_neg_mask
        if L_n < L_c:
            pne = torch.cat([pne, torch.zeros(1,L_c-L_n,pne.shape[2],device=sid_device,dtype=pne.dtype)],1)
            pnm = torch.cat([pnm, torch.zeros(1,L_c-L_n,device=sid_device,dtype=pnm.dtype)],1)
        elif L_n > L_c:
            pne, pnm = pne[:,:L_c], pnm[:,:L_c]
        ue, um = pne.to(pe_list[0][0].dtype), pnm
    return pe_list, ue, um

def make_latents(seed, h, w, dtype, device):
    try:
        g = torch.Generator(device=device).manual_seed(seed)
        return pipe.prepare_latents(1, LATENT_C, h, w, dtype, device, g)
    except Exception:
        torch.manual_seed(seed)
        return torch.randn(1, LATENT_C, h, w, device=device, dtype=dtype)

@torch.no_grad()
def transformer_step(latents, pe, pm, ue, um, t_flat, cfg):
    if cfg == 1.0:
        v = pipe.transformer(hidden_states=latents, encoder_hidden_states=pe,
            encoder_attention_mask=pm, timestep=TIME_SCALE*t_flat, return_dict=False)[0]
        if VARIANCE_SPLIT: v = v.chunk(2, dim=1)[0]
        return v
    else:
        flow_both = pipe.transformer(
            hidden_states=torch.cat([latents,latents]),
            encoder_hidden_states=torch.cat([ue,pe]),
            encoder_attention_mask=torch.cat([um,pm]),
            timestep=TIME_SCALE*torch.cat([t_flat,t_flat]),
            return_dict=False)[0]
        if VARIANCE_SPLIT: flow_both = flow_both.chunk(2, dim=1)[0]
        u, c = flow_both.chunk(2)
        return u + cfg * (c - u)

@torch.no_grad()
def decode_to_pil(D_x, orig_h, orig_w):
    image = pipe.vae.decode(D_x / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = pipe.image_processor.resize_and_crop_tensor(image, orig_h, orig_w)
    return pipe.image_processor.postprocess(image, output_type="pil")[0]

def score_image(prompt, D_x, orig_h, orig_w):
    img = decode_to_pil(D_x, orig_h, orig_w)
    return float(reward_model.score(prompt, img)), img


# ═══════════════════════════════════════════════════════════════════════════════
# MCTS
# ═══════════════════════════════════════════════════════════════════════════════
class MCTSNode:
    """A node in the MCTS search tree."""
    __slots__ = ("step", "D_x", "latents", "children", "N", "Q",
                 "action_N", "action_Q", "parent", "parent_action")

    def __init__(self, step, D_x, latents, parent=None, parent_action=None):
        self.step   = step           # denoising step index (0..steps-1)
        self.D_x    = D_x            # [1,C,H,W] current x̂₀ estimate
        self.latents = latents       # [1,C,H,W] noisy latents at this step
        self.parent = parent
        self.parent_action = parent_action  # (v_idx, cfg) that led here

        # Children: action_key → MCTSNode
        self.children = {}

        # Visit counts and Q-values per action
        self.action_N = {}   # action_key → int
        self.action_Q = {}   # action_key → float (sum of rewards)

        # Total visits to this node
        self.N = 0

    def is_leaf(self, n_steps):
        return self.step >= n_steps

    def untried_actions(self, actions):
        return [a for a in actions if a not in self.action_N]

    def ucb1(self, action, c):
        n = self.action_N.get(action, 0)
        if n == 0:
            return float("inf")
        q = self.action_Q[action] / n
        return q + c * math.sqrt(math.log(self.N) / n)

    def best_action_ucb(self, actions, c):
        return max(actions, key=lambda a: self.ucb1(a, c))

    def best_action_exploit(self, actions):
        """Best action by average reward (no exploration)."""
        best_a, best_q = None, -float("inf")
        for a in actions:
            n = self.action_N.get(a, 0)
            if n > 0:
                q = self.action_Q[a] / n
                if q > best_q:
                    best_q = q
                    best_a = a
        return best_a


@torch.no_grad()
def mcts_step_forward(node, action, pe_list, ue, um, step_schedule):
    """Apply one denoising step action, return the child state (D_x, latents)."""
    v_idx, cfg = action
    pe, pm = pe_list[v_idx]
    t_val, t_4d = step_schedule[node.step]

    flow_pred = transformer_step(node.latents, pe, pm, ue, um, t_val, cfg)
    new_D_x   = node.latents - t_4d * flow_pred

    # Compute latents for next step (if not final)
    next_step = node.step + 1
    if next_step < len(step_schedule):
        t_next_val, t_next_4d = step_schedule[next_step]
        noise = torch.randn_like(new_D_x)
        new_latents = (1.0 - t_next_4d) * new_D_x + t_next_4d * noise
    else:
        new_latents = None  # leaf

    return new_D_x, new_latents


@torch.no_grad()
def run_mcts(prompt, variants, seed, h, w, orig_h, orig_w):
    pe_list, ue, um = encode_variants(variants)
    actions = [(v_idx, cfg) for v_idx in range(len(variants))
                             for cfg in args.cfg_scales]
    n_actions = len(actions)

    # Precompute step schedule: [(t_flat, t_4d), ...]
    latents_init = make_latents(seed, h, w, pe_list[0][0].dtype, sid_device)
    step_schedule = []
    for i in range(args.steps):
        scalar_t = 999.0 * (1.0 - i / args.steps)
        t_flat = torch.full((1,), scalar_t / 999.0,
                             device=sid_device, dtype=latents_init.dtype)
        t_4d   = t_flat.view(1, 1, 1, 1)
        step_schedule.append((t_flat, t_4d))

    # Initial state: step 0, D_x=0, latents=noise
    D_x_init = torch.zeros_like(latents_init)
    t0_val, t0_4d = step_schedule[0]
    noise_init = latents_init  # first step uses initial latents as noise
    latents_0  = (1.0 - t0_4d) * D_x_init + t0_4d * noise_init

    root = MCTSNode(step=0, D_x=D_x_init, latents=latents_0)

    best_global_score = -float("inf")
    best_global_D_x   = None
    best_global_path  = []

    print(f"  MCTS: {args.n_sims} simulations, {n_actions} actions/step, "
          f"{args.steps} steps, c={args.ucb_c}")

    for sim in range(args.n_sims):
        # ── SELECT ────────────────────────────────────────────────────
        node = root
        path = []  # [(node, action)]

        while not node.is_leaf(args.steps):
            untried = node.untried_actions(actions)
            if untried:
                # Pick a random untried action to expand
                action = untried[np.random.randint(len(untried))]
                break
            else:
                # All actions tried — pick by UCB1
                action = node.best_action_ucb(actions, args.ucb_c)
                path.append((node, action))
                node = node.children[action]

        # ── EXPAND (if not at leaf) ───────────────────────────────────
        if not node.is_leaf(args.steps):
            # Check if this action's child is already cached
            if action not in node.children:
                new_D_x, new_latents = mcts_step_forward(
                    node, action, pe_list, ue, um, step_schedule)
                child = MCTSNode(step=node.step + 1, D_x=new_D_x,
                                 latents=new_latents,
                                 parent=node, parent_action=action)
                node.children[action] = child

            path.append((node, action))
            node = node.children[action]

        # ── ROLLOUT (random actions to leaf) ──────────────────────────
        rollout_node_D_x = node.D_x
        rollout_latents  = node.latents
        rollout_step     = node.step

        while rollout_step < args.steps:
            # Random action
            rand_action = actions[np.random.randint(n_actions)]
            v_idx, cfg = rand_action
            pe, pm = pe_list[v_idx]
            t_val, t_4d = step_schedule[rollout_step]

            flow_pred = transformer_step(rollout_latents, pe, pm, ue, um, t_val, cfg)
            rollout_node_D_x = rollout_latents - t_4d * flow_pred

            rollout_step += 1
            if rollout_step < args.steps:
                t_next_val, t_next_4d = step_schedule[rollout_step]
                noise = torch.randn_like(rollout_node_D_x)
                rollout_latents = (1.0 - t_next_4d) * rollout_node_D_x + t_next_4d * noise

        # ── EVALUATE ──────────────────────────────────────────────────
        score, img = score_image(prompt, rollout_node_D_x, orig_h, orig_w)

        if score > best_global_score:
            best_global_score = score
            best_global_D_x   = rollout_node_D_x.clone()
            best_global_path  = [a for _, a in path]

        # ── BACKUP ────────────────────────────────────────────────────
        for (p_node, p_action) in path:
            p_node.N += 1
            p_node.action_N[p_action] = p_node.action_N.get(p_action, 0) + 1
            p_node.action_Q[p_action] = p_node.action_Q.get(p_action, 0.0) + score

        if (sim + 1) % 10 == 0 or sim == 0:
            print(f"    sim {sim+1:3d}/{args.n_sims}  "
                  f"best_IR={best_global_score:.4f}  "
                  f"root_visits={root.N}")

    # ── Extract best trajectory by exploitation ───────────────────────
    print(f"\n  Extracting best trajectory (exploitation) ...")
    best_path_exploit = []
    node = root
    for step_i in range(args.steps):
        best_a = node.best_action_exploit(actions)
        if best_a is None:
            break
        best_path_exploit.append(best_a)
        if best_a in node.children:
            node = node.children[best_a]
        else:
            break

    # Replay the best exploit path to get the final D_x
    replay_D_x = D_x_init
    replay_latents = latents_0
    for step_i, (v_idx, cfg) in enumerate(best_path_exploit):
        pe, pm = pe_list[v_idx]
        t_val, t_4d = step_schedule[step_i]
        flow_pred = transformer_step(replay_latents, pe, pm, ue, um, t_val, cfg)
        replay_D_x = replay_latents - t_4d * flow_pred
        if step_i + 1 < args.steps:
            t_next_val, t_next_4d = step_schedule[step_i + 1]
            noise = torch.randn_like(replay_D_x)
            replay_latents = (1.0 - t_next_4d) * replay_D_x + t_next_4d * noise

    exploit_score, exploit_img = score_image(prompt, replay_D_x, orig_h, orig_w)

    # Pick whichever is better: best simulation or best exploit replay
    if exploit_score >= best_global_score:
        final_img   = exploit_img
        final_score = exploit_score
        final_path  = best_path_exploit
        print(f"  → exploit path IR={exploit_score:.4f}")
    else:
        final_img   = decode_to_pil(best_global_D_x, orig_h, orig_w)
        final_score = best_global_score
        final_path  = best_global_path
        print(f"  → best simulation IR={best_global_score:.4f}")

    # Print chosen actions
    for si, (vi, cfg) in enumerate(final_path):
        print(f"    step {si+1}: v{vi}('{variants[vi][:40]}') γ={cfg:.2f}")

    return final_img, final_score, final_path


# ── Baseline (same as greedy script) ──────────────────────────────────────────
@torch.no_grad()
def run_baseline(prompt, seed, h, w, orig_h, orig_w):
    pe_list, ue, um = encode_variants([prompt])
    pe, pm = pe_list[0]
    latents = make_latents(seed, h, w, pe.dtype, sid_device)
    D_x = torch.zeros_like(latents)
    for i in range(args.steps):
        scalar_t = 999.0 * (1.0 - i / args.steps)
        t_flat = torch.full((1,), scalar_t/999.0, device=sid_device, dtype=pe.dtype)
        t_4d = t_flat.view(1,1,1,1)
        noise = latents if i == 0 else torch.randn_like(latents)
        latents = (1.0 - t_4d) * D_x + t_4d * noise
        v = pipe.transformer(hidden_states=latents, encoder_hidden_states=pe,
            encoder_attention_mask=pm, timestep=TIME_SCALE*t_flat, return_dict=False)[0]
        if VARIANCE_SPLIT: v = v.chunk(2, dim=1)[0]
        D_x = latents - t_4d * v
    img = decode_to_pil(D_x, orig_h, orig_w)
    return img, float(reward_model.score(prompt, img))


# ── Grid comparison ───────────────────────────────────────────────────────────
def _font(size=16):
    try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except: return ImageFont.load_default()

def make_comparison(base_img, mcts_img, base_score, mcts_score, path, variants):
    W, H_img = base_img.size
    HDR = 54
    comp = Image.new("RGB", (W*2, H_img+HDR), (18,18,18))
    draw = ImageDraw.Draw(comp)
    comp.paste(base_img, (0, HDR))
    comp.paste(mcts_img, (W, HDR))
    draw.text((4,4), f"baseline  IR={base_score:.3f}", fill=(200,200,200), font=_font(15))
    col = (100,255,100) if mcts_score >= base_score else (255,100,100)
    draw.text((W+4,4), f"MCTS  IR={mcts_score:.3f}  Δ={mcts_score-base_score:+.3f}",
              fill=col, font=_font(15))
    acts = " → ".join(f"v{vi}/γ{c:.2f}" for vi, c in path)
    draw.text((W+4,28), acts[:90], fill=(255,220,50), font=_font(11))
    return comp


# ── Prompt list ────────────────────────────────────────────────────────────────
if args.prompt_file:
    prompts = [l.strip() for l in open(args.prompt_file) if l.strip()]
else:
    prompts = [args.prompt]

# ── Phase 1: generate all variants ────────────────────────────────────────────
all_variants = {}
for pidx, prompt in enumerate(prompts):
    slug = f"p{pidx:02d}"
    print(f"\n[{slug}] Generating {args.n_variants} variants ...")
    variants = generate_variants(prompt, args.n_variants)
    all_variants[pidx] = variants
    for i, v in enumerate(variants):
        print(f"  v{i}: {v}")
    with open(os.path.join(args.out_dir, f"{slug}_variants.txt"), "w") as f:
        for vi, v in enumerate(variants):
            f.write(f"v{vi}: {v}\n")

# ── Phase 2: run MCTS for all prompts ─────────────────────────────────────────
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

    # MCTS search
    print("\nMCTS search ...")
    mcts_img, mcts_sc, mcts_path = run_mcts(
        prompt, variants, args.seed, h, w, orig_h, orig_w)
    mcts_img.save(os.path.join(args.out_dir, f"{slug}_mcts.png"))

    print(f"\nMCTS IR={mcts_sc:.4f}  Δ={mcts_sc - base_sc:+.4f}")

    comp = make_comparison(base_img, mcts_img, base_sc, mcts_sc, mcts_path, variants)
    comp.save(os.path.join(args.out_dir, f"{slug}_comparison.png"))

    summary.append({
        "slug": slug, "prompt": prompt, "variants": variants,
        "base_IR": base_sc, "mcts_IR": mcts_sc,
        "delta_IR": mcts_sc - base_sc,
        "path": [(vi, cfg) for vi, cfg in mcts_path],
        "n_sims": args.n_sims,
    })

# ── Summary ────────────────────────────────────────────────────────────────────
json_path = os.path.join(args.out_dir, "mcts_summary.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
print(f"{'slug':<6} {'baseline':>10} {'mcts':>10} {'delta':>8}  path")
print("-" * 60)
for row in summary:
    acts = " ".join(f"v{vi}/γ{c:.2f}" for vi, c in row["path"])
    print(f"{row['slug']:<6} {row['base_IR']:>10.4f} "
          f"{row['mcts_IR']:>10.4f} {row['delta_IR']:>+8.4f}  {acts}")

if summary:
    mean_d = sum(r["delta_IR"] for r in summary) / len(summary)
    print(f"\n  mean Δ IR over {len(summary)} prompt(s): {mean_d:+.4f}")
    print(f"  simulations per prompt: {args.n_sims}")

print(f"\nJSON → {json_path}")
print(f"Out  → {os.path.abspath(args.out_dir)}/")