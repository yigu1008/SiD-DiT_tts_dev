"""
Unified test-time scaling sampler for SiD SD3.5-large.

Search space per denoising step:
  action = (prompt_variant_idx, cfg_scale)

Supports:
  - greedy search
  - MCTS search
  - paper UnifiedReward scoring (with blend fallback)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified SD3.5 sampling/search with unified reward.")
    parser.add_argument("--search_method", choices=["greedy", "mcts"], default="greedy")

    parser.add_argument("--model_id", default="YGu1998/SiD-DiT-SD3.5-large")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--prompt", default="a cinematic portrait of a woman in soft rim light, 85mm, ultra detailed")
    parser.add_argument("--prompt_file", default=None)

    parser.add_argument("--n_variants", type=int, default=3)
    parser.add_argument("--no_qwen", action="store_true")
    parser.add_argument("--qwen_id", default="Qwen/Qwen3-4B")
    parser.add_argument("--qwen_python", default="python3")
    parser.add_argument("--qwen_dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--rewrites_file", default=None)

    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--cfg_scales",
        nargs="+",
        type=float,
        default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
    )
    parser.add_argument("--baseline_cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--time_scale", type=float, default=1000.0)
    parser.add_argument("--out_dir", default="./imagereward_sd35_out")

    parser.add_argument("--n_sims", type=int, default=50)
    parser.add_argument("--ucb_c", type=float, default=1.41)
    parser.add_argument("--reward_model", default="CodeGoat24/UnifiedReward-qwen-7b")
    parser.add_argument(
        "--reward_backend",
        choices=["auto", "unifiedreward", "unified", "imagereward", "hpsv2", "blend"],
        default="unifiedreward",
    )
    parser.add_argument(
        "--reward_weights",
        nargs=2,
        type=float,
        default=[1.0, 1.0],
        help="Blend backend weights: imagereward hpsv2",
    )
    parser.add_argument("--reward_api_base", default=None, help="Optional OpenAI-compatible API base for UnifiedReward.")
    parser.add_argument("--reward_api_key", default="unifiedreward")
    parser.add_argument("--reward_api_model", default="UnifiedReward-7b-v1.5")
    parser.add_argument("--reward_max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--reward_prompt_mode",
        choices=["standard", "strict"],
        default="standard",
        help="UnifiedReward prompt template mode.",
    )
    return parser.parse_args(argv)


def _resolve_file(path_str: str, label: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(path)


def _resolve_optional_file(path_str: str | None, label: str) -> str | None:
    if path_str is None:
        return None
    return _resolve_file(path_str, label)


def _resolve_out_dir(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.ckpt = _resolve_optional_file(args.ckpt, "ckpt")
    args.prompt_file = _resolve_optional_file(args.prompt_file, "prompt_file")
    args.rewrites_file = _resolve_optional_file(args.rewrites_file, "rewrites_file")
    args.out_dir = _resolve_out_dir(args.out_dir)
    return args


def _unwrap_state_dict(raw: Any, depth: int = 0) -> Any:
    if not isinstance(raw, dict):
        return raw
    dotted = sum(1 for key in raw if "." in str(key))
    if dotted / max(len(raw), 1) > 0.5:
        return raw
    if depth > 4:
        return raw
    for key in ("ema", "ema_model", "model_ema", "model", "state_dict", "generator", "G_state"):
        if key in raw and isinstance(raw[key], dict):
            return _unwrap_state_dict(raw[key], depth + 1)
    return raw


@dataclass
class PipelineContext:
    pipe: Any
    device: str
    dtype: torch.dtype
    latent_c: int


@dataclass
class EmbeddingContext:
    cond_text: list[torch.Tensor]
    cond_pooled: list[torch.Tensor]
    uncond_text: torch.Tensor
    uncond_pooled: torch.Tensor


@dataclass
class SearchResult:
    image: Image.Image
    score: float
    actions: list[tuple[int, float]]


def load_pipeline(args: argparse.Namespace) -> PipelineContext:
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from sid import SiDSD3Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    print(f"Loading SD3.5 pipeline: {args.model_id}")
    pipe = SiDSD3Pipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)

    if args.ckpt:
        print(f"Loading transformer weights from {args.ckpt}")
        raw = torch.load(args.ckpt, map_location=device, weights_only=False)
        state_dict = _unwrap_state_dict(raw)
        if any(str(k).startswith("module.") for k in state_dict):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        missing, unexpected = pipe.transformer.load_state_dict(state_dict, strict=False)
        print(
            f"  loaded={len(state_dict) - len(unexpected)}/{len(state_dict)} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    pipe.transformer.eval()
    latent_c = pipe.transformer.config.in_channels
    return PipelineContext(pipe=pipe, device=device, dtype=dtype, latent_c=latent_c)


def load_reward_model(args: argparse.Namespace, device: str) -> UnifiedRewardScorer:
    scorer = UnifiedRewardScorer(
        device=device,
        backend=args.reward_backend,
        image_reward_model=args.reward_model,
        unifiedreward_model=args.reward_model,
        unified_weights=(float(args.reward_weights[0]), float(args.reward_weights[1])),
        unifiedreward_api_base=args.reward_api_base,
        unifiedreward_api_key=args.reward_api_key,
        unifiedreward_api_model=args.reward_api_model,
        max_new_tokens=int(args.reward_max_new_tokens),
        unifiedreward_prompt_mode=args.reward_prompt_mode,
    )
    print(f"Reward: {scorer.describe()}")
    return scorer


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file:
        prompts = [line.strip() for line in open(args.prompt_file) if line.strip()]
    else:
        prompts = [args.prompt]
    if not prompts:
        raise RuntimeError("No prompts found.")
    return prompts


def qwen_rewrite(args: argparse.Namespace, prompt: str, instruction: str) -> str:
    dtype_literal = "torch.bfloat16" if args.qwen_dtype == "bfloat16" else "torch.float16"
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
        print(line)
        raise SystemExit(0)
print(sys.argv[2])
"""
    result = subprocess.run(
        [args.qwen_python, "-c", script, instruction, prompt],
        capture_output=True,
        text=True,
    )
    rewritten = result.stdout.strip()
    return rewritten if rewritten else prompt


def generate_variants(args: argparse.Namespace, prompt: str, cache: dict[str, list[str]]) -> list[str]:
    if args.n_variants <= 0 or args.no_qwen:
        return [prompt]
    if prompt in cache:
        cached = cache[prompt][: args.n_variants + 1]
        return cached if cached else [prompt]
    variants = [prompt]
    styles = (REWRITE_STYLES * ((args.n_variants // len(REWRITE_STYLES)) + 1))[: args.n_variants]
    for style in styles:
        variants.append(qwen_rewrite(args, prompt, style))
    return variants


def encode_variants(ctx: PipelineContext, variants: list[str], max_sequence_length: int = 256) -> EmbeddingContext:
    cond_text: list[torch.Tensor] = []
    cond_pooled: list[torch.Tensor] = []
    for variant in variants:
        pe, pp = ctx.pipe.encode_prompt(
            prompt=variant,
            prompt_2=variant,
            prompt_3=variant,
            device=ctx.device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        cond_text.append(pe.detach())
        cond_pooled.append(pp.detach())

    ue, up = ctx.pipe.encode_prompt(
        prompt="",
        prompt_2="",
        prompt_3="",
        device=ctx.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )
    return EmbeddingContext(
        cond_text=cond_text,
        cond_pooled=cond_pooled,
        uncond_text=ue.detach(),
        uncond_pooled=up.detach(),
    )


def make_latents(ctx: PipelineContext, seed: int, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device=ctx.device).manual_seed(seed)
    return ctx.pipe.prepare_latents(
        1,
        ctx.latent_c,
        height,
        width,
        dtype,
        ctx.device,
        generator,
    )


@torch.no_grad()
def transformer_step(
    args: argparse.Namespace,
    ctx: PipelineContext,
    latents: torch.Tensor,
    emb: EmbeddingContext,
    variant_idx: int,
    t_flat: torch.Tensor,
    cfg: float,
) -> torch.Tensor:
    pe = emb.cond_text[variant_idx]
    pp = emb.cond_pooled[variant_idx]

    if cfg == 1.0:
        velocity = ctx.pipe.transformer(
            hidden_states=latents,
            encoder_hidden_states=pe,
            pooled_projections=pp,
            timestep=args.time_scale * t_flat,
            return_dict=False,
        )[0]
        return velocity

    flow = ctx.pipe.transformer(
        hidden_states=torch.cat([latents, latents]),
        encoder_hidden_states=torch.cat([emb.uncond_text, pe]),
        pooled_projections=torch.cat([emb.uncond_pooled, pp]),
        timestep=args.time_scale * torch.cat([t_flat, t_flat]),
        return_dict=False,
    )[0]
    flow_u, flow_c = flow.chunk(2)
    return flow_u + cfg * (flow_c - flow_u)


@torch.no_grad()
def decode_to_pil(ctx: PipelineContext, dx: torch.Tensor) -> Image.Image:
    shift = getattr(ctx.pipe.vae.config, "shift_factor", 0.0)
    image = ctx.pipe.vae.decode(
        (dx / ctx.pipe.vae.config.scaling_factor) + shift,
        return_dict=False,
    )[0]
    return ctx.pipe.image_processor.postprocess(image, output_type="pil")[0]


def score_image(reward_model: UnifiedRewardScorer, prompt: str, image: Image.Image) -> float:
    return float(reward_model.score(prompt, image))


def step_schedule(device: str, dtype: torch.dtype, steps: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    sched: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(steps):
        scalar_t = 999.0 * (1.0 - float(i) / float(steps))
        t_flat = torch.full((1,), scalar_t / 999.0, device=device, dtype=dtype)
        t_4d = t_flat.view(1, 1, 1, 1)
        sched.append((t_flat, t_4d))
    return sched


def run_baseline(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    seed: int,
    cfg_scale: float = 1.0,
) -> tuple[Image.Image, float]:
    latents = make_latents(ctx, seed, args.height, args.width, emb.cond_text[0].dtype)
    dx = torch.zeros_like(latents)
    sched = step_schedule(ctx.device, latents.dtype, args.steps)
    for i, (t_flat, t_4d) in enumerate(sched):
        noise = latents if i == 0 else torch.randn_like(latents)
        latents = (1.0 - t_4d) * dx + t_4d * noise
        flow = transformer_step(args, ctx, latents, emb, 0, t_flat, float(cfg_scale))
        dx = latents - t_4d * flow
    image = decode_to_pil(ctx, dx)
    return image, score_image(reward_model, prompt, image)


def run_greedy(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    variants: list[str],
    seed: int,
) -> SearchResult:
    actions = [(vi, cfg) for vi in range(len(variants)) for cfg in args.cfg_scales]
    latents = make_latents(ctx, seed, args.height, args.width, emb.cond_text[0].dtype)
    dx = torch.zeros_like(latents)
    sched = step_schedule(ctx.device, latents.dtype, args.steps)
    chosen: list[tuple[int, float]] = []
    for step_idx, (t_flat, t_4d) in enumerate(sched):
        noise = latents if step_idx == 0 else torch.randn_like(latents)
        latents = (1.0 - t_4d) * dx + t_4d * noise

        best_score = -float("inf")
        best_action = actions[0]
        best_dx = None

        print(f"  step {step_idx + 1}/{args.steps}: {len(actions)} actions")
        for variant_idx, cfg in actions:
            flow = transformer_step(args, ctx, latents, emb, variant_idx, t_flat, cfg)
            cand_dx = latents - t_4d * flow
            cand_img = decode_to_pil(ctx, cand_dx)
            cand_score = score_image(reward_model, prompt, cand_img)
            marker = ""
            if cand_score > best_score:
                best_score = cand_score
                best_action = (variant_idx, cfg)
                best_dx = cand_dx.clone()
                marker = " <- best"
            print(f"    v{variant_idx} cfg={cfg:.2f} IR={cand_score:.4f}{marker}")

        assert best_dx is not None
        dx = best_dx
        chosen.append(best_action)
        preview = variants[best_action[0]][:56]
        print(
            f"  selected v{best_action[0]} cfg={best_action[1]:.2f} "
            f"prompt='{preview}' score={best_score:.4f}"
        )

    final_img = decode_to_pil(ctx, dx)
    final_score = score_image(reward_model, prompt, final_img)
    return SearchResult(image=final_img, score=final_score, actions=chosen)


class MCTSNode:
    __slots__ = ("step", "dx", "latents", "children", "visits", "action_visits", "action_values")

    def __init__(self, step: int, dx: torch.Tensor, latents: torch.Tensor | None):
        self.step = step
        self.dx = dx
        self.latents = latents
        self.children: dict[tuple[int, float], MCTSNode] = {}
        self.visits = 0
        self.action_visits: dict[tuple[int, float], int] = {}
        self.action_values: dict[tuple[int, float], float] = {}

    def is_leaf(self, max_steps: int) -> bool:
        return self.step >= max_steps

    def untried_actions(self, actions: list[tuple[int, float]]) -> list[tuple[int, float]]:
        return [a for a in actions if a not in self.action_visits]

    def ucb(self, action: tuple[int, float], c: float) -> float:
        n = self.action_visits.get(action, 0)
        if n == 0:
            return float("inf")
        mean = self.action_values[action] / n
        return mean + c * math.sqrt(math.log(max(self.visits, 1)) / n)

    def best_ucb(self, actions: list[tuple[int, float]], c: float) -> tuple[int, float]:
        return max(actions, key=lambda action: self.ucb(action, c))

    def best_exploit(self, actions: list[tuple[int, float]]) -> tuple[int, float] | None:
        best = None
        best_v = -float("inf")
        for action in actions:
            n = self.action_visits.get(action, 0)
            if n <= 0:
                continue
            value = self.action_values[action] / n
            if value > best_v:
                best_v = value
                best = action
        return best


def _expand_child(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    node: MCTSNode,
    action: tuple[int, float],
    sched: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    variant_idx, cfg = action
    t_flat, t_4d = sched[node.step]
    flow = transformer_step(args, ctx, node.latents, emb, variant_idx, t_flat, cfg)
    new_dx = node.latents - t_4d * flow
    next_step = node.step + 1
    if next_step < len(sched):
        _, next_t_4d = sched[next_step]
        noise = torch.randn_like(new_dx)
        new_latents = (1.0 - next_t_4d) * new_dx + next_t_4d * noise
    else:
        new_latents = None
    return new_dx, new_latents


def run_mcts(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    variants: list[str],
    seed: int,
) -> SearchResult:
    actions = [(vi, cfg) for vi in range(len(variants)) for cfg in args.cfg_scales]
    n_actions = len(actions)
    latents0 = make_latents(ctx, seed, args.height, args.width, emb.cond_text[0].dtype)
    dx0 = torch.zeros_like(latents0)
    sched = step_schedule(ctx.device, latents0.dtype, args.steps)
    _, t0_4d = sched[0]
    start_latents = (1.0 - t0_4d) * dx0 + t0_4d * latents0
    root = MCTSNode(0, dx0, start_latents)

    best_global_score = -float("inf")
    best_global_dx = None
    best_global_path: list[tuple[int, float]] = []

    print(f"  mcts: sims={args.n_sims} actions={n_actions} steps={args.steps}")
    for sim in range(args.n_sims):
        node = root
        path: list[tuple[MCTSNode, tuple[int, float]]] = []

        while not node.is_leaf(args.steps):
            untried = node.untried_actions(actions)
            if untried:
                action = untried[np.random.randint(len(untried))]
                break
            action = node.best_ucb(actions, args.ucb_c)
            path.append((node, action))
            node = node.children[action]

        if not node.is_leaf(args.steps):
            if action not in node.children:
                child_dx, child_lat = _expand_child(args, ctx, emb, node, action, sched)
                node.children[action] = MCTSNode(node.step + 1, child_dx, child_lat)
            path.append((node, action))
            node = node.children[action]

        rollout_dx = node.dx
        rollout_latents = node.latents
        rollout_step = node.step
        while rollout_step < args.steps:
            variant_idx, cfg = actions[np.random.randint(n_actions)]
            t_flat, t_4d = sched[rollout_step]
            flow = transformer_step(args, ctx, rollout_latents, emb, variant_idx, t_flat, cfg)
            rollout_dx = rollout_latents - t_4d * flow
            rollout_step += 1
            if rollout_step < args.steps:
                _, next_t_4d = sched[rollout_step]
                noise = torch.randn_like(rollout_dx)
                rollout_latents = (1.0 - next_t_4d) * rollout_dx + next_t_4d * noise

        rollout_img = decode_to_pil(ctx, rollout_dx)
        rollout_score = score_image(reward_model, prompt, rollout_img)
        if rollout_score > best_global_score:
            best_global_score = rollout_score
            best_global_dx = rollout_dx.clone()
            best_global_path = [a for _, a in path]

        for pnode, paction in path:
            pnode.visits += 1
            pnode.action_visits[paction] = pnode.action_visits.get(paction, 0) + 1
            pnode.action_values[paction] = pnode.action_values.get(paction, 0.0) + rollout_score

        if (sim + 1) % 10 == 0 or sim == 0:
            print(f"    sim {sim + 1:3d}/{args.n_sims} best={best_global_score:.4f}")

    exploit_path: list[tuple[int, float]] = []
    node = root
    for _ in range(args.steps):
        action = node.best_exploit(actions)
        if action is None:
            break
        exploit_path.append(action)
        if action in node.children:
            node = node.children[action]
        else:
            break

    replay_dx = dx0
    replay_lat = start_latents
    for step_idx, (variant_idx, cfg) in enumerate(exploit_path):
        t_flat, t_4d = sched[step_idx]
        flow = transformer_step(args, ctx, replay_lat, emb, variant_idx, t_flat, cfg)
        replay_dx = replay_lat - t_4d * flow
        if step_idx + 1 < args.steps:
            _, next_t_4d = sched[step_idx + 1]
            noise = torch.randn_like(replay_dx)
            replay_lat = (1.0 - next_t_4d) * replay_dx + next_t_4d * noise

    exploit_img = decode_to_pil(ctx, replay_dx)
    exploit_score = score_image(reward_model, prompt, exploit_img)

    if exploit_score >= best_global_score:
        return SearchResult(image=exploit_img, score=exploit_score, actions=exploit_path)
    if best_global_dx is None:
        return SearchResult(image=exploit_img, score=exploit_score, actions=exploit_path)
    best_img = decode_to_pil(ctx, best_global_dx)
    return SearchResult(image=best_img, score=best_global_score, actions=best_global_path)


def _font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def save_comparison(
    path: str,
    base_img: Image.Image,
    search_img: Image.Image,
    base_score: float,
    search_score: float,
    actions: list[tuple[int, float]],
) -> None:
    w, h = base_img.size
    header_h = 54
    comp = Image.new("RGB", (w * 2, h + header_h), (18, 18, 18))
    draw = ImageDraw.Draw(comp)
    comp.paste(base_img, (0, header_h))
    comp.paste(search_img, (w, header_h))
    draw.text((4, 4), f"baseline IR={base_score:.3f}", fill=(200, 200, 200), font=_font(15))
    d = search_score - base_score
    col = (100, 255, 100) if d >= 0 else (255, 100, 100)
    draw.text((w + 4, 4), f"search IR={search_score:.3f}  delta={d:+.3f}", fill=col, font=_font(15))
    acts = " ".join(f"s{i+1}:v{v}/cfg{c:.2f}" for i, (v, c) in enumerate(actions))
    draw.text((w + 4, 28), acts[:96], fill=(255, 220, 50), font=_font(11))
    comp.save(path)


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    prompts = load_prompts(args)
    ctx = load_pipeline(args)
    reward_model = load_reward_model(args, ctx.device)

    rewrite_cache: dict[str, list[str]] = {}
    if args.rewrites_file and os.path.exists(args.rewrites_file):
        rewrite_cache = json.load(open(args.rewrites_file))
        print(f"Loaded rewrite cache for {len(rewrite_cache)} prompts.")

    summary: list[dict[str, Any]] = []
    for prompt_idx, prompt in enumerate(prompts):
        slug = f"p{prompt_idx:02d}"
        print(f"\n{'='*72}\n[{slug}] {prompt}\n{'='*72}")
        variants = generate_variants(args, prompt, rewrite_cache)
        with open(os.path.join(args.out_dir, f"{slug}_variants.txt"), "w") as f:
            for vi, text in enumerate(variants):
                f.write(f"v{vi}: {text}\n")
        emb = encode_variants(ctx, variants)

        base_img, base_score = run_baseline(
            args,
            ctx,
            emb,
            reward_model,
            prompt,
            args.seed,
            cfg_scale=float(args.baseline_cfg),
        )
        if args.search_method == "greedy":
            search = run_greedy(args, ctx, emb, reward_model, prompt, variants, args.seed)
        else:
            search = run_mcts(args, ctx, emb, reward_model, prompt, variants, args.seed)

        base_path = os.path.join(args.out_dir, f"{slug}_baseline.png")
        search_path = os.path.join(args.out_dir, f"{slug}_{args.search_method}.png")
        comp_path = os.path.join(args.out_dir, f"{slug}_comparison.png")
        base_img.save(base_path)
        search.image.save(search_path)
        save_comparison(comp_path, base_img, search.image, base_score, search.score, search.actions)

        print(
            f"baseline={base_score:.4f} {args.search_method}={search.score:.4f} "
            f"delta={search.score - base_score:+.4f}"
        )
        summary.append(
            {
                "slug": slug,
                "prompt": prompt,
                "variants": variants,
                "baseline_cfg": float(args.baseline_cfg),
                "baseline_IR": base_score,
                f"{args.search_method}_IR": search.score,
                "delta_IR": search.score - base_score,
                "actions": [[int(v), float(c)] for v, c in search.actions],
            }
        )

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    for row in summary:
        print(f"{row['slug']} delta_IR={row['delta_IR']:+.4f}")
    if summary:
        print(f"mean delta={float(np.mean([r['delta_IR'] for r in summary])):+.4f}")
    print(f"summary json: {summary_path}")


def main(argv: list[str] | None = None) -> None:
    run(normalize_paths(parse_args(argv)))


if __name__ == "__main__":
    main()
