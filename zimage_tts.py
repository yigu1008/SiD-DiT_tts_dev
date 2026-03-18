"""
Test-time scaling pipeline for ZImage (prompt-variant + CFG action space).

Search methods:
  - greedy: per-step one-step lookahead using truncated runs
  - mcts: trajectory search over per-step actions
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZImage test-time scaling search.")
    parser.add_argument("--model", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--prompt", type=str, default="a cinematic portrait, soft rim light, 85mm, ultra detailed")
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--outdir", type=str, default="./zimage_tts_out")

    parser.add_argument("--search_method", choices=["greedy", "mcts"], default="greedy")
    parser.add_argument("--n_sims", type=int, default=30)
    parser.add_argument("--ucb_c", type=float, default=1.41)

    parser.add_argument("--n_variants", type=int, default=3)
    parser.add_argument("--no_qwen", action="store_true")
    parser.add_argument("--qwen_id", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--qwen_python", type=str, default="python3")
    parser.add_argument("--qwen_dtype", type=str, choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--rewrites_file", type=str, default=None)

    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--cfg_scales",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
    )
    parser.add_argument("--baseline_cfg", type=float, default=0.0)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--attention", type=str, default="", choices=["", "flash", "_flash_3"])
    parser.add_argument("--compile_transformer", action="store_true")

    parser.add_argument("--reward_model", type=str, default="ImageReward-v1.0")
    parser.add_argument(
        "--reward_backend",
        type=str,
        choices=["auto", "imagereward", "hpsv2", "unified"],
        default="auto",
        help="Reward backend selector.",
    )
    parser.add_argument(
        "--reward_weights",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        help="Unified backend weights: imagereward hpsv2",
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


def generate_variants(args: argparse.Namespace, prompt: str, cache: Dict[str, List[str]]) -> List[str]:
    if args.n_variants <= 0 or args.no_qwen:
        return [prompt]
    if prompt in cache:
        variants = cache[prompt][: args.n_variants + 1]
        return variants if variants else [prompt]
    variants = [prompt]
    styles = (REWRITE_STYLES * ((args.n_variants // len(REWRITE_STYLES)) + 1))[: args.n_variants]
    for style in styles:
        variants.append(qwen_rewrite(args, prompt, style))
    return variants


def score_image(reward_scorer: UnifiedRewardScorer, prompt: str, image: Image.Image) -> float:
    return float(reward_scorer.score(prompt, image))


def decode_latents_to_pil(pipe: Any, latents: torch.Tensor) -> Image.Image:
    with torch.inference_mode():
        vae_param = next(pipe.vae.parameters())
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


def save_comparison(
    path: str,
    baseline_img: Image.Image,
    search_img: Image.Image,
    baseline_score: float,
    search_score: float,
    actions: List[Tuple[int, float]],
) -> None:
    w, h = baseline_img.size
    hdr = 56
    comp = Image.new("RGB", (w * 2, h + hdr), (18, 18, 18))
    draw = ImageDraw.Draw(comp)
    comp.paste(baseline_img, (0, hdr))
    comp.paste(search_img, (w, hdr))
    draw.text((4, 4), f"baseline IR={baseline_score:.3f}", fill=(200, 200, 200), font=_font(15))
    delta = search_score - baseline_score
    color = (100, 255, 100) if delta >= 0 else (255, 100, 100)
    draw.text((w + 4, 4), f"search IR={search_score:.3f} delta={delta:+.3f}", fill=color, font=_font(15))
    acts = " ".join(f"s{i+1}:v{v}/cfg{c:.2f}" for i, (v, c) in enumerate(actions))
    draw.text((w + 4, 30), acts[:96], fill=(255, 220, 50), font=_font(11))
    comp.save(path)


def make_grid(images: List[Image.Image], cols: int = 3) -> Image.Image:
    w, h = images[0].size
    rows = math.ceil(len(images) / cols)
    canvas = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, image in enumerate(images):
        canvas.paste(image, ((i % cols) * w, (i // cols) * h))
    return canvas


@dataclass
class CandidateEval:
    image: Image.Image
    score: float
    intermediate_records: List[Dict[str, Any]]
    intermediate_images: List[Image.Image]


@dataclass
class EmbeddingBank:
    cond_embeds: List[List[torch.Tensor]]
    neg_embeds: List[torch.Tensor]


def build_embedding_bank(pipe: Any, variants: List[str], negative_prompt: str, max_sequence_length: int) -> EmbeddingBank:
    cond_embeds: List[List[torch.Tensor]] = []
    neg_embeds: Optional[List[torch.Tensor]] = None
    for variant in variants:
        pe, ne = pipe.encode_prompt(
            prompt=variant,
            device=pipe._execution_device,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            max_sequence_length=max_sequence_length,
        )
        cond_embeds.append(pe)
        if neg_embeds is None:
            neg_embeds = ne
    assert neg_embeds is not None
    return EmbeddingBank(cond_embeds=cond_embeds, neg_embeds=neg_embeds)


def schedule_key(schedule: List[Tuple[int, float]]) -> Tuple[Tuple[int, float], ...]:
    return tuple((int(v), float(round(cfg, 6))) for v, cfg in schedule)


def run_with_schedule(
    args: argparse.Namespace,
    pipe: Any,
    reward_scorer: UnifiedRewardScorer,
    prompt_for_reward: str,
    bank: EmbeddingBank,
    schedule: List[Tuple[int, float]],
    seed: int,
    capture_intermediates: bool = False,
    save_intermediate_dir: Optional[str] = None,
    score_intermediates: bool = False,
) -> CandidateEval:
    if not schedule:
        raise ValueError("schedule must be non-empty")

    first_variant, first_cfg = schedule[0]
    generator = torch.Generator("cuda").manual_seed(seed)

    step_records: List[Dict[str, Any]] = []
    step_images: List[Image.Image] = []

    if save_intermediate_dir:
        os.makedirs(save_intermediate_dir, exist_ok=True)

    def _on_step_end(_pipe, step_idx: int, timestep, callback_kwargs):
        if capture_intermediates and "latents" in callback_kwargs:
            image = decode_latents_to_pil(pipe, callback_kwargs["latents"])
            step_ir = None
            if score_intermediates:
                step_ir = score_image(reward_scorer, prompt_for_reward, image)
            record = {
                "step_idx": int(step_idx),
                "timestep": float(timestep) if hasattr(timestep, "__float__") else str(timestep),
                "imagereward": step_ir,
            }
            step_records.append(record)
            label = "n/a" if step_ir is None else f"{step_ir:.4f}"
            canvas = Image.new("RGB", (image.size[0], image.size[1] + 36), (255, 255, 255))
            canvas.paste(image, (0, 36))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 10), f"step={step_idx} t={record['timestep']} IR={label}", fill=(0, 0, 0), font=_font(14))
            step_images.append(canvas)
            if save_intermediate_dir:
                image.save(os.path.join(save_intermediate_dir, f"step_{int(step_idx):03d}.png"))

        next_step = step_idx + 1
        if next_step < len(schedule):
            next_variant, next_cfg = schedule[next_step]
            _pipe._guidance_scale = float(next_cfg)
            callback_kwargs["prompt_embeds"] = bank.cond_embeds[next_variant]
            callback_kwargs["negative_prompt_embeds"] = bank.neg_embeds
        return callback_kwargs

    kwargs: Dict[str, Any] = {
        "prompt": None,
        "prompt_embeds": bank.cond_embeds[first_variant],
        "negative_prompt_embeds": bank.neg_embeds,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": len(schedule),
        "guidance_scale": float(first_cfg),
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
    )


def greedy_search(
    args: argparse.Namespace,
    pipe: Any,
    reward_scorer: UnifiedRewardScorer,
    prompt: str,
    variants: List[str],
    bank: EmbeddingBank,
) -> List[Tuple[int, float]]:
    actions = [(vi, cfg) for vi in range(len(variants)) for cfg in args.cfg_scales]
    cache: Dict[Tuple[int, Tuple[Tuple[int, float], ...]], float] = {}
    chosen: List[Tuple[int, float]] = []

    def eval_prefix(prefix: List[Tuple[int, float]]) -> float:
        key = (len(prefix), schedule_key(prefix))
        if key in cache:
            return cache[key]
        result = run_with_schedule(
            args=args,
            pipe=pipe,
            reward_scorer=reward_scorer,
            prompt_for_reward=prompt,
            bank=bank,
            schedule=prefix,
            seed=args.seed,
            capture_intermediates=False,
            save_intermediate_dir=None,
            score_intermediates=False,
        )
        cache[key] = result.score
        return result.score

    for step_idx in range(args.steps):
        print(f"  greedy step {step_idx + 1}/{args.steps} ({len(actions)} actions)")
        best_action = actions[0]
        best_score = -float("inf")
        for vi, cfg in actions:
            prefix = chosen + [(vi, cfg)]
            score = eval_prefix(prefix)
            marker = ""
            if score > best_score:
                best_score = score
                best_action = (vi, cfg)
                marker = " <- best"
            print(f"    v{vi} cfg={cfg:.2f} IR={score:.4f}{marker}")
        chosen.append(best_action)
        print(f"  selected step {step_idx + 1}: v{best_action[0]} cfg={best_action[1]:.2f} score={best_score:.4f}")
    return chosen


class MCTSNode:
    __slots__ = ("depth", "children", "n", "action_n", "action_q")

    def __init__(self, depth: int):
        self.depth = depth
        self.children: Dict[Tuple[int, float], "MCTSNode"] = {}
        self.n = 0
        self.action_n: Dict[Tuple[int, float], int] = {}
        self.action_q: Dict[Tuple[int, float], float] = {}

    def untried(self, actions: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        return [a for a in actions if a not in self.action_n]

    def ucb(self, action: Tuple[int, float], c: float) -> float:
        n = self.action_n.get(action, 0)
        if n == 0:
            return float("inf")
        mean = self.action_q[action] / n
        return mean + c * math.sqrt(math.log(max(self.n, 1)) / n)

    def best_ucb(self, actions: List[Tuple[int, float]], c: float) -> Tuple[int, float]:
        return max(actions, key=lambda action: self.ucb(action, c))

    def best_exploit(self, actions: List[Tuple[int, float]]) -> Optional[Tuple[int, float]]:
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
    variants: List[str],
    bank: EmbeddingBank,
) -> List[Tuple[int, float]]:
    actions = [(vi, cfg) for vi in range(len(variants)) for cfg in args.cfg_scales]
    root = MCTSNode(depth=0)
    score_cache: Dict[Tuple[Tuple[int, float], ...], float] = {}

    def eval_schedule(schedule: List[Tuple[int, float]]) -> float:
        key = schedule_key(schedule)
        if key in score_cache:
            return score_cache[key]
        result = run_with_schedule(
            args=args,
            pipe=pipe,
            reward_scorer=reward_scorer,
            prompt_for_reward=prompt,
            bank=bank,
            schedule=schedule,
            seed=args.seed,
            capture_intermediates=False,
            save_intermediate_dir=None,
            score_intermediates=False,
        )
        score_cache[key] = result.score
        return result.score

    best_score = -float("inf")
    best_schedule: List[Tuple[int, float]] = []

    print(f"  mcts sims={args.n_sims} actions_per_step={len(actions)} steps={args.steps}")
    for sim in range(args.n_sims):
        node = root
        path: List[Tuple[MCTSNode, Tuple[int, float]]] = []
        schedule: List[Tuple[int, float]] = []

        while node.depth < args.steps:
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
            print(f"    sim {sim + 1:3d}/{args.n_sims} best_IR={best_score:.4f}")

    exploit_schedule: List[Tuple[int, float]] = []
    node = root
    for _ in range(args.steps):
        action = node.best_exploit(actions)
        if action is None:
            break
        exploit_schedule.append(action)
        if action in node.children:
            node = node.children[action]
        else:
            break
    if len(exploit_schedule) < args.steps:
        exploit_schedule.extend(best_schedule[len(exploit_schedule) :])
    return exploit_schedule if exploit_schedule else best_schedule


def write_intermediate_logs(outdir: str, schedule_tag: str, records: List[Dict[str, Any]], images: List[Image.Image]) -> None:
    if records:
        stats_path = os.path.join(outdir, f"{schedule_tag}_intermediate_stats.txt")
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write("step_idx\ttimestep\timagereward\n")
            for rec in records:
                ir = rec["imagereward"]
                ir_text = f"{ir:.8f}" if ir is not None else "nan"
                f.write(f"{rec['step_idx']}\t{rec['timestep']}\t{ir_text}\n")
    if images:
        grid = make_grid(images, cols=min(3, len(images)))
        grid.save(os.path.join(outdir, f"{schedule_tag}_intermediate_grid.png"))


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    from diffusers import ZImagePipeline

    device = "cuda"
    dtype = get_dtype(args.dtype)
    print(f"Loading ZImage pipeline: {args.model}")
    pipe = ZImagePipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    ).to(device)
    if args.attention:
        pipe.transformer.set_attention_backend(args.attention)
    if args.compile_transformer:
        pipe.transformer.compile()

    reward_scorer = UnifiedRewardScorer(
        device=device,
        backend=args.reward_backend,
        image_reward_model=args.reward_model,
        unified_weights=(float(args.reward_weights[0]), float(args.reward_weights[1])),
    )
    print(f"Reward: {reward_scorer.describe()}")
    prompts = load_prompts(args)

    rewrite_cache: Dict[str, List[str]] = {}
    if args.rewrites_file and os.path.exists(args.rewrites_file):
        rewrite_cache = json.load(open(args.rewrites_file))

    summary: List[Dict[str, Any]] = []
    for pidx, prompt in enumerate(prompts):
        slug = f"p{pidx:02d}"
        print(f"\n{'='*72}\n[{slug}] {prompt}\n{'='*72}")
        variants = generate_variants(args, prompt, rewrite_cache)
        with open(os.path.join(args.outdir, f"{slug}_variants.txt"), "w", encoding="utf-8") as f:
            for vi, text in enumerate(variants):
                f.write(f"v{vi}: {text}\n")
        bank = build_embedding_bank(
            pipe=pipe,
            variants=variants,
            negative_prompt=args.negative_prompt,
            max_sequence_length=args.max_sequence_length,
        )

        baseline_schedule = [(0, float(args.baseline_cfg)) for _ in range(args.steps)]
        baseline_result = run_with_schedule(
            args=args,
            pipe=pipe,
            reward_scorer=reward_scorer,
            prompt_for_reward=prompt,
            bank=bank,
            schedule=baseline_schedule,
            seed=args.seed,
            capture_intermediates=False,
            save_intermediate_dir=None,
            score_intermediates=False,
        )

        if args.search_method == "greedy":
            chosen_schedule = greedy_search(args, pipe, reward_scorer, prompt, variants, bank)
        else:
            chosen_schedule = mcts_search(args, pipe, reward_scorer, prompt, variants, bank)

        inter_dir = os.path.join(args.outdir, f"{slug}_{args.search_method}_steps") if args.save_final_intermediate_images else None
        search_result = run_with_schedule(
            args=args,
            pipe=pipe,
            reward_scorer=reward_scorer,
            prompt_for_reward=prompt,
            bank=bank,
            schedule=chosen_schedule,
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
                "variants": variants,
                "baseline_IR": baseline_result.score,
                f"{args.search_method}_IR": search_result.score,
                "delta_IR": search_result.score - baseline_result.score,
                "baseline_schedule": [[int(v), float(c)] for v, c in baseline_schedule],
                "chosen_schedule": [[int(v), float(c)] for v, c in chosen_schedule],
            }
        )

    summary_path = os.path.join(args.outdir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    for row in summary:
        print(f"{row['slug']} delta_IR={row['delta_IR']:+.4f}")
    if summary:
        print(f"mean delta={float(np.mean([row['delta_IR'] for row in summary])):+.4f}")
    print(f"summary json: {summary_path}")


if __name__ == "__main__":
    main()
