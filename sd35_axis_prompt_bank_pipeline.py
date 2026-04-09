#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sampling_unified_sd35 as su


AXES = ("faithful", "composition", "subject", "background", "detail", "style")

AXIS_META: dict[str, dict[str, list[str]]] = {
    "faithful": {
        "changes": ["light cleanup only"],
        "preserves": ["subject", "composition", "background", "style"],
    },
    "composition": {
        "changes": ["framing", "viewpoint", "crop", "spatial arrangement"],
        "preserves": ["subject identity", "background semantics", "rendering style"],
    },
    "subject": {
        "changes": ["subject attributes", "pose", "clothing", "visible props"],
        "preserves": ["framing", "shot type", "background semantics"],
    },
    "background": {
        "changes": ["environment detail", "layout depth", "architecture/objects"],
        "preserves": ["subject identity", "shot framing", "core style"],
    },
    "detail": {
        "changes": ["fine textures", "materials", "micro visual details"],
        "preserves": ["composition", "scene structure", "subject identity"],
    },
    "style": {
        "changes": ["rendering treatment", "palette", "mood/style cues"],
        "preserves": ["scene semantics", "subject identity", "composition"],
    },
}

AXIS_INSTRUCTIONS: dict[str, str] = {
    "faithful": (
        "Axis=faithful. Do minimal cleanup only. Preserve scene, subject identity, composition, and style."
    ),
    "composition": (
        "Axis=composition. Change framing/crop/viewpoint/camera distance/spatial arrangement. "
        "Do not significantly change subject identity or background semantics."
    ),
    "subject": (
        "Axis=subject. Sharpen visible subject attributes: face, hair, clothing, pose, held props. "
        "Do not mainly change framing or background."
    ),
    "background": (
        "Axis=background. Enrich environment/layout/depth/background objects. "
        "Keep subject identity and shot type stable."
    ),
    "detail": (
        "Axis=detail. Add local textures/materials/fine visible attributes "
        "(wrinkles, reflections, embroidery, folds, strands). "
        "Do not change composition or large scene structure."
    ),
    "style": (
        "Axis=style. Change rendering treatment/palette/mood/photographic style "
        "while keeping core scene semantics stable."
    ),
}

AXIS_SYSTEM_PROMPT = (
    "You are an expert image-prompt editor for text-to-image generation. "
    "Return ONE rewritten prompt only, no explanation, no JSON, no quotes. "
    "Global constraints: preserve original scene intent and main subject identity; "
    "do not add major new objects unless background enrichment justifies it; "
    "avoid vague words like 'beautiful', 'masterpiece', 'highly detailed', 'cinematic', 'more detail'; "
    "use concrete visible attributes only."
)

STEP_SUBBANK_AXES = {
    "early": ["faithful", "composition", "background"],
    "mid": ["faithful", "subject", "background"],
    "late": ["faithful", "detail", "style"],
}


@dataclass
class PromptAxisRecord:
    axis: str
    prompt: str
    changes: list[str]
    preserves: list[str]


@dataclass
class PromptRunArtifacts:
    prompt_index: int
    prompt: str
    records: list[PromptAxisRecord]
    selected_axes: list[str]
    selected_prompts: list[str]
    text_distance_mean: float
    text_distance_min: float
    text_similarity_max: float
    text_logdet: float
    image_across_diversity: float | None
    image_within_mean_diversity: float | None


@dataclass
class ClipEmbedder:
    model: Any
    tokenizer: Any
    processor: Any
    device: torch.device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Axis-tagged SD3.5 prompt-bank pipeline (no reward model; diversity tracking only)."
    )
    p.add_argument("--prompt_file", required=True, help="Prompt txt file (one prompt per line).")
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--end_index", type=int, default=-1, help="Exclusive end index; -1 means all.")
    p.add_argument("--max_prompts", type=int, default=-1, help="Optional cap after slicing.")

    p.add_argument("--out_dir", default="./sd35_axis_prompt_bank_out")
    p.add_argument("--run_tag", default=None)

    p.add_argument("--backend", choices=["sid", "sd35_base", "senseflow_large", "senseflow_medium"], default="sid")
    p.add_argument("--model_id", default=None)
    p.add_argument("--transformer_id", default=None)
    p.add_argument("--transformer_subfolder", default=None)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default=None)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--sigmas", nargs="+", type=float, default=None)
    p.add_argument("--baseline_cfg", type=float, default=1.0)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--time_scale", type=float, default=1000.0)
    p.add_argument("--max_sequence_length", type=int, default=256)

    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--num_seeds", type=int, default=3)
    p.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--stepaware_policy",
        choices=["cycle", "random"],
        default="cycle",
        help="How to choose one axis from each phase subbank per step.",
    )
    p.add_argument("--run_stepaware", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--no_qwen", action="store_true")
    p.add_argument("--qwen_id", default="Qwen/Qwen3-4B")
    p.add_argument("--qwen_dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--qwen_device", default="auto", help="auto|cpu|cuda|cuda:N")
    p.add_argument("--max_new_tokens", type=int, default=140)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--rewrites_cache_file", default=None)
    p.add_argument("--rewrites_overwrite", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--text_embed_backend", choices=["sd35", "clip"], default="sd35")
    p.add_argument("--clip_model_id", default="openai/clip-vit-large-patch14")
    p.add_argument("--text_device", default="auto", help="Device for CLIP text/image embedding (auto|cpu|cuda|cuda:N).")
    p.add_argument("--selection_k", type=int, default=6, help="Final bank size (3-6 recommended).")

    p.add_argument("--compute_image_diversity", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--image_embed_backend", choices=["clip"], default="clip")
    return p.parse_args()


def _resolve_path(path_str: str) -> str:
    return str(Path(path_str).expanduser().resolve())


def _load_prompts(path: str, start: int, end: int, max_prompts: int) -> list[tuple[int, str]]:
    path_obj = Path(path).expanduser().resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"prompt_file not found: {path_obj}")
    prompts = [line.strip() for line in path_obj.read_text(encoding="utf-8").splitlines() if line.strip()]
    if end < 0:
        end = len(prompts)
    lo = max(0, min(int(start), len(prompts)))
    hi = max(lo, min(int(end), len(prompts)))
    pairs = [(i, prompts[i]) for i in range(lo, hi)]
    if int(max_prompts) > 0:
        pairs = pairs[: int(max_prompts)]
    return pairs


def _normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _pairwise_distance_metrics(norm_emb: np.ndarray) -> dict[str, Any]:
    sim = np.clip(norm_emb @ norm_emb.T, -1.0, 1.0)
    dist = 1.0 - sim
    n = int(dist.shape[0])
    off = dist[np.triu_indices(n, k=1)] if n > 1 else np.array([0.0], dtype=np.float32)
    off_sim = sim[np.triu_indices(n, k=1)] if n > 1 else np.array([1.0], dtype=np.float32)
    gram = sim + (1e-6 * np.eye(n, dtype=sim.dtype))
    sign, logdet = np.linalg.slogdet(gram)
    if sign <= 0:
        logdet_val = float("-inf")
    else:
        logdet_val = float(logdet)
    return {
        "distance_matrix": dist,
        "similarity_matrix": sim,
        "mean_pairwise_distance": float(np.mean(off)),
        "min_pairwise_distance": float(np.min(off)),
        "max_pairwise_similarity": float(np.max(off_sim)),
        "logdet_diversity": logdet_val,
    }


def _write_matrix_csv(path: str, matrix: np.ndarray, labels: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["axis"] + labels)
        for i, lab in enumerate(labels):
            w.writerow([lab] + [f"{float(v):.8f}" for v in matrix[i].tolist()])


def _qwen_device_map(device: str) -> Any:
    d = str(device).strip().lower()
    if d == "auto":
        return "auto"
    if d == "cpu":
        return {"": "cpu"}
    if d == "cuda":
        return {"": "cuda:0"}
    if d.startswith("cuda:"):
        return {"": d}
    return "auto"


class AxisRewriter:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.use_qwen = not bool(args.no_qwen)
        self.tokenizer = None
        self.model = None
        self.device = None
        if not self.use_qwen:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dtype = torch.bfloat16 if args.qwen_dtype == "bfloat16" else torch.float16
            device_map = _qwen_device_map(args.qwen_device)
            print(
                f"[axis-rewrite] loading Qwen: {args.qwen_id} "
                f"(dtype={args.qwen_dtype} device_map={device_map})"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(args.qwen_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                args.qwen_id,
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
            )
            self.model.eval()
            self.device = _first_model_device(self.model)
        except Exception as exc:
            print(f"[axis-rewrite] Qwen load failed; fallback to heuristic rewrites. err={exc}")
            self.use_qwen = False
            self.tokenizer = None
            self.model = None
            self.device = None

    @torch.inference_mode()
    def rewrite_one(self, prompt: str, axis: str) -> str:
        if not self.use_qwen or self.model is None or self.tokenizer is None:
            return _heuristic_axis_rewrite(prompt, axis)
        instruction = AXIS_INSTRUCTIONS[axis]
        messages = [
            {"role": "system", "content": AXIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{instruction}\n\n"
                    f"Original prompt: {prompt}\n\n"
                    "Return only one rewritten prompt. /no_think"
                ),
            },
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model.generate(
            **inputs,
            max_new_tokens=int(self.args.max_new_tokens),
            temperature=float(self.args.temperature),
            top_p=float(self.args.top_p),
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        in_len = int(inputs["input_ids"].shape[1])
        decoded = self.tokenizer.decode(out[0][in_len:], skip_special_tokens=True)
        return su.sanitize_rewrite_text(decoded, prompt)


def _first_model_device(model: Any) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _heuristic_axis_rewrite(prompt: str, axis: str) -> str:
    base = prompt.strip()
    suffix = {
        "faithful": "Preserve the same scene and identity with lightly cleaned wording.",
        "composition": "Wide 3/4 view with stronger foreground-midground-background depth and cleaner framing.",
        "subject": "Clarify visible subject traits: facial features, hair, clothing texture, and pose.",
        "background": "Expand environment cues: architecture, spatial layers, and background objects consistent with the scene.",
        "detail": "Add concrete micro details: fabric seams, reflections, material grain, and fine strands.",
        "style": "Shift rendering treatment with a restrained palette and coherent mood while preserving scene semantics.",
    }[axis]
    return su.sanitize_rewrite_text(f"{base}. {suffix}", base)


def _load_rewrite_cache(path: str | None, overwrite: bool) -> dict[str, dict[str, str]]:
    if not path or overwrite:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, dict[str, str]] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                out[str(k)] = {str(kk): str(vv) for kk, vv in v.items() if isinstance(kk, str)}
    return out


def _save_rewrite_cache(path: str | None, cache: dict[str, dict[str, str]]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, str(p))


def generate_axis_records(
    prompt: str,
    rewriter: AxisRewriter,
    rewrite_cache: dict[str, dict[str, str]],
) -> list[PromptAxisRecord]:
    cached = rewrite_cache.get(prompt)
    records: list[PromptAxisRecord] = []
    used: set[str] = set()
    for axis in AXES:
        if isinstance(cached, dict) and axis in cached:
            txt = su.sanitize_rewrite_text(cached[axis], prompt)
        else:
            txt = su.sanitize_rewrite_text(rewriter.rewrite_one(prompt, axis), prompt)
        # Ensure each axis has a meaningfully distinct text, else use axis heuristic fallback.
        if txt in used:
            txt = _heuristic_axis_rewrite(prompt, axis)
        used.add(txt)
        meta = AXIS_META[axis]
        records.append(
            PromptAxisRecord(
                axis=axis,
                prompt=txt,
                changes=list(meta["changes"]),
                preserves=list(meta["preserves"]),
            )
        )
    rewrite_cache[prompt] = {r.axis: r.prompt for r in records}
    return records


def _select_bank_maxmin(
    axes: list[str],
    prompts: list[str],
    norm_emb: np.ndarray,
    k: int,
) -> tuple[list[int], list[str], list[str]]:
    n = len(axes)
    k = max(1, min(int(k), n))
    axis_to_idx = {a: i for i, a in enumerate(axes)}
    selected: list[int] = []
    if "faithful" in axis_to_idx:
        selected.append(axis_to_idx["faithful"])
    else:
        selected.append(0)
    if k == 1:
        idxs = selected
        return idxs, [axes[i] for i in idxs], [prompts[i] for i in idxs]

    sim = np.clip(norm_emb @ norm_emb.T, -1.0, 1.0)
    dist = 1.0 - sim
    remaining = [i for i in range(n) if i not in selected]
    while len(selected) < k and remaining:
        best_i = remaining[0]
        best_val = -float("inf")
        for i in remaining:
            min_dist = min(float(dist[i, j]) for j in selected)
            if min_dist > best_val:
                best_val = min_dist
                best_i = i
        selected.append(best_i)
        remaining = [i for i in remaining if i != best_i]

    idxs = selected
    return idxs, [axes[i] for i in idxs], [prompts[i] for i in idxs]


def _subbank_from_axes(selected_axes: list[str]) -> dict[str, list[str]]:
    available = set(selected_axes)
    out: dict[str, list[str]] = {}
    for phase, target_axes in STEP_SUBBANK_AXES.items():
        picked = [a for a in target_axes if a in available]
        if len(picked) == 0:
            if "faithful" in available:
                picked = ["faithful"]
            else:
                picked = [selected_axes[0]]
        out[phase] = picked
    return out


def _phase_for_step(step_idx: int, total_steps: int) -> str:
    if total_steps <= 1:
        return "early"
    a = float(step_idx) / float(max(1, total_steps))
    if a < (1.0 / 3.0):
        return "early"
    if a < (2.0 / 3.0):
        return "mid"
    return "late"


def _build_stepaware_axis_schedule(
    selected_axes: list[str],
    subbanks: dict[str, list[str]],
    steps: int,
    seed: int,
    policy: str,
) -> list[str]:
    rng = np.random.default_rng(int(seed) + 1234)
    axis_schedule: list[str] = []
    for s in range(int(steps)):
        phase = _phase_for_step(s, int(steps))
        pool = subbanks.get(phase, selected_axes)
        if len(pool) <= 0:
            pool = selected_axes
        if policy == "random":
            axis = pool[int(rng.integers(len(pool)))]
        else:
            axis = pool[int((seed + s) % len(pool))]
        axis_schedule.append(axis)
    return axis_schedule


def _unpack_schedule_step(step: Any, device: str | torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept multiple step-schedule formats and return (t_flat, t_4d).

    Compatible with:
    - (t_flat, t_4d)
    - (t_flat, t_4d, ...)
    - scalar sigma/timestep values
    """
    if isinstance(step, (tuple, list)):
        if len(step) >= 2:
            t_flat, t_4d = step[0], step[1]
            if not isinstance(t_flat, torch.Tensor):
                t_flat = torch.as_tensor(t_flat, device=device, dtype=dtype).reshape(1)
            if not isinstance(t_4d, torch.Tensor):
                t_4d = torch.as_tensor(t_4d, device=device, dtype=dtype).reshape(1, 1, 1, 1)
            return t_flat, t_4d
        if len(step) == 1:
            step = step[0]

    if not isinstance(step, torch.Tensor):
        t_flat = torch.as_tensor(step, device=device, dtype=dtype).reshape(1)
    else:
        t_flat = step.to(device=device, dtype=dtype).reshape(1)
    t_4d = t_flat.view(1, 1, 1, 1)
    return t_flat, t_4d


@torch.no_grad()
def sample_image_with_axis_schedule(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    emb: su.EmbeddingContext,
    axis_to_variant: dict[str, int],
    axis_schedule: list[str],
    seed: int,
) -> Any:
    dtype = emb.cond_text[0].dtype
    latents = su.make_latents(ctx, int(seed), args.height, args.width, dtype)
    dx = torch.zeros_like(latents)
    sched = su.step_schedule(ctx.device, latents.dtype, args.steps, getattr(args, "sigmas", None))
    for step_idx, raw_step in enumerate(sched):
        t_flat, t_4d = _unpack_schedule_step(raw_step, ctx.device, latents.dtype)
        noise = latents if step_idx == 0 else torch.randn_like(latents)
        latents = (1.0 - t_4d) * dx + t_4d * noise
        axis = axis_schedule[step_idx]
        variant_idx = int(axis_to_variant[axis])
        flow = su.transformer_step(args, ctx, latents, emb, variant_idx, t_flat, float(args.baseline_cfg))
        dx = su._pred_x0(latents, t_4d, flow, bool(args.x0_sampler))
    return su.decode_to_pil(ctx, dx)


def _sd35_text_embeddings(
    ctx: su.PipelineContext,
    prompts: list[str],
    max_sequence_length: int,
) -> tuple[np.ndarray, su.EmbeddingContext]:
    emb = su.encode_variants(ctx, prompts, max_sequence_length=max_sequence_length)
    rows: list[np.ndarray] = []
    for i in range(len(prompts)):
        pe = emb.cond_text[i].detach().float().cpu().numpy()  # [1, L, D]
        pp = emb.cond_pooled[i].detach().float().cpu().numpy()  # [1, Dp]
        vec = np.concatenate([np.mean(pe, axis=1).reshape(-1), pp.reshape(-1)], axis=0)
        rows.append(vec.astype(np.float32))
    mat = np.stack(rows, axis=0)
    return _normalize_rows(mat), emb


@torch.inference_mode()
def _clip_text_embeddings(clip: ClipEmbedder, prompts: list[str]) -> np.ndarray:
    inputs = clip.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(clip.device)
    feats = clip.model.get_text_features(**inputs).detach().float().cpu().numpy()
    return _normalize_rows(feats.astype(np.float32))


@torch.inference_mode()
def _clip_image_embeddings(clip: ClipEmbedder, images: list[Any]) -> np.ndarray:
    inputs = clip.processor(images=images, return_tensors="pt").to(clip.device)
    feats = clip.model.get_image_features(**inputs).detach().float().cpu().numpy()
    return _normalize_rows(feats.astype(np.float32))


def _load_clip_embedder(model_id: str, device: str) -> ClipEmbedder:
    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

    clip_device = _resolve_torch_device(device)
    tok = CLIPTokenizer.from_pretrained(model_id)
    proc = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(clip_device)
    model.eval()
    return ClipEmbedder(model=model, tokenizer=tok, processor=proc, device=clip_device)


def _resolve_torch_device(device: str) -> torch.device:
    d = str(device).strip().lower()
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    if d == "cuda":
        return torch.device("cuda:0")
    return torch.device(d)


def _mean_pairwise_dist(x: np.ndarray) -> float:
    if x.shape[0] <= 1:
        return 0.0
    sim = np.clip(x @ x.T, -1.0, 1.0)
    d = 1.0 - sim
    off = d[np.triu_indices(d.shape[0], k=1)]
    return float(np.mean(off)) if off.size > 0 else 0.0


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    if float(np.std(xv)) < 1e-12 or float(np.std(yv)) < 1e-12:
        return None
    return float(np.corrcoef(xv, yv)[0, 1])


def _ensure_out_dir(base: str, run_tag: str | None) -> str:
    root = Path(base).expanduser().resolve()
    if run_tag:
        out = root / run_tag
    else:
        out = root / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def _build_su_args(cfg: argparse.Namespace) -> argparse.Namespace:
    argv: list[str] = [
        "--backend", str(cfg.backend),
        "--steps", str(int(cfg.steps)),
        "--baseline_cfg", str(float(cfg.baseline_cfg)),
        "--width", str(int(cfg.width)),
        "--height", str(int(cfg.height)),
        "--time_scale", str(float(cfg.time_scale)),
        "--search_method", "greedy",
        "--no_qwen",
        "--reward_backend", "imagereward",
        "--reward_model", "CodeGoat24/UnifiedReward-qwen-7b",
        "--unifiedreward_model", "CodeGoat24/UnifiedReward-qwen-7b",
        "--image_reward_model", "ImageReward-v1.0",
        "--pickscore_model", "yuvalkirstain/PickScore_v1",
        "--correction_strengths", "0.0",
    ]
    if cfg.model_id:
        argv += ["--model_id", str(cfg.model_id)]
    if cfg.dtype:
        argv += ["--dtype", str(cfg.dtype)]
    if cfg.transformer_id:
        argv += ["--transformer_id", str(cfg.transformer_id)]
    if cfg.transformer_subfolder:
        argv += ["--transformer_subfolder", str(cfg.transformer_subfolder)]
    if cfg.sigmas:
        argv += ["--sigmas"] + [str(float(s)) for s in cfg.sigmas]

    args = su.parse_args(argv)
    args.max_sequence_length = int(cfg.max_sequence_length)
    args.out_dir = str(cfg.out_dir)
    args.baseline_cfg = float(cfg.baseline_cfg)
    args.steps = int(cfg.steps)
    args.width = int(cfg.width)
    args.height = int(cfg.height)
    args.time_scale = float(cfg.time_scale)
    args.x0_sampler = bool(getattr(args, "x0_sampler", False))
    return args


def _prompt_dir(base_dir: str, prompt_index: int) -> str:
    pdir = Path(base_dir) / f"p{int(prompt_index):04d}"
    pdir.mkdir(parents=True, exist_ok=True)
    return str(pdir)


def run() -> None:
    cfg = parse_args()
    cfg.prompt_file = _resolve_path(cfg.prompt_file)
    if cfg.rewrites_cache_file:
        cfg.rewrites_cache_file = _resolve_path(cfg.rewrites_cache_file)
    cfg.out_dir = _ensure_out_dir(cfg.out_dir, cfg.run_tag)

    prompts = _load_prompts(cfg.prompt_file, cfg.start_index, cfg.end_index, cfg.max_prompts)
    if len(prompts) <= 0:
        raise RuntimeError("No prompts selected.")
    seeds = [int(cfg.seed_base) + i for i in range(max(1, int(cfg.num_seeds)))]

    print(f"[axis-pipeline] out_dir={cfg.out_dir}")
    print(
        f"[axis-pipeline] prompts={len(prompts)} seeds={len(seeds)} backend={cfg.backend} "
        f"text_embed_backend={cfg.text_embed_backend} image_div={cfg.compute_image_diversity}"
    )

    rewrite_cache = _load_rewrite_cache(cfg.rewrites_cache_file, bool(cfg.rewrites_overwrite))
    rewriter = AxisRewriter(cfg)

    su_args = _build_su_args(cfg)
    ctx = su.load_pipeline(su_args)
    clip_embedder = None
    if cfg.text_embed_backend == "clip" or cfg.compute_image_diversity:
        print(f"[axis-pipeline] loading CLIP embedder: {cfg.clip_model_id} (device={cfg.text_device})")
        clip_embedder = _load_clip_embedder(cfg.clip_model_id, cfg.text_device)

    run_summary: list[dict[str, Any]] = []
    prompt_artifacts: list[PromptRunArtifacts] = []
    text_diversities: list[float] = []
    image_diversities: list[float] = []

    for order, (p_idx, prompt) in enumerate(prompts, start=1):
        print(f"\n[{order}/{len(prompts)}] p{p_idx:04d} {prompt}")
        pdir = _prompt_dir(cfg.out_dir, p_idx)
        records = generate_axis_records(prompt, rewriter, rewrite_cache)
        axes = [r.axis for r in records]
        variants = [r.prompt for r in records]

        if cfg.text_embed_backend == "clip":
            if clip_embedder is None:
                raise RuntimeError("clip_embedder is required for text_embed_backend=clip")
            norm_text_emb = _clip_text_embeddings(clip_embedder, variants)
            emb_ctx = su.encode_variants(ctx, variants, max_sequence_length=cfg.max_sequence_length)
        else:
            norm_text_emb, emb_ctx = _sd35_text_embeddings(ctx, variants, max_sequence_length=cfg.max_sequence_length)

        metrics = _pairwise_distance_metrics(norm_text_emb)
        keep_idxs, keep_axes, keep_prompts = _select_bank_maxmin(
            axes=axes,
            prompts=variants,
            norm_emb=norm_text_emb,
            k=int(cfg.selection_k),
        )
        subbanks = _subbank_from_axes(keep_axes)

        # Prepare embedding context for selected bank.
        emb_sel = su.encode_variants(ctx, keep_prompts, max_sequence_length=cfg.max_sequence_length)
        axis_to_variant = {axis: i for i, axis in enumerate(keep_axes)}

        fixed_manifest: list[dict[str, Any]] = []
        image_feats_by_axis: dict[str, list[np.ndarray]] = {a: [] for a in keep_axes}

        for axis in keep_axes:
            axis_dir = Path(pdir) / "images" / "fixed" / axis
            axis_dir.mkdir(parents=True, exist_ok=True)
            for seed in seeds:
                axis_schedule = [axis] * int(su_args.steps)
                img = sample_image_with_axis_schedule(
                    su_args,
                    ctx,
                    emb_sel,
                    axis_to_variant=axis_to_variant,
                    axis_schedule=axis_schedule,
                    seed=int(seed),
                )
                out_path = axis_dir / f"seed_{seed}.png"
                if cfg.save_images:
                    img.save(out_path)
                rec = {
                    "mode": "fixed",
                    "axis": axis,
                    "seed": int(seed),
                    "path": str(out_path),
                    "axis_schedule": axis_schedule,
                }
                fixed_manifest.append(rec)
                if cfg.compute_image_diversity:
                    if clip_embedder is None:
                        raise RuntimeError("clip_embedder is required when compute_image_diversity=True")
                    feat = _clip_image_embeddings(clip_embedder, [img])[0]
                    image_feats_by_axis[axis].append(feat)

        stepaware_manifest: list[dict[str, Any]] = []
        if cfg.run_stepaware:
            sw_dir = Path(pdir) / "images" / "stepaware"
            sw_dir.mkdir(parents=True, exist_ok=True)
            for seed in seeds:
                axis_schedule = _build_stepaware_axis_schedule(
                    selected_axes=keep_axes,
                    subbanks=subbanks,
                    steps=int(su_args.steps),
                    seed=int(seed),
                    policy=str(cfg.stepaware_policy),
                )
                img = sample_image_with_axis_schedule(
                    su_args,
                    ctx,
                    emb_sel,
                    axis_to_variant=axis_to_variant,
                    axis_schedule=axis_schedule,
                    seed=int(seed),
                )
                out_path = sw_dir / f"seed_{seed}.png"
                if cfg.save_images:
                    img.save(out_path)
                stepaware_manifest.append(
                    {
                        "mode": "stepaware",
                        "seed": int(seed),
                        "path": str(out_path),
                        "axis_schedule": axis_schedule,
                    }
                )

        image_within: dict[str, float] = {}
        image_across = None
        image_within_mean = None
        if cfg.compute_image_diversity:
            centers: list[np.ndarray] = []
            for axis in keep_axes:
                arr = np.asarray(image_feats_by_axis.get(axis, []), dtype=np.float32)
                if arr.shape[0] <= 0:
                    image_within[axis] = 0.0
                    continue
                arr = _normalize_rows(arr)
                image_within[axis] = _mean_pairwise_dist(arr)
                centers.append(_normalize_rows(np.mean(arr, axis=0, keepdims=True))[0])
            image_within_mean = float(np.mean(list(image_within.values()))) if image_within else 0.0
            if len(centers) > 1:
                centers_arr = _normalize_rows(np.stack(centers, axis=0))
                image_across = _mean_pairwise_dist(centers_arr)
            else:
                image_across = 0.0

        bank_json = {
            "id": f"p{int(p_idx):04d}",
            "prompt_index": int(p_idx),
            "prompt": prompt,
            "records": [
                {
                    "axis": r.axis,
                    "prompt": r.prompt,
                    "changes": r.changes,
                    "preserves": r.preserves,
                }
                for r in records
            ],
            "selected_indices": [int(i) for i in keep_idxs],
            "selected_axes": keep_axes,
            "selected_prompts": keep_prompts,
            "step_subbanks": subbanks,
            "text_diversity": {
                "mean_pairwise_distance": metrics["mean_pairwise_distance"],
                "min_pairwise_distance": metrics["min_pairwise_distance"],
                "max_pairwise_similarity": metrics["max_pairwise_similarity"],
                "logdet_diversity": metrics["logdet_diversity"],
            },
            "image_diversity": {
                "within_per_axis": image_within,
                "within_mean": image_within_mean,
                "across_variants": image_across,
            },
            "sampling": {
                "steps": int(su_args.steps),
                "baseline_cfg": float(su_args.baseline_cfg),
                "seeds": seeds,
                "fixed_manifest": fixed_manifest,
                "stepaware_manifest": stepaware_manifest,
            },
        }

        with open(Path(pdir) / "prompt_bank.json", "w", encoding="utf-8") as f:
            json.dump(bank_json, f, indent=2, ensure_ascii=False)

        np.save(Path(pdir) / "text_embeddings_norm.npy", norm_text_emb)
        np.save(Path(pdir) / "text_distance_matrix.npy", metrics["distance_matrix"])
        np.save(Path(pdir) / "text_similarity_matrix.npy", metrics["similarity_matrix"])
        _write_matrix_csv(
            str(Path(pdir) / "text_distance_matrix.csv"),
            metrics["distance_matrix"],
            axes,
        )

        rec = PromptRunArtifacts(
            prompt_index=int(p_idx),
            prompt=prompt,
            records=records,
            selected_axes=keep_axes,
            selected_prompts=keep_prompts,
            text_distance_mean=float(metrics["mean_pairwise_distance"]),
            text_distance_min=float(metrics["min_pairwise_distance"]),
            text_similarity_max=float(metrics["max_pairwise_similarity"]),
            text_logdet=float(metrics["logdet_diversity"]),
            image_across_diversity=image_across,
            image_within_mean_diversity=image_within_mean,
        )
        prompt_artifacts.append(rec)
        text_diversities.append(rec.text_distance_mean)
        if image_across is not None:
            image_diversities.append(float(image_across))
        run_summary.append(
            {
                "prompt_index": rec.prompt_index,
                "prompt": rec.prompt,
                "selected_axes": rec.selected_axes,
                "text_mean_pairwise_distance": rec.text_distance_mean,
                "text_min_pairwise_distance": rec.text_distance_min,
                "text_max_pairwise_similarity": rec.text_similarity_max,
                "text_logdet_diversity": rec.text_logdet,
                "image_across_diversity": rec.image_across_diversity,
                "image_within_mean_diversity": rec.image_within_mean_diversity,
            }
        )
        print(
            f"  selected_axes={keep_axes} text_mean={rec.text_distance_mean:.4f} "
            f"text_min={rec.text_distance_min:.4f} image_across={rec.image_across_diversity}"
        )

    _save_rewrite_cache(cfg.rewrites_cache_file, rewrite_cache)

    corr = None
    if len(text_diversities) >= 2 and len(image_diversities) == len(text_diversities):
        corr = _pearson(text_diversities, image_diversities)

    aggregate = {
        "config": {
            "backend": cfg.backend,
            "model_id": cfg.model_id,
            "transformer_id": cfg.transformer_id,
            "steps": int(su_args.steps),
            "baseline_cfg": float(su_args.baseline_cfg),
            "width": int(su_args.width),
            "height": int(su_args.height),
            "num_prompts": len(prompts),
            "seeds": seeds,
            "axes": list(AXES),
            "selection_k": int(cfg.selection_k),
            "text_embed_backend": cfg.text_embed_backend,
            "image_embed_backend": cfg.image_embed_backend if cfg.compute_image_diversity else None,
            "run_stepaware": bool(cfg.run_stepaware),
            "stepaware_policy": cfg.stepaware_policy,
        },
        "prompt_summaries": run_summary,
        "text_image_correlation": {
            "pearson_text_mean_vs_image_across": corr,
            "num_points": len(text_diversities),
        },
    }
    with open(Path(cfg.out_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    with open(Path(cfg.out_dir) / "summary.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "prompt_index",
                "text_mean_pairwise_distance",
                "text_min_pairwise_distance",
                "text_max_pairwise_similarity",
                "text_logdet_diversity",
                "image_across_diversity",
                "image_within_mean_diversity",
                "selected_axes",
            ]
        )
        for r in prompt_artifacts:
            w.writerow(
                [
                    r.prompt_index,
                    f"{r.text_distance_mean:.6f}",
                    f"{r.text_distance_min:.6f}",
                    f"{r.text_similarity_max:.6f}",
                    f"{r.text_logdet:.6f}",
                    "" if r.image_across_diversity is None else f"{float(r.image_across_diversity):.6f}",
                    "" if r.image_within_mean_diversity is None else f"{float(r.image_within_mean_diversity):.6f}",
                    " ".join(r.selected_axes),
                ]
            )

    print(f"\n[axis-pipeline] done. out_dir={cfg.out_dir}")


if __name__ == "__main__":
    run()
