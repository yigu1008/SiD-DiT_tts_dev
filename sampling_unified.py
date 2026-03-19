"""
Unified test-time sampling/search for SiD SANA.

This script merges the former sampling entrypoints:
- greedy_search.py
- greedy_prompt_cfg_search.py
- mcts_prompt_cfg_search.py
- geneval_greedy.py
- geneval_mcts.py

Core knobs:
- search method: greedy, mcts, or ga
- reward type: imagereward or geneval
- action space: (prompt_variant_idx, cfg_scale)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


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

GA_PROMPT_FOCUS = {
    "balanced": "",
    "subject": "Focus on subject identity, face, and outfit fidelity.",
    "prop": "Focus on key props, object pose, and hand-object interaction.",
    "background": "Focus on composition, scene layout, and background structure.",
    "detail": "Focus on high-frequency details, textures, and fine attributes.",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified SiD sampler for prompt/cfg action-space search."
    )

    # High-level mode
    parser.add_argument("--search_method", choices=["greedy", "mcts", "ga"], default="greedy")
    parser.add_argument("--reward_type", choices=["imagereward", "geneval"], default="imagereward")
    parser.add_argument(
        "--reward_device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for ImageReward backend.",
    )

    # Model + generation
    parser.add_argument("--model_id", default="YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--neg_embed", default=None)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--cfg_scales",
        nargs="+",
        type=float,
        default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--time_scale", type=float, default=1000.0)
    parser.add_argument("--out_dir", default="./sampling_unified_out")
    parser.add_argument(
        "--cuda_alloc_conf",
        default="expandable_segments:True",
        help="Set PYTORCH_CUDA_ALLOC_CONF when not already defined.",
    )
    parser.add_argument(
        "--vae_slicing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable VAE sliced decode for lower peak memory.",
    )
    parser.add_argument(
        "--vae_tiling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable VAE tiled decode for lower peak memory.",
    )
    parser.add_argument(
        "--empty_cache_after_decode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Call torch.cuda.empty_cache() after each decode.",
    )
    parser.add_argument(
        "--offload_text_encoder_after_encode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move text encoder to CPU after prompt embeddings are computed.",
    )

    # Prompt sources (regular)
    parser.add_argument(
        "--prompt",
        default="a studio portrait of an elderly woman smiling, soft window light, 85mm lens, photorealistic",
    )
    parser.add_argument("--prompt_file", default=None)

    # Prompt sources (GenEval)
    parser.add_argument("--geneval_prompts", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--n_samples", type=int, default=1)

    # Prompt variants
    parser.add_argument("--n_variants", type=int, default=0)
    parser.add_argument("--no_qwen", action="store_true")
    parser.add_argument("--qwen_id", default="Qwen/Qwen3-4B")
    parser.add_argument("--qwen_python", default="python3")
    parser.add_argument("--qwen_dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--qwen_device", default="auto")  # legacy arg; not used in subprocess flow
    parser.add_argument("--rewrites_file", default=None)

    # MCTS
    parser.add_argument("--n_sims", type=int, default=50)
    parser.add_argument("--ucb_c", type=float, default=1.41)

    # GA
    parser.add_argument("--ga_population", type=int, default=24)
    parser.add_argument("--ga_generations", type=int, default=12)
    parser.add_argument("--ga_elites", type=int, default=3)
    parser.add_argument("--ga_mutation_prob", type=float, default=0.10)
    parser.add_argument("--ga_tournament_k", type=int, default=3)
    parser.add_argument("--ga_crossover", choices=["uniform", "one_point"], default="uniform")
    parser.add_argument("--ga_log_topk", type=int, default=3)
    parser.add_argument("--ga_random_trials", type=int, default=32)
    parser.add_argument("--ga_phase_constraints", action="store_true")
    parser.add_argument(
        "--ga_cfg_scales",
        nargs="+",
        type=float,
        default=[1.0, 1.25, 1.5],
        help="Small CFG bank for GA genomes.",
    )

    # GenEval reward backends
    parser.add_argument("--reward_url", default=None)
    parser.add_argument("--geneval_python", default=None)
    parser.add_argument("--geneval_repo", default=None)
    parser.add_argument("--detector_path", default=None)

    # Legacy compatibility knobs (ignored by unified logic where not needed)
    parser.add_argument("--max_batch", type=int, default=28)
    parser.add_argument("--neg_prompt", default="")
    return parser.parse_args(argv)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _unwrap_state_dict(raw: Any, depth: int = 0) -> Any:
    if not isinstance(raw, dict):
        return raw
    dotted = sum(1 for k in raw if "." in str(k))
    if dotted / max(len(raw), 1) > 0.5:
        return raw
    if depth > 4:
        return raw
    for key in ("ema", "ema_model", "model_ema", "model", "state_dict", "generator", "G_state"):
        if key in raw and isinstance(raw[key], dict):
            return _unwrap_state_dict(raw[key], depth + 1)
    return raw


def _load_prompt_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if args.geneval_prompts:
        with open(args.geneval_prompts) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        end = len(entries) if args.end_index == -1 else min(args.end_index, len(entries))
        entries = entries[args.start_index:end]
        out = []
        for local_idx, meta in enumerate(entries):
            global_idx = args.start_index + local_idx
            out.append(
                {
                    "index": global_idx,
                    "slug": f"p{global_idx:05d}",
                    "prompt": meta["prompt"],
                    "metadata": meta,
                }
            )
        return out

    if args.prompt_file:
        prompts = [line.strip() for line in open(args.prompt_file) if line.strip()]
    else:
        prompts = [args.prompt]

    for i, prompt in enumerate(prompts):
        entries.append({"index": i, "slug": f"p{i:02d}", "prompt": prompt, "metadata": None})
    return entries


@dataclass
class PipelineContext:
    pipe: Any
    device: str
    dtype: torch.dtype
    latent_c: int
    variance_split: bool
    aspect_ratio_bins: dict[int, dict[str, Any]]
    empty_cache_after_decode: bool
    decode_counts: dict[str, int]


def load_pipeline(args: argparse.Namespace) -> PipelineContext:
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Compatibility shim:
    # Some dependency combos (older huggingface_hub + newer imports) expect HF_HOME.
    # Define it dynamically when missing to avoid import-time failures.
    try:
        import huggingface_hub.constants as hhc

        if not hasattr(hhc, "HF_HOME"):
            cache_root = getattr(hhc, "HUGGINGFACE_HUB_CACHE", None)
            if cache_root:
                hhc.HF_HOME = str(Path(cache_root).expanduser().parent)
            else:
                hhc.HF_HOME = str(Path.home() / ".cache" / "huggingface")
    except Exception:
        pass

    try:
        from sid import SiDSanaPipeline
    except ImportError as e:
        raise RuntimeError(f"Cannot import SiDSanaPipeline: {e}") from e

    from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
        ASPECT_RATIO_512_BIN,
        ASPECT_RATIO_1024_BIN,
    )
    try:
        from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
    except ImportError:
        ASPECT_RATIO_2048_BIN = ASPECT_RATIO_1024_BIN

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    print("Loading SiD pipeline ...")
    pipe = SiDSanaPipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)

    if args.ckpt:
        print(f"Loading checkpoint {args.ckpt} ...")
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

    if args.vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        try:
            pipe.enable_vae_slicing()
            print("Enabled VAE slicing.")
        except Exception as exc:
            print(f"Warning: could not enable VAE slicing: {exc}")
    if args.vae_tiling and hasattr(pipe, "enable_vae_tiling"):
        try:
            pipe.enable_vae_tiling()
            print("Enabled VAE tiling.")
        except Exception as exc:
            print(f"Warning: could not enable VAE tiling: {exc}")

    latent_c = pipe.transformer.config.in_channels
    out_c = pipe.transformer.config.out_channels
    variance_split = (out_c // 2 == latent_c)
    bins = {16: ASPECT_RATIO_512_BIN, 32: ASPECT_RATIO_1024_BIN, 64: ASPECT_RATIO_2048_BIN}

    print(f"Loaded. device={device} variance_split={variance_split}")
    return PipelineContext(
        pipe=pipe,
        device=device,
        dtype=dtype,
        latent_c=latent_c,
        variance_split=variance_split,
        aspect_ratio_bins=bins,
        empty_cache_after_decode=args.empty_cache_after_decode,
        decode_counts={},
    )


@dataclass
class RewardContext:
    kind: str
    score_images: Any
    geneval_mode: str | None = None
    scorer_path: str | None = None


GENEVAL_SCORER_SCRIPT = r'''
import argparse
import glob
import json
import os
import sys
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
            config_file = os.path.join(
                mmdet_root, "configs", "mask2former",
                "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py")
            if not os.path.exists(config_file):
                config_file = os.path.join(
                    mmdet_root, "configs", "mask2former",
                    "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco-panoptic.py")
        except:
            config_file = None
else:
    configs = glob.glob(os.path.join(det_path, "*.py"))
    ckpts = glob.glob(os.path.join(det_path, "*.pth"))
    config_file = configs[0] if configs else None
    ckpt_file = ckpts[0] if ckpts else None

if not config_file or not os.path.exists(str(config_file)):
    print(json.dumps([{"file":"error","score":0.0,"correct":False}]))
    sys.exit(0)
if not ckpt_file or not os.path.exists(str(ckpt_file)):
    print(json.dumps([{"file":"error","score":0.0,"correct":False}]))
    sys.exit(0)

from mmdet.apis import init_detector, inference_detector
model = init_detector(config_file, ckpt_file, device="cuda:0")
classes = model.CLASSES if hasattr(model, "CLASSES") else []
class_to_idx = {name.lower(): idx for idx, name in enumerate(classes)}

def score_image(image_path):
    result = inference_detector(model, image_path)
    bbox_results = result[0] if isinstance(result, tuple) else result
    scores = []
    is_correct = True
    for obj_spec in include_spec:
        cls_name = obj_spec["class"].lower()
        required = obj_spec.get("count", 1)
        cls_idx = class_to_idx.get(cls_name, -1)
        if cls_idx == -1:
            for cn, ci in class_to_idx.items():
                if cls_name in cn or cn in cls_name:
                    cls_idx = ci
                    break
        if cls_idx == -1 or cls_idx >= len(bbox_results):
            scores.append(0.0)
            is_correct = False
            continue
        dets = bbox_results[cls_idx]
        if len(dets) > 0:
            dets = dets[dets[:, 4] > 0.3]
        n_det = len(dets)
        max_conf = float(dets[:, 4].max()) if n_det > 0 else 0.0
        count_score = max(0.0, 1.0 - abs(n_det - required) / required)
        obj_score = count_score * max(max_conf, 0.5 if n_det > 0 else 0.0)
        scores.append(obj_score)
        if n_det < required:
            is_correct = False
        if tag == "counting" and n_det != required:
            is_correct = False
    soft = float(np.mean(scores)) if scores else 0.0
    return {"score": soft, "correct": is_correct}

image_files = sorted(glob.glob(os.path.join(args.image_dir, "candidate_*.png")))
results = []
for img_path in image_files:
    info = score_image(img_path)
    results.append({
        "file": os.path.basename(img_path),
        "score": info["score"],
        "correct": info["correct"],
    })
print(json.dumps(results))
'''


def load_reward(args: argparse.Namespace, ctx: PipelineContext) -> RewardContext:
    if args.reward_type == "imagereward":
        print("Loading ImageReward ...")
        # Compatibility shim:
        # Some ImageReward releases expect BertModel.all_tied_weights_keys,
        # while newer transformers expose _tied_weights_keys instead.
        try:
            import transformers

            bert_cls = getattr(transformers, "BertModel", None)
            if bert_cls is not None and not hasattr(bert_cls, "all_tied_weights_keys"):
                def _all_tied_keys(self):
                    return getattr(self, "_tied_weights_keys", None)

                bert_cls.all_tied_weights_keys = property(_all_tied_keys)  # type: ignore[attr-defined]
                print("  Applied transformers/ImageReward BertModel compatibility shim.")
        except Exception as exc:
            print(f"  Warning: could not apply ImageReward compatibility shim: {exc}")

        import ImageReward as RM

        reward_device = args.reward_device
        if reward_device == "auto":
            if ctx.device == "cuda":
                free_gb = None
                try:
                    free_bytes, _ = torch.cuda.mem_get_info()
                    free_gb = free_bytes / (1024 ** 3)
                except Exception:
                    pass
                reward_device = "cpu" if free_gb is not None and free_gb < 14.0 else "cuda"
            else:
                reward_device = "cpu"
        print(f"  ImageReward device={reward_device}")

        reward_model = RM.load("ImageReward-v1.0", device=reward_device)
        reward_model.eval()

        def score_images(prompt: str, images: list[Image.Image], metadata: dict[str, Any] | None = None) -> list[float]:
            return [float(reward_model.score(prompt, img)) for img in images]

        return RewardContext(kind="imagereward", score_images=score_images)

    # Geneval
    geneval_mode = None
    if args.reward_url:
        try:
            import requests

            requests.get(f"{args.reward_url}/", timeout=3)
            geneval_mode = "http"
            print(f"Using GenEval HTTP reward server: {args.reward_url}")
        except Exception:
            print(f"GenEval reward server unreachable at {args.reward_url}; trying subprocess backend.")

    if geneval_mode is None:
        if not (args.geneval_python and args.geneval_repo and args.detector_path):
            raise RuntimeError(
                "GenEval reward requires either --reward_url or all of "
                "--geneval_python --geneval_repo --detector_path."
            )
        geneval_mode = "subprocess"

    scorer_path = os.path.join(args.out_dir, "_geneval_scorer.py")
    if geneval_mode == "subprocess":
        with open(scorer_path, "w") as f:
            f.write(GENEVAL_SCORER_SCRIPT)
        print(f"Wrote GenEval scorer helper: {scorer_path}")

    def score_images_geneval(
        prompt: str, images: list[Image.Image], metadata: dict[str, Any] | None = None
    ) -> list[float]:
        del prompt
        if metadata is None:
            raise RuntimeError("GenEval scoring requires metadata per prompt entry.")
        if geneval_mode == "http":
            import requests

            payload = {"images": [pil_to_b64(img) for img in images], "metadata": metadata}
            try:
                response = requests.post(f"{args.reward_url}/geneval", json=payload, timeout=120)
                response.raise_for_status()
                return response.json()["scores"]
            except Exception as exc:
                print(f"  HTTP scoring error: {exc}")
                return [0.0] * len(images)

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, img in enumerate(images):
                img.save(os.path.join(tmpdir, f"candidate_{i:03d}.png"))
            cmd = [
                args.geneval_python,
                scorer_path,
                "--image_dir",
                tmpdir,
                "--metadata_json",
                json.dumps(metadata),
                "--detector_path",
                args.detector_path,
                "--geneval_repo",
                args.geneval_repo,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"  subprocess scoring failed rc={result.returncode}")
                if result.stderr:
                    print(f"  stderr: {result.stderr[:240]}")
                return [0.0] * len(images)

            for line in result.stdout.strip().splitlines():
                if line.strip().startswith("["):
                    try:
                        parsed = json.loads(line.strip())
                        return [float(item["score"]) for item in parsed]
                    except json.JSONDecodeError:
                        continue
            print("  unable to parse scorer output")
            return [0.0] * len(images)

    return RewardContext(
        kind="geneval",
        score_images=score_images_geneval,
        geneval_mode=geneval_mode,
        scorer_path=scorer_path,
    )


def pil_to_b64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@dataclass
class EmbeddingContext:
    pe_list: list[tuple[torch.Tensor, torch.Tensor]]
    ue: torch.Tensor
    um: torch.Tensor


def load_neg_embed(args: argparse.Namespace, ctx: PipelineContext) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not args.neg_embed:
        return None, None
    checkpoint = torch.load(args.neg_embed, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "neg_embeds" in checkpoint:
        neg_embeds = checkpoint["neg_embeds"].to(ctx.device, ctx.dtype)
        neg_mask = checkpoint["neg_mask"].to(ctx.device)
    else:
        neg_embeds = checkpoint.to(ctx.device, ctx.dtype)
        neg_mask = torch.ones(neg_embeds.shape[:2], device=ctx.device, dtype=torch.long)
    print(f"Loaded negative embedding: {tuple(neg_embeds.shape)}")
    return neg_embeds, neg_mask


def qwen_rewrite(args: argparse.Namespace, prompt: str, instruction: str) -> str:
    dtype_literal = "torch.bfloat16" if args.qwen_dtype == "bfloat16" else "torch.float16"
    script = f"""
import re
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
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
    candidate = result.stdout.strip()
    return candidate if candidate else prompt


def generate_variants(args: argparse.Namespace, prompt: str, cache: dict[str, list[str]]) -> list[str]:
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


def build_ga_prompt_bank(prompt: str) -> list[tuple[str, str]]:
    bank: list[tuple[str, str]] = []
    for label in ("balanced", "subject", "prop", "background", "detail"):
        suffix = GA_PROMPT_FOCUS[label]
        text = prompt if not suffix else f"{prompt} {suffix}"
        bank.append((label, text))
    return bank


def encode_variants(
    args: argparse.Namespace,
    ctx: PipelineContext,
    variants: list[str],
    neg_embeds: torch.Tensor | None,
    neg_mask: torch.Tensor | None,
    max_seq: int = 256,
) -> EmbeddingContext:
    text_encoder = getattr(ctx.pipe, "text_encoder", None)
    if (
        args.offload_text_encoder_after_encode
        and text_encoder is not None
        and ctx.device == "cuda"
    ):
        try:
            text_encoder.to(ctx.device)
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"Warning: could not move text encoder to CUDA before encode: {exc}")

    pe_list: list[tuple[torch.Tensor, torch.Tensor]] = []
    ue = None
    um = None
    with torch.inference_mode():
        for i, variant in enumerate(variants):
            pe, pm, ne, nm = ctx.pipe.encode_prompt(
                prompt=variant,
                do_classifier_free_guidance=True,
                negative_prompt="",
                device=ctx.device,
                num_images_per_prompt=1,
                max_sequence_length=max_seq,
            )
            pe_list.append((pe.detach(), pm.detach()))
            if i == 0:
                ue = ne.detach()
                um = nm.detach()

    assert ue is not None
    assert um is not None

    if neg_embeds is not None and neg_mask is not None:
        cond_len = pe_list[0][0].shape[1]
        cur_ne = neg_embeds
        cur_nm = neg_mask
        neg_len = cur_ne.shape[1]
        if neg_len < cond_len:
            cur_ne = torch.cat(
                [
                    cur_ne,
                    torch.zeros(
                        1,
                        cond_len - neg_len,
                        cur_ne.shape[2],
                        device=ctx.device,
                        dtype=cur_ne.dtype,
                    ),
                ],
                dim=1,
            )
            cur_nm = torch.cat(
                [
                    cur_nm,
                    torch.zeros(1, cond_len - neg_len, device=ctx.device, dtype=cur_nm.dtype),
                ],
                dim=1,
            )
        elif neg_len > cond_len:
            cur_ne = cur_ne[:, :cond_len]
            cur_nm = cur_nm[:, :cond_len]
        ue = cur_ne.to(dtype=pe_list[0][0].dtype, device=ctx.device)
        um = cur_nm.to(device=ctx.device)

    if (
        args.offload_text_encoder_after_encode
        and text_encoder is not None
        and ctx.device == "cuda"
    ):
        try:
            text_encoder.to("cpu")
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"Warning: could not offload text encoder after encode: {exc}")

    return EmbeddingContext(pe_list=pe_list, ue=ue, um=um)


def make_latents(ctx: PipelineContext, seed: int, h: int, w: int, dtype: torch.dtype) -> torch.Tensor:
    try:
        generator = torch.Generator(device=ctx.device).manual_seed(seed)
        return ctx.pipe.prepare_latents(1, ctx.latent_c, h, w, dtype, ctx.device, generator)
    except Exception:
        torch.manual_seed(seed)
        return torch.randn(1, ctx.latent_c, h, w, device=ctx.device, dtype=dtype)


@torch.no_grad()
def transformer_step(
    args: argparse.Namespace,
    ctx: PipelineContext,
    latents: torch.Tensor,
    pe: torch.Tensor,
    pm: torch.Tensor,
    ue: torch.Tensor,
    um: torch.Tensor,
    t_flat: torch.Tensor,
    cfg_scale: float,
) -> torch.Tensor:
    if cfg_scale == 1.0:
        velocity = ctx.pipe.transformer(
            hidden_states=latents,
            encoder_hidden_states=pe,
            encoder_attention_mask=pm,
            timestep=args.time_scale * t_flat,
            return_dict=False,
        )[0]
        if ctx.variance_split:
            velocity = velocity.chunk(2, dim=1)[0]
        return velocity

    flow_both = ctx.pipe.transformer(
        hidden_states=torch.cat([latents, latents]),
        encoder_hidden_states=torch.cat([ue, pe]),
        encoder_attention_mask=torch.cat([um, pm]),
        timestep=args.time_scale * torch.cat([t_flat, t_flat]),
        return_dict=False,
    )[0]
    if ctx.variance_split:
        flow_both = flow_both.chunk(2, dim=1)[0]
    flow_uncond, flow_cond = flow_both.chunk(2)
    return flow_uncond + cfg_scale * (flow_cond - flow_uncond)


@torch.no_grad()
def decode_to_pil(
    ctx: PipelineContext,
    dx: torch.Tensor,
    orig_h: int,
    orig_w: int,
    tag: str = "generic",
) -> Image.Image:
    ctx.decode_counts[tag] = ctx.decode_counts.get(tag, 0) + 1
    scaled = (dx / ctx.pipe.vae.config.scaling_factor).to(dtype=ctx.dtype)
    image = ctx.pipe.vae.decode(scaled, return_dict=False)[0]
    image = ctx.pipe.image_processor.resize_and_crop_tensor(image, orig_h, orig_w)
    pil = ctx.pipe.image_processor.postprocess(image, output_type="pil")[0]
    if ctx.empty_cache_after_decode and ctx.device == "cuda":
        torch.cuda.empty_cache()
    return pil


def maybe_resize_to_bin(ctx: PipelineContext, height: int, width: int) -> tuple[int, int]:
    sample_size = ctx.pipe.transformer.config.sample_size
    if sample_size in ctx.aspect_ratio_bins:
        return ctx.pipe.image_processor.classify_height_width_bin(
            height, width, ratios=ctx.aspect_ratio_bins[sample_size]
        )
    return height, width


@dataclass
class SearchResult:
    image: Image.Image
    score: float
    actions: list[tuple[int, float]]


def _step_tensors(ctx: PipelineContext, steps: int, dtype: torch.dtype) -> list[tuple[torch.Tensor, torch.Tensor]]:
    schedule = []
    for i in range(steps):
        scalar_t = 999.0 * (1.0 - float(i) / float(steps))
        t_flat = torch.full((1,), scalar_t / 999.0, device=ctx.device, dtype=dtype)
        t_4d = t_flat.view(1, 1, 1, 1)
        schedule.append((t_flat, t_4d))
    return schedule


def run_action_sequence(
    args: argparse.Namespace,
    ctx: PipelineContext,
    reward_ctx: RewardContext,
    prompt: str,
    metadata: dict[str, Any] | None,
    seed: int,
    h: int,
    w: int,
    orig_h: int,
    orig_w: int,
    emb: EmbeddingContext,
    actions: list[tuple[int, float]],
) -> SearchResult:
    if len(actions) != args.steps:
        raise ValueError(f"Expected {args.steps} actions, got {len(actions)}")
    latents = make_latents(ctx, seed, h, w, emb.pe_list[0][0].dtype)
    schedule = _step_tensors(ctx, args.steps, latents.dtype)
    rng = torch.Generator(device=ctx.device).manual_seed(seed + 2048)
    dx = torch.zeros_like(latents)

    for step_idx, ((t_flat, t_4d), (variant_idx, cfg)) in enumerate(zip(schedule, actions)):
        if step_idx == 0:
            noise = latents
        else:
            noise = torch.randn(
                latents.shape,
                device=latents.device,
                dtype=latents.dtype,
                generator=rng,
            )
        latents = (1.0 - t_4d) * dx + t_4d * noise
        pe, pm = emb.pe_list[variant_idx]
        velocity = transformer_step(args, ctx, latents, pe, pm, emb.ue, emb.um, t_flat, cfg)
        dx = latents - t_4d * velocity

    image = decode_to_pil(ctx, dx, orig_h, orig_w, tag="action_sequence_final")
    score = float(reward_ctx.score_images(prompt, [image], metadata)[0])
    return SearchResult(image=image, score=score, actions=list(actions))


def _ga_step_phase(step_idx: int, steps: int) -> str:
    return "early" if step_idx < max(1, steps // 2) else "late"


def _ga_allowed_prompt_indices(
    step_idx: int,
    steps: int,
    prompt_bank: list[tuple[str, str]],
    use_constraints: bool,
) -> list[int]:
    all_ids = list(range(len(prompt_bank)))
    if not use_constraints:
        return all_ids
    label_to_idx = {label: i for i, (label, _) in enumerate(prompt_bank)}
    phase = _ga_step_phase(step_idx, steps)
    if phase == "early":
        prefer = ["balanced", "background", "subject"]
    else:
        prefer = ["detail", "balanced", "prop", "subject"]
    out = [label_to_idx[name] for name in prefer if name in label_to_idx]
    return out if out else all_ids


def _ga_default_actions(
    args: argparse.Namespace,
    prompt_bank: list[tuple[str, str]],
    cfg_bank: list[float],
) -> list[tuple[int, float]]:
    label_to_idx = {label: i for i, (label, _) in enumerate(prompt_bank)}
    pidx = label_to_idx.get("balanced", 0)
    cfg_target = 1.0
    cfg = min(cfg_bank, key=lambda x: abs(float(x) - cfg_target))
    return [(pidx, float(cfg)) for _ in range(args.steps)]


def _ga_random_genome(
    rng: np.random.Generator,
    steps: int,
    prompt_bank: list[tuple[str, str]],
    cfg_bank: list[float],
    use_constraints: bool,
) -> list[int]:
    genome: list[int] = []
    for step in range(steps):
        allowed = _ga_allowed_prompt_indices(step, steps, prompt_bank, use_constraints)
        p_gene = int(allowed[int(rng.integers(0, len(allowed)))])
        c_gene = int(rng.integers(0, len(cfg_bank)))
        genome.extend([p_gene, c_gene])
    return genome


def _ga_decode_genome(
    genome: list[int],
    args: argparse.Namespace,
    prompt_bank: list[tuple[str, str]],
    cfg_bank: list[float],
) -> tuple[list[int], list[tuple[int, float]]]:
    repaired = list(genome)
    actions: list[tuple[int, float]] = []
    for step in range(args.steps):
        p_raw = int(repaired[2 * step])
        c_raw = int(repaired[2 * step + 1])
        allowed = _ga_allowed_prompt_indices(step, args.steps, prompt_bank, args.ga_phase_constraints)
        p_idx = allowed[abs(p_raw) % len(allowed)]
        c_idx = abs(c_raw) % len(cfg_bank)
        repaired[2 * step] = int(p_idx)
        repaired[2 * step + 1] = int(c_idx)
        actions.append((int(p_idx), float(cfg_bank[c_idx])))
    return repaired, actions


def _ga_mutate(
    genome: list[int],
    rng: np.random.Generator,
    args: argparse.Namespace,
    prompt_bank: list[tuple[str, str]],
    cfg_bank: list[float],
) -> list[int]:
    out = list(genome)
    for step in range(args.steps):
        if rng.random() < args.ga_mutation_prob:
            allowed = _ga_allowed_prompt_indices(step, args.steps, prompt_bank, args.ga_phase_constraints)
            out[2 * step] = int(allowed[int(rng.integers(0, len(allowed)))])
        if rng.random() < args.ga_mutation_prob:
            out[2 * step + 1] = int(rng.integers(0, len(cfg_bank)))
    return out


def _ga_crossover(
    a: list[int],
    b: list[int],
    rng: np.random.Generator,
    mode: str,
) -> tuple[list[int], list[int]]:
    if len(a) != len(b):
        raise ValueError("Genome length mismatch in crossover.")
    n = len(a)
    if n < 2:
        return list(a), list(b)
    if mode == "one_point":
        point = int(rng.integers(1, n))
        return list(a[:point] + b[point:]), list(b[:point] + a[point:])
    # uniform
    child1: list[int] = []
    child2: list[int] = []
    for ga, gb in zip(a, b):
        if rng.random() < 0.5:
            child1.append(int(ga))
            child2.append(int(gb))
        else:
            child1.append(int(gb))
            child2.append(int(ga))
    return child1, child2


def _ga_tournament_select(
    scored: list[dict[str, Any]],
    rng: np.random.Generator,
    k: int,
) -> list[int]:
    if not scored:
        raise RuntimeError("Tournament selection received empty population.")
    picks = [scored[int(rng.integers(0, len(scored)))] for _ in range(max(1, k))]
    best = max(picks, key=lambda row: float(row["score"]))
    return list(best["genome"])


def run_ga(
    args: argparse.Namespace,
    ctx: PipelineContext,
    reward_ctx: RewardContext,
    prompt: str,
    metadata: dict[str, Any] | None,
    seed: int,
    h: int,
    w: int,
    orig_h: int,
    orig_w: int,
    emb: EmbeddingContext,
    prompt_bank: list[tuple[str, str]],
    cfg_bank: list[float],
    log_root: str,
) -> tuple[SearchResult, dict[str, Any]]:
    os.makedirs(log_root, exist_ok=True)
    rng = np.random.default_rng(seed + 9001)
    pop_size = max(4, int(args.ga_population))
    elites = min(max(1, int(args.ga_elites)), pop_size)
    topk = max(1, int(args.ga_log_topk))
    cfg_bank = [float(c) for c in cfg_bank]

    population = [
        _ga_random_genome(rng, args.steps, prompt_bank, cfg_bank, args.ga_phase_constraints)
        for _ in range(pop_size)
    ]
    default_actions = _ga_default_actions(args, prompt_bank, cfg_bank)
    cache: dict[tuple[int, ...], dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    global_best: dict[str, Any] | None = None

    def _eval(genome: list[int], need_image: bool = False) -> dict[str, Any]:
        repaired, actions = _ga_decode_genome(genome, args, prompt_bank, cfg_bank)
        key = tuple(repaired)
        cached = cache.get(key)
        if cached is not None and (not need_image or cached.get("image") is not None):
            return cached
        result = run_action_sequence(
            args,
            ctx,
            reward_ctx,
            prompt,
            metadata,
            seed,
            h,
            w,
            orig_h,
            orig_w,
            emb,
            actions,
        )
        payload = {
            "genome": repaired,
            "score": float(result.score),
            "actions": actions,
            "image": result.image if need_image else None,
        }
        cache[key] = payload
        return payload

    for gen in range(int(args.ga_generations)):
        scored = [_eval(genome, need_image=False) for genome in population]
        scored.sort(key=lambda row: float(row["score"]), reverse=True)
        best = scored[0]
        if global_best is None or float(best["score"]) > float(global_best["score"]):
            global_best = _eval(best["genome"], need_image=True)

        gen_dir = os.path.join(log_root, f"gen_{gen:03d}")
        os.makedirs(gen_dir, exist_ok=True)
        top_records: list[dict[str, Any]] = []
        for rank, row in enumerate(scored[:topk]):
            with_img = _eval(row["genome"], need_image=True)
            img_path = os.path.join(gen_dir, f"rank_{rank:02d}.png")
            with_img["image"].save(img_path)
            decoded = []
            for step_i, (vi, cfg) in enumerate(with_img["actions"]):
                label = prompt_bank[vi][0]
                decoded.append(
                    {
                        "step": step_i,
                        "prompt_idx": int(vi),
                        "prompt_label": label,
                        "cfg": float(cfg),
                    }
                )
            top_records.append(
                {
                    "rank": rank,
                    "score": float(with_img["score"]),
                    "genome": [int(x) for x in with_img["genome"]],
                    "actions": decoded,
                    "image": os.path.basename(img_path),
                }
            )

        gen_summary = {
            "generation": gen,
            "best": float(scored[0]["score"]),
            "mean": float(np.mean([float(row["score"]) for row in scored])),
            "median": float(np.median([float(row["score"]) for row in scored])),
            "worst": float(scored[-1]["score"]),
            "top": top_records,
        }
        history.append(gen_summary)
        with open(os.path.join(gen_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(gen_summary, f, indent=2)
        print(
            f"    [ga] gen={gen + 1:02d}/{args.ga_generations} "
            f"best={gen_summary['best']:.4f} mean={gen_summary['mean']:.4f}"
        )

        if gen == int(args.ga_generations) - 1:
            break

        next_population: list[list[int]] = [list(row["genome"]) for row in scored[:elites]]
        while len(next_population) < pop_size:
            p1 = _ga_tournament_select(scored, rng, args.ga_tournament_k)
            p2 = _ga_tournament_select(scored, rng, args.ga_tournament_k)
            c1, c2 = _ga_crossover(p1, p2, rng, args.ga_crossover)
            c1 = _ga_mutate(c1, rng, args, prompt_bank, cfg_bank)
            c2 = _ga_mutate(c2, rng, args, prompt_bank, cfg_bank)
            next_population.append(c1)
            if len(next_population) < pop_size:
                next_population.append(c2)
        population = next_population

    assert global_best is not None
    best_actions = [(int(vi), float(cfg)) for vi, cfg in global_best["actions"]]
    final_result = SearchResult(image=global_best["image"], score=float(global_best["score"]), actions=best_actions)

    # Baselines against the same action banks.
    no_search_result = run_action_sequence(
        args, ctx, reward_ctx, prompt, metadata, seed, h, w, orig_h, orig_w, emb, default_actions
    )
    random_best = no_search_result
    for _ in range(max(1, int(args.ga_random_trials))):
        genome = _ga_random_genome(rng, args.steps, prompt_bank, cfg_bank, args.ga_phase_constraints)
        row = _eval(genome, need_image=False)
        if row["score"] > random_best.score:
            random_best = run_action_sequence(
                args, ctx, reward_ctx, prompt, metadata, seed, h, w, orig_h, orig_w, emb, row["actions"]
            )

    # Greedy baseline over GA banks (step-wise with full rollout scoring).
    greedy_actions = list(default_actions)
    for step in range(args.steps):
        best_score = -float("inf")
        best = greedy_actions[step]
        prompt_ids = _ga_allowed_prompt_indices(step, args.steps, prompt_bank, args.ga_phase_constraints)
        for vi in prompt_ids:
            for cfg in cfg_bank:
                cand = list(greedy_actions)
                cand[step] = (int(vi), float(cfg))
                cand_result = run_action_sequence(
                    args, ctx, reward_ctx, prompt, metadata, seed, h, w, orig_h, orig_w, emb, cand
                )
                if cand_result.score > best_score:
                    best_score = cand_result.score
                    best = (int(vi), float(cfg))
        greedy_actions[step] = best
    greedy_bank_result = run_action_sequence(
        args, ctx, reward_ctx, prompt, metadata, seed, h, w, orig_h, orig_w, emb, greedy_actions
    )

    # Current MCTS baseline with same banks.
    old_cfg = list(args.cfg_scales)
    args.cfg_scales = [float(c) for c in cfg_bank]
    try:
        mcts_result = run_mcts(args, ctx, reward_ctx, prompt, metadata, seed, h, w, orig_h, orig_w, emb)
    finally:
        args.cfg_scales = old_cfg

    # Save baseline images.
    no_search_result.image.save(os.path.join(log_root, "baseline_no_search.png"))
    random_best.image.save(os.path.join(log_root, "baseline_random.png"))
    greedy_bank_result.image.save(os.path.join(log_root, "baseline_greedy.png"))
    mcts_result.image.save(os.path.join(log_root, "baseline_mcts.png"))
    final_result.image.save(os.path.join(log_root, "ga_best.png"))

    diagnostics = {
        "prompt_bank": [{"label": label, "text": text} for label, text in prompt_bank],
        "cfg_bank": [float(c) for c in cfg_bank],
        "history": history,
        "best_genome": [int(x) for x in global_best["genome"]],
        "baselines": {
            "no_search": float(no_search_result.score),
            "random": float(random_best.score),
            "greedy": float(greedy_bank_result.score),
            "mcts": float(mcts_result.score),
            "ga": float(final_result.score),
        },
        "baseline_actions": {
            "no_search": [[int(vi), float(cfg)] for vi, cfg in no_search_result.actions],
            "random": [[int(vi), float(cfg)] for vi, cfg in random_best.actions],
            "greedy": [[int(vi), float(cfg)] for vi, cfg in greedy_bank_result.actions],
            "mcts": [[int(vi), float(cfg)] for vi, cfg in mcts_result.actions],
            "ga": [[int(vi), float(cfg)] for vi, cfg in final_result.actions],
        },
    }
    with open(os.path.join(log_root, "ga_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    return final_result, diagnostics


def run_baseline(
    args: argparse.Namespace,
    ctx: PipelineContext,
    reward_ctx: RewardContext,
    prompt: str,
    metadata: dict[str, Any] | None,
    seed: int,
    h: int,
    w: int,
    orig_h: int,
    orig_w: int,
    emb: EmbeddingContext,
) -> tuple[Image.Image, float]:
    pe, pm = emb.pe_list[0]
    latents = make_latents(ctx, seed, h, w, pe.dtype)
    schedule = _step_tensors(ctx, args.steps, latents.dtype)

    dx = torch.zeros_like(latents)
    for step_idx, (t_flat, t_4d) in enumerate(schedule):
        noise = latents if step_idx == 0 else torch.randn_like(latents)
        latents = (1.0 - t_4d) * dx + t_4d * noise
        velocity = transformer_step(args, ctx, latents, pe, pm, emb.ue, emb.um, t_flat, 1.0)
        dx = latents - t_4d * velocity

    image = decode_to_pil(ctx, dx, orig_h, orig_w, tag="baseline_final")
    score = reward_ctx.score_images(prompt, [image], metadata)[0]
    return image, float(score)


def run_greedy(
    args: argparse.Namespace,
    ctx: PipelineContext,
    reward_ctx: RewardContext,
    prompt: str,
    metadata: dict[str, Any] | None,
    seed: int,
    h: int,
    w: int,
    orig_h: int,
    orig_w: int,
    emb: EmbeddingContext,
    variants: list[str],
) -> SearchResult:
    actions = [(vi, cfg) for vi in range(len(emb.pe_list)) for cfg in args.cfg_scales]
    latents = make_latents(ctx, seed, h, w, emb.pe_list[0][0].dtype)
    schedule = _step_tensors(ctx, args.steps, latents.dtype)
    dx = torch.zeros_like(latents)
    chosen: list[tuple[int, float]] = []

    for step_idx, (t_flat, t_4d) in enumerate(schedule):
        noise = latents if step_idx == 0 else torch.randn_like(latents)
        latents = (1.0 - t_4d) * dx + t_4d * noise

        best_score = -float("inf")
        best_action = actions[0]
        best_dx = None

        print(f"  step {step_idx + 1}/{args.steps}: evaluating {len(actions)} actions")
        for variant_idx, cfg in actions:
            pe, pm = emb.pe_list[variant_idx]
            velocity = transformer_step(args, ctx, latents, pe, pm, emb.ue, emb.um, t_flat, cfg)
            candidate_dx = latents - t_4d * velocity
            candidate_img = decode_to_pil(ctx, candidate_dx, orig_h, orig_w, tag="greedy_candidate")
            score = float(reward_ctx.score_images(prompt, [candidate_img], metadata)[0])
            mark = ""
            if score > best_score:
                best_score = score
                best_action = (variant_idx, cfg)
                best_dx = candidate_dx.clone()
                mark = " <- best"
            print(f"    v{variant_idx} cfg={cfg:.2f} score={score:.4f}{mark}")

        assert best_dx is not None
        dx = best_dx
        chosen.append(best_action)
        preview_prompt = variants[best_action[0]][:56]
        print(
            f"  selected: step={step_idx + 1} v={best_action[0]} cfg={best_action[1]:.2f} "
            f"prompt='{preview_prompt}' score={best_score:.4f}"
        )

    final_image = decode_to_pil(ctx, dx, orig_h, orig_w, tag="greedy_final")
    final_score = float(reward_ctx.score_images(prompt, [final_image], metadata)[0])
    return SearchResult(image=final_image, score=final_score, actions=chosen)


class MCTSNode:
    __slots__ = ("step", "dx", "latents", "children", "n", "action_n", "action_q")

    def __init__(self, step: int, dx: torch.Tensor, latents: torch.Tensor | None):
        self.step = step
        self.dx = dx
        self.latents = latents
        self.children: dict[tuple[int, float], "MCTSNode"] = {}
        self.n = 0
        self.action_n: dict[tuple[int, float], int] = {}
        self.action_q: dict[tuple[int, float], float] = {}

    def is_leaf(self, steps: int) -> bool:
        return self.step >= steps

    def untried_actions(self, actions: list[tuple[int, float]]) -> list[tuple[int, float]]:
        return [action for action in actions if action not in self.action_n]

    def ucb1(self, action: tuple[int, float], c: float) -> float:
        action_visits = self.action_n.get(action, 0)
        if action_visits == 0:
            return float("inf")
        avg = self.action_q[action] / action_visits
        return avg + c * math.sqrt(math.log(max(self.n, 1)) / action_visits)

    def best_action_ucb(self, actions: list[tuple[int, float]], c: float) -> tuple[int, float]:
        return max(actions, key=lambda action: self.ucb1(action, c))

    def best_action_exploit(self, actions: list[tuple[int, float]]) -> tuple[int, float] | None:
        best_action = None
        best_avg = -float("inf")
        for action in actions:
            action_visits = self.action_n.get(action, 0)
            if action_visits == 0:
                continue
            avg = self.action_q[action] / action_visits
            if avg > best_avg:
                best_avg = avg
                best_action = action
        return best_action


def _mcts_forward_child(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    node: MCTSNode,
    action: tuple[int, float],
    schedule: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    variant_idx, cfg = action
    pe, pm = emb.pe_list[variant_idx]
    t_flat, t_4d = schedule[node.step]
    velocity = transformer_step(args, ctx, node.latents, pe, pm, emb.ue, emb.um, t_flat, cfg)
    new_dx = node.latents - t_4d * velocity
    next_step = node.step + 1
    if next_step < len(schedule):
        _, next_t_4d = schedule[next_step]
        noise = torch.randn_like(new_dx)
        new_latents = (1.0 - next_t_4d) * new_dx + next_t_4d * noise
    else:
        new_latents = None
    return new_dx, new_latents


def run_mcts(
    args: argparse.Namespace,
    ctx: PipelineContext,
    reward_ctx: RewardContext,
    prompt: str,
    metadata: dict[str, Any] | None,
    seed: int,
    h: int,
    w: int,
    orig_h: int,
    orig_w: int,
    emb: EmbeddingContext,
) -> SearchResult:
    actions = [(vi, cfg) for vi in range(len(emb.pe_list)) for cfg in args.cfg_scales]
    n_actions = len(actions)
    latents_init = make_latents(ctx, seed, h, w, emb.pe_list[0][0].dtype)
    schedule = _step_tensors(ctx, args.steps, latents_init.dtype)

    dx_init = torch.zeros_like(latents_init)
    _, t0_4d = schedule[0]
    latents_0 = (1.0 - t0_4d) * dx_init + t0_4d * latents_init
    root = MCTSNode(step=0, dx=dx_init, latents=latents_0)

    best_global_score = -float("inf")
    best_global_dx = None
    best_global_path: list[tuple[int, float]] = []

    print(
        f"  mcts sims={args.n_sims} actions_per_step={n_actions} steps={args.steps} c={args.ucb_c:.2f}"
    )
    for sim in range(args.n_sims):
        node = root
        path: list[tuple[MCTSNode, tuple[int, float]]] = []

        # Select
        while not node.is_leaf(args.steps):
            untried = node.untried_actions(actions)
            if untried:
                action = untried[np.random.randint(len(untried))]
                break
            action = node.best_action_ucb(actions, args.ucb_c)
            path.append((node, action))
            node = node.children[action]

        # Expand
        if not node.is_leaf(args.steps):
            if action not in node.children:
                new_dx, new_latents = _mcts_forward_child(args, ctx, emb, node, action, schedule)
                node.children[action] = MCTSNode(step=node.step + 1, dx=new_dx, latents=new_latents)
            path.append((node, action))
            node = node.children[action]

        # Rollout
        rollout_dx = node.dx
        rollout_latents = node.latents
        rollout_step = node.step
        while rollout_step < args.steps:
            variant_idx, cfg = actions[np.random.randint(n_actions)]
            pe, pm = emb.pe_list[variant_idx]
            t_flat, t_4d = schedule[rollout_step]
            velocity = transformer_step(args, ctx, rollout_latents, pe, pm, emb.ue, emb.um, t_flat, cfg)
            rollout_dx = rollout_latents - t_4d * velocity
            rollout_step += 1
            if rollout_step < args.steps:
                _, next_t_4d = schedule[rollout_step]
                noise = torch.randn_like(rollout_dx)
                rollout_latents = (1.0 - next_t_4d) * rollout_dx + next_t_4d * noise

        rollout_img = decode_to_pil(ctx, rollout_dx, orig_h, orig_w, tag="mcts_rollout")
        rollout_score = float(reward_ctx.score_images(prompt, [rollout_img], metadata)[0])

        if rollout_score > best_global_score:
            best_global_score = rollout_score
            best_global_dx = rollout_dx.clone()
            best_global_path = [action for _, action in path]

        for parent_node, parent_action in path:
            parent_node.n += 1
            parent_node.action_n[parent_action] = parent_node.action_n.get(parent_action, 0) + 1
            parent_node.action_q[parent_action] = parent_node.action_q.get(parent_action, 0.0) + rollout_score

        if (sim + 1) % 10 == 0 or sim == 0:
            print(f"    sim={sim + 1:3d}/{args.n_sims} best_score={best_global_score:.4f}")

    # Exploit best average path from the tree
    exploit_path: list[tuple[int, float]] = []
    node = root
    for _ in range(args.steps):
        action = node.best_action_exploit(actions)
        if action is None:
            break
        exploit_path.append(action)
        if action in node.children:
            node = node.children[action]
        else:
            break

    replay_dx = dx_init
    replay_latents = latents_0
    for step_idx, (variant_idx, cfg) in enumerate(exploit_path):
        pe, pm = emb.pe_list[variant_idx]
        t_flat, t_4d = schedule[step_idx]
        velocity = transformer_step(args, ctx, replay_latents, pe, pm, emb.ue, emb.um, t_flat, cfg)
        replay_dx = replay_latents - t_4d * velocity
        if step_idx + 1 < args.steps:
            _, next_t_4d = schedule[step_idx + 1]
            noise = torch.randn_like(replay_dx)
            replay_latents = (1.0 - next_t_4d) * replay_dx + next_t_4d * noise

    exploit_img = decode_to_pil(ctx, replay_dx, orig_h, orig_w, tag="mcts_exploit")
    exploit_score = float(reward_ctx.score_images(prompt, [exploit_img], metadata)[0])

    if exploit_score >= best_global_score:
        return SearchResult(image=exploit_img, score=exploit_score, actions=exploit_path)
    if best_global_dx is None:
        return SearchResult(image=exploit_img, score=exploit_score, actions=exploit_path)
    best_global_img = decode_to_pil(ctx, best_global_dx, orig_h, orig_w, tag="mcts_best_global")
    return SearchResult(image=best_global_img, score=best_global_score, actions=best_global_path)


def _font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def save_comparison(
    out_path: str,
    baseline_img: Image.Image,
    search_img: Image.Image,
    baseline_score: float,
    search_score: float,
    actions: list[tuple[int, float]],
) -> None:
    width, height = baseline_img.size
    header_h = 54
    grid = Image.new("RGB", (width * 2, height + header_h), (18, 18, 18))
    draw = ImageDraw.Draw(grid)
    grid.paste(baseline_img, (0, header_h))
    grid.paste(search_img, (width, header_h))
    draw.text((4, 4), f"baseline score={baseline_score:.3f}", fill=(200, 200, 200), font=_font(15))
    delta = search_score - baseline_score
    score_col = (100, 255, 100) if delta >= 0 else (255, 100, 100)
    draw.text(
        (width + 4, 4),
        f"search score={search_score:.3f} delta={delta:+.3f}",
        fill=score_col,
        font=_font(15),
    )
    path_text = " ".join(f"s{idx + 1}:v{vi}/cfg{cfg:.2f}" for idx, (vi, cfg) in enumerate(actions))
    draw.text((width + 4, 28), path_text[:96], fill=(255, 220, 50), font=_font(11))
    grid.save(out_path)


def save_geneval_layout(image: Image.Image, metadata: dict[str, Any], prompt_index: int, root: str, sample_index: int) -> None:
    prompt_dir = os.path.join(root, f"{prompt_index:05d}")
    sample_dir = os.path.join(prompt_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    with open(os.path.join(prompt_dir, "metadata.jsonl"), "w") as f:
        json.dump(metadata, f)
    image.save(os.path.join(sample_dir, f"{sample_index:04d}.png"))


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    entries = _load_prompt_entries(args)
    if not entries:
        raise RuntimeError("No prompts found to evaluate.")
    if args.reward_type == "geneval" and args.geneval_prompts is None:
        raise RuntimeError("Geneval mode requires --geneval_prompts metadata jsonl input.")

    ctx = load_pipeline(args)
    reward_ctx = load_reward(args, ctx)
    neg_embeds, neg_mask = load_neg_embed(args, ctx)

    rewrite_cache: dict[str, list[str]] = {}
    if args.rewrites_file and os.path.exists(args.rewrites_file):
        with open(args.rewrites_file) as f:
            rewrite_cache = json.load(f)
        print(f"Loaded rewrites cache for {len(rewrite_cache)} prompts from {args.rewrites_file}")

    # Pre-generate prompt banks once per prompt.
    bank_map: dict[int, list[tuple[str, str]]] = {}
    for entry in entries:
        if args.search_method == "ga":
            bank = build_ga_prompt_bank(entry["prompt"])
        else:
            variants = generate_variants(args, entry["prompt"], rewrite_cache)
            bank = [(f"v{i}", text) for i, text in enumerate(variants)]
        bank_map[entry["index"]] = bank

        variants_path = os.path.join(args.out_dir, f"{entry['slug']}_variants.txt")
        with open(variants_path, "w", encoding="utf-8") as f:
            for idx, (label, variant) in enumerate(bank):
                f.write(f"v{idx}[{label}]: {variant}\n")

    summary: list[dict[str, Any]] = []
    geneval_root = os.path.join(args.out_dir, "geneval_images")
    if args.reward_type == "geneval":
        os.makedirs(geneval_root, exist_ok=True)

    for entry in entries:
        prompt = entry["prompt"]
        metadata = entry["metadata"]
        slug = entry["slug"]
        index = entry["index"]
        bank = bank_map[index]
        variants = [text for _, text in bank]
        print(f"\n{'=' * 72}\n[{slug}] {prompt}\n{'=' * 72}")

        orig_h, orig_w = args.height, args.width
        h, w = maybe_resize_to_bin(ctx, orig_h, orig_w)

        prompt_samples: list[dict[str, Any]] = []
        for sample_i in range(args.n_samples):
            seed = args.seed + sample_i
            print(f"\n  sample {sample_i + 1}/{args.n_samples} seed={seed}")

            emb = encode_variants(args, ctx, variants, neg_embeds, neg_mask)
            baseline_img, baseline_score = run_baseline(
                args, ctx, reward_ctx, prompt, metadata, seed, h, w, orig_h, orig_w, emb
            )

            ga_diag = None
            if args.search_method == "greedy":
                search_result = run_greedy(
                    args,
                    ctx,
                    reward_ctx,
                    prompt,
                    metadata,
                    seed,
                    h,
                    w,
                    orig_h,
                    orig_w,
                    emb,
                    variants,
                )
            elif args.search_method == "mcts":
                search_result = run_mcts(
                    args, ctx, reward_ctx, prompt, metadata, seed, h, w, orig_h, orig_w, emb
                )
            else:
                ga_log_root = os.path.join(args.out_dir, f"{slug}_s{sample_i}_ga")
                search_result, ga_diag = run_ga(
                    args,
                    ctx,
                    reward_ctx,
                    prompt,
                    metadata,
                    seed,
                    h,
                    w,
                    orig_h,
                    orig_w,
                    emb,
                    prompt_bank=bank,
                    cfg_bank=[float(v) for v in args.ga_cfg_scales],
                    log_root=ga_log_root,
                )

            search_name = args.search_method
            baseline_path = os.path.join(args.out_dir, f"{slug}_s{sample_i}_baseline.png")
            search_path = os.path.join(args.out_dir, f"{slug}_s{sample_i}_{search_name}.png")
            comparison_path = os.path.join(args.out_dir, f"{slug}_s{sample_i}_comparison.png")
            baseline_img.save(baseline_path)
            search_result.image.save(search_path)
            save_comparison(
                comparison_path,
                baseline_img,
                search_result.image,
                baseline_score,
                search_result.score,
                search_result.actions,
            )

            if args.reward_type == "geneval" and metadata is not None:
                save_geneval_layout(search_result.image, metadata, index, geneval_root, sample_i)

            print(
                f"  baseline={baseline_score:.4f} {search_name}={search_result.score:.4f} "
                f"delta={search_result.score - baseline_score:+.4f}"
            )

            sample_payload = {
                "seed": seed,
                "baseline_score": baseline_score,
                "search_score": search_result.score,
                "delta_score": search_result.score - baseline_score,
                "actions": [[int(vi), float(cfg)] for vi, cfg in search_result.actions],
            }
            if ga_diag is not None:
                sample_payload["ga_baselines"] = ga_diag["baselines"]
                sample_payload["ga_log_dir"] = os.path.join(args.out_dir, f"{slug}_s{sample_i}_ga")
            if args.reward_type == "geneval":
                sample_payload["baseline_pass"] = baseline_score >= 0.99
                sample_payload["search_pass"] = search_result.score >= 0.99
            prompt_samples.append(sample_payload)

        summary.append(
            {
                "slug": slug,
                "index": index,
                "prompt": prompt,
                "reward_type": args.reward_type,
                "search_method": args.search_method,
                "variants": variants,
                "variant_bank": [{"label": label, "text": text} for label, text in bank],
                "samples": prompt_samples,
            }
        )

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    decode_counts_path = os.path.join(args.out_dir, "decode_counts.json")
    with open(decode_counts_path, "w", encoding="utf-8") as f:
        json.dump(ctx.decode_counts, f, indent=2, sort_keys=True)

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    deltas = []
    for row in summary:
        sample_deltas = [sample["delta_score"] for sample in row["samples"]]
        mean_delta = float(np.mean(sample_deltas)) if sample_deltas else 0.0
        deltas.extend(sample_deltas)
        print(f"{row['slug']}: mean_delta={mean_delta:+.4f} n_samples={len(row['samples'])}")
    if deltas:
        print(f"overall mean delta: {float(np.mean(deltas)):+.4f}")
    print(f"summary json: {summary_path}")
    print(f"decode counts: {ctx.decode_counts}")
    print(f"decode counts json: {decode_counts_path}")
    if args.reward_type == "geneval":
        print(f"geneval images: {os.path.abspath(geneval_root)}/")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.cuda_alloc_conf and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf
        print(f"Set PYTORCH_CUDA_ALLOC_CONF={args.cuda_alloc_conf}")
    run(args)


if __name__ == "__main__":
    main()
