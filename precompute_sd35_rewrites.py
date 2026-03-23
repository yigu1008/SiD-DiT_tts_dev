#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch precompute Qwen rewrites cache for SD3.5 DDP runs.")
    p.add_argument("--prompt_file", required=True)
    p.add_argument("--rewrites_file", required=True)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--end_index", type=int, default=-1)
    p.add_argument("--n_variants", type=int, default=3)
    p.add_argument("--qwen_id", default="Qwen/Qwen3-4B")
    p.add_argument("--qwen_dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:N")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--save_every_batches", type=int, default=1)
    p.add_argument("--clear_cache_each_batch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def resolve_path(path_str: str) -> str:
    p = Path(path_str).expanduser().resolve()
    return str(p)


def load_prompts(path: str, start: int, end: int) -> list[str]:
    prompts = [line.strip() for line in open(path, encoding="utf-8") if line.strip()]
    if end < 0:
        end = len(prompts)
    start = max(0, min(start, end))
    end = max(start, min(end, len(prompts)))
    return prompts[start:end]


def load_cache(path: str, overwrite: bool) -> dict[str, list[str]]:
    if overwrite:
        return {}
    try:
        raw = json.load(open(path, encoding="utf-8"))
        if isinstance(raw, dict):
            out: dict[str, list[str]] = {}
            for k, v in raw.items():
                if isinstance(v, list):
                    out[str(k)] = [str(x) for x in v]
            return out
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"[rewrites] warning: failed to load cache {path}: {exc}")
    return {}


def save_cache(path: str, cache: dict[str, list[str]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    os.replace(tmp, str(out_path))


def clean_generation(text: str, fallback: str) -> str:
    decoded = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL).strip()
    for line in decoded.splitlines():
        line = line.strip()
        if line:
            return line
    return fallback


def resolve_device(req: str) -> tuple[Any, str]:
    key = str(req).strip().lower()
    if key == "auto":
        target = "cuda:0" if torch.cuda.is_available() else "cpu"
        return {"": target}, target
    if key == "cuda":
        target = "cuda:0"
        return {"": target}, target
    if key.startswith("cuda:"):
        target = str(req)
        return {"": target}, target
    target = str(req)
    return {"": target}, target


def build_chat_prompts(tokenizer: Any, prompts: list[str], instruction: str) -> list[str]:
    rows = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": instruction + "\n\nOriginal prompt: " + prompt + " /no_think"},
        ]
        rows.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return rows


@torch.inference_mode()
def rewrite_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    instruction: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool = True,
) -> list[str]:
    if len(prompts) == 0:
        return []
    texts = build_chat_prompts(tokenizer, prompts, instruction)
    model_device = None
    for param in model.parameters():
        if param.device.type != "meta":
            model_device = param.device
            break
    if model_device is None:
        model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model_device)
    out = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        do_sample=bool(do_sample),
        pad_token_id=tokenizer.eos_token_id,
    )
    attn = inputs.attention_mask
    results: list[str] = []
    for i, prompt in enumerate(prompts):
        in_len = int(attn[i].sum().item())
        decoded = tokenizer.decode(out[i][in_len:], skip_special_tokens=True)
        results.append(clean_generation(decoded, fallback=prompt))
    return results


def batched(items: list[str], batch_size: int) -> list[list[str]]:
    bs = max(1, int(batch_size))
    return [items[i : i + bs] for i in range(0, len(items), bs)]


def main() -> None:
    args = parse_args()
    args.prompt_file = resolve_path(args.prompt_file)
    args.rewrites_file = resolve_path(args.rewrites_file)
    device_map, target_device = resolve_device(args.device)
    dtype = torch.bfloat16 if args.qwen_dtype == "bfloat16" else torch.float16

    prompts = load_prompts(args.prompt_file, args.start_index, args.end_index)
    cache = load_cache(args.rewrites_file, args.overwrite)
    pending = [p for p in prompts if p not in cache]
    print(
        f"[rewrites] prompts_total={len(prompts)} pending={len(pending)} "
        f"cached={len(prompts)-len(pending)} device={target_device} dtype={args.qwen_dtype}"
    , flush=True)
    if len(pending) == 0:
        save_cache(args.rewrites_file, cache)
        print(f"[rewrites] nothing to do. cache={args.rewrites_file}", flush=True)
        return

    t0 = time.perf_counter()
    # Guard against launcher-injected distributed env; this script must stay single-process.
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "NODE_RANK"):
        if key in os.environ:
            os.environ.pop(key, None)
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.qwen_id,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()

    batches = batched(pending, args.batch_size)
    for bi, batch_prompts in enumerate(batches, start=1):
        for prompt in batch_prompts:
            cache[prompt] = [prompt]

        styles = (REWRITE_STYLES * ((int(args.n_variants) // len(REWRITE_STYLES)) + 1))[: int(args.n_variants)]
        for style in styles:
            rewrites = rewrite_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                instruction=style,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
            )
            for prompt, rewrite in zip(batch_prompts, rewrites):
                cache[prompt].append(str(rewrite).strip() or prompt)

        if args.save_every_batches > 0 and (bi % int(args.save_every_batches) == 0):
            save_cache(args.rewrites_file, cache)

        if args.clear_cache_each_batch and str(model.device).startswith("cuda"):
            gc.collect()
            torch.cuda.empty_cache()

        print(f"[rewrites] batch {bi}/{len(batches)} done ({len(batch_prompts)} prompts)", flush=True)

    save_cache(args.rewrites_file, cache)
    elapsed = time.perf_counter() - t0
    print(f"[rewrites] done. cache={args.rewrites_file} prompts={len(pending)} elapsed_sec={elapsed:.2f}", flush=True)


if __name__ == "__main__":
    main()
