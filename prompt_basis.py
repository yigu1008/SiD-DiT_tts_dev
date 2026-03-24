from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Any

import torch

import sampling_unified as su


BASIS_LABELS = ("balanced", "subject", "composition", "texture")

QWEN_BASIS_SYSTEM = (
    "You are an image prompt editor. "
    "Given one text-to-image prompt, return JSON only with keys: "
    "balanced, subject, composition, texture. "
    "All variants must preserve the same scene and identity. "
    "balanced: faithful full prompt rewrite. "
    "subject: emphasize subject attributes/details. "
    "composition: emphasize layout/background/spatial composition. "
    "texture: emphasize materials/surface/fine texture details."
)


@dataclass
class PromptCandidate:
    label: str
    text: str


@dataclass
class PromptBasis:
    original: str
    candidates: list[PromptCandidate]


@dataclass
class PromptBasisEmbeddings:
    basis: PromptBasis
    labels: list[str]
    texts: list[str]
    pe_list: list[tuple[torch.Tensor, torch.Tensor]]
    ue: torch.Tensor
    um: torch.Tensor
    orig_pe: torch.Tensor
    orig_pm: torch.Tensor
    orig_ue: torch.Tensor
    orig_um: torch.Tensor


def load_basis_cache(path: str | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            out: dict[str, dict[str, str]] = {}
            for prompt, payload in raw.items():
                if not isinstance(payload, dict):
                    continue
                cur = {k: str(v) for k, v in payload.items() if isinstance(k, str)}
                out[str(prompt)] = cur
            return out
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"[prompt_basis] warning: failed to load cache {path}: {exc}")
    return {}


def save_basis_cache(path: str | None, cache: dict[str, dict[str, str]]) -> None:
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"[prompt_basis] warning: failed to save cache {path}: {exc}")


def _extract_json_obj(raw: str) -> dict[str, Any] | None:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _normalize_text(s: str, fallback: str) -> str:
    return su.sanitize_rewrite_text(str(s), fallback)


def _qwen_basis_once(args: Any, prompt: str) -> PromptBasis | None:
    dtype_literal = "torch.bfloat16" if str(args.qwen_dtype) == "bfloat16" else "torch.float16"
    script = f"""
import json, re, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained({repr(args.qwen_id)})
mdl = AutoModelForCausalLM.from_pretrained(
    {repr(args.qwen_id)},
    torch_dtype={dtype_literal},
    device_map="auto",
)
mdl.eval()
msgs = [
    {{"role":"system","content":{repr(QWEN_BASIS_SYSTEM)}}},
    {{"role":"user","content":"Prompt: " + sys.argv[1] + "\\nReturn strict JSON only."}},
]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inp = tok([text], return_tensors="pt").to(mdl.device)
with torch.no_grad():
    out = mdl.generate(
        **inp,
        max_new_tokens=320,
        temperature=0.4,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tok.eos_token_id,
    )
decoded = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()
print(decoded)
"""
    proc = subprocess.run([args.qwen_python, "-c", script, prompt], capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    obj = _extract_json_obj(proc.stdout)
    if obj is None:
        return None
    vals = {k: _normalize_text(obj.get(k, ""), prompt) for k in BASIS_LABELS}
    candidates = [PromptCandidate(label=k, text=vals[k]) for k in BASIS_LABELS]
    return PromptBasis(original=prompt, candidates=candidates)


def _heuristic_basis(prompt: str) -> PromptBasis:
    variants = {
        "balanced": prompt,
        "subject": f"{prompt}. Emphasize subject identity, face, and key subject attributes.",
        "composition": f"{prompt}. Emphasize composition, spatial layout, and background scene structure.",
        "texture": f"{prompt}. Emphasize material texture, surface detail, and fine-grained visual details.",
    }
    candidates = [PromptCandidate(label=k, text=variants[k]) for k in BASIS_LABELS]
    return PromptBasis(original=prompt, candidates=candidates)


def _rewrite_fallback_basis(args: Any, prompt: str) -> PromptBasis:
    instructions = {
        "balanced": (
            "Rewrite minimally while preserving scene and composition. Keep balanced full-scene description."
        ),
        "subject": (
            "Rewrite to emphasize subject details and attributes while preserving the same scene/composition."
        ),
        "composition": (
            "Rewrite to emphasize composition, layout, and background context while preserving subject identity."
        ),
        "texture": (
            "Rewrite to emphasize texture/material/fine details while preserving scene and composition."
        ),
    }
    candidates: list[PromptCandidate] = []
    for label in BASIS_LABELS:
        text = su.qwen_rewrite(args, prompt, instructions[label])
        candidates.append(PromptCandidate(label=label, text=_normalize_text(text, prompt)))
    return PromptBasis(original=prompt, candidates=candidates)


def build_prompt_basis(
    args: Any,
    prompt: str,
    cache: dict[str, dict[str, str]],
) -> PromptBasis:
    cached = cache.get(prompt)
    if isinstance(cached, dict):
        vals = {k: _normalize_text(cached.get(k, ""), prompt) for k in BASIS_LABELS}
        if all(vals[k] for k in BASIS_LABELS):
            return PromptBasis(
                original=prompt,
                candidates=[PromptCandidate(label=k, text=vals[k]) for k in BASIS_LABELS],
            )

    if getattr(args, "no_qwen", False):
        basis = _heuristic_basis(prompt)
    else:
        basis = _qwen_basis_once(args, prompt)
        if basis is None:
            basis = _rewrite_fallback_basis(args, prompt)

    cache[prompt] = {c.label: c.text for c in basis.candidates}
    return basis


def encode_prompt_basis(
    args: Any,
    ctx: su.PipelineContext,
    basis: PromptBasis,
    neg_embeds: torch.Tensor | None,
    neg_mask: torch.Tensor | None,
    max_seq: int = 256,
) -> PromptBasisEmbeddings:
    texts = [c.text for c in basis.candidates]
    labels = [c.label for c in basis.candidates]
    emb = su.encode_variants(args, ctx, texts, neg_embeds, neg_mask, max_seq=max_seq)
    emb_orig = su.encode_variants(args, ctx, [basis.original], neg_embeds, neg_mask, max_seq=max_seq)
    return PromptBasisEmbeddings(
        basis=basis,
        labels=labels,
        texts=texts,
        pe_list=emb.pe_list,
        ue=emb.ue,
        um=emb.um,
        orig_pe=emb_orig.pe_list[0][0],
        orig_pm=emb_orig.pe_list[0][1],
        orig_ue=emb_orig.ue,
        orig_um=emb_orig.um,
    )
