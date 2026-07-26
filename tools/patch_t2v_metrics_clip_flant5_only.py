#!/usr/bin/env python3
"""Restrict legacy t2v-metrics 3.0 imports to CLIP-FlanT5 VQAScore.

The 3.0 wheel eagerly imports every VQA/CLIP/ITM implementation, making
optional packages such as FlashAttention and InternVideo2 mandatory even
when the requested model is only clip-flant5-xxl. This patch changes registry
imports only; the CLIP-FlanT5 model and scoring implementation are untouched.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
from pathlib import Path


PATCH_MARKER = "SiD compatibility registry: legacy t2v-metrics, CLIP-FlanT5 only."
VQA_PATCH_MARKER = "SiD compatibility registry: expose only CLIP-FlanT5 VQAScore."

TOP_LEVEL = '''\
"""SiD compatibility registry: legacy t2v-metrics, CLIP-FlanT5 only."""
import shutil
import subprocess

if shutil.which("ffmpeg") is None:
    raise RuntimeError("ffmpeg is required by t2v-metrics")
subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)

from .constants import HF_CACHE_DIR
from .vqascore import VQAScore, list_all_vqascore_models


def list_all_models():
    return list_all_vqascore_models()


def get_score_model(model="clip-flant5-xxl", device="cuda", cache_dir=HF_CACHE_DIR, **kwargs):
    if model not in list_all_vqascore_models():
        raise NotImplementedError(
            f"This isolated reward environment supports CLIP-FlanT5 only, not {model!r}"
        )
    return VQAScore(model, device=device, cache_dir=cache_dir, **kwargs)
'''


VQA_REGISTRY = '''\
"""SiD compatibility registry: expose only CLIP-FlanT5 VQAScore."""
from .clip_t5_model import CLIP_T5_MODELS, CLIPT5Model
from ...constants import HF_CACHE_DIR

ALL_VQA_MODELS = [CLIP_T5_MODELS]


def list_all_vqascore_models():
    return [model for models in ALL_VQA_MODELS for model in models]


def get_vqascore_model(model_name, device="cuda", cache_dir=HF_CACHE_DIR, **kwargs):
    if model_name not in CLIP_T5_MODELS:
        raise NotImplementedError(
            f"This isolated reward environment supports CLIP-FlanT5 only, not {model_name!r}"
        )
    return CLIPT5Model(model_name, device=device, cache_dir=cache_dir, **kwargs)
'''


def _atomic_replace(path: Path, content: str) -> None:
    backup = path.with_suffix(path.suffix + ".sid_full_registry.bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    temporary = path.with_suffix(path.suffix + ".sid_tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _package_root(override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()

    version = metadata.version("t2v-metrics")
    if version.split(".", 1)[0] != "3":
        raise RuntimeError(f"expected t2v-metrics 3.x, found {version}")
    distribution = metadata.distribution("t2v-metrics")
    return Path(distribution.locate_file("t2v_metrics")).resolve()


def _registry_paths(package_root: Path) -> tuple[Path, Path]:
    top_level = package_root / "__init__.py"
    vqa_registry = package_root / "models" / "vqascore_models" / "__init__.py"
    for path in (top_level, vqa_registry):
        if not path.is_file():
            raise FileNotFoundError(path)
    return top_level, vqa_registry


def _check_registry(top_level: Path, vqa_registry: Path) -> None:
    top_text = top_level.read_text(encoding="utf-8")
    vqa_text = vqa_registry.read_text(encoding="utf-8")
    problems = []
    if PATCH_MARKER not in top_text:
        problems.append(f"{top_level} lacks the SiD CLIP-FlanT5-only marker")
    if VQA_PATCH_MARKER not in vqa_text:
        problems.append(f"{vqa_registry} lacks the SiD CLIP-FlanT5-only marker")

    forbidden = (
        "clipscore",
        "itmscore",
        "llava",
        "internvideo",
        "flash_attn",
        "qwen2vl",
    )
    combined = f"{top_text}\n{vqa_text}".lower()
    leaked = [name for name in forbidden if name in combined]
    if leaked:
        problems.append(
            "patched registries still reference unrelated backends: "
            + ", ".join(leaked)
        )
    if problems:
        raise RuntimeError("; ".join(problems))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=None,
        help="Testing override for the installed t2v_metrics package directory.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that the installed registries are patched; do not modify files.",
    )
    args = parser.parse_args()

    package_root = _package_root(args.package_root)
    top_level, vqa_registry = _registry_paths(package_root)
    if not args.check:
        _atomic_replace(top_level, TOP_LEVEL)
        _atomic_replace(vqa_registry, VQA_REGISTRY)
    _check_registry(top_level, vqa_registry)
    action = "verified" if args.check else "installed and verified"
    print(f"[t2v-patch] CLIP-FlanT5-only registry {action} under {package_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
