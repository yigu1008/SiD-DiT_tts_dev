from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEFAULT_UNIFIEDREWARD_MODEL = "CodeGoat24/UnifiedReward-qwen-7b"
UNIFIEDREWARD_MODEL_ALIASES = {
    "CodeGoat24/UnifiedReward-qwen-4b": DEFAULT_UNIFIEDREWARD_MODEL,
    "CodeGoat24/UnifiedReward-2.0-qwen3vl-4b": DEFAULT_UNIFIEDREWARD_MODEL,
}
UNIFIEDREWARD_MODEL_FALLBACKS = (
    DEFAULT_UNIFIEDREWARD_MODEL,
    "CodeGoat24/UnifiedReward-qwen-3b",
)


@dataclass
class _BackendState:
    imagereward: Optional[object] = None
    hps_model: Optional[object] = None
    hps_module: Optional[object] = None  # fallback for newer hpsv2 (module-level score())
    open_clip: Optional[object] = None
    hpsv3_inferencer: Optional[object] = None
    pickscore_model: Optional[object] = None
    pickscore_processor: Optional[object] = None
    unifiedreward_model: Optional[object] = None
    unifiedreward_processor: Optional[object] = None
    unifiedreward_process_vision_info: Optional[object] = None
    unifiedreward_api_base: Optional[str] = None
    unifiedreward_api_key: Optional[str] = None
    unifiedreward_api_model: Optional[str] = None


class UnifiedRewardScorer:
    """
    Unified reward wrapper for TTS.

    Backends:
      - unifiedreward: paper model scoring (CodeGoat24 UnifiedReward)
      - unified      : alias of unifiedreward (kept for compatibility)
      - imagereward
      - pickscore
      - hpsv3
      - hpsv2
      - blend        : weighted average of ImageReward and HPSv2
      - auto         : prefer unifiedreward, then imagereward, then pickscore, then hpsv3, then hpsv2
    """

    _clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    _clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])

    _point_prompt_standard = (
        "You are given a text caption and a generated image based on that caption. "
        "Your task is to evaluate this image based on two key criteria:\n"
        "1. Alignment with the Caption: Assess how well this image aligns with the provided caption. "
        "Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n"
        "2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, "
        "color accuracy, and overall aesthetic appeal.\n"
        "Based on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\n"
        "Your task is provided as follows:\n"
        "Text Caption: [{prompt}]"
    )

    _point_prompt_strict = (
        "You are an image reward model.\n"
        "Evaluate one generated image against the given caption.\n"
        "Consider both prompt alignment and overall visual quality.\n"
        "Output exactly one line in this format and nothing else:\n"
        "Final Score: <float from 1.0 to 5.0>\n"
        "Caption: [{prompt}]"
    )

    _final_score_re = re.compile(r"Final Score:\s*([-+]?\d*\.?\d+)", re.IGNORECASE)
    _align_re = re.compile(r"Alignment Score[^:]*:\s*([-+]?\d*\.?\d+)", re.IGNORECASE)
    _coherence_re = re.compile(r"Coherence Score[^:]*:\s*([-+]?\d*\.?\d+)", re.IGNORECASE)
    _style_re = re.compile(r"Style Score[^:]*:\s*([-+]?\d*\.?\d+)", re.IGNORECASE)
    _unifiedreward_install_hint = (
        "Install/update dependencies for local UnifiedReward (pinned), e.g.:\n"
        "  pip install -U --no-deps \"transformers==4.52.4\" \"qwen-vl-utils==0.0.14\""
        "\nUse a valid checkpoint id, e.g.:\n"
        f"  --unifiedreward_model {DEFAULT_UNIFIEDREWARD_MODEL}"
        "\nIf private, authenticate first:\n"
        "  huggingface-cli login"
    )
    _imagereward_install_hint = (
        "Install ImageReward dependencies, e.g.:\n"
        "  export SID_FORCE_WANDB_STUB=1\n"
        "  export WANDB_DISABLED=true\n"
        "  pip install -U setuptools\n"
        "  pip install -U xxhash\n"
        "  pip install -U --force-reinstall wandb\n"
        "  pip install -U git+https://github.com/THUDM/ImageReward.git\n"
        "  pip install -U ftfy regex tqdm\n"
        "  pip install -U git+https://github.com/openai/CLIP.git"
    )
    _pickscore_install_hint = (
        "Install PickScore dependencies, e.g.:\n"
        "  pip install -U transformers accelerate sentencepiece\n"
        "  pip install -U \"timm>=1.0.15\""
    )
    _hpsv3_install_hint = (
        "Install HPSv3 dependencies, e.g.:\n"
        "  pip install -U hpsv3 open_clip_torch omegaconf hydra-core"
    )

    def __init__(
        self,
        device: str = "cuda",
        backend: str = "auto",
        image_reward_model: str = "ImageReward-v1.0",
        pickscore_model: str = "yuvalkirstain/PickScore_v1",
        unifiedreward_model: str = DEFAULT_UNIFIEDREWARD_MODEL,
        unified_weights: Tuple[float, float] = (1.0, 1.0),
        unifiedreward_api_base: Optional[str] = None,
        unifiedreward_api_key: str = "unifiedreward",
        unifiedreward_api_model: str = "UnifiedReward-7b-v1.5",
        max_new_tokens: int = 512,
        unifiedreward_prompt_mode: str = "standard",
    ) -> None:
        valid = {"auto", "imagereward", "pickscore", "hpsv3", "hpsv2", "blend", "unified", "unifiedreward"}
        if backend not in valid:
            raise ValueError(f"Unsupported backend: {backend}")
        if unifiedreward_prompt_mode not in {"standard", "strict"}:
            raise ValueError(f"Unsupported unifiedreward_prompt_mode: {unifiedreward_prompt_mode}")

        self.device = device
        self._hf_device = self._normalize_hf_device(device)
        self.backend = backend
        self.image_reward_model = image_reward_model
        self.pickscore_model = pickscore_model
        self.unifiedreward_model = unifiedreward_model
        self.unified_weights = unified_weights
        self.unifiedreward_api_base = unifiedreward_api_base
        self.unifiedreward_api_key = unifiedreward_api_key
        self.unifiedreward_api_model = unifiedreward_api_model
        self.max_new_tokens = int(max_new_tokens)
        self.unifiedreward_prompt_mode = unifiedreward_prompt_mode
        self.state = _BackendState()
        self.available: List[str] = []
        self.unifiedreward_last_error: Optional[str] = None
        self.imagereward_last_error: Optional[str] = None
        self.pickscore_last_error: Optional[str] = None
        self.hpsv3_last_error: Optional[str] = None
        self._hpsv3_fp32_forced = False
        self._hpsv3_bf16_forced = False
        self.debug = str(os.environ.get("UNIFIEDREWARD_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on"}
        self._load_backends()

    @staticmethod
    def _normalize_hf_device(device: str) -> str:
        if device.startswith("cuda") and ":" not in device:
            return "cuda:0"
        return device

    def _load_backends(self) -> None:
        target = "unifiedreward" if self.backend == "unified" else self.backend
        need_unifiedreward = target in {"unifiedreward", "auto"}
        need_imagereward = target in {"imagereward", "blend", "auto", "all"}
        need_pickscore = target in {"pickscore", "auto", "all"}
        need_hpsv3 = target in {"hpsv3", "blend", "auto", "all"}
        need_hpsv2 = target in {"hpsv2", "blend", "auto", "all"}

        if need_unifiedreward:
            self._try_load_unifiedreward()
        if need_imagereward:
            self._try_load_imagereward()
        if need_pickscore:
            self._try_load_pickscore()
        if need_hpsv3:
            self._try_load_hpsv3()
        if need_hpsv2:
            self._try_load_hpsv2()

        if target == "unifiedreward" and "unifiedreward" not in self.available:
            detail = f" Last error: {self.unifiedreward_last_error}" if self.unifiedreward_last_error else ""
            raise RuntimeError(
                "Requested backend=unifiedreward, but UnifiedReward is unavailable."
                f"{detail}\n{self._unifiedreward_install_hint}"
            )
        if target == "imagereward" and "imagereward" not in self.available:
            detail = f" Last error: {self.imagereward_last_error}" if self.imagereward_last_error else ""
            raise RuntimeError(
                "Requested backend=imagereward, but ImageReward is unavailable."
                f"{detail}\n{self._imagereward_install_hint}"
            )
        if target == "pickscore" and "pickscore" not in self.available:
            detail = f" Last error: {self.pickscore_last_error}" if self.pickscore_last_error else ""
            raise RuntimeError(
                "Requested backend=pickscore, but PickScore is unavailable."
                f"{detail}\n{self._pickscore_install_hint}"
            )
        if target == "hpsv3" and "hpsv3" not in self.available:
            detail = f" Last error: {self.hpsv3_last_error}" if self.hpsv3_last_error else ""
            raise RuntimeError(
                "Requested backend=hpsv3, but HPSv3 is unavailable."
                f"{detail}\n{self._hpsv3_install_hint}"
            )
        if target == "hpsv2" and "hpsv2" not in self.available:
            raise RuntimeError("Requested backend=hpsv2, but HPSv2 is unavailable.")
        if target == "blend":
            if "imagereward" not in self.available and "hpsv2" not in self.available and "hpsv3" not in self.available:
                raise RuntimeError("Requested backend=blend, but ImageReward/HPS backends are unavailable.")
        if target == "auto" and not self.available:
            raise RuntimeError("No reward backend available.")

    def _candidate_unifiedreward_model_ids(self) -> List[str]:
        requested = str(self.unifiedreward_model).strip() or DEFAULT_UNIFIEDREWARD_MODEL
        remapped = UNIFIEDREWARD_MODEL_ALIASES.get(requested, requested)
        if remapped != requested:
            print(f"[Reward] Remapping deprecated UnifiedReward id '{requested}' -> '{remapped}'.")
        candidates: List[str] = [remapped]
        for fallback in UNIFIEDREWARD_MODEL_FALLBACKS:
            if fallback not in candidates:
                candidates.append(fallback)
        return candidates

    def _try_load_unifiedreward(self) -> None:
        if self.unifiedreward_api_base:
            self.state.unifiedreward_api_base = self.unifiedreward_api_base.rstrip("/")
            self.state.unifiedreward_api_key = self.unifiedreward_api_key
            self.state.unifiedreward_api_model = self.unifiedreward_api_model
            self.available.append("unifiedreward")
            return

        try:
            import transformers
            from transformers import AutoConfig, AutoProcessor

            try:
                from qwen_vl_utils import process_vision_info  # type: ignore
            except Exception:
                process_vision_info = None

            model = None
            requested_model = self.unifiedreward_model
            selected_model = None
            load_errors: List[str] = []
            for model_id in self._candidate_unifiedreward_model_ids():
                model_type = ""
                archs: List[str] = []
                try:
                    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                    model_type = str(getattr(cfg, "model_type", "") or "").strip().lower()
                    raw_archs = getattr(cfg, "architectures", None)
                    if isinstance(raw_archs, (list, tuple)):
                        archs = [str(x) for x in raw_archs]
                except Exception as exc:
                    load_errors.append(f"{model_id} :: AutoConfig: {type(exc).__name__}: {str(exc)[:240]}")

                preferred_classes: tuple[str, ...]
                if model_type == "qwen3_vl" or any("Qwen3VLForConditionalGeneration" in x for x in archs):
                    preferred_classes = (
                        "Qwen3VLForConditionalGeneration",
                        "AutoModelForImageTextToText",
                        "AutoModelForVision2Seq",
                        "AutoModelForCausalLM",
                        "AutoModel",
                    )
                elif model_type == "qwen2_5_vl":
                    preferred_classes = (
                        "Qwen2_5_VLForConditionalGeneration",
                        "AutoModelForImageTextToText",
                        "AutoModelForVision2Seq",
                        "AutoModelForCausalLM",
                        "AutoModel",
                    )
                elif model_type == "qwen2_vl":
                    preferred_classes = (
                        "Qwen2VLForConditionalGeneration",
                        "AutoModelForImageTextToText",
                        "AutoModelForVision2Seq",
                        "AutoModelForCausalLM",
                        "AutoModel",
                    )
                elif model_type == "qwen3_5":
                    preferred_classes = (
                        "Qwen3_5ForConditionalGeneration",
                        "AutoModelForImageTextToText",
                        "AutoModelForVision2Seq",
                        "AutoModelForCausalLM",
                        "AutoModel",
                    )
                else:
                    preferred_classes = (
                        "Qwen3VLForConditionalGeneration",
                        "Qwen3_5ForConditionalGeneration",
                        "Qwen2_5_VLForConditionalGeneration",
                        "Qwen2VLForConditionalGeneration",
                        "AutoModelForImageTextToText",
                        "AutoModelForVision2Seq",
                        "AutoModelForCausalLM",
                        "AutoModel",
                    )

                if model_type == "qwen3_vl" and not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
                    tv = str(getattr(transformers, "__version__", "unknown"))
                    load_errors.append(
                        f"{model_id} :: model_type=qwen3_vl requires Qwen3VLForConditionalGeneration "
                        f"(transformers={tv}, need >=4.57.1)"
                    )
                    # Do not force-load qwen3_vl checkpoints via Qwen2.5/Auto fallbacks.
                    continue
                if model_type == "qwen3_5" and not hasattr(transformers, "Qwen3_5ForConditionalGeneration"):
                    tv = str(getattr(transformers, "__version__", "unknown"))
                    load_errors.append(
                        f"{model_id} :: model_type=qwen3_5 requires Qwen3_5ForConditionalGeneration "
                        f"(transformers={tv})"
                    )
                    # Do not force-load qwen3.5 checkpoints via Qwen2.5/Auto fallbacks.
                    continue

                for cls_name in preferred_classes:
                    cls = getattr(transformers, cls_name, None)
                    if cls is None:
                        load_errors.append(f"{model_id} :: {cls_name}: class not found in transformers")
                        continue
                    for dm in ({"": self._hf_device}, "auto", None):
                        kwargs: Dict[str, Any] = {
                            "torch_dtype": "auto",
                            "trust_remote_code": True,
                        }
                        if dm is not None:
                            kwargs["device_map"] = dm
                        try:
                            candidate = cls.from_pretrained(model_id, **kwargs)
                            if dm is None and hasattr(candidate, "to"):
                                try:
                                    candidate = candidate.to(self._hf_device)
                                except Exception:
                                    pass
                            if not hasattr(candidate, "generate"):
                                load_errors.append(
                                    f"{model_id} :: {cls_name}(device_map={dm}): loaded model has no generate()"
                                )
                                continue
                            model = candidate
                            selected_model = model_id
                            break
                        except Exception as exc:
                            err = f"{model_id} :: {cls_name}(device_map={dm}): {type(exc).__name__}: {str(exc)[:240]}"
                            load_errors.append(err)
                    if model is not None:
                        break
                if model is not None:
                    break
            if model is None:
                preview = " | ".join(load_errors[:4]) if load_errors else "no loader attempts captured"
                raise RuntimeError(f"No compatible transformers loader worked for UnifiedReward. {preview}")
            if selected_model is None:
                selected_model = requested_model

            processor = AutoProcessor.from_pretrained(selected_model, trust_remote_code=True)
            model.eval()
            self.unifiedreward_model = selected_model
            self.state.unifiedreward_model = model
            self.state.unifiedreward_processor = processor
            self.state.unifiedreward_process_vision_info = process_vision_info
            self.available.append("unifiedreward")
            self.unifiedreward_last_error = None
            if selected_model != requested_model:
                print(
                    f"[Reward] UnifiedReward fallback loaded '{selected_model}' "
                    f"(requested '{requested_model}')."
                )
        except Exception as exc:
            self.unifiedreward_last_error = str(exc)
            print(f"[Reward] UnifiedReward unavailable: {exc}")
            print(f"[Reward] {self._unifiedreward_install_hint}")

    def _try_load_imagereward(self) -> None:
        def _force_wandb_stub_enabled() -> bool:
            raw = str(os.environ.get("SID_FORCE_WANDB_STUB", "1")).strip().lower()
            return raw not in {"0", "false", "no", "off"}

        def _inject_wandb_stub(reason: str) -> None:
            import importlib.machinery
            import types

            existing = sys.modules.get("wandb")
            if existing is not None and bool(getattr(existing, "__codex_stub__", False)):
                # Keep old stub object but ensure importlib can inspect it safely.
                if getattr(existing, "__spec__", None) is None:
                    existing.__spec__ = importlib.machinery.ModuleSpec(
                        "wandb", loader=None, is_package=True
                    )
                if getattr(existing, "__package__", None) is None:
                    existing.__package__ = "wandb"
                if not hasattr(existing, "__path__"):
                    existing.__path__ = []  # type: ignore[attr-defined]
                return

            def _noop(*args, **kwargs):
                return None

            class _DummyRun:
                def log(self, *args, **kwargs):
                    return None

                def finish(self, *args, **kwargs):
                    return None

                def __getattr__(self, _name):
                    return _noop

            stub = types.ModuleType("wandb")
            stub.__codex_stub__ = True
            stub.init = lambda *args, **kwargs: _DummyRun()
            stub.log = _noop
            stub.finish = _noop
            stub.login = _noop
            stub.watch = _noop
            stub.define_metric = _noop
            stub.config = {}
            stub.run = None
            stub.__package__ = "wandb"
            stub.__path__ = []  # type: ignore[attr-defined]
            stub.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None, is_package=True)

            # Minimal submodule for libs that probe/import wandb.sdk.
            sdk_stub = types.ModuleType("wandb.sdk")
            sdk_stub.__package__ = "wandb"
            sdk_stub.__path__ = []  # type: ignore[attr-defined]
            sdk_stub.__spec__ = importlib.machinery.ModuleSpec(
                "wandb.sdk", loader=None, is_package=True
            )
            stub.sdk = sdk_stub
            sys.modules["wandb"] = stub
            sys.modules.setdefault("wandb.sdk", sdk_stub)
            print(f"[Reward] Using wandb stub for ImageReward inference ({reason}).")

        def _ensure_wandb_importable() -> None:
            if _force_wandb_stub_enabled():
                _inject_wandb_stub("SID_FORCE_WANDB_STUB=1")
                return
            try:
                import wandb  # noqa: F401
                # Some broken envs leave wandb importable but with __spec__ missing.
                if getattr(wandb, "__spec__", None) is None:
                    _inject_wandb_stub("wandb.__spec__ is None")
            except Exception as exc:
                _inject_wandb_stub(str(exc))

        def _load() -> tuple[object, Optional[str]]:
            # Compatibility shims for ImageReward vs newer transformers:
            try:
                import transformers

                # 1. BertModel.all_tied_weights_keys removed in newer transformers.
                bert_cls = getattr(transformers, "BertModel", None)
                if bert_cls is not None and not hasattr(bert_cls, "all_tied_weights_keys"):

                    def _all_tied_keys(self):
                        return getattr(self, "_tied_weights_keys", None)

                    bert_cls.all_tied_weights_keys = property(_all_tied_keys)  # type: ignore[attr-defined]

                # 2. apply_chunking_to_forward removed from transformers.modeling_utils in newer versions.
                import transformers.modeling_utils as _tmu
                if not hasattr(_tmu, "apply_chunking_to_forward"):
                    def _apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
                        if chunk_size > 0:
                            tensor_shape = input_tensors[0].shape[chunk_dim]
                            if tensor_shape % chunk_size != 0:
                                raise ValueError(
                                    f"tensor shape {tensor_shape} not divisible by chunk_size {chunk_size}"
                                )
                            num_chunks = tensor_shape // chunk_size
                            return torch.cat(
                                [forward_fn(*[t.narrow(chunk_dim, c * chunk_size, chunk_size) for t in input_tensors])
                                 for c in range(num_chunks)],
                                dim=chunk_dim,
                            )
                        return forward_fn(*input_tensors)
                    _tmu.apply_chunking_to_forward = _apply_chunking_to_forward

                # 3. find_pruneable_heads_and_indices removed from transformers.modeling_utils in newer versions.
                if not hasattr(_tmu, "find_pruneable_heads_and_indices"):
                    def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
                        mask = torch.ones(n_heads, head_size)
                        heads = set(heads) - already_pruned_heads
                        for head in sorted(heads):
                            head -= sum(1 for h in sorted(already_pruned_heads) if h < head)
                            mask[head] = 0
                        mask = mask.view(-1).eq(1)
                        index = torch.arange(len(mask))[mask].long()
                        return heads, index
                    _tmu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

                # 4. prune_linear_layer removed from transformers.modeling_utils in newer versions.
                if not hasattr(_tmu, "prune_linear_layer"):
                    import torch.nn as _nn
                    def _prune_linear_layer(layer, index, dim=0):
                        index = index.to(layer.weight.device)
                        W = layer.weight.index_select(dim, index).clone().detach()
                        if layer.bias is not None:
                            b = layer.bias[index].clone().detach() if dim == 0 else layer.bias.clone().detach()
                        new_size = list(layer.weight.size())
                        new_size[dim] = len(index)
                        new_layer = _nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
                        new_layer.weight.requires_grad = False
                        new_layer.weight.copy_(W.contiguous())
                        new_layer.weight.requires_grad = True
                        if layer.bias is not None:
                            new_layer.bias.requires_grad = False
                            new_layer.bias.copy_(b.contiguous())
                            new_layer.bias.requires_grad = True
                        return new_layer
                    _tmu.prune_linear_layer = _prune_linear_layer
            except Exception:
                pass

            _ensure_wandb_importable()
            import ImageReward as RM

            tried_errors: List[str] = []
            candidates = [self.image_reward_model]
            if self.image_reward_model != "ImageReward-v1.0":
                candidates.append("ImageReward-v1.0")

            download_root = os.environ.get("IMAGEREWARD_CACHE") or None

            model = None
            selected = None
            for candidate in candidates:
                try:
                    model = RM.load(candidate, device=self.device, download_root=download_root)
                    selected = candidate
                    break
                except Exception as exc:
                    tried_errors.append(f"{candidate}: {type(exc).__name__}: {exc}")

            if model is None:
                joined = " | ".join(tried_errors) if tried_errors else "unknown error"
                raise RuntimeError(f"Unable to load ImageReward model. {joined}")
            return model, selected

        try:
            try:
                model, selected = _load()
            except Exception as exc:
                msg = str(exc)
                if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", "") == "xxhash":
                    print("[Reward] ImageReward missing dependency 'xxhash'. Installing and retrying once ...")
                    try:
                        subprocess.run(
                            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "xxhash>=3.4.1"],
                            check=True,
                        )
                    except Exception as install_exc:
                        raise RuntimeError(f"Failed to install xxhash automatically: {install_exc}") from exc
                    model, selected = _load()
                elif "wandb_telemetry_pb2" in msg or (
                    "cannot import name 'Imports'" in msg and "wandb" in msg
                ):
                    _inject_wandb_stub(msg)
                    model, selected = _load()
                else:
                    raise

            model.eval()
            self.state.imagereward = model
            self.available.append("imagereward")
            self.imagereward_last_error = None
            if selected is not None and selected != self.image_reward_model:
                print(
                    f"[Reward] ImageReward fallback loaded '{selected}' "
                    f"(requested '{self.image_reward_model}')."
                )
        except Exception as exc:
            self.imagereward_last_error = str(exc)
            print(f"[Reward] ImageReward unavailable: {exc}")

    def _try_load_hpsv2(self) -> None:
        try:
            import hpsv2 as hm

            # Strategy 1: old hpsv2 API — hm.utils.initialize_model() returns a CLIP model.
            if hasattr(getattr(hm, "utils", None), "initialize_model"):
                import open_clip
                model = hm.utils.initialize_model().to(self.device).eval()
                for p in model.parameters():
                    p.requires_grad_(False)
                self.state.hps_model = model
                self.state.open_clip = open_clip
            # Strategy 2: newer hpsv2 — module exposes hpsv2.score([img], prompt, hps_version=...).
            elif callable(getattr(hm, "score", None)):
                # Eagerly trigger the internal imports so a broken install
                # (e.g. missing BPE vocab file) is caught here, not at score time.
                import hpsv2.img_score as _hps_check  # noqa: F401
                self.state.hps_module = hm
            else:
                raise RuntimeError(
                    "hpsv2 has neither utils.initialize_model nor a callable score(); "
                    "unsupported hpsv2 version."
                )
            self.available.append("hpsv2")
        except Exception as exc:
            print(f"[Reward] HPSv2 unavailable: {exc}")

    def _try_load_hpsv3(self) -> None:
        try:
            import hpsv3

            inferencer_cls = getattr(hpsv3, "HPSv3RewardInferencer", None)
            if inferencer_cls is None:
                raise RuntimeError("hpsv3 module does not expose HPSv3RewardInferencer")

            inferencer = None
            errors: List[str] = []
            for kwargs in (
                {"device": self._hf_device},
                {"device": self.device},
                {},
            ):
                try:
                    inferencer = inferencer_cls(**kwargs)
                    break
                except Exception as exc:
                    errors.append(f"{kwargs}: {type(exc).__name__}: {str(exc)[:200]}")
            if inferencer is None:
                msg = " | ".join(errors[:3]) if errors else "constructor failed"
                raise RuntimeError(f"Unable to initialize HPSv3RewardInferencer. {msg}")

            self.state.hpsv3_inferencer = inferencer
            self.available.append("hpsv3")
            self.hpsv3_last_error = None
        except Exception as exc:
            self.hpsv3_last_error = str(exc)
            print(f"[Reward] HPSv3 unavailable: {exc}")

    def _try_load_pickscore(self) -> None:
        try:
            from transformers import AutoModel, AutoProcessor

            processor = AutoProcessor.from_pretrained(self.pickscore_model, trust_remote_code=True)
            model = AutoModel.from_pretrained(self.pickscore_model, trust_remote_code=True)
            model = model.to(self.device).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self.state.pickscore_model = model
            self.state.pickscore_processor = processor
            self.available.append("pickscore")
            self.pickscore_last_error = None
        except Exception as exc:
            self.pickscore_last_error = str(exc)
            print(f"[Reward] PickScore unavailable: {exc}")

    def describe(self) -> str:
        target = "unifiedreward" if self.backend == "unified" else self.backend
        if target == "auto":
            return f"backend=auto(selected={self._auto_backend()}) available={self.available}"
        if target == "blend":
            return (
                f"backend=blend available={self.available} "
                f"weights=(imagereward={self.unified_weights[0]}, hps={self.unified_weights[1]})"
            )
        if target == "pickscore":
            return f"backend=pickscore model={self.pickscore_model} available={self.available}"
        if target == "unifiedreward":
            mode = "api" if self.state.unifiedreward_api_base else "local"
            detail = self.state.unifiedreward_api_model if mode == "api" else self.unifiedreward_model
            return f"backend=unifiedreward({mode}) model={detail} available={self.available}"
        return f"backend={target} available={self.available}"

    def _auto_backend(self) -> str:
        if "unifiedreward" in self.available:
            return "unifiedreward"
        if "imagereward" in self.available:
            return "imagereward"
        if "pickscore" in self.available:
            return "pickscore"
        if "hpsv3" in self.available:
            return "hpsv3"
        if "hpsv2" in self.available:
            return "hpsv2"
        raise RuntimeError("No available backend for auto mode.")

    def _prep_hps_image(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = self._clip_mean.to(self.device).view(1, 3, 1, 1)
        std = self._clip_std.to(self.device).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def _score_imagereward(self, prompt: str, image: Image.Image) -> float:
        assert self.state.imagereward is not None
        return float(self.state.imagereward.score(prompt, image))

    @torch.no_grad()
    def _score_hpsv2(self, prompt: str, image: Image.Image) -> float:
        # Fast path: model loaded directly via old hpsv2 API.
        if self.state.hps_model is not None:
            assert self.state.open_clip is not None
            toks = self.state.open_clip.tokenize([prompt]).to(self.device)
            tf = self.state.hps_model.encode_text(toks)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            vf = self.state.hps_model.encode_image(self._prep_hps_image(image))
            vf = vf / vf.norm(dim=-1, keepdim=True)
            return float((vf * tf).sum(dim=-1).item())
        # Fallback: newer hpsv2 module-level score() API.
        assert self.state.hps_module is not None
        result = self.state.hps_module.score([image], prompt, hps_version="v2.1")
        return float(result[0])

    @staticmethod
    def _extract_scalar_from_hpsv3_output(obj: Any) -> float:
        if isinstance(obj, dict):
            for key in ("score", "scores", "reward", "rewards", "preference", "preferences", "logits"):
                if key in obj:
                    return UnifiedRewardScorer._extract_scalar_from_hpsv3_output(obj[key])
            if len(obj) > 0:
                return UnifiedRewardScorer._extract_scalar_from_hpsv3_output(next(iter(obj.values())))
            raise RuntimeError("Empty dict returned by HPSv3.")
        if isinstance(obj, torch.Tensor):
            arr = obj.detach().float().cpu().reshape(-1)
            if arr.numel() == 0:
                raise RuntimeError("Empty tensor returned by HPSv3.")
            # HPSv3 commonly returns [N,2], use the first scalar.
            return float(arr[0].item())
        if isinstance(obj, np.ndarray):
            arr = obj.reshape(-1)
            if arr.size == 0:
                raise RuntimeError("Empty ndarray returned by HPSv3.")
            return float(arr[0])
        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                raise RuntimeError("Empty list/tuple returned by HPSv3.")
            return UnifiedRewardScorer._extract_scalar_from_hpsv3_output(obj[0])
        if isinstance(obj, (float, int)):
            return float(obj)
        raise RuntimeError(f"Unsupported HPSv3 output type: {type(obj).__name__}")

    @torch.no_grad()
    def _score_hpsv3(self, prompt: str, image: Image.Image) -> float:
        inferencer = self.state.hpsv3_inferencer
        if inferencer is None:
            raise RuntimeError("HPSv3 inferencer is not loaded.")

        def _run_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
            # Avoid inheriting outer autocast state from diffusion path.
            if str(self.device).startswith("cuda"):
                with torch.autocast(device_type="cuda", enabled=False):
                    return fn(*args, **kwargs)
            if str(self.device).startswith("cpu"):
                try:
                    with torch.autocast(device_type="cpu", enabled=False):
                        return fn(*args, **kwargs)
                except Exception:
                    return fn(*args, **kwargs)
            return fn(*args, **kwargs)

        def _force_fp32_modules() -> int:
            touched = 0
            candidates = []
            if isinstance(inferencer, torch.nn.Module):
                candidates.append(inferencer)
            for name in ("model", "clip_model", "hps_model", "image_encoder", "text_encoder", "backbone", "net", "module"):
                mod = getattr(inferencer, name, None)
                if isinstance(mod, torch.nn.Module):
                    candidates.append(mod)
            try:
                for mod in vars(inferencer).values():
                    if isinstance(mod, torch.nn.Module):
                        candidates.append(mod)
            except Exception:
                pass
            seen = set()
            for mod in candidates:
                mid = id(mod)
                if mid in seen:
                    continue
                seen.add(mid)
                try:
                    mod.to(dtype=torch.float32)
                    touched += 1
                except Exception:
                    pass
            return touched

        def _force_bf16_modules() -> int:
            if not str(self.device).startswith("cuda"):
                return 0
            touched = 0
            candidates = []
            if isinstance(inferencer, torch.nn.Module):
                candidates.append(inferencer)
            for name in ("model", "clip_model", "hps_model", "image_encoder", "text_encoder", "backbone", "net", "module"):
                mod = getattr(inferencer, name, None)
                if isinstance(mod, torch.nn.Module):
                    candidates.append(mod)
            try:
                for mod in vars(inferencer).values():
                    if isinstance(mod, torch.nn.Module):
                        candidates.append(mod)
            except Exception:
                pass
            seen = set()
            for mod in candidates:
                mid = id(mod)
                if mid in seen:
                    continue
                seen.add(mid)
                try:
                    mod.to(dtype=torch.bfloat16)
                    touched += 1
                except Exception:
                    pass
            return touched

        def _is_dtype_mismatch(msg: str) -> bool:
            low = msg.lower()
            return (
                "same dtype" in low
                or "expected scalar type" in low
                or ("bfloat16" in low and "float" in low)
                or ("half" in low and "float" in low)
            )

        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                tmp_path = f.name
            image.convert("RGB").save(tmp_path)

            call_errors: List[str] = []
            for fn_name in ("reward", "score", "__call__"):
                fn = getattr(inferencer, fn_name, None)
                if fn is None:
                    continue
                for kwargs in (
                    {"prompts": [prompt], "image_paths": [tmp_path]},
                    {"image_paths": [tmp_path], "prompts": [prompt]},
                ):
                    try:
                        out = _run_call(fn, **kwargs)
                        return self._extract_scalar_from_hpsv3_output(out)
                    except Exception as exc:
                        msg = str(exc)
                        if _is_dtype_mismatch(msg):
                            if not self._hpsv3_fp32_forced:
                                touched = _force_fp32_modules()
                                if touched > 0:
                                    self._hpsv3_fp32_forced = True
                                    try:
                                        out = _run_call(fn, **kwargs)
                                        return self._extract_scalar_from_hpsv3_output(out)
                                    except Exception as retry_exc:
                                        call_errors.append(
                                            f"{fn_name}({list(kwargs.keys())})[fp32-retry]: {type(retry_exc).__name__}: {str(retry_exc)[:160]}"
                                        )
                            if not self._hpsv3_bf16_forced:
                                touched = _force_bf16_modules()
                                if touched > 0:
                                    self._hpsv3_bf16_forced = True
                                    try:
                                        out = _run_call(fn, **kwargs)
                                        return self._extract_scalar_from_hpsv3_output(out)
                                    except Exception as retry_exc:
                                        call_errors.append(
                                            f"{fn_name}({list(kwargs.keys())})[bf16-retry]: {type(retry_exc).__name__}: {str(retry_exc)[:160]}"
                                        )
                        call_errors.append(f"{fn_name}({list(kwargs.keys())}): {type(exc).__name__}: {msg[:160]}")
                for args in (
                    ([prompt], [tmp_path]),
                    ([tmp_path], [prompt]),
                ):
                    try:
                        out = _run_call(fn, *args)
                        return self._extract_scalar_from_hpsv3_output(out)
                    except Exception as exc:
                        msg = str(exc)
                        if _is_dtype_mismatch(msg):
                            if not self._hpsv3_fp32_forced:
                                touched = _force_fp32_modules()
                            else:
                                touched = 0
                            if touched > 0:
                                self._hpsv3_fp32_forced = True
                                try:
                                    out = _run_call(fn, *args)
                                    return self._extract_scalar_from_hpsv3_output(out)
                                except Exception as retry_exc:
                                    call_errors.append(
                                        f"{fn_name}(positional)[fp32-retry]: {type(retry_exc).__name__}: {str(retry_exc)[:160]}"
                                    )
                            if not self._hpsv3_bf16_forced:
                                touched = _force_bf16_modules()
                                if touched > 0:
                                    self._hpsv3_bf16_forced = True
                                    try:
                                        out = _run_call(fn, *args)
                                        return self._extract_scalar_from_hpsv3_output(out)
                                    except Exception as retry_exc:
                                        call_errors.append(
                                            f"{fn_name}(positional)[bf16-retry]: {type(retry_exc).__name__}: {str(retry_exc)[:160]}"
                                        )
                        call_errors.append(f"{fn_name}(positional): {type(exc).__name__}: {msg[:160]}")
            msg = " | ".join(call_errors[:4]) if call_errors else "no callable API found"
            raise RuntimeError(f"Unable to score with HPSv3 inferencer. {msg}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    @torch.no_grad()
    def _score_pickscore(self, prompt: str, image: Image.Image) -> float:
        model = self.state.pickscore_model
        processor = self.state.pickscore_processor
        if model is None or processor is None:
            raise RuntimeError("PickScore model is not loaded.")

        inputs = processor(
            text=[prompt],
            images=[image.convert("RGB")],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        prepared = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = value

        if hasattr(model, "get_image_features") and hasattr(model, "get_text_features"):
            image_embs = model.get_image_features(pixel_values=prepared["pixel_values"])
            text_embs = model.get_text_features(
                input_ids=prepared["input_ids"],
                attention_mask=prepared.get("attention_mask"),
            )
        else:
            raise RuntimeError("PickScore model missing CLIP-style get_image_features/get_text_features.")
        image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        logits = image_embs @ text_embs.T
        if hasattr(model, "logit_scale"):
            try:
                logits = logits * model.logit_scale.exp()
            except Exception:
                pass
        return float(logits.squeeze().item())

    def _unifiedreward_question(self, prompt: str) -> str:
        if self.unifiedreward_prompt_mode == "strict":
            return self._point_prompt_strict.format(prompt=prompt)
        return self._point_prompt_standard.format(prompt=prompt)

    def _image_to_data_url(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=95)
        payload = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{payload}"

    def _extract_unifiedreward_score(self, text: str) -> float:
        text = text.replace("\r", "\n").strip()
        # Some UnifiedReward checkpoints return a bare float only.
        bare = re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if bare:
            return float(bare.group(0))
        match = self._final_score_re.search(text)
        if not match:
            match = re.search(r"Final\s*score\s*[:=]\s*([-+]?\d*\.?\d+)", text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        a = self._align_re.search(text)
        c = self._coherence_re.search(text)
        s = self._style_re.search(text)
        if a and c and s:
            return (float(a.group(1)) + float(c.group(1)) + float(s.group(1))) / 3.0
        scored_fields = list(
            re.finditer(r"(?:^|\n)\s*[^:\n]*score[^:\n]*[:=]\s*([-+]?\d*\.?\d+)", text, re.IGNORECASE)
        )
        if scored_fields:
            # Prefer the last explicit score field if "Final Score" is absent.
            return float(scored_fields[-1].group(1))
        # Last fallback: return the last numeric line if the model output is terse.
        numeric_lines = re.findall(r"(?:^|\n)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?=\n|$)", text)
        if numeric_lines:
            return float(numeric_lines[-1])
        raise RuntimeError(f"Unable to parse UnifiedReward score from output: {text[:200]!r}")

    @torch.no_grad()
    def _score_unifiedreward_local(self, prompt: str, image: Image.Image) -> float:
        model = self.state.unifiedreward_model
        processor = self.state.unifiedreward_processor
        if model is None or processor is None:
            raise RuntimeError("UnifiedReward local model is not loaded.")

        question = self._unifiedreward_question(prompt)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        process_vision_info = self.state.unifiedreward_process_vision_info
        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(
                text=[text],
                images=[image.convert("RGB")],
                padding=True,
                return_tensors="pt",
            )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        score = self._extract_unifiedreward_score(output_text)
        if self.debug:
            print(f"[UnifiedReward][local] raw={output_text[:300]!r} score={score:.4f}")
        return score

    def _score_unifiedreward_api(self, prompt: str, image: Image.Image) -> float:
        if not self.state.unifiedreward_api_base:
            raise RuntimeError("UnifiedReward API base URL is missing.")
        url = f"{self.state.unifiedreward_api_base}/chat/completions"
        payload = {
            "model": self.state.unifiedreward_api_model or "UnifiedReward-7b-v1.5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self._image_to_data_url(image)}},
                        {"type": "text", "text": self._unifiedreward_question(prompt)},
                    ],
                }
            ],
            "temperature": 0,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.state.unifiedreward_api_key or 'unifiedreward'}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        content = raw["choices"][0]["message"]["content"]
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    maybe_text = item.get("text")
                    if isinstance(maybe_text, str):
                        text_parts.append(maybe_text)
            content = "\n".join(text_parts)
        if not isinstance(content, str):
            content = str(content)
        score = self._extract_unifiedreward_score(content)
        if self.debug:
            print(f"[UnifiedReward][api] raw={content[:300]!r} score={score:.4f}")
        return score

    def _score_unifiedreward(self, prompt: str, image: Image.Image) -> float:
        if self.state.unifiedreward_api_base:
            return self._score_unifiedreward_api(prompt, image)
        return self._score_unifiedreward_local(prompt, image)

    def _score_all(self, prompt: str, image: Image.Image) -> float:
        """Equal-weight mean of every available backend: imagereward, pickscore, hpsv2, hpsv3."""
        scores: List[float] = []
        if "imagereward" in self.available:
            scores.append(self._score_imagereward(prompt, image))
        if "pickscore" in self.available:
            scores.append(self._score_pickscore(prompt, image))
        if "hpsv2" in self.available:
            scores.append(self._score_hpsv2(prompt, image))
        if "hpsv3" in self.available:
            scores.append(self._score_hpsv3(prompt, image))
        if not scores:
            raise RuntimeError("No backend available in 'all' mode.")
        return float(sum(scores) / len(scores))

    def _score_blend(self, prompt: str, image: Image.Image) -> float:
        scored: Dict[str, float] = {}
        if "imagereward" in self.available:
            scored["imagereward"] = self._score_imagereward(prompt, image)
        if "hpsv3" in self.available:
            scored["hps"] = self._score_hpsv3(prompt, image)
        elif "hpsv2" in self.available:
            scored["hps"] = self._score_hpsv2(prompt, image)
        if not scored:
            raise RuntimeError("No backend available in blend mode.")

        w_ir, w_hps = self.unified_weights
        numer = 0.0
        denom = 0.0
        if "imagereward" in scored:
            numer += w_ir * scored["imagereward"]
            denom += w_ir
        if "hps" in scored:
            numer += w_hps * scored["hps"]
            denom += w_hps
        if denom <= 0:
            raise RuntimeError("Invalid blend weights; sum must be > 0.")
        return numer / denom

    def score(self, prompt: str, image: Image.Image) -> float:
        target = "unifiedreward" if self.backend == "unified" else self.backend
        if target == "imagereward":
            return self._score_imagereward(prompt, image)
        if target == "pickscore":
            return self._score_pickscore(prompt, image)
        if target == "hpsv3":
            return self._score_hpsv3(prompt, image)
        if target == "hpsv2":
            return self._score_hpsv2(prompt, image)
        if target == "blend":
            return self._score_blend(prompt, image)
        if target == "all":
            return self._score_all(prompt, image)
        if target == "unifiedreward":
            return self._score_unifiedreward(prompt, image)
        if target == "auto":
            selected = self._auto_backend()
            if selected == "unifiedreward":
                return self._score_unifiedreward(prompt, image)
            if selected == "imagereward":
                return self._score_imagereward(prompt, image)
            if selected == "pickscore":
                return self._score_pickscore(prompt, image)
            if selected == "hpsv3":
                return self._score_hpsv3(prompt, image)
            return self._score_hpsv2(prompt, image)
        raise RuntimeError(f"Unexpected backend: {self.backend}")
