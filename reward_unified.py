from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class _BackendState:
    imagereward: Optional[object] = None
    hps_model: Optional[object] = None
    open_clip: Optional[object] = None


class UnifiedRewardScorer:
    """
    Unified reward wrapper.

    Backends:
      - imagereward
      - hpsv2
      - auto    : prefer ImageReward, then HPSv2
      - unified : weighted average of all available backends
    """

    _clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    _clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])

    def __init__(
        self,
        device: str = "cuda",
        backend: str = "auto",
        image_reward_model: str = "ImageReward-v1.0",
        unified_weights: Tuple[float, float] = (1.0, 1.0),
    ) -> None:
        if backend not in {"auto", "imagereward", "hpsv2", "unified"}:
            raise ValueError(f"Unsupported backend: {backend}")
        self.device = device
        self.backend = backend
        self.image_reward_model = image_reward_model
        self.unified_weights = unified_weights
        self.state = _BackendState()
        self.available: List[str] = []
        self._load_backends()

    def _load_backends(self) -> None:
        # Try ImageReward
        try:
            import ImageReward as RM

            model = RM.load(self.image_reward_model, device=self.device)
            model.eval()
            self.state.imagereward = model
            self.available.append("imagereward")
        except Exception as exc:
            print(f"[Reward] ImageReward unavailable: {exc}")

        # Try HPSv2
        try:
            import hpsv2 as hm
            import open_clip

            model = hm.utils.initialize_model().to(self.device).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self.state.hps_model = model
            self.state.open_clip = open_clip
            self.available.append("hpsv2")
        except Exception as exc:
            print(f"[Reward] HPSv2 unavailable: {exc}")

        if self.backend == "imagereward" and "imagereward" not in self.available:
            raise RuntimeError("Requested backend=imagereward, but ImageReward is unavailable.")
        if self.backend == "hpsv2" and "hpsv2" not in self.available:
            raise RuntimeError("Requested backend=hpsv2, but HPSv2 is unavailable.")
        if self.backend in {"auto", "unified"} and not self.available:
            raise RuntimeError("No reward backend available (ImageReward/HPSv2 both unavailable).")

    def describe(self) -> str:
        if self.backend == "auto":
            selected = self._auto_backend()
            return f"backend=auto(selected={selected}) available={self.available}"
        if self.backend == "unified":
            return (
                f"backend=unified available={self.available} "
                f"weights=(imagereward={self.unified_weights[0]}, hpsv2={self.unified_weights[1]})"
            )
        return f"backend={self.backend} available={self.available}"

    def _auto_backend(self) -> str:
        if "imagereward" in self.available:
            return "imagereward"
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
        assert self.state.hps_model is not None
        assert self.state.open_clip is not None
        toks = self.state.open_clip.tokenize([prompt]).to(self.device)
        tf = self.state.hps_model.encode_text(toks)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        vf = self.state.hps_model.encode_image(self._prep_hps_image(image))
        vf = vf / vf.norm(dim=-1, keepdim=True)
        return float((vf * tf).sum(dim=-1).item())

    def score(self, prompt: str, image: Image.Image) -> float:
        if self.backend == "imagereward":
            return self._score_imagereward(prompt, image)
        if self.backend == "hpsv2":
            return self._score_hpsv2(prompt, image)

        if self.backend == "auto":
            selected = self._auto_backend()
            if selected == "imagereward":
                return self._score_imagereward(prompt, image)
            return self._score_hpsv2(prompt, image)

        # unified
        scored: Dict[str, float] = {}
        if "imagereward" in self.available:
            scored["imagereward"] = self._score_imagereward(prompt, image)
        if "hpsv2" in self.available:
            scored["hpsv2"] = self._score_hpsv2(prompt, image)
        if not scored:
            raise RuntimeError("No backend available in unified mode.")

        w_ir, w_hps = self.unified_weights
        numer = 0.0
        denom = 0.0
        if "imagereward" in scored:
            numer += w_ir * scored["imagereward"]
            denom += w_ir
        if "hpsv2" in scored:
            numer += w_hps * scored["hpsv2"]
            denom += w_hps
        if denom <= 0:
            raise RuntimeError("Invalid unified reward weights; sum must be > 0.")
        return numer / denom
