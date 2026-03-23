from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WeightParams:
    a: np.ndarray
    b: np.ndarray

    @classmethod
    def from_vector(cls, vec: np.ndarray, k: int) -> "WeightParams":
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        if arr.size != 2 * int(k):
            raise ValueError(f"Expected vector size={2 * int(k)}, got {arr.size}")
        a = arr[:k].copy()
        b = arr[k:].copy()
        return cls(a=a, b=b)

    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.a, self.b], axis=0)


def softmax(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    z = arr - np.max(arr)
    ez = np.exp(z)
    denom = np.sum(ez)
    if not np.isfinite(denom) or denom <= 0.0:
        return np.full_like(arr, 1.0 / float(arr.size))
    return ez / denom


def weights_for_progress(params: WeightParams, progress: float) -> np.ndarray:
    u = float(np.clip(progress, 0.0, 1.0))
    logits = params.b + params.a * u
    return softmax(logits)


def progress_from_sigmas(sigmas: list[float]) -> list[float]:
    if len(sigmas) == 0:
        return []
    smax = float(max(sigmas))
    smin = float(min(sigmas))
    denom = smax - smin
    if denom <= 1e-8:
        n = max(1, len(sigmas) - 1)
        return [float(i) / float(n) for i in range(len(sigmas))]
    return [float(np.clip((smax - float(s)) / denom, 0.0, 1.0)) for s in sigmas]
