from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class CEMIteration:
    iteration: int
    best_score: float
    mean_score: float
    elite_mean_score: float
    eval_calls_total: int
    std_mean: float


@dataclass
class CEMResult:
    best_x: np.ndarray
    best_score: float
    best_aux: Any
    eval_calls: int
    history: list[CEMIteration]


def optimize_cem(
    objective: Callable[[np.ndarray], tuple[float, Any]],
    dim: int,
    seed: int,
    n_iters: int,
    pop_size: int,
    elite_frac: float = 0.25,
    init_std: float = 1.0,
    min_std: float = 0.05,
    clip_value: float = 3.0,
) -> CEMResult:
    d = int(dim)
    if d <= 0:
        raise ValueError(f"dim must be > 0, got {dim}")
    n_iters = max(1, int(n_iters))
    pop_size = max(2, int(pop_size))
    elite_frac = float(np.clip(float(elite_frac), 0.05, 0.95))
    elite_k = max(1, int(round(pop_size * elite_frac)))

    rng = np.random.default_rng(int(seed))
    mean = np.zeros((d,), dtype=np.float64)
    std = np.full((d,), float(max(init_std, min_std)), dtype=np.float64)

    best_x = mean.copy()
    best_score = -float("inf")
    best_aux = None
    eval_calls = 0
    history: list[CEMIteration] = []

    for it in range(n_iters):
        samples = mean[None, :] + std[None, :] * rng.standard_normal((pop_size, d))
        if clip_value > 0:
            samples = np.clip(samples, -float(clip_value), float(clip_value))

        scored: list[tuple[float, np.ndarray, Any]] = []
        for x in samples:
            score, aux = objective(x)
            eval_calls += 1
            s = float(score)
            scored.append((s, x.copy(), aux))
            if s > best_score:
                best_score = s
                best_x = x.copy()
                best_aux = aux

        scored.sort(key=lambda row: row[0], reverse=True)
        elite = np.stack([row[1] for row in scored[:elite_k]], axis=0)
        mean = elite.mean(axis=0)
        std = elite.std(axis=0)
        std = np.maximum(std, float(min_std))

        scores = np.asarray([row[0] for row in scored], dtype=np.float64)
        elite_scores = np.asarray([row[0] for row in scored[:elite_k]], dtype=np.float64)
        history.append(
            CEMIteration(
                iteration=int(it),
                best_score=float(scores.max()),
                mean_score=float(scores.mean()),
                elite_mean_score=float(elite_scores.mean()),
                eval_calls_total=int(eval_calls),
                std_mean=float(std.mean()),
            )
        )

    return CEMResult(
        best_x=best_x,
        best_score=float(best_score),
        best_aux=best_aux,
        eval_calls=int(eval_calls),
        history=history,
    )
