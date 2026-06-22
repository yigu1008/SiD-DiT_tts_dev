#!/usr/bin/env python3
"""Reward-space conditional MI diagnostic for SiD/SD3.5 controls.

This script has two stages:
1) Generate reward dataset over (prompt, seed, variant, cfg) controls.
2) Train conditional MINE critics and report MI diagnostics.

Target channels (as requested):
  - seed only
  - cfg only
  - prompt variant only
  - variant + cfg
  - seed + variant + cfg
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import sampling_unified_sd35 as su


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SD3.5 reward-space MI diagnostic (generate + MINE critics).")
    p.add_argument(
        "--mode",
        choices=[
            "generate",
            "train",
            "full",
            "per_step_generate",
            "per_step_train",
            "per_step",
            "per_step_plot",
            "per_step_calibrate",
            "decompose",
            "per_step_decompose",
        ],
        default="full",
        help=(
            "generate/train/full: root-version constant-action channels. "
            "per_step_generate/per_step_train/per_step: per-step marginal MI "
            "(sweep one step's action on a shared root trajectory). "
            "decompose: assumption-free variance decomposition (eta^2 per control "
            "axis) on the existing dataset CSV; no critic training, cannot diverge."
        ),
    )

    # Data generation knobs
    p.add_argument("--prompt_file", default=None, help="Text file with one original prompt per line (required for generate modes).")
    p.add_argument("--n_prompts", type=int, default=50)
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--n_seeds", type=int, default=16)
    p.add_argument("--n_rewrites", type=int, default=3, help="Number of minimal rewrites per prompt.")
    p.add_argument(
        "--cfg_scales",
        nargs="+",
        type=float,
        default=[1.0, 3.0, 5.0, 7.0, 9.0],
    )
    p.add_argument("--default_cfg", type=float, default=5.0)
    p.add_argument("--reward_noise_std", type=float, default=0.01)
    p.add_argument("--dataset_csv", default="./mi_diag_sd35_dataset.csv")
    p.add_argument("--rewrites_file", default=None)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_every", type=int, default=100)

    # Sampler/model/reward knobs (aligned to sampling_unified_sd35)
    p.add_argument("--backend", choices=["sid", "sd35_base", "senseflow_large", "senseflow_medium"], default="sid")
    p.add_argument("--model_id", default=os.environ.get("MODEL_ID"))
    p.add_argument("--transformer_id", default=os.environ.get("TRANSFORMER_ID"))
    p.add_argument("--transformer_subfolder", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--lora_path", default=None)
    p.add_argument("--lora_scale", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--sigmas", nargs="+", type=float, default=None)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--time_scale", type=float, default=1000.0)
    p.add_argument("--max_sequence_length", type=int, default=256)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default=None)
    p.add_argument("--x0_sampler", action="store_true", default=False)
    p.add_argument("--euler_sampler", action="store_true", default=False)

    p.add_argument("--no_qwen", action="store_true")
    p.add_argument("--qwen_id", default="Qwen/Qwen3-4B")
    p.add_argument("--qwen_python", default="python3")
    p.add_argument("--qwen_dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--qwen_timeout_sec", type=float, default=240.0)

    p.add_argument(
        "--reward_backend",
        choices=["auto", "unifiedreward", "unified", "imagereward", "pickscore", "hpsv3", "hpsv2", "blend", "all"],
        default="imagereward",
    )
    p.add_argument("--reward_model", default="CodeGoat24/UnifiedReward-qwen-7b")
    p.add_argument("--unifiedreward_model", default=None)
    p.add_argument("--image_reward_model", default="ImageReward-v1.0")
    p.add_argument("--pickscore_model", default="yuvalkirstain/PickScore_v1")
    p.add_argument("--reward_weights", nargs=2, type=float, default=[1.0, 1.0])
    p.add_argument("--reward_api_base", default=None)
    p.add_argument("--reward_api_key", default="unifiedreward")
    p.add_argument("--reward_api_model", default="UnifiedReward-7b-v1.5")
    p.add_argument("--reward_max_new_tokens", type=int, default=512)
    p.add_argument("--reward_prompt_mode", choices=["standard", "strict"], default="standard")

    # MINE training knobs
    p.add_argument("--mine_hidden_dim", type=int, default=256)
    p.add_argument("--mine_embedding_dim", type=int, default=32)
    p.add_argument("--mine_batch_size", type=int, default=1024)
    p.add_argument("--mine_lr", type=float, default=1e-4)
    p.add_argument("--mine_steps", type=int, default=10000)
    p.add_argument("--mine_eval_every", type=int, default=250)
    p.add_argument("--mine_grad_clip", type=float, default=1.0)
    p.add_argument("--mine_val_frac", type=float, default=0.2)
    p.add_argument("--mine_restarts", type=int, default=3)
    p.add_argument("--mine_device", default="cuda")
    p.add_argument("--mine_seed", type=int, default=1234)
    p.add_argument("--mine_report_json", default="./mi_diag_sd35_report.json")
    p.add_argument("--mine_table_csv", default="./mi_diag_sd35_table.csv")
    p.add_argument(
        "--mine_estimator",
        choices=["smile", "dv", "infonce", "ross"],
        default="infonce",
        help=(
            "infonce (default): contrastive lower bound, provably <= log K = "
            "H(C|P) with within-prompt negatives, so it cannot diverge "
            "(theoretically safe). smile: tau-clipped DV (lower variance, but can "
            "still blow up on high-cardinality joint controls). dv: raw "
            "Donsker-Varadhan (unbounded log-partition; diverges easily). "
            "ross: Ross (2014) k-NN mixed discrete-continuous MI -- no critic, no "
            "training, low variance for the small discrete action grid. Computes "
            "per-prompt I(action; R | prompt), averages, with a within-prompt "
            "label-permuted null and a prompt-bootstrap CI. Recommended for the "
            "per-step diagnostic where neural critics are high-variance."
        ),
    )
    p.add_argument("--mine_ross_k", type=int, default=3, help="k-NN neighbors for the Ross estimator.")
    p.add_argument("--mine_ross_bootstrap", type=int, default=200, help="Prompt-bootstrap resamples for the Ross CI (0 disables).")
    p.add_argument("--mine_smile_tau", type=float, default=5.0, help="Clip threshold for SMILE critic outputs.")
    p.add_argument(
        "--mine_infonce_max_neg",
        type=int,
        default=512,
        help="Max within-prompt negatives per block for the InfoNCE estimator (caps log K ceiling). "
        "Capped by available within-prompt rows; higher = more negatives = tighter, lower-variance bound.",
    )
    p.add_argument(
        "--mine_infonce_blocks",
        type=int,
        default=8,
        help="Prompt-blocks sampled per InfoNCE training step.",
    )
    p.add_argument(
        "--mine_early_stop_patience",
        type=int,
        default=8,
        help="Stop a restart after this many eval points without EMA val-MI improvement (<=0 disables).",
    )

    # Variance-decomposition (decompose) knobs
    p.add_argument("--decompose_table_csv", default="./mi_diag_sd35_decompose.csv")
    p.add_argument("--decompose_report_json", default="./mi_diag_sd35_decompose.json")
    p.add_argument("--per_step_decompose_table_csv", default="./mi_perstep_sd35_decompose.csv")
    p.add_argument("--per_step_decompose_report_json", default="./mi_perstep_sd35_decompose.json")

    # Per-step (per_step_*) marginal-MI knobs
    p.add_argument("--per_step_dataset_csv", default="./mi_perstep_sd35_dataset.csv")
    p.add_argument("--per_step_report_json", default="./mi_perstep_sd35_report.json")
    p.add_argument("--per_step_table_csv", default="./mi_perstep_sd35_table.csv")
    p.add_argument("--per_step_baseline_variant", type=int, default=0, help="Variant id held on non-branched steps.")
    p.add_argument(
        "--per_step_n_noise", type=int, default=0,
        help="Noise-channel grid size: number of CRN noise realizations swept at the probed step "
        "(0 disables the noise channel). Step 0 sweeps the root latent; interior steps sweep the "
        "per-step injected noise (inert for deterministic/Euler backbones).",
    )
    p.add_argument(
        "--per_step_noise_seed_base", type=int, default=900000,
        help="Base seed for noise-channel realizations (kept disjoint from sample seeds).",
    )
    p.add_argument(
        "--per_step_test_frac", type=float, default=0.2,
        help="Held-out test fraction for the 3-way split; MI is reported on this split only. 0 -> legacy 2-way.",
    )
    p.add_argument(
        "--per_step_unclamped", action=argparse.BooleanOptionalAction, default=True,
        help="Report mi_corrected = mi_real - mi_null without the max(0,.) floor or H(C) clamp.",
    )
    p.add_argument(
        "--per_step_crn_residualize", action=argparse.BooleanOptionalAction, default=True,
        help="Center reward within each (prompt_id, seed_id) CRN block before the per-step "
        "MINE/InfoNCE estimator. This removes the per-root baseline (the dominant seed-variance "
        "nuisance, eta2_seed~0.92-0.999) so the estimator measures conditional MI "
        "I(action; R | prompt, seed) instead of the seed-swamped marginal I(action; R). "
        "Off -> legacy raw-reward marginal MI.",
    )
    p.add_argument("--per_step_plot_png", default="./mi_perstep_sd35_curve.png")
    return p.parse_args(argv)


def _nearest_cfg_id(cfg_values: list[float], default_cfg: float) -> int:
    return int(min(range(len(cfg_values)), key=lambda i: abs(float(cfg_values[i]) - float(default_cfg))))


def _read_prompts(path: str, n_prompts: int) -> list[str]:
    with open(path, encoding="utf-8") as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    if n_prompts > 0:
        return all_prompts[:n_prompts]
    return all_prompts


def _build_sampler_args(args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace(
        backend=args.backend,
        model_id=args.model_id,
        ckpt=args.ckpt,
        lora_path=args.lora_path,
        lora_scale=float(args.lora_scale),
        transformer_id=args.transformer_id,
        transformer_subfolder=args.transformer_subfolder,
        prompt="",
        prompt_file=None,
        n_variants=int(args.n_rewrites),
        no_qwen=bool(args.no_qwen),
        qwen_id=args.qwen_id,
        qwen_python=args.qwen_python,
        qwen_dtype=args.qwen_dtype,
        qwen_timeout_sec=float(args.qwen_timeout_sec),
        rewrites_file=args.rewrites_file,
        steps=int(args.steps),
        sigmas=args.sigmas,
        cfg_scales=[float(v) for v in args.cfg_scales],
        baseline_cfg=float(args.default_cfg),
        seed=int(args.seed_base),
        width=int(args.width),
        height=int(args.height),
        time_scale=float(args.time_scale),
        out_dir=str(Path(args.dataset_csv).resolve().parent),
        correction_strengths=[0.0],
        x0_sampler=bool(args.x0_sampler),
        euler_sampler=bool(args.euler_sampler),
        reward_model=args.reward_model,
        unifiedreward_model=args.unifiedreward_model,
        image_reward_model=args.image_reward_model,
        pickscore_model=args.pickscore_model,
        reward_backend=args.reward_backend,
        reward_weights=[float(args.reward_weights[0]), float(args.reward_weights[1])],
        reward_api_base=args.reward_api_base,
        reward_api_key=args.reward_api_key,
        reward_api_model=args.reward_api_model,
        reward_max_new_tokens=int(args.reward_max_new_tokens),
        reward_prompt_mode=args.reward_prompt_mode,
        dtype=args.dtype,
    )
    ns = su._apply_backend_defaults(ns)
    return ns


def _csv_safe(text: str) -> str:
    """Strip newlines / CRs / NULs so one logical row can never span multiple
    physical lines, regardless of how downstream tools quote/read the file.
    Embedded newlines have caused shard-CSV column-shift on resume."""
    if text is None:
        return ""
    return str(text).replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\x00", "")


def _existing_key_set(csv_path: str) -> set[tuple[int, int, int, int]]:
    out: set[tuple[int, int, int, int]] = set()
    if not os.path.exists(csv_path):
        return out
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out.add((int(row["prompt_id"]), int(row["seed_id"]), int(row["variant_id"]), int(row["cfg_id"])))
            except (ValueError, TypeError, KeyError):
                # Skip malformed rows (column-shift from a prior partial write).
                continue
    return out


def generate_dataset(args: argparse.Namespace) -> None:
    sampler_args = _build_sampler_args(args)
    prompts = _read_prompts(args.prompt_file, int(args.n_prompts))
    if len(prompts) == 0:
        raise RuntimeError("No prompts found after filtering.")

    out_csv = str(Path(args.dataset_csv).expanduser().resolve())
    os.makedirs(str(Path(out_csv).parent), exist_ok=True)

    rewrite_cache: dict[str, Any] = {}
    if args.rewrites_file and os.path.exists(args.rewrites_file):
        with open(args.rewrites_file, encoding="utf-8") as f:
            rewrite_cache = json.load(f)
        print(f"[generate] loaded rewrite cache entries={len(rewrite_cache)}")

    done_keys: set[tuple[int, int, int, int]] = set()
    if bool(args.resume) and os.path.exists(out_csv):
        done_keys = _existing_key_set(out_csv)
        print(f"[generate] resume enabled: {len(done_keys)} rows already present.")
    else:
        if os.path.exists(out_csv):
            os.remove(out_csv)
        print("[generate] starting fresh dataset file.")

    header = [
        "prompt_id",
        "original_prompt",
        "variant_id",
        "variant_text",
        "seed_id",
        "cfg_id",
        "cfg_value",
        "reward",
    ]

    sampler_args.out_dir = str(Path(out_csv).parent)
    ctx = su.load_pipeline(sampler_args)
    reward_model = su.load_reward_model(sampler_args, ctx.device)

    cfg_values = [float(v) for v in args.cfg_scales]
    target_variant_count = int(args.n_rewrites) + 1
    rng = np.random.default_rng(int(args.seed_base) + 777)
    total_target_rows = len(prompts) * int(args.n_seeds) * target_variant_count * len(cfg_values)
    written = 0
    skipped = 0
    t0 = time.time()

    file_exists = os.path.exists(out_csv)
    with open(out_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists or os.path.getsize(out_csv) == 0:
            writer.writeheader()

        for prompt_id, original_prompt in enumerate(prompts):
            variants = su.generate_variants(sampler_args, original_prompt, rewrite_cache)
            if len(variants) < target_variant_count:
                variants = variants + [original_prompt] * (target_variant_count - len(variants))
            elif len(variants) > target_variant_count:
                variants = variants[:target_variant_count]

            emb = su.encode_variants(ctx, variants, max_sequence_length=int(args.max_sequence_length))
            for variant_id, variant_text in enumerate(variants):
                for cfg_id, cfg_val in enumerate(cfg_values):
                    actions = [(int(variant_id), float(cfg_val), 0.0)] * int(sampler_args.steps)
                    for seed_id in range(int(args.n_seeds)):
                        key = (int(prompt_id), int(seed_id), int(variant_id), int(cfg_id))
                        if key in done_keys:
                            skipped += 1
                            continue
                        sample_seed = int(args.seed_base) + int(seed_id)
                        result = su.run_schedule_actions(
                            sampler_args,
                            ctx,
                            emb,
                            reward_model,
                            original_prompt,  # reward is always against original task prompt
                            sample_seed,
                            actions,
                            deterministic_noise=True,
                        )
                        reward_noisy = float(result.score) + float(rng.normal(0.0, float(args.reward_noise_std)))
                        writer.writerow(
                            {
                                "prompt_id": int(prompt_id),
                                "original_prompt": _csv_safe(original_prompt),
                                "variant_id": int(variant_id),
                                "variant_text": _csv_safe(variant_text),
                                "seed_id": int(seed_id),
                                "cfg_id": int(cfg_id),
                                "cfg_value": float(cfg_val),
                                "reward": float(reward_noisy),
                            }
                        )
                        written += 1
                        if written % max(1, int(args.save_every)) == 0:
                            f.flush()
                            os.fsync(f.fileno())
                            elapsed = time.time() - t0
                            print(
                                f"[generate] written={written} skipped={skipped} "
                                f"done={written + skipped}/{total_target_rows} elapsed={elapsed:.1f}s"
                            )

        f.flush()
        os.fsync(f.fileno())
    print(
        f"[generate] complete: file={out_csv} written={written} skipped={skipped} "
        f"total={written + skipped} target={total_target_rows}"
    )


def _safe_unique_sorted(values: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(values))


def _standardize_reward(arr: dict[str, np.ndarray], tag: str) -> dict[str, np.ndarray]:
    """Z-score the reward column in place (global affine transform).

    The MINE critic concatenates the raw scalar reward next to unit-scale learned
    embeddings; an un-normalized reward leaves the objective poorly conditioned
    (high gradient variance, slow convergence, finite-sample bias). A single
    global affine transform leaves the conditional MI I(C;R|P) exactly invariant
    while fixing the input scale, so this changes only estimator quality, never
    the quantity being estimated. mean/std are stored for audit/reproducibility.
    """
    r = arr["reward"].astype(np.float32, copy=False)
    mean = float(r.mean())
    std = float(r.std())
    if std > 1e-8:
        arr["reward"] = ((r - mean) / std).astype(np.float32)
    else:
        # Degenerate (constant) reward -> no information; leave centered at 0.
        arr["reward"] = (r - mean).astype(np.float32)
    arr["reward_mean"] = np.float32(mean)
    arr["reward_std"] = np.float32(std)
    print(f"[{tag}] standardized reward: mean={mean:.4f} std={std:.4f} (z-scored for critic conditioning)")
    return arr


def _crn_residualize_reward(
    reward: np.ndarray, prompt_id: np.ndarray, seed_id: np.ndarray
) -> np.ndarray:
    """Center reward within each (prompt_id, seed_id) CRN block.

    The per-step OSD design fixes the root trajectory and sweeps only the
    action at step k, so within one (prompt, seed) block every row shares the
    same root. Subtracting that block's mean removes the per-root baseline --
    the seed-variance nuisance that dominates total reward variance
    (eta2_seed ~ 0.92-0.999) and that biases the marginal MI estimate toward 0.
    What remains is the action-attributable residual, so a downstream estimator
    measures the conditional MI I(action; R | prompt, seed). Variance reduction
    by exact differencing -- the MI analogue of the paired CRN-delta / reachable
    -gap statistics in run_per_step_decompose.
    """
    out = reward.astype(np.float32, copy=True)
    pair = np.stack([prompt_id.astype(np.int64), seed_id.astype(np.int64)], axis=1)
    _, inv = np.unique(pair, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)  # numpy>=2.0 may return inv with shape (N,1)
    n_groups = int(inv.max()) + 1 if inv.size else 0
    if n_groups == 0:
        return out
    sums = np.zeros(n_groups, dtype=np.float64)
    cnts = np.zeros(n_groups, dtype=np.float64)
    np.add.at(sums, inv, out.astype(np.float64))
    np.add.at(cnts, inv, 1.0)
    means = (sums / np.maximum(cnts, 1.0)).astype(np.float32)
    out -= means[inv]
    return out


def load_dataset(csv_path: str) -> dict[str, Any]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"dataset_csv not found: {csv_path}")
    prompt_ids: list[int] = []
    rewards: list[float] = []
    seed_ids: list[int] = []
    variant_ids: list[int] = []
    cfg_ids: list[int] = []
    cfg_values: list[float] = []

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt_ids.append(int(row["prompt_id"]))
            rewards.append(float(row["reward"]))
            seed_ids.append(int(row["seed_id"]))
            variant_ids.append(int(row["variant_id"]))
            cfg_ids.append(int(row["cfg_id"]))
            cfg_values.append(float(row["cfg_value"]))

    arr = {
        "prompt_id": np.asarray(prompt_ids, dtype=np.int64),
        "reward": np.asarray(rewards, dtype=np.float32),
        "seed_id": np.asarray(seed_ids, dtype=np.int64),
        "variant_id": np.asarray(variant_ids, dtype=np.int64),
        "cfg_id": np.asarray(cfg_ids, dtype=np.int64),
        "cfg_value": np.asarray(cfg_values, dtype=np.float32),
    }
    n = int(arr["reward"].shape[0])
    if n <= 0:
        raise RuntimeError("Dataset is empty.")
    _standardize_reward(arr, tag="root")
    return arr


def _split_indices(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    n_val = int(max(1, min(n - 1, round(n * val_frac))))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def _split_indices_3way(
    n: int, val_frac: float, test_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Disjoint train / val (early-stop) / test (report) split. The MI reported
    for a channel is measured on `test`, which the critic never saw during
    training or checkpoint selection -> removes the optimistic bias of reading
    MI on the same val split used for early stopping."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    n_test = int(max(1, min(n - 2, round(n * test_frac))))
    n_val = int(max(1, min(n - 1 - n_test, round(n * val_frac))))
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx


def _subset_array(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return arr[idx]


def _permute_within_prompt_ids(
    prompt_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.arange(prompt_ids.shape[0], dtype=np.int64)
    by_prompt: dict[int, list[int]] = {}
    for i, p in enumerate(prompt_ids.tolist()):
        by_prompt.setdefault(int(p), []).append(i)
    for _p, pos in by_prompt.items():
        if len(pos) <= 1:
            continue
        pos_arr = np.asarray(pos, dtype=np.int64)
        out[pos_arr] = pos_arr[rng.permutation(len(pos_arr))]
    return out


def _permute_controls_globally_within_prompt(
    prompt_ids: np.ndarray,
    controls: dict[str, np.ndarray],
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = _permute_within_prompt_ids(prompt_ids, rng)
    out: dict[str, np.ndarray] = {}
    for name, v in controls.items():
        out[name] = v[perm]
    return out


def _entropy_nats(discrete_values: np.ndarray) -> float:
    if discrete_values.size <= 0:
        return 0.0
    unique, counts = np.unique(discrete_values, return_counts=True)
    del unique
    probs = counts.astype(np.float64) / float(np.sum(counts))
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def _mi_explained_var(mi: float) -> float:
    """Gaussian I<->rho^2 map: rho^2 = 1 - exp(-2I). Reads the (debiased) MI as
    the fraction of (CRN-residualized) reward variance the channel explains --
    cardinality-free (unlike MI/H(C)), in [0,1), ~2I for small I, and directly
    comparable to the variance/eta^2 decompose. Negative debiased MI floors to 0.
    """
    m = max(0.0, float(mi))
    return float(1.0 - np.exp(-2.0 * m))


def _joint_code(cols: list[np.ndarray]) -> np.ndarray:
    if len(cols) == 1:
        return cols[0].astype(np.int64)
    # Stable tuple encoding via mixed radix.
    out = cols[0].astype(np.int64).copy()
    base = int(np.max(out)) + 1
    for col in cols[1:]:
        c = col.astype(np.int64)
        out = out * int(np.max(c) + 1) + c
        base *= int(np.max(c) + 1)
    del base
    return out


def _auroc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)

    # Tie correction to average ranks.
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg_rank = float((i + 1 + j) / 2.0)
            ranks[order[i:j]] = avg_rank
        i = j

    sum_pos = float(np.sum(ranks[labels == 1]))
    auc = (sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


@dataclass
class CriticSpec:
    channel: str
    controls: list[str]
    variant_eq: int | None
    cfg_eq: int | None


class MineCritic(nn.Module):
    def __init__(
        self,
        n_prompts: int,
        control_sizes: dict[str, int],
        control_names: list[str],
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.control_names = list(control_names)
        self.prompt_emb = nn.Embedding(int(n_prompts), int(embedding_dim))
        self.control_embs = nn.ModuleDict(
            {name: nn.Embedding(int(control_sizes[name]), int(embedding_dim)) for name in self.control_names}
        )
        in_dim = 1 + int(embedding_dim) * (1 + len(self.control_names))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        reward: torch.Tensor,
        prompt_id: torch.Tensor,
        controls: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        feats = [reward.view(-1, 1), self.prompt_emb(prompt_id)]
        for name in self.control_names:
            feats.append(self.control_embs[name](controls[name]))
        x = torch.cat(feats, dim=-1)
        out = self.mlp(x)
        return out.view(-1)


def _max_or_zero(arr: np.ndarray) -> int:
    if arr.size <= 0:
        return 0
    return int(np.max(arr))


def _build_filtered_data(
    data: dict[str, np.ndarray],
    spec: CriticSpec,
) -> dict[str, np.ndarray]:
    mask = np.ones_like(data["prompt_id"], dtype=bool)
    if spec.variant_eq is not None:
        mask &= (data["variant_id"] == int(spec.variant_eq))
    if spec.cfg_eq is not None:
        mask &= (data["cfg_id"] == int(spec.cfg_eq))
    if int(mask.sum()) <= 0:
        raise RuntimeError(f"No rows left for critic channel={spec.channel}.")

    out = {
        "prompt_id": data["prompt_id"][mask],
        "reward": data["reward"][mask],
    }
    for c in spec.controls:
        out[c] = data[c][mask]
    return out


def _to_tensors(batch: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {
        "prompt_id": torch.as_tensor(batch["prompt_id"], dtype=torch.long, device=device),
        "reward": torch.as_tensor(batch["reward"], dtype=torch.float32, device=device),
    }
    for k, v in batch.items():
        if k in {"prompt_id", "reward"}:
            continue
        out[k] = torch.as_tensor(v, dtype=torch.long, device=device)
    return out


def _permute_controls_in_batch(
    prompt_id: torch.Tensor,
    controls: dict[str, torch.Tensor],
    rng: np.random.Generator,
) -> dict[str, torch.Tensor]:
    p = prompt_id.detach().cpu().numpy()
    perm = _permute_within_prompt_ids(p, rng)
    perm_t = torch.as_tensor(perm, dtype=torch.long, device=prompt_id.device)
    out: dict[str, torch.Tensor] = {}
    for k, v in controls.items():
        out[k] = v.index_select(0, perm_t)
    return out


def _log_mean_exp(t: torch.Tensor, estimator: str, tau: float) -> torch.Tensor:
    """log E[e^t] estimate. For SMILE, clip t to [-tau, tau] before the
    logsumexp so a single large negative-sample logit cannot blow up the
    variance of the bound (the classic MINE/DV instability)."""
    if estimator == "smile" and tau > 0:
        t = torch.clamp(t, min=-float(tau), max=float(tau))
    return torch.logsumexp(t, dim=0) - math.log(float(t.shape[0]))


def _critic_bound(
    model: MineCritic,
    reward: torch.Tensor,
    prompt_id: torch.Tensor,
    controls_pos: dict[str, torch.Tensor],
    controls_neg: dict[str, torch.Tensor],
    estimator: str,
    tau: float,
) -> torch.Tensor:
    t_pos = model(reward, prompt_id, controls_pos)
    t_neg = model(reward, prompt_id, controls_neg)
    return t_pos.mean() - _log_mean_exp(t_neg, estimator, tau)


def _prompt_to_rows(prompt_ids: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, list[int]] = {}
    for i, p in enumerate(prompt_ids.tolist()):
        out.setdefault(int(p), []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def _block_scores(
    model: MineCritic,
    reward: torch.Tensor,
    prompt_id: torch.Tensor,
    controls: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Score every (reward_i, control_j) pair in a same-prompt block of size K.
    Returns S[i, j] = f(R_i, P, C_j); the positive is the diagonal (i == j)."""
    k = int(reward.shape[0])
    reward_rep = reward.repeat_interleave(k)  # i-major
    prompt_rep = prompt_id.repeat_interleave(k)
    controls_tile = {name: v.repeat(k) for name, v in controls.items()}  # j-tiled
    s = model(reward_rep, prompt_rep, controls_tile)
    return s.view(k, k)


def _infonce_from_scores(scores: torch.Tensor) -> torch.Tensor:
    """InfoNCE lower bound from a K x K same-prompt score matrix (positive on the
    diagonal). Bounded above by log K, so with within-prompt negatives the
    estimate can never exceed H(C|P) -- it cannot diverge."""
    k = int(scores.shape[0])
    diag = torch.diagonal(scores)
    lse = torch.logsumexp(scores, dim=1)  # over candidate controls j (incl. positive)
    return (diag - lse).mean() + math.log(float(k))


def _infonce_train_step(
    model: MineCritic,
    data_t: dict[str, torch.Tensor],
    control_names: list[str],
    prompt_rows: dict[int, np.ndarray],
    rng: np.random.Generator,
    n_blocks: int,
    max_neg: int,
) -> torch.Tensor:
    device = data_t["reward"].device
    keys = [p for p, r in prompt_rows.items() if r.shape[0] >= 2]
    if not keys:
        return torch.zeros((), device=device, requires_grad=True)
    pick = rng.choice(len(keys), size=min(int(n_blocks), len(keys)), replace=False)
    inces: list[torch.Tensor] = []
    for ci in pick:
        rows = prompt_rows[keys[int(ci)]]
        if rows.shape[0] > max_neg + 1:
            rows = rng.choice(rows, size=max_neg + 1, replace=False)
        idx = torch.as_tensor(rows, dtype=torch.long, device=device)
        reward = data_t["reward"].index_select(0, idx)
        prompt_id = data_t["prompt_id"].index_select(0, idx)
        controls = {k: data_t[k].index_select(0, idx) for k in control_names}
        inces.append(_infonce_from_scores(_block_scores(model, reward, prompt_id, controls)))
    return -torch.stack(inces).mean()


def _estimate_infonce(
    model: MineCritic,
    data_t: dict[str, torch.Tensor],
    control_names: list[str],
    max_neg: int,
    seed: int,
) -> tuple[float, float]:
    """Full per-prompt InfoNCE, averaged over prompt blocks. AUROC from diagonal
    (positive) vs off-diagonal (within-prompt negative) scores."""
    model.eval()
    device = data_t["reward"].device
    rng = np.random.default_rng(seed)
    prompt_rows = _prompt_to_rows(data_t["prompt_id"].detach().cpu().numpy())
    inces: list[float] = []
    pos_scores: list[np.ndarray] = []
    neg_scores: list[np.ndarray] = []
    with torch.no_grad():
        for rows in prompt_rows.values():
            if rows.shape[0] < 2:
                continue
            r = rows
            if r.shape[0] > max_neg + 1:
                r = rng.choice(r, size=max_neg + 1, replace=False)
            idx = torch.as_tensor(r, dtype=torch.long, device=device)
            reward = data_t["reward"].index_select(0, idx)
            prompt_id = data_t["prompt_id"].index_select(0, idx)
            controls = {k: data_t[k].index_select(0, idx) for k in control_names}
            scores = _block_scores(model, reward, prompt_id, controls)
            inces.append(float(_infonce_from_scores(scores).item()))
            k = int(scores.shape[0])
            s_np = scores.detach().cpu().numpy()
            off = ~np.eye(k, dtype=bool)
            pos_scores.append(np.diagonal(s_np))
            neg_scores.append(s_np[off])
    if not inces:
        return 0.0, 0.5
    mi = float(np.mean(inces))
    s_pos = np.concatenate(pos_scores, axis=0)
    s_neg = np.concatenate(neg_scores, axis=0)
    auc = _auroc_from_scores(
        np.concatenate([np.ones_like(s_pos, dtype=np.int64), np.zeros_like(s_neg, dtype=np.int64)], axis=0),
        np.concatenate([s_pos, s_neg], axis=0),
    )
    return mi, auc


def _estimate_mi_and_auroc(
    model: MineCritic,
    data_t: dict[str, torch.Tensor],
    control_names: list[str],
    batch_size: int,
    seed: int,
    estimator: str = "smile",
    tau: float = 5.0,
    infonce_max_neg: int = 256,
) -> tuple[float, float]:
    """Full-set DV/SMILE estimate: accumulate t_pos over every row and take a
    SINGLE log-mean-exp over ALL negatives. Averaging per-minibatch DV bounds
    (the old behavior) is Jensen-biased upward and high-variance, so the whole
    set is scored in one pass against one within-prompt permuted negative draw."""
    model.eval()
    n = int(data_t["reward"].shape[0])
    if n <= 0:
        return 0.0, 0.5

    if estimator == "infonce":
        return _estimate_infonce(model, data_t, control_names, max_neg=int(infonce_max_neg), seed=seed)

    # One global within-prompt permutation -> conditional-MI negatives p(R|P)p(C|P).
    rng = np.random.default_rng(seed)
    prompt_np = data_t["prompt_id"].detach().cpu().numpy()
    perm = _permute_within_prompt_ids(prompt_np, rng)
    perm_t = torch.as_tensor(perm, dtype=torch.long, device=data_t["prompt_id"].device)

    t_pos_chunks: list[torch.Tensor] = []
    t_neg_chunks: list[torch.Tensor] = []
    bs = max(1, batch_size)
    with torch.no_grad():
        for start in range(0, n, bs):
            end = min(n, start + bs)
            reward = data_t["reward"][start:end]
            prompt_id = data_t["prompt_id"][start:end]
            controls_pos = {k: data_t[k][start:end] for k in control_names}
            neg_idx = perm_t[start:end]
            controls_neg = {k: data_t[k].index_select(0, neg_idx) for k in control_names}
            t_pos_chunks.append(model(reward, prompt_id, controls_pos))
            t_neg_chunks.append(model(reward, prompt_id, controls_neg))

    t_pos = torch.cat(t_pos_chunks, dim=0)
    t_neg = torch.cat(t_neg_chunks, dim=0)
    mi = float((t_pos.mean() - _log_mean_exp(t_neg, estimator, tau)).item())

    s_pos = t_pos.detach().cpu().numpy()
    s_neg = t_neg.detach().cpu().numpy()
    auc = _auroc_from_scores(
        np.concatenate([np.ones_like(s_pos, dtype=np.int64), np.zeros_like(s_neg, dtype=np.int64)], axis=0),
        np.concatenate([s_pos, s_neg], axis=0),
    )
    return mi, auc


def _train_one_restart(
    train_np: dict[str, np.ndarray],
    val_np: dict[str, np.ndarray],
    control_names: list[str],
    n_prompts: int,
    args: argparse.Namespace,
    restart_seed: int,
    test_np: dict[str, np.ndarray] | None = None,
) -> tuple[float, float, float]:
    """Returns (train_mi, report_mi, auroc). report_mi/auroc are measured on
    `test_np` when given (held-out 3-way split) else on `val_np` (legacy 2-way).
    Early stopping always tracks val MI regardless."""
    device = args.mine_device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    torch.manual_seed(restart_seed)
    np.random.seed(restart_seed)
    random.seed(restart_seed)

    control_sizes = {k: int(max(_max_or_zero(train_np[k]), _max_or_zero(val_np[k])) + 1) for k in control_names}
    model = MineCritic(
        n_prompts=n_prompts,
        control_sizes=control_sizes,
        control_names=control_names,
        embedding_dim=int(args.mine_embedding_dim),
        hidden_dim=int(args.mine_hidden_dim),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.mine_lr))
    rng = np.random.default_rng(restart_seed + 17)

    estimator = str(getattr(args, "mine_estimator", "smile"))
    tau = float(getattr(args, "mine_smile_tau", 5.0))
    infonce_max_neg = int(getattr(args, "mine_infonce_max_neg", 512))
    infonce_blocks = int(getattr(args, "mine_infonce_blocks", 8))
    patience_limit = int(getattr(args, "mine_early_stop_patience", 0))

    train_t = _to_tensors(train_np, device=device)
    val_t = _to_tensors(val_np, device=device)
    n_train = int(train_t["reward"].shape[0])
    bs = max(1, int(args.mine_batch_size))
    train_prompt_rows = _prompt_to_rows(train_np["prompt_id"]) if estimator == "infonce" else {}

    # EMA-smoothed early stopping. Selecting the single max val-MI checkpoint
    # maximizes over noisy estimates -> upward bias (worst for near-zero
    # channels). Track an EMA of val MI and keep the best-EMA state instead.
    best_ema = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    ema_val: float | None = None
    no_improve = 0

    for step in range(1, int(args.mine_steps) + 1):
        model.train()
        if estimator == "infonce":
            loss = _infonce_train_step(
                model, train_t, control_names, train_prompt_rows, rng, infonce_blocks, infonce_max_neg
            )
        else:
            idx = rng.integers(0, n_train, size=bs, endpoint=False)
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
            reward = train_t["reward"].index_select(0, idx_t)
            prompt_id = train_t["prompt_id"].index_select(0, idx_t)
            controls_pos = {k: train_t[k].index_select(0, idx_t) for k in control_names}
            controls_neg = _permute_controls_in_batch(prompt_id, controls_pos, rng)
            bound = _critic_bound(model, reward, prompt_id, controls_pos, controls_neg, estimator, tau)
            loss = -bound
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if float(args.mine_grad_clip) > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.mine_grad_clip))
        opt.step()

        if step % max(1, int(args.mine_eval_every)) == 0 or step == int(args.mine_steps):
            val_mi, _ = _estimate_mi_and_auroc(
                model,
                val_t,
                control_names,
                batch_size=bs,
                seed=restart_seed + step * 3,
                estimator=estimator,
                tau=tau,
                infonce_max_neg=infonce_max_neg,
            )
            ema_val = float(val_mi) if ema_val is None else 0.7 * ema_val + 0.3 * float(val_mi)
            if ema_val > best_ema + 1e-4:
                best_ema = float(ema_val)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if patience_limit > 0 and no_improve >= patience_limit:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_mi, _ = _estimate_mi_and_auroc(
        model, train_t, control_names, batch_size=bs, seed=restart_seed + 20001,
        estimator=estimator, tau=tau, infonce_max_neg=infonce_max_neg,
    )
    if test_np is not None:
        report_t = _to_tensors(test_np, device=device)
        report_seed = restart_seed + 40001
    else:
        report_t = val_t
        report_seed = restart_seed + 30001
    report_mi, auroc = _estimate_mi_and_auroc(
        model, report_t, control_names, batch_size=bs, seed=report_seed,
        estimator=estimator, tau=tau, infonce_max_neg=infonce_max_neg,
    )
    return float(train_mi), float(report_mi), float(auroc)


def _digamma(x: np.ndarray) -> np.ndarray:
    """Vectorized digamma (psi) for x > 0, scipy-free.

    Recurrence psi(x) = psi(x+1) - 1/x shifts every entry to x >= 6, then the
    standard asymptotic series. Accurate to ~1e-8 over the counts we feed it
    (sample sizes / class counts), which is far below the estimator's sampling
    noise. Avoids a hard scipy dependency in the generation env.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    result = np.zeros_like(x)
    small = x < 6.0
    while np.any(small):
        result[small] -= 1.0 / x[small]
        x[small] += 1.0
        small = x < 6.0
    inv = 1.0 / x
    inv2 = inv * inv
    result += np.log(x) - 0.5 * inv + inv2 * (-1.0 / 12.0 + inv2 * (1.0 / 120.0 - inv2 / 252.0))
    return result


def _ross_mi_1d(y: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Ross (2014) MI between a discrete label and a 1-D continuous y.

    I = psi(N) + <psi(k_i)> - <psi(N_class_i)> - <psi(m_i + 1)>, where for each
    point i: k_i is the (possibly class-size-limited) neighbor count, the radius
    d_i is the distance to its k_i-th nearest same-class neighbor in y, and m_i
    counts all points (any class) with |y_j - y_i| <= d_i. Because y is scalar
    we do every nearest-neighbor / radius query by sorting -- no KD-tree, no
    scipy. Matches sklearn's mutual_info_classif discrete-target formula.
    """
    n = int(y.shape[0])
    if n < 3:
        return 0.0
    y = y.astype(np.float64)
    order = np.argsort(y, kind="mergesort")
    ys = y[order]  # sorted reward; for radius counts via searchsorted

    radii = np.empty(n, dtype=np.float64)
    k_used = np.empty(n, dtype=np.float64)
    label_counts = np.empty(n, dtype=np.float64)
    uniq = np.unique(labels)
    for lab in uniq:
        cls = np.where(labels == lab)[0]
        nc = int(cls.shape[0])
        label_counts[cls] = float(nc)
        if nc < 2:
            radii[cls] = 0.0
            k_used[cls] = 0.0  # excluded below (psi undefined); keep separate
            continue
        kk = min(int(k), nc - 1)
        yc = y[cls]
        corder = np.argsort(yc, kind="mergesort")
        yc_sorted = yc[corder]
        # distance to the kk-th nearest same-class neighbor for each class point
        for rank, ci in enumerate(corder):
            lo = max(0, rank - kk)
            hi = min(nc - 1, rank + kk)
            window = np.abs(yc_sorted[lo:hi + 1] - yc_sorted[rank])
            window = np.sort(window)
            # window[0] is self (0); the kk-th neighbor distance is window[kk]
            radii[cls[ci]] = float(window[kk]) if kk < window.shape[0] else float(window[-1])
        k_used[cls] = float(kk)

    valid = label_counts >= 2.0
    if not np.any(valid):
        return 0.0
    # m_i: count of all points within radius d_i (exclusive of self), 1-D via
    # searchsorted on the globally sorted reward.
    lo_idx = np.searchsorted(ys, y - radii, side="left")
    hi_idx = np.searchsorted(ys, y + radii, side="right")
    m = (hi_idx - lo_idx - 1).astype(np.float64)  # subtract self
    m = np.maximum(m, 0.0)

    vy = valid
    mi = (
        _digamma(np.array([float(n)]))[0]
        + float(np.mean(_digamma(k_used[vy])))
        - float(np.mean(_digamma(label_counts[vy])))
        - float(np.mean(_digamma(m[vy] + 1.0)))
    )
    return float(mi)


def _ross_conditional_mi(
    prompt_id: np.ndarray, reward: np.ndarray, label_code: np.ndarray, k: int
) -> float:
    """Per-prompt Ross MI, averaged over prompts (equal weight, matching the
    InfoNCE per-prompt-block average). Conditions on prompt; with CRN-residualized
    reward this is I(action; R | prompt, seed)."""
    rows_by_prompt = _prompt_to_rows(prompt_id)
    vals: list[float] = []
    for rows in rows_by_prompt.values():
        if rows.shape[0] < 3:
            continue
        lab = label_code[rows]
        if np.unique(lab).shape[0] < 2:
            continue  # no action variation in this prompt -> MI undefined
        vals.append(_ross_mi_1d(reward[rows], lab, k))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _run_channel_ross(
    subset: dict[str, np.ndarray],
    controls: list[str],
    args: argparse.Namespace,
    seed_offset: int,
    clamp_corrected: bool = True,
) -> dict[str, Any]:
    """Ross k-NN estimator path: no critic, no train/val/test split, no restarts.
    Real MI minus a within-prompt label-permuted null, with a prompt-bootstrap CI.
    Returns the same row schema as `_run_channel` (plus CI keys for the report)."""
    k = int(getattr(args, "mine_ross_k", 3))
    n_rows = int(subset["reward"].shape[0])
    reward = subset["reward"].astype(np.float64)
    prompt_id = subset["prompt_id"]
    label_code = _joint_code([subset[c] for c in controls])

    mi_real = _ross_conditional_mi(prompt_id, reward, label_code, k)

    # Null: permute the controls within prompt, recompute. Ross is low-bias, so
    # the null sits near 0 -- but we keep the same debiasing contract as the
    # neural path for an apples-to-apples comparison.
    rng = np.random.default_rng(int(args.mine_seed) + 999 * seed_offset)
    perm = _permute_within_prompt_ids(prompt_id, rng)
    null_code = label_code[perm]
    mi_null = _ross_conditional_mi(prompt_id, reward, null_code, k)

    h_control = _entropy_nats(label_code)
    if clamp_corrected:
        mi_corrected = float(max(0.0, mi_real - mi_null))
        if h_control > 0.0:
            mi_corrected = min(mi_corrected, h_control)
            mi_norm = float(min(1.0, mi_corrected / h_control))
        else:
            mi_norm = 0.0
    else:
        mi_corrected = float(mi_real - mi_null)
        mi_norm = float(mi_corrected / h_control) if h_control > 0.0 else 0.0

    # Prompt-bootstrap CI on the debiased MI (resample whole prompts).
    n_boot = int(getattr(args, "mine_ross_bootstrap", 0))
    ci_lo = ci_hi = float("nan")
    boot_std = 0.0
    if n_boot > 0:
        prompts = np.array(sorted(set(int(p) for p in prompt_id)))
        rows_by_prompt = _prompt_to_rows(prompt_id)
        boot_rng = np.random.default_rng(int(args.mine_seed) + 7 * seed_offset + 1)
        boots: list[float] = []
        for _ in range(n_boot):
            pick = boot_rng.choice(prompts, size=prompts.shape[0], replace=True)
            mi_vals = []
            for p in pick:
                rows = rows_by_prompt.get(int(p))
                if rows is None or rows.shape[0] < 3:
                    continue
                lab = label_code[rows]
                if np.unique(lab).shape[0] < 2:
                    continue
                mi_vals.append(_ross_mi_1d(reward[rows], lab, k))
            if mi_vals:
                boots.append(float(np.mean(mi_vals)) - mi_null)
        if boots:
            ci_lo = float(np.percentile(boots, 2.5))
            ci_hi = float(np.percentile(boots, 97.5))
            boot_std = float(np.std(boots))

    return {
        "controls": list(controls),
        "rows_used": int(n_rows),
        "MI": float(mi_real),
        "MI_raw": float(mi_real),
        "MI_median": float(mi_real),
        "MI_corrected": float(mi_corrected),
        "normalized_MI": float(mi_norm),
        "mi_explained_var": _mi_explained_var(mi_corrected),
        "Null_MI": float(mi_null),
        "Null_std": 0.0,
        "AUROC": float("nan"),  # not defined for the k-NN estimator
        "Restart_std": boot_std,
        "diverged": False,
        "train_mi_mean": float(mi_real),
        "h_control_nats": float(h_control),
        "report_split": "full",
        "MI_ci_lo": ci_lo,
        "MI_ci_hi": ci_hi,
        "estimator": "ross",
    }


def _run_channel(
    subset: dict[str, np.ndarray],
    controls: list[str],
    n_prompts: int,
    args: argparse.Namespace,
    seed_offset: int,
    test_frac: float = 0.0,
    clamp_corrected: bool = True,
) -> dict[str, Any]:
    """Train the real + null critics over `mine_restarts` restarts for one
    channel (subset already filtered, must hold prompt_id, reward, and every
    control). Returns a row dict with raw MI, null-debiased MI, AUROC, std.

    test_frac > 0 switches to a 3-way train/val/test split and reports MI on the
    held-out test split (the val split is used only for early stopping).
    clamp_corrected=False reports the raw debiased difference mi_real - mi_null
    without the max(0,.) floor or the H(C) clamp -- so a null-band-overlapping
    channel can read slightly negative, which is the honest finite-sample
    behaviour the per-step diagnostic wants for its error bands."""
    if str(getattr(args, "mine_estimator", "")) == "ross":
        return _run_channel_ross(
            subset, controls, args, seed_offset=seed_offset, clamp_corrected=clamp_corrected
        )
    n_rows_subset = int(subset["reward"].shape[0])

    controls_np = {c: subset[c] for c in controls}
    null_controls = _permute_controls_globally_within_prompt(
        subset["prompt_id"], controls_np, seed=int(args.mine_seed) + 999 * seed_offset
    )
    null_subset = {
        "prompt_id": subset["prompt_id"].copy(),
        "reward": subset["reward"].copy(),
        **{c: null_controls[c] for c in controls},
    }

    use_3way = float(test_frac) > 0.0
    if use_3way:
        train_idx, val_idx, test_idx = _split_indices_3way(
            n_rows_subset, float(args.mine_val_frac), float(test_frac),
            seed=int(args.mine_seed) + 31 * seed_offset,
        )
    else:
        train_idx, val_idx = _split_indices(
            n_rows_subset, float(args.mine_val_frac), seed=int(args.mine_seed) + 31 * seed_offset
        )
        test_idx = None

    def _slice(src: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
        out = {"prompt_id": _subset_array(src["prompt_id"], idx), "reward": _subset_array(src["reward"], idx)}
        for c in controls:
            out[c] = _subset_array(src[c], idx)
        return out

    train_np = _slice(subset, train_idx)
    val_np = _slice(subset, val_idx)
    test_np = _slice(subset, test_idx) if test_idx is not None else None
    null_train_np = _slice(null_subset, train_idx)
    null_val_np = _slice(null_subset, val_idx)
    null_test_np = _slice(null_subset, test_idx) if test_idx is not None else None

    train_mis: list[float] = []
    val_mis: list[float] = []
    aucs: list[float] = []
    null_val_mis: list[float] = []
    for r in range(int(args.mine_restarts)):
        restart_seed = int(args.mine_seed) + 1000 * seed_offset + r
        tr_mi, va_mi, auroc = _train_one_restart(
            train_np, val_np, controls, n_prompts, args, restart_seed=restart_seed, test_np=test_np
        )
        train_mis.append(float(tr_mi))
        val_mis.append(float(va_mi))
        aucs.append(float(auroc))
        _ntr, nva, _na = _train_one_restart(
            null_train_np, null_val_np, controls, n_prompts, args,
            restart_seed=restart_seed + 50000, test_np=null_test_np,
        )
        null_val_mis.append(float(nva))

    h_control = _entropy_nats(_joint_code([subset[c] for c in controls]))
    # val_mis/null_val_mis hold the *reported* per-restart MI: test-split MI when
    # 3-way, else val-split MI (legacy). Naming kept for back-compat.
    mi_mean_raw = float(np.mean(val_mis))
    mi_median_raw = float(np.median(val_mis))
    null_mean = float(np.mean(null_val_mis))
    null_std = float(np.std(null_val_mis))
    # Hard information-theoretic ceiling: I(C;R|P) <= H(C|P) <= H(C). Any
    # estimate above H(C) is an estimator divergence (DV/SMILE log-partition
    # blow-up on near-unique joint codes, e.g. seed x variant x cfg), not signal.
    diverged = bool(h_control > 0.0 and mi_mean_raw > h_control * 1.05)
    if clamp_corrected and h_control > 0.0:
        mi_mean = min(mi_mean_raw, h_control)
        mi_median = min(mi_median_raw, h_control)
    else:
        mi_mean = mi_mean_raw
        mi_median = mi_median_raw
    # Debias by the within-prompt-shuffled null: it shares the estimator's
    # finite-sample bias floor, so MI - Null is the real conditional signal.
    if clamp_corrected:
        mi_corrected = float(max(0.0, mi_mean - null_mean))
        if h_control > 0.0:
            mi_corrected = min(mi_corrected, h_control)
            mi_norm = float(min(1.0, mi_corrected / h_control))
        else:
            mi_norm = 0.0
    else:
        mi_corrected = float(mi_mean_raw - null_mean)
        mi_norm = float(mi_corrected / h_control) if h_control > 0.0 else 0.0
    return {
        "controls": list(controls),
        "rows_used": int(n_rows_subset),
        "MI": mi_mean,
        "MI_raw": mi_mean_raw,
        "MI_median": mi_median,
        "MI_corrected": mi_corrected,
        "normalized_MI": mi_norm,
        "mi_explained_var": _mi_explained_var(mi_corrected),
        "Null_MI": null_mean,
        "Null_std": null_std,
        "AUROC": float(np.mean(aucs)),
        "Restart_std": float(np.std(val_mis)),
        "diverged": diverged,
        "train_mi_mean": float(np.mean(train_mis)),
        "h_control_nats": float(h_control),
        "report_split": "test" if use_3way else "val",
    }


def run_mine_critics(args: argparse.Namespace) -> None:
    data = load_dataset(args.dataset_csv)
    cfg_values = _safe_unique_sorted(data["cfg_id"])
    if len(cfg_values) <= 0:
        raise RuntimeError("No cfg_id values found in dataset.")
    nearest_default_cfg_id = _nearest_cfg_id(
        sorted(float(v) for v in np.unique(data["cfg_value"])),
        float(args.default_cfg),
    )
    # nearest id in sorted unique ids from dataset
    default_cfg_id = int(min(cfg_values, key=lambda cid: abs(int(cid) - int(nearest_default_cfg_id))))

    specs = [
        CriticSpec(
            channel="Seed only",
            controls=["seed_id"],
            variant_eq=0,
            cfg_eq=default_cfg_id,
        ),
        CriticSpec(
            channel="CFG only",
            controls=["cfg_id"],
            variant_eq=0,
            cfg_eq=None,
        ),
        CriticSpec(
            channel="Prompt variant",
            controls=["variant_id"],
            variant_eq=None,
            cfg_eq=default_cfg_id,
        ),
        CriticSpec(
            channel="Variant+CFG",
            controls=["variant_id", "cfg_id"],
            variant_eq=None,
            cfg_eq=None,
        ),
        CriticSpec(
            channel="Seed+Variant+CFG",
            controls=["seed_id", "variant_id", "cfg_id"],
            variant_eq=None,
            cfg_eq=None,
        ),
    ]

    all_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "dataset_csv": str(Path(args.dataset_csv).expanduser().resolve()),
        "n_rows": int(data["reward"].shape[0]),
        "n_prompts": int(np.unique(data["prompt_id"]).shape[0]),
        "default_cfg": float(args.default_cfg),
        "default_cfg_id_used": int(default_cfg_id),
        "specs": [],
    }

    n_prompts = int(_max_or_zero(data["prompt_id"]) + 1)
    for spec_idx, spec in enumerate(specs):
        subset = _build_filtered_data(data, spec)
        n_rows_subset = int(subset["reward"].shape[0])
        if n_rows_subset < 8:
            raise RuntimeError(f"Too few rows for critic '{spec.channel}': {n_rows_subset}")

        ch = _run_channel(subset, spec.controls, n_prompts, args, seed_offset=spec_idx + 1)
        row = {
            "Channel": spec.channel,
            "Control": ",".join(spec.controls),
            "MI": ch["MI"],
            "MI_raw": ch["MI_raw"],
            "MI_corrected": ch["MI_corrected"],
            "MI_median": ch["MI_median"],
            "normalized_MI": ch["normalized_MI"],
            "Null_MI": ch["Null_MI"],
            "AUROC": ch["AUROC"],
            "Restart_std": ch["Restart_std"],
            "diverged": int(ch["diverged"]),
        }
        all_rows.append(row)

        report["specs"].append(
            {
                "channel": spec.channel,
                "controls": list(spec.controls),
                "rows_used": ch["rows_used"],
                "train_mi_mean": ch["train_mi_mean"],
                "val_mi_mean": ch["MI"],
                "val_mi_mean_raw": ch["MI_raw"],
                "val_mi_median": ch["MI_median"],
                "val_mi_std": ch["Restart_std"],
                "null_val_mi_mean": ch["Null_MI"],
                "mi_corrected": ch["MI_corrected"],
                "auroc_mean": ch["AUROC"],
                "h_control_nats": ch["h_control_nats"],
                "mi_over_h_control": ch["normalized_MI"],
                "diverged": bool(ch["diverged"]),
                "restarts": int(args.mine_restarts),
            }
        )

        warn = (
            f"  [WARN diverged: raw MI={ch['MI_raw']:.1f} exceeded H(C)={ch['h_control_nats']:.3f} "
            f"ceiling -> clamped; rerun with --mine_estimator infonce]"
            if ch["diverged"] else ""
        )
        print(
            f"[train] {spec.channel:18s} MI={row['MI']:.4f} MIc={row['MI_corrected']:.4f} "
            f"MI/H={row['normalized_MI']:.4f} Null={row['Null_MI']:.4f} "
            f"AUROC={row['AUROC']:.4f} std={row['Restart_std']:.4f}{warn}"
        )

    table_csv = str(Path(args.mine_table_csv).expanduser().resolve())
    os.makedirs(str(Path(table_csv).parent), exist_ok=True)
    with open(table_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Channel", "Control", "MI", "MI_raw", "MI_corrected", "MI_median",
                "normalized_MI", "Null_MI", "AUROC", "Restart_std", "diverged",
            ],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    report_json = str(Path(args.mine_report_json).expanduser().resolve())
    os.makedirs(str(Path(report_json).parent), exist_ok=True)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[train] wrote table: {table_csv}")
    print(f"[train] wrote report: {report_json}")


def _existing_perstep_key_set(csv_path: str) -> set[tuple[int, int, int, str, int, int, int]]:
    out: set[tuple[int, int, int, str, int, int, int]] = set()
    if not os.path.exists(csv_path):
        return out
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out.add(
                    (
                        int(row["prompt_id"]),
                        int(row["seed_id"]),
                        int(row["step_idx"]),
                        str(row.get("channel", "grid")),
                        int(row["variant_id"]),
                        int(row["cfg_id"]),
                        int(row.get("noise_id", 0)),
                    )
                )
            except (ValueError, TypeError, KeyError):
                continue
    return out


def generate_dataset_per_step(args: argparse.Namespace) -> None:
    """Per-step marginal-MI dataset. For each (prompt, seed) the root latent and
    per-step noise are fixed by the seed (run_schedule_actions re-seeds to the
    same sample_seed). At one step k we sweep the full action grid while every
    other step is held at baseline -> the reward delta is attributable purely to
    the action injected at step k, on a shared root trajectory. No backup."""
    sampler_args = _build_sampler_args(args)
    prompts = _read_prompts(args.prompt_file, int(args.n_prompts))
    if len(prompts) == 0:
        raise RuntimeError("No prompts found after filtering.")

    out_csv = str(Path(args.per_step_dataset_csv).expanduser().resolve())
    os.makedirs(str(Path(out_csv).parent), exist_ok=True)

    rewrite_cache: dict[str, Any] = {}
    if args.rewrites_file and os.path.exists(args.rewrites_file):
        with open(args.rewrites_file, encoding="utf-8") as f:
            rewrite_cache = json.load(f)
        print(f"[per_step] loaded rewrite cache entries={len(rewrite_cache)}")

    done_keys: set[tuple[int, int, int, int, int]] = set()
    if bool(args.resume) and os.path.exists(out_csv):
        done_keys = _existing_perstep_key_set(out_csv)
        print(f"[per_step] resume enabled: {len(done_keys)} rows already present.")
    else:
        if os.path.exists(out_csv):
            os.remove(out_csv)
        print("[per_step] starting fresh dataset file.")

    header = [
        "prompt_id",
        "original_prompt",
        "seed_id",
        "step_idx",
        "channel",
        "variant_id",
        "cfg_id",
        "cfg_value",
        "noise_id",
        "reward",
    ]

    sampler_args.out_dir = str(Path(out_csv).parent)
    ctx = su.load_pipeline(sampler_args)
    reward_model = su.load_reward_model(sampler_args, ctx.device)

    cfg_values = [float(v) for v in args.cfg_scales]
    default_cfg_id = _nearest_cfg_id(cfg_values, float(args.default_cfg))
    base_variant = int(args.per_step_baseline_variant)
    base_cfg_val = float(cfg_values[default_cfg_id])
    n_variants = int(args.n_rewrites) + 1
    steps = int(sampler_args.steps)
    n_noise = int(args.per_step_n_noise)
    noise_seed_base = int(args.per_step_noise_seed_base)
    rng = np.random.default_rng(int(args.seed_base) + 13)

    grid_rows = len(prompts) * int(args.n_seeds) * steps * n_variants * len(cfg_values)
    noise_rows = len(prompts) * int(args.n_seeds) * steps * n_noise
    total_target_rows = grid_rows + noise_rows
    written = 0
    skipped = 0
    t0 = time.time()

    file_exists = os.path.exists(out_csv)
    with open(out_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists or os.path.getsize(out_csv) == 0:
            writer.writeheader()

        def _run_and_write(
            prompt_id: int,
            original_prompt: str,
            emb: Any,
            seed_id: int,
            sample_seed: int,
            k: int,
            channel: str,
            variant_id: int,
            cfg_id: int,
            cfg_val: float,
            noise_id: int,
            step_noise_seeds: dict[int, int] | None,
        ) -> None:
            nonlocal written, skipped
            key = (int(prompt_id), int(seed_id), int(k), str(channel),
                   int(variant_id), int(cfg_id), int(noise_id))
            if key in done_keys:
                skipped += 1
                return
            actions = [(int(base_variant), base_cfg_val, 0.0)] * steps
            actions[k] = (int(variant_id), float(cfg_val), 0.0)
            result = su.run_schedule_actions(
                sampler_args,
                ctx,
                emb,
                reward_model,
                original_prompt,
                sample_seed,
                actions,
                deterministic_noise=True,
                step_noise_seeds=step_noise_seeds,
                score_prompt=original_prompt,  # always score against c0
            )
            reward_noisy = float(result.score) + float(rng.normal(0.0, float(args.reward_noise_std)))
            writer.writerow(
                {
                    "prompt_id": int(prompt_id),
                    "original_prompt": _csv_safe(original_prompt),
                    "seed_id": int(seed_id),
                    "step_idx": int(k),
                    "channel": str(channel),
                    "variant_id": int(variant_id),
                    "cfg_id": int(cfg_id),
                    "cfg_value": float(cfg_val),
                    "noise_id": int(noise_id),
                    "reward": float(reward_noisy),
                }
            )
            written += 1
            if written % max(1, int(args.save_every)) == 0:
                f.flush()
                os.fsync(f.fileno())
                elapsed = time.time() - t0
                print(
                    f"[per_step] written={written} skipped={skipped} "
                    f"done={written + skipped}/{total_target_rows} elapsed={elapsed:.1f}s"
                )

        for prompt_id, original_prompt in enumerate(prompts):
            variants = su.generate_variants(sampler_args, original_prompt, rewrite_cache)
            if len(variants) < n_variants:
                variants = variants + [original_prompt] * (n_variants - len(variants))
            elif len(variants) > n_variants:
                variants = variants[:n_variants]
            emb = su.encode_variants(ctx, variants, max_sequence_length=int(args.max_sequence_length))

            for seed_id in range(int(args.n_seeds)):
                sample_seed = int(args.seed_base) + int(seed_id)
                for k in range(steps):
                    # variant x cfg factorial ("grid"): variant/cfg/joint channels
                    # are read as slices of this shared block downstream.
                    for variant_id in range(n_variants):
                        for cfg_id, cfg_val in enumerate(cfg_values):
                            _run_and_write(
                                prompt_id, original_prompt, emb, seed_id, sample_seed, k,
                                "grid", variant_id, cfg_id, cfg_val, 0, None,
                            )
                    # noise channel: sweep CRN noise realizations at step k with
                    # variant=base, cfg=default; the only thing varying within the
                    # block is the step-k noise -> I(noise@k ; R | P).
                    for noise_id in range(n_noise):
                        noise_seed = (
                            noise_seed_base
                            + 1_000_000 * int(prompt_id)
                            + 10_000 * int(seed_id)
                            + 100 * int(k)
                            + int(noise_id)
                        )
                        _run_and_write(
                            prompt_id, original_prompt, emb, seed_id, sample_seed, k,
                            "noise", base_variant, default_cfg_id, base_cfg_val,
                            noise_id, {int(k): int(noise_seed)},
                        )
        f.flush()
        os.fsync(f.fileno())
    print(
        f"[per_step] complete: file={out_csv} written={written} skipped={skipped} "
        f"total={written + skipped} target={total_target_rows}"
    )


def load_per_step_dataset(csv_path: str) -> dict[str, np.ndarray]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"per_step_dataset_csv not found: {csv_path}")
    cols: dict[str, list[float]] = {
        "prompt_id": [],
        "reward": [],
        "seed_id": [],
        "step_idx": [],
        "variant_id": [],
        "cfg_id": [],
        "cfg_value": [],
        "noise_id": [],
    }
    channels: list[str] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cols["prompt_id"].append(int(row["prompt_id"]))
                cols["reward"].append(float(row["reward"]))
                cols["seed_id"].append(int(row["seed_id"]))
                cols["step_idx"].append(int(row["step_idx"]))
                cols["variant_id"].append(int(row["variant_id"]))
                cols["cfg_id"].append(int(row["cfg_id"]))
                cols["cfg_value"].append(float(row["cfg_value"]))
                cols["noise_id"].append(int(row.get("noise_id", 0)))
                channels.append(str(row.get("channel", "grid")))
            except (ValueError, TypeError, KeyError):
                continue
    arr = {
        "prompt_id": np.asarray(cols["prompt_id"], dtype=np.int64),
        "reward": np.asarray(cols["reward"], dtype=np.float32),
        "seed_id": np.asarray(cols["seed_id"], dtype=np.int64),
        "step_idx": np.asarray(cols["step_idx"], dtype=np.int64),
        "variant_id": np.asarray(cols["variant_id"], dtype=np.int64),
        "cfg_id": np.asarray(cols["cfg_id"], dtype=np.int64),
        "cfg_value": np.asarray(cols["cfg_value"], dtype=np.float32),
        "noise_id": np.asarray(cols["noise_id"], dtype=np.int64),
        "channel": np.asarray(channels, dtype=object),
    }
    if int(arr["reward"].shape[0]) <= 0:
        raise RuntimeError("Per-step dataset is empty.")
    _standardize_reward(arr, tag="per_step")
    return arr


def run_per_step_critics(args: argparse.Namespace) -> None:
    data = load_per_step_dataset(args.per_step_dataset_csv)
    cfg_ids = _safe_unique_sorted(data["cfg_id"])
    nearest_default_cfg_id = _nearest_cfg_id(
        sorted(float(v) for v in np.unique(data["cfg_value"])), float(args.default_cfg)
    )
    default_cfg_id = int(min(cfg_ids, key=lambda cid: abs(int(cid) - int(nearest_default_cfg_id))))
    base_variant = int(args.per_step_baseline_variant)
    n_prompts = int(_max_or_zero(data["prompt_id"]) + 1)
    step_ids = _safe_unique_sorted(data["step_idx"])

    # (name, controls, row-mask): variant_k pins cfg=default, cfg_k pins
    # variant=baseline, joint_k sweeps both -- all read from the "grid"
    # (variant x cfg factorial) rows. noise_k reads the "noise" rows (CRN
    # step-noise sweep). Each control stays scalar so the MI/H normalization
    # remains valid (unlike a full per-step action sequence).
    chan_col = data.get("channel")
    grid_mask = (chan_col == "grid") if chan_col is not None else np.ones_like(data["prompt_id"], dtype=bool)
    noise_mask = (chan_col == "noise") if chan_col is not None else np.zeros_like(data["prompt_id"], dtype=bool)
    channels: list[tuple[str, list[str], np.ndarray]] = [
        ("variant", ["variant_id"], grid_mask & (data["cfg_id"] == default_cfg_id)),
        ("cfg", ["cfg_id"], grid_mask & (data["variant_id"] == base_variant)),
        ("joint", ["variant_id", "cfg_id"], grid_mask),
    ]
    if bool(noise_mask.any()):
        channels.append(("noise", ["noise_id"], noise_mask))

    test_frac = float(args.per_step_test_frac)
    clamp_corrected = not bool(args.per_step_unclamped)
    crn_residualize = bool(args.per_step_crn_residualize)

    all_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "per_step_dataset_csv": str(Path(args.per_step_dataset_csv).expanduser().resolve()),
        "n_rows": int(data["reward"].shape[0]),
        "n_prompts": n_prompts,
        "default_cfg": float(args.default_cfg),
        "default_cfg_id_used": int(default_cfg_id),
        "baseline_variant": int(base_variant),
        "estimator": str(args.mine_estimator),
        "test_frac": test_frac,
        "clamp_corrected": clamp_corrected,
        "crn_residualize": crn_residualize,
        "steps": [],
    }
    if crn_residualize:
        print(
            "[per_step] CRN residualization ON: reward centered within (prompt,seed) blocks "
            "-> estimator reports conditional MI I(action; R | prompt, seed)."
        )

    seed_off = 0
    for k in step_ids:
        step_mask = data["step_idx"] == int(k)
        for cname, controls, chan_mask in channels:
            seed_off += 1
            mask = step_mask & chan_mask
            subset: dict[str, np.ndarray] = {
                "prompt_id": data["prompt_id"][mask],
                "reward": data["reward"][mask],
            }
            for c in dict.fromkeys(controls):
                subset[c] = data[c][mask]
            n_rows = int(subset["reward"].shape[0])
            if n_rows < 8:
                print(f"[per_step] skip step={k} channel={cname}: only {n_rows} rows")
                continue

            if crn_residualize:
                # Difference out the per-root (prompt,seed) baseline, then
                # re-standardize the residual to unit scale so the MINE/InfoNCE
                # input stays well-conditioned (affine -> MI-invariant).
                resid = _crn_residualize_reward(
                    subset["reward"], subset["prompt_id"], data["seed_id"][mask]
                )
                rstd = float(resid.std())
                if rstd > 1e-8:
                    resid = (resid / rstd).astype(np.float32)
                subset["reward"] = resid

            ch = _run_channel(
                subset, controls, n_prompts, args, seed_offset=seed_off,
                test_frac=test_frac, clamp_corrected=clamp_corrected,
            )
            row = {
                "Step": int(k),
                "Channel": cname,
                "Control": ",".join(controls),
                "MI": ch["MI"],
                "MI_corrected": ch["MI_corrected"],
                "MI_median": ch["MI_median"],
                "normalized_MI": ch["normalized_MI"],
                "mi_explained_var": ch.get("mi_explained_var", _mi_explained_var(ch["MI_corrected"])),
                "Null_MI": ch["Null_MI"],
                "Null_std": ch.get("Null_std", 0.0),
                "AUROC": ch["AUROC"],
                "Restart_std": ch["Restart_std"],
                "report_split": ch.get("report_split", "val"),
                "MI_ci_lo": ch.get("MI_ci_lo", float("nan")),
                "MI_ci_hi": ch.get("MI_ci_hi", float("nan")),
            }
            all_rows.append(row)
            report["steps"].append({"step": int(k), "channel": cname, "rows_used": ch["rows_used"], **ch})
            ci_lo = ch.get("MI_ci_lo", float("nan"))
            ci_hi = ch.get("MI_ci_hi", float("nan"))
            ci_str = (
                f" CI=[{ci_lo:+.4f},{ci_hi:+.4f}]"
                if (ci_lo == ci_lo and ci_hi == ci_hi)  # not NaN
                else ""
            )
            rho2 = ch.get("mi_explained_var", _mi_explained_var(ch["MI_corrected"]))
            print(
                f"[per_step] step={int(k):2d} {cname:8s} MIc={ch['MI_corrected']:.4f} nats "
                f"rho2={rho2:.4f} (MI_raw={ch['MI']:.4f} MI/H={ch['normalized_MI']:.4f}) "
                f"Null={ch['Null_MI']:.4f}±{ch.get('Null_std', 0.0):.4f} "
                f"AUROC={ch['AUROC']:.4f} std={ch['Restart_std']:.4f}{ci_str}"
            )

    table_csv = str(Path(args.per_step_table_csv).expanduser().resolve())
    os.makedirs(str(Path(table_csv).parent), exist_ok=True)
    with open(table_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Step", "Channel", "Control", "MI", "MI_corrected", "MI_median",
                "normalized_MI", "mi_explained_var", "Null_MI", "Null_std", "AUROC",
                "Restart_std", "report_split", "MI_ci_lo", "MI_ci_hi",
            ],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    report_json = str(Path(args.per_step_report_json).expanduser().resolve())
    os.makedirs(str(Path(report_json).parent), exist_ok=True)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[per_step] wrote table: {table_csv}")
    print(f"[per_step] wrote report: {report_json}")
    if bool(getattr(args, "per_step_plot_png", "")):
        try:
            plot_per_step_mi(args, all_rows)
        except Exception as exc:  # plotting must never break the numeric run
            print(f"[per_step] plot skipped ({type(exc).__name__}: {exc})")


def _rows_from_table_csv(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out: dict[str, Any] = {"Step": int(row["Step"]), "Channel": str(row["Channel"])}
            for key in ("MI", "MI_corrected", "MI_median", "normalized_MI",
                        "mi_explained_var", "Null_MI", "Null_std", "AUROC", "Restart_std"):
                try:
                    out[key] = float(row.get(key, "nan"))
                except (TypeError, ValueError):
                    out[key] = float("nan")
            rows.append(out)
    return rows


def plot_per_step_mi(args: argparse.Namespace, rows: list[dict[str, Any]] | None = None) -> None:
    """Per-step MI curve: I(channel; R | P) (raw nats, debiased) vs step k, one
    line per channel, with a shaded restart/null band and y=0 drawn so a flat-zero
    (e.g. deterministic-backbone noise) channel is visible. Secondary panel shows
    the explained-variance equivalent rho^2 = 1 - exp(-2I) (cardinality-free,
    decompose-comparable). Single-step intervention; all-default continuation;
    reward scored against c0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if rows is None:
        rows = _rows_from_table_csv(str(Path(args.per_step_table_csv).expanduser().resolve()))
    if not rows:
        print("[per_step] plot: no rows to plot.")
        return

    channels = list(dict.fromkeys(str(r["Channel"]) for r in rows))
    steps = sorted({int(r["Step"]) for r in rows})

    def _series(cname: str, field: str) -> list[float]:
        by_step = {int(r["Step"]): float(r.get(field, float("nan"))) for r in rows if str(r["Channel"]) == cname}
        return [by_step.get(s, float("nan")) for s in steps]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for cname in channels:
        mi = np.asarray(_series(cname, "MI_corrected"), dtype=np.float64)
        band = np.asarray(_series(cname, "Restart_std"), dtype=np.float64)
        null_band = np.asarray(_series(cname, "Null_std"), dtype=np.float64)
        total_band = np.sqrt(np.nan_to_num(band) ** 2 + np.nan_to_num(null_band) ** 2)
        line = ax0.plot(steps, mi, marker="o", label=cname)[0]
        ax0.fill_between(steps, mi - total_band, mi + total_band, alpha=0.18, color=line.get_color())
        rho2 = np.asarray(_series(cname, "mi_explained_var"), dtype=np.float64)
        if not np.any(np.isfinite(rho2)):  # backward-compat with pre-rho2 tables
            rho2 = 1.0 - np.exp(-2.0 * np.maximum(0.0, mi))
        ax1.plot(steps, rho2, marker="o", label=cname)
    ax0.axhline(0.0, color="k", lw=0.8, ls="--")
    ax0.set_ylabel("I(channel; R | P)  [nats, debiased]")
    ax0.set_title("Per-step conditional MI (single-step intervention, all-default continuation, scored vs c0)")
    ax0.legend(fontsize=8)
    ax1.axhline(0.0, color="k", lw=0.8, ls="--")
    ax1.set_ylabel("explained var  rho^2 = 1 - exp(-2I)")
    ax1.set_xlabel("denoising step k")
    ax1.set_xticks(steps)
    ax1.legend(fontsize=8)
    fig.tight_layout()

    out_png = str(Path(args.per_step_plot_png).expanduser().resolve())
    os.makedirs(str(Path(out_png).parent), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[per_step] wrote plot: {out_png}")


def _synth_channel(
    n_prompts: int, k_levels: int, reps: int, signal: float, seed: int
) -> dict[str, np.ndarray]:
    """Synthetic (prompt_id, control, reward) block for the calibration guard.
    reward = prompt_offset + signal * centered(control) + noise. signal=0 gives a
    control independent of reward given P (-> debiased MI must sit in the null
    band of 0); signal>0 gives a control that genuinely drives reward (-> clearly
    positive MI)."""
    rng = np.random.default_rng(seed)
    prompt_ids: list[int] = []
    controls: list[int] = []
    rewards: list[float] = []
    for p in range(n_prompts):
        offset = float(rng.normal(0.0, 1.0))
        for c in range(k_levels):
            c_centered = (float(c) - (k_levels - 1) / 2.0)
            for _ in range(reps):
                prompt_ids.append(p)
                controls.append(c)
                rewards.append(offset + signal * c_centered + float(rng.normal(0.0, 1.0)))
    arr = {
        "prompt_id": np.asarray(prompt_ids, dtype=np.int64),
        "control_id": np.asarray(controls, dtype=np.int64),
        "reward": np.asarray(rewards, dtype=np.float32),
    }
    _standardize_reward(arr, tag=f"calib_signal={signal}")
    return arr


def run_per_step_calibrate(args: argparse.Namespace) -> None:
    """Calibration guard for the estimator + null-debias + split, BEFORE burning
    GPU on the full grid. random control (signal=0) must give mi_corrected within
    its band of 0; a reward-correlated control (signal>0) must give clearly
    positive MI. If random-control MI is not ~0, the null debias or the split is
    wrong -- fix that first."""
    n_prompts = max(2, int(args.n_prompts) if int(args.n_prompts) > 0 else 50)
    k_levels = max(2, int(args.n_rewrites) + 1)
    reps = max(2, int(args.n_seeds))
    test_frac = float(args.per_step_test_frac)
    clamp_corrected = not bool(args.per_step_unclamped)

    cases = [("random(signal=0)", 0.0), ("correlated(signal=0.8)", 0.8)]
    results: list[dict[str, Any]] = []
    for off, (name, signal) in enumerate(cases):
        arr = _synth_channel(n_prompts, k_levels, reps, signal, seed=int(args.mine_seed) + 7 * off)
        ch = _run_channel(
            arr, ["control_id"], n_prompts, args, seed_offset=100 + off,
            test_frac=test_frac, clamp_corrected=clamp_corrected,
        )
        band = float(ch["Restart_std"]) + float(ch.get("Null_std", 0.0))
        within_null = abs(float(ch["MI_corrected"])) <= 2.0 * band + 1e-3
        results.append({"case": name, "signal": signal, "within_null_band": within_null, **ch})
        print(
            f"[calibrate] {name:24s} MIc={ch['MI_corrected']:+.4f} "
            f"MI={ch['MI']:.4f} Null={ch['Null_MI']:.4f}±{ch.get('Null_std', 0.0):.4f} "
            f"band={band:.4f} AUROC={ch['AUROC']:.4f}"
        )

    rnd = next(r for r in results if r["signal"] == 0.0)
    cor = next(r for r in results if r["signal"] > 0.0)
    rnd_ok = bool(rnd["within_null_band"])
    cor_ok = bool(float(cor["MI_corrected"]) > 3.0 * (float(cor["Restart_std"]) + float(cor.get("Null_std", 0.0)) + 1e-3))
    print(f"[calibrate] random-control ~0: {'PASS' if rnd_ok else 'FAIL'}; "
          f"correlated-control >0: {'PASS' if cor_ok else 'FAIL'}")
    if not (rnd_ok and cor_ok):
        raise SystemExit("[calibrate] FAILED: estimator/null/split miscalibrated -- do not scale to the full grid.")
    print("[calibrate] PASS: estimator calibrated; safe to scale.")


def _prompt_factorial(arr: dict[str, Any], pid: int) -> np.ndarray | None:
    """Balanced seed×variant×cfg reward tensor for one prompt.

    Returns a (ns, nv, nc) array, or None if the prompt's grid is not a
    complete factorial (missing or duplicate cells -> skip the prompt).
    """
    mask = arr["prompt_id"] == int(pid)
    s = arr["seed_id"][mask]
    v = arr["variant_id"][mask]
    c = arr["cfg_id"][mask]
    r = arr["reward"][mask]
    s_levels = np.unique(s)
    v_levels = np.unique(v)
    c_levels = np.unique(c)
    ns, nv, nc = int(s_levels.size), int(v_levels.size), int(c_levels.size)
    if int(r.size) != ns * nv * nc:
        return None
    s_idx = {int(x): i for i, x in enumerate(s_levels)}
    v_idx = {int(x): i for i, x in enumerate(v_levels)}
    c_idx = {int(x): i for i, x in enumerate(c_levels)}
    grid = np.full((ns, nv, nc), np.nan, dtype=np.float64)
    for si, vi, ci, ri in zip(s.tolist(), v.tolist(), c.tolist(), r.tolist()):
        grid[s_idx[int(si)], v_idx[int(vi)], c_idx[int(ci)]] = float(ri)
    if np.isnan(grid).any():
        return None
    return grid


def _decompose_one(grid: np.ndarray) -> dict[str, float]:
    """Balanced two-way (variant×cfg) ANOVA with seeds as replicates.

    The reward at a fixed (variant, cfg) cell varies only with the seed, so the
    within-cell sum of squares is the *seed-driven* (per-instance noise) variance
    that cross-prompt conditional MI cannot see. The between-cell variance splits
    into cfg, variant, and variant×cfg systematic effects. eta^2 are SS fractions
    of the per-prompt total (sum to 1 up to float clamping of the interaction).
    """
    ns, nv, nc = grid.shape
    n_cells = int(grid.size)
    grand = float(grid.mean())
    ss_total = float(((grid - grand) ** 2).sum())

    m_vc = grid.mean(axis=0)                       # (nv, nc) cell means over seeds
    ss_seed = float(((grid - m_vc[None, :, :]) ** 2).sum())  # within-cell, seed-driven
    ss_systematic = ss_total - ss_seed
    m_v = m_vc.mean(axis=1)                         # variant marginal
    m_c = m_vc.mean(axis=0)                         # cfg marginal
    ss_variant = float(ns * nc * ((m_v - grand) ** 2).sum())
    ss_cfg = float(ns * nv * ((m_c - grand) ** 2).sum())
    ss_inter = max(0.0, float(ss_systematic - ss_variant - ss_cfg))

    out = {
        "ss_total": ss_total,
        "ss_seed": ss_seed,
        "ss_variant": ss_variant,
        "ss_cfg": ss_cfg,
        "ss_variant_x_cfg": ss_inter,
        "sd_total": math.sqrt(ss_total / max(1, n_cells)),
        "sd_seed": math.sqrt(ss_seed / max(1, n_cells)),
        "dims": [int(ns), int(nv), int(nc)],
    }
    denom = ss_total if ss_total > 0 else 1.0
    out["eta2_seed"] = ss_seed / denom if ss_total > 0 else 0.0
    out["eta2_variant"] = ss_variant / denom if ss_total > 0 else 0.0
    out["eta2_cfg"] = ss_cfg / denom if ss_total > 0 else 0.0
    out["eta2_variant_x_cfg"] = ss_inter / denom if ss_total > 0 else 0.0
    return out


def run_decompose(args: argparse.Namespace) -> None:
    """Prompt-stratified variance decomposition (eta^2 per control axis).

    Assumption-free alternative to the MINE/InfoNCE critics: each prompt is its
    own stratum (prompt treated as a random effect; per-prompt eta^2 is
    scale-invariant, so within-prompt z-scoring is implicit and unnecessary).
    Reports mean ± SEM of eta^2 across prompts. Cannot diverge.
    """
    data = load_dataset(args.dataset_csv)
    pids = _safe_unique_sorted(data["prompt_id"])

    axes = ["seed", "cfg", "variant", "variant_x_cfg"]
    label = {
        "seed": "Seed (within-cell)",
        "cfg": "CFG",
        "variant": "Prompt variant",
        "variant_x_cfg": "Variant×CFG",
    }
    per_prompt: dict[str, list[float]] = {a: [] for a in axes}
    sd_total: list[float] = []
    sd_seed: list[float] = []
    used = 0
    skipped: list[int] = []
    dims_seen: set[tuple[int, int, int]] = set()

    # First pass: find the modal (most common) complete-factorial grid shape and
    # decompose only prompts matching it, so the design is balanced across prompts
    # and degenerate single-cell prompts don't inject structural-zero eta^2.
    grids: dict[int, np.ndarray] = {}
    shape_counts: dict[tuple[int, int, int], int] = {}
    for pid in pids:
        grid = _prompt_factorial(data, int(pid))
        if grid is None:
            skipped.append(int(pid))
            continue
        grids[int(pid)] = grid
        shape_counts[grid.shape] = shape_counts.get(grid.shape, 0) + 1
    if not shape_counts:
        modal_shape = None
    else:
        modal_shape = max(shape_counts, key=lambda s: (shape_counts[s], s[0] * s[1] * s[2]))

    for pid in pids:
        grid = grids.get(int(pid))
        if grid is None or grid.shape != modal_shape:
            if int(pid) not in skipped:
                skipped.append(int(pid))
            continue
        d = _decompose_one(grid)
        used += 1
        dims_seen.add(tuple(d["dims"]))
        per_prompt["seed"].append(d["eta2_seed"])
        per_prompt["cfg"].append(d["eta2_cfg"])
        per_prompt["variant"].append(d["eta2_variant"])
        per_prompt["variant_x_cfg"].append(d["eta2_variant_x_cfg"])
        sd_total.append(d["sd_total"])
        sd_seed.append(d["sd_seed"])

    if used == 0:
        raise RuntimeError(
            "No complete-factorial prompts found for decomposition "
            f"(checked {len(pids)} prompts). The dataset may be partial/unbalanced."
        )

    def _mean_sem(xs: list[float]) -> tuple[float, float, float]:
        a = np.asarray(xs, dtype=np.float64)
        m = float(a.mean())
        sem = float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else 0.0
        return m, sem, float(np.median(a))

    rows: list[dict[str, Any]] = []
    report_axes: dict[str, Any] = {}
    for a in axes:
        m, sem, med = _mean_sem(per_prompt[a])
        rows.append({
            "Axis": label[a],
            "eta2_mean": round(m, 6),
            "eta2_sem": round(sem, 6),
            "eta2_median": round(med, 6),
        })
        report_axes[a] = {"label": label[a], "eta2_mean": m, "eta2_sem": sem, "eta2_median": med}

    sdt_m, sdt_sem, _ = _mean_sem(sd_total)
    sds_m, sds_sem, _ = _mean_sem(sd_seed)

    table_csv = str(Path(args.decompose_table_csv).expanduser().resolve())
    os.makedirs(str(Path(table_csv).parent), exist_ok=True)
    with open(table_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Axis", "eta2_mean", "eta2_sem", "eta2_median"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    report = {
        "kind": "variance_decomposition",
        "note": (
            "Prompt-stratified balanced ANOVA (variant×cfg with seeds as "
            "replicates). eta^2 are per-prompt SS fractions averaged across "
            "prompts; 'Seed (within-cell)' is the per-instance noise effect that "
            "cross-prompt conditional MI cannot detect."
        ),
        "dataset_csv": str(Path(args.dataset_csv).expanduser().resolve()),
        "n_prompts_total": int(len(pids)),
        "n_prompts_used": int(used),
        "n_prompts_skipped": int(len(skipped)),
        "skipped_prompt_ids": [int(x) for x in skipped[:50]],
        "modal_grid_dims": list(modal_shape) if modal_shape is not None else None,
        "grid_dims_seen": sorted([list(d) for d in dims_seen]),
        "axes": report_axes,
        "sd_total_reward_mean": sdt_m,
        "sd_total_reward_sem": sdt_sem,
        "sd_seed_reward_mean": sds_m,
        "sd_seed_reward_sem": sds_sem,
    }
    report_json = str(Path(args.decompose_report_json).expanduser().resolve())
    os.makedirs(str(Path(report_json).parent), exist_ok=True)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"[decompose] prompts used={used}/{len(pids)} skipped={len(skipped)} "
        f"grid_dims(s,v,c)={sorted([list(d) for d in dims_seen])}"
    )
    for a in axes:
        info = report_axes[a]
        print(
            f"[decompose] {info['label']:20s} eta^2 = {info['eta2_mean']:.4f} "
            f"± {info['eta2_sem']:.4f}  (median {info['eta2_median']:.4f})"
        )
    print(
        f"[decompose] reward SD: total={sdt_m:.4f}±{sdt_sem:.4f}  "
        f"seed-only(within-cell)={sds_m:.4f}±{sds_sem:.4f}"
    )
    print(f"[decompose] wrote table: {table_csv}")
    print(f"[decompose] wrote report: {report_json}")


def _perstep_factorial(
    data: dict[str, np.ndarray], step: int, pid: int, grid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Balanced seed×variant×cfg reward tensor for one (step, prompt) cell.

    Reads only the "grid"-channel one-step-deviation rows at the probed step:
    every other step is held at baseline, so the variance across this tensor is
    attributable purely to the action injected at step k on a shared root.
    Returns (grid, v_levels, c_levels) with grid of shape (ns, nv, nc), or None
    if the cell is not a complete factorial. The level arrays give the
    variant_id / cfg_id along the grid's axes (needed to locate the baseline
    action cell for the paired CRN-delta test).
    """
    mask = grid_mask & (data["step_idx"] == int(step)) & (data["prompt_id"] == int(pid))
    s = data["seed_id"][mask]
    v = data["variant_id"][mask]
    c = data["cfg_id"][mask]
    r = data["reward"][mask]
    s_levels = np.unique(s)
    v_levels = np.unique(v)
    c_levels = np.unique(c)
    ns, nv, nc = int(s_levels.size), int(v_levels.size), int(c_levels.size)
    if int(r.size) != ns * nv * nc or ns == 0 or nv == 0 or nc == 0:
        return None
    s_idx = {int(x): i for i, x in enumerate(s_levels)}
    v_idx = {int(x): i for i, x in enumerate(v_levels)}
    c_idx = {int(x): i for i, x in enumerate(c_levels)}
    grid = np.full((ns, nv, nc), np.nan, dtype=np.float64)
    for si, vi, ci, ri in zip(s.tolist(), v.tolist(), c.tolist(), r.tolist()):
        grid[s_idx[int(si)], v_idx[int(vi)], c_idx[int(ci)]] = float(ri)
    if np.isnan(grid).any():
        return None
    return grid, v_levels, c_levels


def _paired_delta_stats(
    delta_rows: list[np.ndarray],
    baseline_flat_idx: int | None,
    rng: np.random.Generator,
    n_splits: int = 40,
) -> dict[str, float]:
    """Paired CRN-delta statistics at one step.

    Each row of ``delta_rows`` is a (prompt, seed) pair's reward delta vector
    R(action) - R(baseline action), flattened over the (variant×cfg) action
    cells. Because both terms share the same root noise (CRN), the per-instance
    seed noise differences out, so a small consistent action effect that the
    unpaired MI/variance views drown in seed variance becomes detectable.

      - best_delta / best_t : the single non-baseline action with the largest
        mean paired delta, and its one-sample paired t-statistic (in-sample
        selection, so best_delta is mildly optimistic -- read it with best_t).
      - heldout_gain : winner's-curse-free reachable gain. Repeatedly split the
        pairs in half, pick the best action on one half, evaluate its mean delta
        on the other; average. Negative -> no action reliably beats baseline.
      - frac_sig : fraction of non-baseline action cells with |paired t| > 2.
    All values are in z (reward-SD) units; the caller rescales to raw.
    """
    out = {"best_delta": 0.0, "best_t": 0.0, "heldout_gain": 0.0, "frac_sig": 0.0, "n_pairs": 0}
    if not delta_rows or baseline_flat_idx is None:
        return out
    D = np.concatenate(delta_rows, axis=0)
    n, ncells = D.shape
    out["n_pairs"] = int(n)
    if n < 2 or ncells < 2:
        return out
    mean_d = D.mean(axis=0)
    se = D.std(axis=0, ddof=1) / math.sqrt(n)
    t = np.divide(mean_d, se, out=np.zeros_like(mean_d), where=se > 1e-12)
    nb = np.ones(ncells, dtype=bool)
    nb[int(baseline_flat_idx)] = False
    nb_mean = mean_d[nb]
    nb_t = t[nb]
    best_local = int(np.argmax(nb_mean))
    out["best_delta"] = float(nb_mean[best_local])
    out["best_t"] = float(nb_t[best_local])
    out["frac_sig"] = float(np.mean(np.abs(nb_t) > 2.0))
    gains: list[float] = []
    h = n // 2
    if h >= 1:
        for _ in range(int(n_splits)):
            perm = rng.permutation(n)
            a, b = perm[:h], perm[h:]
            ma = D[a].mean(axis=0).copy()
            mb = D[b].mean(axis=0)
            ma[int(baseline_flat_idx)] = -np.inf  # never select baseline
            gains.append(float(mb[int(np.argmax(ma))]))
            mb_sel = mb.copy()
            mb_sel[int(baseline_flat_idx)] = -np.inf
            gains.append(float(ma[int(np.argmax(mb_sel))]) if np.isfinite(ma[int(np.argmax(mb_sel))]) else 0.0)
    out["heldout_gain"] = float(np.mean(gains)) if gains else 0.0
    return out


def run_per_step_decompose(args: argparse.Namespace) -> None:
    """Per-step variance-based effect sizes on the existing one-step-deviation data.

    An assumption-free complement to the per-step MINE/InfoNCE critics, which are
    biased low in nats and read near-zero here. For each step k we read the
    "grid"-channel rows (variant×cfg factorial swept at step k on a shared root,
    all other steps at baseline) and report, per step:
      - Sobol first-order indices (eta^2 SS fractions) for variant / cfg /
        variant×cfg / seed-noise, via the same balanced two-way ANOVA as
        `decompose`; these are scale-invariant and cannot diverge.
      - reachable_gap = best minus worst action at step k after averaging over
        seeds FIRST (cancels per-instance noise) -> the expected-reward gain from
        always steering to the best action; the honest, noise-debiased gap.
      - oracle_gap = mean over seeds of (max_a R - min_a R): the clairvoyant
        per-seed spread. Inflated by seed noise (max-of-noise bias), so kept only
        as an upper bound next to reachable_gap.
      - gap_snr = reachable_gap / seed-noise SD: reachable gain in units of the
        irreducible per-instance noise.
      - cfg slope = dR/dcfg at step k (baseline variant), least-squares.
    Reward is z-scored on load, so *_z columns are in SD units; *_raw columns
    multiply back by the stored reward SD for natural reward units.
    """
    data = load_per_step_dataset(args.per_step_dataset_csv)
    reward_std = float(data.get("reward_std", np.float32(1.0)))
    chan_col = data.get("channel")
    grid_mask = (
        (chan_col == "grid") if chan_col is not None
        else np.ones_like(data["prompt_id"], dtype=bool)
    )
    base_variant = int(args.per_step_baseline_variant)
    pids = _safe_unique_sorted(data["prompt_id"])
    step_ids = _safe_unique_sorted(data["step_idx"])

    # Baseline cfg_id = the swept cfg level whose value is nearest the default
    # cfg held on non-branched steps. Defines the reference action cell that the
    # paired CRN-delta test differences against.
    cfg_id_to_val: dict[int, float] = {}
    for cid, cv in zip(data["cfg_id"].tolist(), data["cfg_value"].tolist()):
        cfg_id_to_val.setdefault(int(cid), float(cv))
    default_cfg_id = (
        min(cfg_id_to_val, key=lambda cid: abs(cfg_id_to_val[cid] - float(args.default_cfg)))
        if cfg_id_to_val else 0
    )
    rng = np.random.default_rng(20260620)

    def _mean_sem(xs: list[float]) -> tuple[float, float, float]:
        a = np.asarray(xs, dtype=np.float64)
        if a.size == 0:
            return 0.0, 0.0, 0.0
        m = float(a.mean())
        sem = float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else 0.0
        return m, sem, float(np.median(a))

    all_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "kind": "per_step_variance_decomposition",
        "note": (
            "Per-step one-step-deviation effect sizes. eta^2 are per-prompt "
            "balanced-ANOVA SS fractions (Sobol first-order indices) averaged "
            "across prompts; reachable_gap is the noise-debiased best-vs-worst "
            "action reward spread at step k (seed-averaged); paired_* are CRN "
            "paired-delta statistics (R(action)-R(baseline) per shared root), "
            "with heldout_gain the winner's-curse-free reachable gain; cfg_slope "
            "is dR/dcfg at step k. *_z in reward-SD units, *_raw rescaled by SD."
        ),
        "per_step_dataset_csv": str(Path(args.per_step_dataset_csv).expanduser().resolve()),
        "reward_std": reward_std,
        "baseline_variant": base_variant,
        "steps": [],
    }

    for k in step_ids:
        # eta^2 over modal complete-factorial prompts (balanced design)
        grids: dict[int, np.ndarray] = {}
        levels: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        shape_counts: dict[tuple[int, int, int], int] = {}
        for pid in pids:
            res = _perstep_factorial(data, int(k), int(pid), grid_mask)
            if res is None:
                continue
            g, v_levels, c_levels = res
            grids[int(pid)] = g
            levels[int(pid)] = (v_levels, c_levels)
            shape_counts[g.shape] = shape_counts.get(g.shape, 0) + 1
        modal_shape = (
            max(shape_counts, key=lambda s: (shape_counts[s], s[0] * s[1] * s[2]))
            if shape_counts else None
        )

        eta2 = {a: [] for a in ("seed", "cfg", "variant", "variant_x_cfg")}
        gaps_z: list[float] = []
        gaps_sys_z: list[float] = []
        snrs: list[float] = []
        slopes_z: list[float] = []
        delta_rows: list[np.ndarray] = []
        baseline_flat_idx: int | None = None
        used = 0
        for pid in pids:
            g = grids.get(int(pid))
            if g is None or g.shape != modal_shape:
                continue
            d = _decompose_one(g)
            eta2["seed"].append(d["eta2_seed"])
            eta2["cfg"].append(d["eta2_cfg"])
            eta2["variant"].append(d["eta2_variant"])
            eta2["variant_x_cfg"].append(d["eta2_variant_x_cfg"])
            # Clairvoyant per-seed gap: best minus worst action within each seed.
            # Inflated by seed noise (max-of-noise bias), so report it only as an
            # upper bound alongside the debiased systematic gap below.
            per_seed = g.reshape(g.shape[0], -1)
            gaps_z.append(float((per_seed.max(axis=1) - per_seed.min(axis=1)).mean()))
            # Debiased reachable gap: average over seeds FIRST (cancels per-instance
            # noise), then best minus worst action cell mean -> the expected-reward
            # improvement from always steering to the best step-k action.
            cell_means = g.mean(axis=0).reshape(-1)
            gap_sys = float(cell_means.max() - cell_means.min())
            gaps_sys_z.append(gap_sys)
            sd_seed = float(d.get("sd_seed", 0.0))
            if sd_seed > 1e-8:
                snrs.append(gap_sys / sd_seed)
            # Paired CRN-delta rows: R(action) - R(baseline action) per seed,
            # differencing out the shared root noise.
            v_lv, c_lv = levels[int(pid)]
            bvi = np.where(v_lv == base_variant)[0]
            bci = np.where(c_lv == default_cfg_id)[0]
            if bvi.size and bci.size:
                nc = g.shape[2]
                base_col = g[:, int(bvi[0]), int(bci[0])]
                delta_rows.append((g - base_col[:, None, None]).reshape(g.shape[0], -1))
                baseline_flat_idx = int(bvi[0]) * nc + int(bci[0])
            used += 1

        paired = _paired_delta_stats(delta_rows, baseline_flat_idx, rng)

        # cfg slope at baseline variant: dR/dcfg per (prompt, seed), least-squares.
        slope_mask = (
            grid_mask
            & (data["step_idx"] == int(k))
            & (data["variant_id"] == base_variant)
        )
        sp = data["prompt_id"][slope_mask]
        ss = data["seed_id"][slope_mask]
        scfg = data["cfg_value"][slope_mask].astype(np.float64)
        sr = data["reward"][slope_mask].astype(np.float64)
        for pid in pids:
            pm = sp == int(pid)
            if not bool(pm.any()):
                continue
            seeds_here = np.unique(ss[pm])
            for sd_ in seeds_here:
                sm = pm & (ss == int(sd_))
                x = scfg[sm]
                y = sr[sm]
                if x.size >= 2 and float(np.ptp(x)) > 1e-8:
                    slopes_z.append(float(np.polyfit(x, y, 1)[0]))

        ev_m, ev_sem, _ = _mean_sem(eta2["variant"])
        ec_m, ec_sem, _ = _mean_sem(eta2["cfg"])
        ei_m, ei_sem, _ = _mean_sem(eta2["variant_x_cfg"])
        es_m, es_sem, _ = _mean_sem(eta2["seed"])
        gap_m, gap_sem, _ = _mean_sem(gaps_z)
        gsys_m, gsys_sem, _ = _mean_sem(gaps_sys_z)
        snr_m, snr_sem, _ = _mean_sem(snrs)
        slope_m, slope_sem, _ = _mean_sem(slopes_z)

        row = {
            "Step": int(k),
            "n_prompts_used": int(used),
            "grid_dims": "x".join(str(d) for d in modal_shape) if modal_shape else "",
            "eta2_variant": round(ev_m, 6),
            "eta2_cfg": round(ec_m, 6),
            "eta2_variant_x_cfg": round(ei_m, 6),
            "eta2_seed": round(es_m, 6),
            "oracle_gap_z": round(gap_m, 6),
            "oracle_gap_raw": round(gap_m * reward_std, 6),
            "reachable_gap_z": round(gsys_m, 6),
            "reachable_gap_raw": round(gsys_m * reward_std, 6),
            "gap_snr": round(snr_m, 6),
            "paired_best_delta_z": round(paired["best_delta"], 6),
            "paired_best_delta_raw": round(paired["best_delta"] * reward_std, 6),
            "paired_best_t": round(paired["best_t"], 4),
            "paired_heldout_gain_z": round(paired["heldout_gain"], 6),
            "paired_heldout_gain_raw": round(paired["heldout_gain"] * reward_std, 6),
            "paired_frac_sig": round(paired["frac_sig"], 4),
            "paired_n_pairs": int(paired["n_pairs"]),
            "cfg_slope_z": round(slope_m, 6),
            "cfg_slope_raw": round(slope_m * reward_std, 6),
        }
        all_rows.append(row)
        report["steps"].append({
            "step": int(k),
            "n_prompts_used": int(used),
            "grid_dims": list(modal_shape) if modal_shape else None,
            "eta2_variant": {"mean": ev_m, "sem": ev_sem},
            "eta2_cfg": {"mean": ec_m, "sem": ec_sem},
            "eta2_variant_x_cfg": {"mean": ei_m, "sem": ei_sem},
            "eta2_seed": {"mean": es_m, "sem": es_sem},
            "oracle_gap_z": {"mean": gap_m, "sem": gap_sem},
            "oracle_gap_raw": {"mean": gap_m * reward_std, "sem": gap_sem * reward_std},
            "reachable_gap_z": {"mean": gsys_m, "sem": gsys_sem},
            "reachable_gap_raw": {"mean": gsys_m * reward_std, "sem": gsys_sem * reward_std},
            "gap_snr": {"mean": snr_m, "sem": snr_sem},
            "paired": {
                "best_delta_z": paired["best_delta"],
                "best_delta_raw": paired["best_delta"] * reward_std,
                "best_t": paired["best_t"],
                "heldout_gain_z": paired["heldout_gain"],
                "heldout_gain_raw": paired["heldout_gain"] * reward_std,
                "frac_sig": paired["frac_sig"],
                "n_pairs": paired["n_pairs"],
            },
            "cfg_slope_z": {"mean": slope_m, "sem": slope_sem},
            "cfg_slope_raw": {"mean": slope_m * reward_std, "sem": slope_sem * reward_std},
        })
        print(
            f"[per_step_decompose] step={int(k):2d} used={used:3d} "
            f"eta2(var={ev_m:.4f} cfg={ec_m:.4f} vxc={ei_m:.4f} seed={es_m:.4f}) "
            f"reach_gap_raw={gsys_m * reward_std:.4f} snr={snr_m:.4f} "
            f"paired(best_raw={paired['best_delta'] * reward_std:.4f} t={paired['best_t']:.2f} "
            f"heldout_raw={paired['heldout_gain'] * reward_std:.4f} sig={paired['frac_sig']:.2f}) "
            f"cfg_slope_raw={slope_m * reward_std:.4f}"
        )

    table_csv = str(Path(args.per_step_decompose_table_csv).expanduser().resolve())
    os.makedirs(str(Path(table_csv).parent), exist_ok=True)
    with open(table_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Step", "n_prompts_used", "grid_dims",
                "eta2_variant", "eta2_cfg", "eta2_variant_x_cfg", "eta2_seed",
                "oracle_gap_z", "oracle_gap_raw",
                "reachable_gap_z", "reachable_gap_raw", "gap_snr",
                "paired_best_delta_z", "paired_best_delta_raw", "paired_best_t",
                "paired_heldout_gain_z", "paired_heldout_gain_raw",
                "paired_frac_sig", "paired_n_pairs",
                "cfg_slope_z", "cfg_slope_raw",
            ],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    report_json = str(Path(args.per_step_decompose_report_json).expanduser().resolve())
    os.makedirs(str(Path(report_json).parent), exist_ok=True)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[per_step_decompose] wrote table: {table_csv}")
    print(f"[per_step_decompose] wrote report: {report_json}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode in {"generate", "full", "per_step_generate", "per_step"} and not args.prompt_file:
        raise SystemExit(f"--prompt_file is required for mode '{args.mode}'.")
    if args.mode in {"generate", "full"}:
        generate_dataset(args)
    if args.mode in {"train", "full"}:
        run_mine_critics(args)
    if args.mode in {"per_step_generate", "per_step"}:
        generate_dataset_per_step(args)
    if args.mode in {"per_step_train", "per_step"}:
        run_per_step_critics(args)
    if args.mode == "per_step_plot":
        plot_per_step_mi(args)
    if args.mode == "per_step_calibrate":
        run_per_step_calibrate(args)
    if args.mode == "decompose":
        run_decompose(args)
    if args.mode == "per_step_decompose":
        run_per_step_decompose(args)


if __name__ == "__main__":
    main()
