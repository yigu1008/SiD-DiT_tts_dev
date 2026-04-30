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
        choices=["generate", "train", "full"],
        default="full",
        help="generate: build dataset only, train: train critics from dataset only, full: do both.",
    )

    # Data generation knobs
    p.add_argument("--prompt_file", required=True, help="Text file with one original prompt per line.")
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
    return arr


def _split_indices(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    n_val = int(max(1, min(n - 1, round(n * val_frac))))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


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


def _dv_bound(
    model: MineCritic,
    reward: torch.Tensor,
    prompt_id: torch.Tensor,
    controls_pos: dict[str, torch.Tensor],
    controls_neg: dict[str, torch.Tensor],
) -> torch.Tensor:
    t_pos = model(reward, prompt_id, controls_pos)
    t_neg = model(reward, prompt_id, controls_neg)
    log_mean_exp_neg = torch.logsumexp(t_neg, dim=0) - math.log(float(t_neg.shape[0]))
    return t_pos.mean() - log_mean_exp_neg


def _estimate_mi_and_auroc(
    model: MineCritic,
    data_t: dict[str, torch.Tensor],
    control_names: list[str],
    batch_size: int,
    seed: int,
) -> tuple[float, float]:
    model.eval()
    rng = np.random.default_rng(seed)
    n = int(data_t["reward"].shape[0])
    if n <= 0:
        return 0.0, 0.5

    mi_vals: list[float] = []
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, max(1, batch_size)):
            end = min(n, start + max(1, batch_size))
            reward = data_t["reward"][start:end]
            prompt_id = data_t["prompt_id"][start:end]
            controls_pos = {k: data_t[k][start:end] for k in control_names}
            controls_neg = _permute_controls_in_batch(prompt_id, controls_pos, rng)

            t_pos = model(reward, prompt_id, controls_pos)
            t_neg = model(reward, prompt_id, controls_neg)
            log_mean_exp_neg = torch.logsumexp(t_neg, dim=0) - math.log(float(max(1, t_neg.shape[0])))
            mi_vals.append(float((t_pos.mean() - log_mean_exp_neg).item()))

            s_pos = t_pos.detach().cpu().numpy()
            s_neg = t_neg.detach().cpu().numpy()
            scores = np.concatenate([s_pos, s_neg], axis=0)
            labels = np.concatenate(
                [
                    np.ones_like(s_pos, dtype=np.int64),
                    np.zeros_like(s_neg, dtype=np.int64),
                ],
                axis=0,
            )
            all_scores.append(scores)
            all_labels.append(labels)

    mi = float(np.mean(mi_vals)) if mi_vals else 0.0
    auc = _auroc_from_scores(
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_scores, axis=0),
    )
    return mi, auc


def _train_one_restart(
    train_np: dict[str, np.ndarray],
    val_np: dict[str, np.ndarray],
    control_names: list[str],
    n_prompts: int,
    args: argparse.Namespace,
    restart_seed: int,
) -> tuple[float, float, float]:
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

    train_t = _to_tensors(train_np, device=device)
    val_t = _to_tensors(val_np, device=device)
    n_train = int(train_t["reward"].shape[0])
    bs = max(1, int(args.mine_batch_size))
    best_val = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for step in range(1, int(args.mine_steps) + 1):
        model.train()
        idx = rng.integers(0, n_train, size=bs, endpoint=False)
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
        reward = train_t["reward"].index_select(0, idx_t)
        prompt_id = train_t["prompt_id"].index_select(0, idx_t)
        controls_pos = {k: train_t[k].index_select(0, idx_t) for k in control_names}
        controls_neg = _permute_controls_in_batch(prompt_id, controls_pos, rng)
        dv = _dv_bound(model, reward, prompt_id, controls_pos, controls_neg)
        loss = -dv
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
            )
            if val_mi > best_val:
                best_val = float(val_mi)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    train_mi, _ = _estimate_mi_and_auroc(
        model, train_t, control_names, batch_size=bs, seed=restart_seed + 20001
    )
    val_mi, auroc = _estimate_mi_and_auroc(
        model, val_t, control_names, batch_size=bs, seed=restart_seed + 30001
    )
    return float(train_mi), float(val_mi), float(auroc)


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

    for spec_idx, spec in enumerate(specs):
        subset = _build_filtered_data(data, spec)
        n_rows_subset = int(subset["reward"].shape[0])
        if n_rows_subset < 8:
            raise RuntimeError(f"Too few rows for critic '{spec.channel}': {n_rows_subset}")

        # Null setup: permute controls (jointly) within prompt before training.
        controls_np = {c: subset[c] for c in spec.controls}
        null_controls = _permute_controls_globally_within_prompt(
            subset["prompt_id"], controls_np, seed=int(args.mine_seed) + 999 * (spec_idx + 1)
        )
        null_subset = {
            "prompt_id": subset["prompt_id"].copy(),
            "reward": subset["reward"].copy(),
            **{c: null_controls[c] for c in spec.controls},
        }

        # Train/val split fixed across restarts.
        train_idx, val_idx = _split_indices(
            n_rows_subset,
            float(args.mine_val_frac),
            seed=int(args.mine_seed) + 31 * (spec_idx + 1),
        )
        train_np = {"prompt_id": _subset_array(subset["prompt_id"], train_idx), "reward": _subset_array(subset["reward"], train_idx)}
        val_np = {"prompt_id": _subset_array(subset["prompt_id"], val_idx), "reward": _subset_array(subset["reward"], val_idx)}
        for c in spec.controls:
            train_np[c] = _subset_array(subset[c], train_idx)
            val_np[c] = _subset_array(subset[c], val_idx)

        null_train_np = {
            "prompt_id": _subset_array(null_subset["prompt_id"], train_idx),
            "reward": _subset_array(null_subset["reward"], train_idx),
        }
        null_val_np = {
            "prompt_id": _subset_array(null_subset["prompt_id"], val_idx),
            "reward": _subset_array(null_subset["reward"], val_idx),
        }
        for c in spec.controls:
            null_train_np[c] = _subset_array(null_subset[c], train_idx)
            null_val_np[c] = _subset_array(null_subset[c], val_idx)

        n_prompts = int(max(_max_or_zero(subset["prompt_id"]), _max_or_zero(data["prompt_id"])) + 1)

        train_mis: list[float] = []
        val_mis: list[float] = []
        aucs: list[float] = []
        null_val_mis: list[float] = []

        for r in range(int(args.mine_restarts)):
            restart_seed = int(args.mine_seed) + 1000 * (spec_idx + 1) + r
            tr_mi, va_mi, auroc = _train_one_restart(
                train_np, val_np, spec.controls, n_prompts, args, restart_seed=restart_seed
            )
            train_mis.append(float(tr_mi))
            val_mis.append(float(va_mi))
            aucs.append(float(auroc))

            _null_tr, null_va, _null_auc = _train_one_restart(
                null_train_np,
                null_val_np,
                spec.controls,
                n_prompts,
                args,
                restart_seed=restart_seed + 50000,
            )
            null_val_mis.append(float(null_va))

        # Entropy denominator H(control)
        h_control = _entropy_nats(_joint_code([subset[c] for c in spec.controls]))
        mi_mean = float(np.mean(val_mis))
        mi_norm = float(mi_mean / h_control) if h_control > 0 else 0.0
        row = {
            "Channel": spec.channel,
            "Control": ",".join(spec.controls),
            "MI": mi_mean,
            "normalized_MI": mi_norm,
            "Null_MI": float(np.mean(null_val_mis)),
            "AUROC": float(np.mean(aucs)),
            "Restart_std": float(np.std(val_mis)),
        }
        all_rows.append(row)

        report["specs"].append(
            {
                "channel": spec.channel,
                "controls": list(spec.controls),
                "rows_used": n_rows_subset,
                "train_mi_mean": float(np.mean(train_mis)),
                "val_mi_mean": float(np.mean(val_mis)),
                "val_mi_std": float(np.std(val_mis)),
                "null_val_mi_mean": float(np.mean(null_val_mis)),
                "auroc_mean": float(np.mean(aucs)),
                "h_control_nats": float(h_control),
                "mi_over_h_control": float(mi_norm),
                "restarts": int(args.mine_restarts),
            }
        )

        print(
            f"[train] {spec.channel:18s} MI={row['MI']:.4f} MI/H={row['normalized_MI']:.4f} "
            f"Null={row['Null_MI']:.4f} AUROC={row['AUROC']:.4f} std={row['Restart_std']:.4f}"
        )

    table_csv = str(Path(args.mine_table_csv).expanduser().resolve())
    os.makedirs(str(Path(table_csv).parent), exist_ok=True)
    with open(table_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Channel", "Control", "MI", "normalized_MI", "Null_MI", "AUROC", "Restart_std"],
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode in {"generate", "full"}:
        generate_dataset(args)
    if args.mode in {"train", "full"}:
        run_mine_critics(args)


if __name__ == "__main__":
    main()
