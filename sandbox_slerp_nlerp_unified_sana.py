"""
SANA SLERP/NLERP sandbox with UnifiedReward defaults.

This script focuses on two things:
  1) fixed blend-profile sweep over (family, profile, cfg)
  2) optional GA/MCTS search over per-step (family, profile, cfg) actions

GA here is intentionally simple (discrete action genomes) and reuses the
same rank/tournament selection style used elsewhere in this repo.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import sampling_unified as su
from blend_ops import blend_prompt_embeddings
from eval_and_log import ensure_dir, save_json
from ga_prompt_search import crossover_uniform, rank_select, tournament_select
from prompt_basis import build_prompt_basis, encode_prompt_basis, load_basis_cache, save_basis_cache
from rollout_runner import resolve_resolution, run_single_prompt_rollout
from weight_policy import progress_from_sigmas


@dataclass
class ActionRollout:
    final_score: float
    final_image: Image.Image
    preview_rewards: list[float]
    step_actions: list[dict[str, Any]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SANA SLERP/NLERP test with UnifiedReward defaults and optional GA/MCTS action search."
    )

    # I/O
    p.add_argument("--prompt", type=str, default="a studio portrait of an elderly woman smiling, soft window light, 85mm lens, photorealistic")
    p.add_argument("--prompt_file", type=str, default=None)
    p.add_argument("--max_prompts", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="./sandbox_slerp_nlerp_unified_sana_out")
    p.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_first_k", type=int, default=10)
    p.add_argument("--basis_cache_json", type=str, default=None)

    # Pipeline
    p.add_argument("--model_id", type=str, default="YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--neg_embed", type=str, default=None)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument("--gpu_id", type=int, default=-1)
    p.add_argument("--auto_select_gpu", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min_free_gb", type=float, default=12.0)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--resolution_binning", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--time_scale", type=float, default=1000.0)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--baseline_cfg", type=float, default=None)
    p.add_argument("--cfg_scales", nargs="+", type=float, default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5])
    p.add_argument("--seed", type=int, default=42)

    # Runtime/memory
    p.add_argument("--cuda_alloc_conf", type=str, default="expandable_segments:True")
    p.add_argument("--vae_slicing", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--vae_tiling", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--empty_cache_after_decode", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--offload_text_encoder_after_encode", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sana_no_fp32_attn", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--decode_device", type=str, default="auto")
    p.add_argument("--decode_cpu_dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
    p.add_argument("--decode_cpu_if_free_below_gb", type=float, default=20.0)

    # Reward (UnifiedReward defaults)
    p.add_argument(
        "--reward_type",
        choices=["imagereward", "auto", "unifiedreward", "unified", "pickscore", "hpsv3", "hpsv2", "blend"],
        default="unifiedreward",
    )
    p.add_argument("--reward_device", type=str, default="cpu")
    p.add_argument("--reward_model", type=str, default="CodeGoat24/UnifiedReward-qwen-7b")
    p.add_argument("--unifiedreward_model", type=str, default=None)
    p.add_argument("--image_reward_model", type=str, default="ImageReward-v1.0")
    p.add_argument("--pickscore_model", type=str, default="yuvalkirstain/PickScore_v1")
    p.add_argument("--reward_weights", nargs=2, type=float, default=[1.0, 1.0])
    p.add_argument("--reward_api_base", type=str, default=None)
    p.add_argument("--reward_api_key", type=str, default="unifiedreward")
    p.add_argument("--reward_api_model", type=str, default="UnifiedReward-7b-v1.5")
    p.add_argument("--reward_max_new_tokens", type=int, default=512)
    p.add_argument("--reward_prompt_mode", choices=["standard", "strict"], default="standard")

    # Qwen prompt basis
    p.add_argument("--no_qwen", action="store_true")
    p.add_argument("--qwen_id", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--qwen_python", type=str, default="python3")
    p.add_argument("--qwen_dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--qwen_device", type=str, default="auto", help="auto/cpu/cuda[:idx] for Qwen rewrite subprocess.")
    p.add_argument("--qwen_timeout_sec", type=float, default=240.0, help="Timeout for one Qwen rewrite subprocess.")

    # Interp setup
    p.add_argument(
        "--interp_labels",
        nargs="+",
        default=["balanced", "subject", "composition", "texture"],
        help="Basis labels to blend (defaults to four prompt levels).",
    )
    p.add_argument("--interp_k", type=int, default=4, help="How many basis prompts to blend.")
    p.add_argument("--interp_values", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument(
        "--mix_weight_vectors",
        nargs="+",
        default=None,
        help="Optional explicit weight vectors over interp_k prompts, e.g. '1,0,0,0 0.25,0.25,0.25,0.25'.",
    )
    p.add_argument("--families", nargs="+", default=["nlerp", "slerp"])
    p.add_argument("--preview_every", type=int, default=-1, help="Set -1 to disable preview decodes for speed.")

    # Which evaluations to run
    p.add_argument("--run_sweep", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run_ga", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run_mcts", action=argparse.BooleanOptionalAction, default=True)

    # GA over per-step discrete actions (action bank = families x profiles x cfg_scales)
    p.add_argument("--ga_population", type=int, default=24)
    p.add_argument("--ga_generations", type=int, default=8)
    p.add_argument("--ga_elites", type=int, default=3)
    p.add_argument("--ga_mutation_prob", type=float, default=0.10)
    p.add_argument("--ga_tournament_k", type=int, default=3)
    p.add_argument("--ga_selection", choices=["rank", "tournament"], default="rank")
    p.add_argument("--ga_rank_pressure", type=float, default=1.7)
    p.add_argument("--ga_crossover", choices=["uniform", "one_point"], default="uniform")
    p.add_argument("--ga_log_topk", type=int, default=3)
    p.add_argument("--ga_cache", action=argparse.BooleanOptionalAction, default=True)

    # MCTS over per-step discrete actions (action bank = families x profiles x cfg_scales)
    p.add_argument("--mcts_n_sims", type=int, default=50)
    p.add_argument("--mcts_ucb_c", type=float, default=1.41)
    p.add_argument("--mcts_log_every", type=int, default=10)
    p.add_argument("--mcts_cache", action=argparse.BooleanOptionalAction, default=True)

    # Root-level SPSA over 4 prompt-weight scalars during MCTS
    p.add_argument(
        "--enable_prompt_weight_spsa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adapt 4 global prompt-weight scalars online during MCTS blocks.",
    )
    p.add_argument(
        "--num_prompt_weights",
        type=int,
        default=4,
        help="Number of global prompt-weight scalars adapted by SPSA.",
    )
    p.add_argument(
        "--spsa_block_rollouts",
        type=int,
        default=8,
        help="Number of MCTS rollouts run with the current prompt weights before one SPSA update.",
    )
    p.add_argument("--spsa_c", type=float, default=0.07, help="SPSA perturbation magnitude in log-weight space.")
    p.add_argument("--spsa_eta", type=float, default=0.05, help="SPSA learning rate in log-weight space.")
    p.add_argument("--weight_clip_min", type=float, default=0.5, help="Minimum prompt-weight multiplier after exp(theta).")
    p.add_argument("--weight_clip_max", type=float, default=2.0, help="Maximum prompt-weight multiplier after exp(theta).")
    p.add_argument(
        "--run_mcts_weight_ablation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run both fixed-weight MCTS and SPSA-weighted MCTS for comparison.",
    )
    p.add_argument(
        "--run_mcts_design_ablation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compare variant-only MCTS against global 4-weight blend MCTS (fixed vs SPSA).",
    )
    p.add_argument(
        "--run_mcts_family_spsa_ablation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Within design ablation, also run family-specific SPSA arms: nlerp-only and slerp-only.",
    )
    return p.parse_args(argv)


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [str(args.prompt).strip()]
    if int(args.max_prompts) > 0:
        prompts = prompts[: int(args.max_prompts)]
    if len(prompts) == 0:
        raise RuntimeError("No prompts found.")
    return prompts


def _pick_basis_indices(labels: list[str], wanted: list[str], k: int) -> list[int]:
    key_to_idx = {str(lbl).strip().lower(): i for i, lbl in enumerate(labels)}
    picks: list[int] = []
    for raw in wanted:
        idx = key_to_idx.get(str(raw).strip().lower(), -1)
        if idx >= 0 and idx not in picks:
            picks.append(int(idx))
    for i in range(len(labels)):
        if len(picks) >= int(k):
            break
        if i not in picks:
            picks.append(int(i))
    if len(picks) == 0:
        raise RuntimeError("No basis indices available to blend.")
    return [int(x) for x in picks[: max(1, int(k))]]


def _family_list(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    for fam in args.families:
        f = str(fam).strip().lower()
        if f in {"nlerp", "slerp"} and f not in out:
            out.append(f)
    if len(out) == 0:
        out = ["nlerp", "slerp"]
    return out


def _normalize_weights(weights: list[float]) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    arr = np.clip(arr, 0.0, None)
    s = float(arr.sum())
    if (not np.isfinite(s)) or s <= 0.0:
        arr = np.full_like(arr, 1.0 / float(max(1, arr.size)))
    else:
        arr = arr / s
    return arr


def _ones_prompt_weight_vector(n_weights: int) -> np.ndarray:
    return np.ones((int(n_weights),), dtype=np.float64)


def _weight_signature(weights: np.ndarray | None) -> tuple[float, ...]:
    if weights is None:
        return ()
    arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    return tuple(float(f"{float(x):.8f}") for x in arr.tolist())


def _effective_profile_weights(
    profile_weights: list[float],
    prompt_weight_multipliers: np.ndarray | None,
) -> np.ndarray:
    base = np.asarray(profile_weights, dtype=np.float64).reshape(-1)
    if prompt_weight_multipliers is None:
        return base
    mul = np.asarray(prompt_weight_multipliers, dtype=np.float64).reshape(-1)
    if base.shape[0] != mul.shape[0]:
        raise RuntimeError(
            f"Prompt-weight multiplier length mismatch: profile={base.shape[0]} vs spsa={mul.shape[0]}"
        )
    return base * mul


def _clip_prompt_weights(theta: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    return np.clip(np.exp(np.asarray(theta, dtype=np.float64)), float(clip_min), float(clip_max))


def _build_weight_profiles(
    args: argparse.Namespace,
    selected_labels: list[str],
    t_values: list[float],
) -> list[dict[str, Any]]:
    k = int(len(selected_labels))
    if k <= 0:
        raise RuntimeError("Expected at least one selected label for profiles.")

    raw_profiles: list[dict[str, Any]] = []
    explicit = args.mix_weight_vectors if args.mix_weight_vectors is not None else []
    if len(explicit) > 0:
        for i, spec in enumerate(explicit):
            toks = [x.strip() for x in str(spec).split(",") if x.strip()]
            if len(toks) != k:
                raise RuntimeError(
                    f"Invalid mix_weight_vectors entry '{spec}': expected {k} comma-separated values."
                )
            vec = _normalize_weights([float(x) for x in toks])
            raw_profiles.append(
                {
                    "name": f"custom_{i:02d}",
                    "weights": [float(x) for x in vec.tolist()],
                    "source": "custom",
                    "t": None,
                    "focus_label": None,
                }
            )
    else:
        # Uniform profile.
        raw_profiles.append(
            {
                "name": "uniform",
                "weights": [float(1.0 / float(k)) for _ in range(k)],
                "source": "auto",
                "t": None,
                "focus_label": None,
            }
        )
        # Focus-label profiles: interpolate target label vs others.
        for idx, label in enumerate(selected_labels):
            for t in t_values:
                tt = float(np.clip(t, 0.0, 1.0))
                if k == 1:
                    w = np.asarray([1.0], dtype=np.float64)
                else:
                    other = (1.0 - tt) / float(k - 1)
                    w = np.full((k,), other, dtype=np.float64)
                    w[idx] = tt
                w = _normalize_weights([float(x) for x in w.tolist()])
                raw_profiles.append(
                    {
                        "name": f"focus_{str(label)}_t{tt:.2f}",
                        "weights": [float(x) for x in w.tolist()],
                        "source": "auto",
                        "t": float(tt),
                        "focus_label": str(label),
                    }
                )

    # Deduplicate by exact weight vector after normalization.
    dedup: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for row in raw_profiles:
        key = tuple(float(f"{float(x):.8f}") for x in row["weights"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)
    return dedup


def _build_variant_only_weight_profiles(selected_labels: list[str]) -> list[dict[str, Any]]:
    k = int(len(selected_labels))
    if k <= 0:
        raise RuntimeError("Expected at least one selected label for variant-only profiles.")
    out: list[dict[str, Any]] = []
    for idx, label in enumerate(selected_labels):
        vec = np.zeros((k,), dtype=np.float64)
        vec[idx] = 1.0
        out.append(
            {
                "name": f"variant_{str(label)}",
                "weights": [float(x) for x in vec.tolist()],
                "source": "variant_only",
                "t": 1.0,
                "focus_label": str(label),
            }
        )
    return out


def _build_global_blend_weight_profiles(selected_labels: list[str]) -> list[dict[str, Any]]:
    k = int(len(selected_labels))
    if k <= 0:
        raise RuntimeError("Expected at least one selected label for global blend profile.")
    return [
        {
            "name": "global_blend_uniform",
            "weights": [float(1.0 / float(k)) for _ in range(k)],
            "source": "global_blend",
            "t": None,
            "focus_label": None,
        }
    ]


def _action_bank(families: list[str], n_profiles: int, cfg_values: list[float]) -> list[tuple[str, int, float]]:
    out: list[tuple[str, int, float]] = []
    for fam in families:
        for profile_idx in range(int(n_profiles)):
            for cfg in cfg_values:
                out.append((str(fam), int(profile_idx), float(cfg)))
    return out


def _action_detail(
    action_idx: int,
    action_bank: list[tuple[str, int, float]],
    weight_profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    fam, profile_idx, cfg = action_bank[int(action_idx)]
    profile = weight_profiles[int(profile_idx)]
    out: dict[str, Any] = {
        "action_idx": int(action_idx),
        "family": str(fam),
        "profile_idx": int(profile_idx),
        "profile": str(profile["name"]),
        "cfg": float(cfg),
        "weights": [float(x) for x in profile["weights"]],
    }
    if profile.get("t") is not None:
        out["t"] = float(profile["t"])
    if profile.get("focus_label") is not None:
        out["focus_label"] = str(profile["focus_label"])
    return out


def _should_preview(step_idx: int, steps: int, every: int) -> bool:
    if every < 0:
        return False
    if every <= 0:
        return step_idx == steps - 1
    return ((step_idx + 1) % every == 0) or (step_idx == steps - 1)


def _font(size: int = 15) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _build_summary_panel(
    baseline_img: Image.Image,
    baseline_score: float,
    sweep_items: list[dict[str, Any]],
    mcts_item: dict[str, Any] | None,
    ga_item: dict[str, Any] | None,
    out_path: str,
) -> None:
    # Keep panel compact: baseline + top 4 sweep + mcts/ga best (if present).
    ranked = sorted(sweep_items, key=lambda x: float(x["score"]), reverse=True)
    ranked = ranked[:4]
    entries: list[tuple[str, Image.Image, float, float]] = [("baseline", baseline_img, baseline_score, 0.0)]
    for row in ranked:
        label = f"{row['family']} {row['profile']} cfg={float(row['cfg']):.2f}"
        entries.append((label, row["image"], float(row["score"]), float(row["score"]) - baseline_score))
    if mcts_item is not None:
        entries.append(("mcts_best", mcts_item["image"], float(mcts_item["score"]), float(mcts_item["score"]) - baseline_score))
    if ga_item is not None:
        entries.append(("ga_best", ga_item["image"], float(ga_item["score"]), float(ga_item["score"]) - baseline_score))

    w, h = baseline_img.size
    label_h = 46
    cols = len(entries)
    canvas = Image.new("RGB", (cols * w, h + label_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    f1 = _font(15)
    f2 = _font(13)
    for i, (title, img, score, delta) in enumerate(entries):
        x0 = i * w
        canvas.paste(img, (x0, label_h))
        draw.text((x0 + 8, 6), title, fill=(230, 230, 230), font=f1)
        col = (200, 200, 200) if i == 0 else ((110, 255, 110) if delta >= 0.0 else (255, 120, 120))
        draw.text((x0 + 8, 24), f"s={score:.4f} d={delta:+.4f}", fill=col, font=f2)
    canvas.save(out_path)


def _write_mcts_weight_history_rows(
    writer: csv.writer,
    slug: str,
    prompt: str,
    method_name: str,
    result: dict[str, Any] | None,
) -> None:
    if not isinstance(result, dict):
        return
    search_space = str(result.get("search_space_name", "unknown"))
    mode = str(result.get("mode", "fixed"))
    history = result.get("spsa_history", [])
    if isinstance(history, list) and len(history) > 0:
        for row in history:
            r_plus = row.get("R_plus")
            r_minus = row.get("R_minus")
            delta_r = None
            if r_plus is not None and r_minus is not None:
                delta_r = float(r_plus) - float(r_minus)
            writer.writerow(
                [
                    slug,
                    prompt,
                    method_name,
                    search_space,
                    mode,
                    "spsa_block",
                    int(row.get("block", -1)),
                    int(row.get("after_sim", -1)),
                    f"{float(r_plus):.6f}" if r_plus is not None else "",
                    f"{float(r_minus):.6f}" if r_minus is not None else "",
                    f"{float(delta_r):+.6f}" if delta_r is not None else "",
                    json.dumps(row.get("w_before", [])),
                    json.dumps(row.get("w_after", [])),
                    json.dumps(row.get("gradient", [])),
                    json.dumps(result.get("final_prompt_weights", [])),
                ]
            )
    else:
        writer.writerow(
            [
                slug,
                prompt,
                method_name,
                search_space,
                mode,
                "final_only",
                -1,
                int(result.get("eval_calls_total", -1)),
                "",
                "",
                "",
                json.dumps(result.get("best_prompt_weights", [])),
                json.dumps(result.get("final_prompt_weights", [])),
                json.dumps([]),
                json.dumps(result.get("final_prompt_weights", [])),
            ]
        )


def _summarize_step_actions(step_actions: Any) -> tuple[str, str, str, str]:
    if not isinstance(step_actions, list) or len(step_actions) <= 0:
        return ("mixed", "mixed", "mixed", "mixed")

    def _pick_str(key: str) -> str:
        vals = [str(x.get(key, "")).strip() for x in step_actions if isinstance(x, dict)]
        vals = [v for v in vals if len(v) > 0]
        if len(vals) <= 0:
            return "mixed"
        uniq = sorted(set(vals))
        return vals[0] if len(uniq) == 1 else "mixed"

    def _pick_float_str(key: str) -> str:
        vals: list[float] = []
        for x in step_actions:
            if isinstance(x, dict) and x.get(key) is not None:
                vals.append(float(x.get(key)))
        if len(vals) <= 0:
            return "mixed"
        rounded = [round(v, 6) for v in vals]
        if len(set(rounded)) == 1:
            return f"{vals[0]:.2f}"
        return "mixed"

    return (
        _pick_str("family"),
        _pick_str("profile"),
        _pick_float_str("t"),
        _pick_float_str("cfg"),
    )


@torch.no_grad()
def run_stepwise_action_rollout(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    reward_ctx: su.RewardContext,
    prompt: str,
    metadata: dict[str, Any] | None,
    seed: int,
    h: int,
    w: int,
    orig_h: int,
    orig_w: int,
    basis_emb: Any,
    basis_indices: list[int],
    weight_profiles: list[dict[str, Any]],
    action_bank: list[tuple[str, int, float]],
    genome: list[int],
    preview_every: int,
    prompt_weight_multipliers: np.ndarray | None = None,
    save_path: str | None = None,
    tag: str = "stepwise",
) -> ActionRollout:
    if len(basis_indices) <= 0:
        raise ValueError("Expected non-empty basis_indices.")
    if len(weight_profiles) <= 0:
        raise ValueError("Expected non-empty weight_profiles.")
    if len(action_bank) <= 0:
        raise ValueError("action_bank must be non-empty.")

    pe_bank = [basis_emb.pe_list[int(i)][0] for i in basis_indices]
    pm_bank = [basis_emb.pe_list[int(i)][1] for i in basis_indices]
    prompt_weights = (
        _ones_prompt_weight_vector(len(basis_indices))
        if prompt_weight_multipliers is None
        else np.asarray(prompt_weight_multipliers, dtype=np.float64).reshape(-1)
    )
    if prompt_weights.shape[0] != len(basis_indices):
        raise RuntimeError(
            f"Expected {len(basis_indices)} prompt weights, got {prompt_weights.shape[0]}"
        )

    latents = su.make_latents(ctx, int(seed), int(h), int(w), basis_emb.orig_pe.dtype)
    sched = su._step_tensors(ctx, int(args.steps), latents.dtype)
    sigmas = [float(t_flat[0].item()) for t_flat, _ in sched]
    progress = progress_from_sigmas(sigmas)

    dx = torch.zeros_like(latents)
    rng = torch.Generator(device=ctx.device).manual_seed(int(seed) + 2048)

    preview_rewards: list[float] = []
    step_actions: list[dict[str, Any]] = []
    prev_reward = 0.0

    for step_idx, ((t_flat, t_4d), sigma_t, prog_t) in enumerate(zip(sched, sigmas, progress)):
        noise = latents if step_idx == 0 else torch.randn(
            latents.shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=rng,
        )
        latents = (1.0 - t_4d) * dx + t_4d * noise

        a_idx = int(genome[step_idx % len(genome)]) % len(action_bank)
        fam, profile_idx, cfg = action_bank[a_idx]
        profile = weight_profiles[int(profile_idx)]
        effective_weights = _effective_profile_weights(profile["weights"], prompt_weights)
        weights_t = torch.tensor(effective_weights, dtype=pe_bank[0].dtype, device=ctx.device)
        blend = blend_prompt_embeddings(pe_bank, pm_bank, weights_t, fam)
        velocity = su.transformer_step(
            args,
            ctx,
            latents,
            blend.prompt_embed,
            blend.prompt_mask,
            basis_emb.ue,
            basis_emb.um,
            t_flat,
            float(cfg),
        )
        dx = latents - t_4d * velocity

        cur_reward = prev_reward
        delta = 0.0
        if _should_preview(step_idx, int(args.steps), int(preview_every)):
            preview_img = su.decode_to_pil(ctx, dx, orig_h, orig_w, tag=f"{tag}_preview")
            cur_reward = float(reward_ctx.score_images(prompt, [preview_img], metadata)[0])
            delta = float(cur_reward - prev_reward)
            preview_rewards.append(float(cur_reward))

        step_actions.append(
            {
                "step": int(step_idx),
                "sigma": float(sigma_t),
                "progress": float(prog_t),
                "action_idx": int(a_idx),
                "family": str(fam),
                "profile_idx": int(profile_idx),
                "profile": str(profile["name"]),
                "t": float(profile["t"]) if profile.get("t") is not None else None,
                "focus_label": str(profile["focus_label"]) if profile.get("focus_label") is not None else None,
                "cfg": float(cfg),
                "profile_weights": [float(x) for x in profile["weights"]],
                "prompt_weight_multipliers": [float(x) for x in prompt_weights.tolist()],
                "effective_weights": [float(x) for x in effective_weights.tolist()],
                "selected_indices": [int(basis_indices[i]) for i in blend.selected_indices],
                "selected_labels": [str([basis_emb.labels[j] for j in basis_indices][i]) for i in blend.selected_indices],
                "selected_weights": [float(x) for x in blend.selected_weights],
                "preview_reward": float(cur_reward),
                "delta_reward": float(delta),
            }
        )
        prev_reward = cur_reward

    final_image = su.decode_to_pil(ctx, dx, orig_h, orig_w, tag=f"{tag}_final")
    final_score = float(reward_ctx.score_images(prompt, [final_image], metadata)[0])
    if save_path:
        final_image.save(save_path)
    return ActionRollout(
        final_score=float(final_score),
        final_image=final_image,
        preview_rewards=[float(x) for x in preview_rewards],
        step_actions=step_actions,
    )


def _random_genome(rng: np.random.Generator, steps: int, n_actions: int) -> list[int]:
    return [int(rng.integers(0, n_actions)) for _ in range(int(steps))]


def _mutate_genome(
    genome: list[int],
    rng: np.random.Generator,
    n_actions: int,
    mutation_prob: float,
) -> list[int]:
    out = list(genome)
    for i in range(len(out)):
        if rng.random() < float(mutation_prob):
            out[i] = int(rng.integers(0, n_actions))
    return out


def _one_point_crossover(a: list[int], b: list[int], rng: np.random.Generator) -> tuple[list[int], list[int]]:
    if len(a) != len(b):
        raise ValueError("Genome length mismatch in one-point crossover.")
    if len(a) < 2:
        return list(a), list(b)
    point = int(rng.integers(1, len(a)))
    return list(a[:point] + b[point:]), list(b[:point] + a[point:])


def run_ga_search(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    reward_ctx: su.RewardContext,
    prompt: str,
    metadata: dict[str, Any] | None,
    seed: int,
    h: int,
    w: int,
    orig_h: int,
    orig_w: int,
    basis_emb: Any,
    basis_indices: list[int],
    weight_profiles: list[dict[str, Any]],
    action_bank: list[tuple[str, int, float]],
    prompt_dir: str,
    save_images: bool,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed) + 911)
    pop_size = max(4, int(args.ga_population))
    elites = max(1, min(int(args.ga_elites), pop_size))
    n_actions = len(action_bank)
    steps = int(args.steps)

    # Anchor genome: first action repeated.
    population: list[list[int]] = [[0 for _ in range(steps)]]
    while len(population) < pop_size:
        population.append(_random_genome(rng, steps, n_actions))

    cache: dict[tuple[int, ...], dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    global_best: dict[str, Any] | None = None
    eval_calls_total = 0
    prev_eval_calls_total = 0

    def evaluate(genome: list[int], need_image: bool, generation: int, phase: str) -> dict[str, Any]:
        nonlocal eval_calls_total
        repaired = [int(x) % n_actions for x in genome]
        key = tuple(repaired)
        eval_calls_total += 1
        if bool(args.ga_cache):
            cached = cache.get(key)
            if cached is not None and (not need_image or cached.get("image") is not None):
                return cached
        trace = run_stepwise_action_rollout(
            args=args,
            ctx=ctx,
            reward_ctx=reward_ctx,
            prompt=prompt,
            metadata=metadata,
            seed=int(seed),
            h=h,
            w=w,
            orig_h=orig_h,
            orig_w=orig_w,
            basis_emb=basis_emb,
            basis_indices=basis_indices,
            weight_profiles=weight_profiles,
            action_bank=action_bank,
            genome=repaired,
            preview_every=int(args.preview_every),
            save_path=None,
            tag=f"ga_g{generation}_{phase}",
        )
        payload = {
            "genome": repaired,
            "score": float(trace.final_score),
            "image": trace.final_image if need_image else None,
            "preview_rewards": [float(x) for x in trace.preview_rewards],
            "step_actions": trace.step_actions,
        }
        if bool(args.ga_cache):
            cache[key] = payload
        return payload

    for gen in range(int(args.ga_generations)):
        scored: list[dict[str, Any]] = [evaluate(genome, need_image=False, generation=gen, phase="population") for genome in population]
        scored.sort(key=lambda row: float(row["score"]), reverse=True)
        if global_best is None or float(scored[0]["score"]) > float(global_best["score"]):
            global_best = evaluate(scored[0]["genome"], need_image=True, generation=gen, phase="best")

        topk = max(1, int(args.ga_log_topk))
        top_rows: list[dict[str, Any]] = []
        for rk, row in enumerate(scored[:topk]):
            detailed = evaluate(row["genome"], need_image=(save_images and rk == 0), generation=gen, phase=f"top{rk}")
            top_rows.append(
                {
                    "rank": int(rk + 1),
                    "score": float(detailed["score"]),
                    "genome": [int(x) for x in detailed["genome"]],
                    "actions": [
                        dict({"step": int(i)}, **_action_detail(int(aidx), action_bank, weight_profiles))
                        for i, aidx in enumerate(detailed["genome"])
                    ],
                }
            )

        scores_np = np.asarray([float(row["score"]) for row in scored], dtype=np.float64)
        unique_genomes = len({tuple(row["genome"]) for row in scored})
        eval_calls_generation = int(eval_calls_total - prev_eval_calls_total)
        prev_eval_calls_total = eval_calls_total
        gen_summary = {
            "generation": int(gen),
            "best_score": float(scores_np.max()),
            "mean_score": float(scores_np.mean()),
            "worst_score": float(scores_np.min()),
            "unique_genomes": int(unique_genomes),
            "population_size": int(len(scored)),
            "eval_calls_total": int(eval_calls_total),
            "eval_calls_generation": int(eval_calls_generation),
            "nfe_per_generation": int(eval_calls_generation * steps),
            "nfe_total": int(eval_calls_total * steps),
            "top": top_rows,
        }
        history.append(gen_summary)
        print(
            f"  [ga] gen={gen+1:02d}/{args.ga_generations} "
            f"best={gen_summary['best_score']:.4f} mean={gen_summary['mean_score']:.4f} "
            f"uniq={gen_summary['unique_genomes']}/{gen_summary['population_size']} "
            f"nfe/gen={gen_summary['nfe_per_generation']}"
        )

        if gen == int(args.ga_generations) - 1:
            break

        scored_pairs = [(float(row["score"]), [int(x) for x in row["genome"]]) for row in scored]
        next_population: list[list[int]] = [[int(x) for x in row["genome"]] for row in scored[:elites]]

        while len(next_population) < pop_size:
            if str(args.ga_selection).lower() == "rank":
                p1 = rank_select(scored_pairs, rng, float(args.ga_rank_pressure))
                p2 = rank_select(scored_pairs, rng, float(args.ga_rank_pressure))
            else:
                p1 = tournament_select(scored_pairs, rng, int(args.ga_tournament_k))
                p2 = tournament_select(scored_pairs, rng, int(args.ga_tournament_k))

            if str(args.ga_crossover).lower() == "one_point":
                c1, c2 = _one_point_crossover(p1, p2, rng)
            else:
                c1 = crossover_uniform(p1, p2, rng)
                c2 = crossover_uniform(p2, p1, rng)

            c1 = _mutate_genome(c1, rng, n_actions, float(args.ga_mutation_prob))
            c2 = _mutate_genome(c2, rng, n_actions, float(args.ga_mutation_prob))
            next_population.append(c1)
            if len(next_population) < pop_size:
                next_population.append(c2)
        population = next_population

    assert global_best is not None
    best_genome = [int(x) for x in global_best["genome"]]
    best_actions = [
        dict({"step": int(i)}, **_action_detail(int(aidx), action_bank, weight_profiles))
        for i, aidx in enumerate(best_genome)
    ]
    best_img_path = None
    if save_images and global_best.get("image") is not None:
        best_img_path = os.path.join(prompt_dir, "ga_best.png")
        global_best["image"].save(best_img_path)

    result = {
        "best_score": float(global_best["score"]),
        "best_genome": best_genome,
        "best_actions": best_actions,
        "best_preview_rewards": [float(x) for x in global_best.get("preview_rewards", [])],
        "best_step_actions": global_best.get("step_actions", []),
        "best_image_file": best_img_path,
        "history": history,
        "eval_calls_total": int(eval_calls_total),
        "nfe_total": int(eval_calls_total * steps),
    }
    save_json(os.path.join(prompt_dir, "ga_action_search.json"), result)
    return result


class _MCTSNode:
    __slots__ = ("step", "prefix", "N", "children", "action_N", "action_Q")

    def __init__(self, step: int, prefix: tuple[int, ...]) -> None:
        self.step = int(step)
        self.prefix = tuple(int(x) for x in prefix)
        self.N = 0
        self.children: dict[int, _MCTSNode] = {}
        self.action_N: dict[int, int] = {}
        self.action_Q: dict[int, float] = {}

    def untried_actions(self, n_actions: int) -> list[int]:
        return [a for a in range(int(n_actions)) if a not in self.action_N]

    def ucb(self, action: int, c: float) -> float:
        n = int(self.action_N.get(int(action), 0))
        if n <= 0:
            return float("inf")
        q = float(self.action_Q.get(int(action), 0.0)) / float(n)
        return q + float(c) * math.sqrt(math.log(max(1, int(self.N))) / float(n))


def _mcts_best_exploit_genome(root: _MCTSNode, steps: int, n_actions: int) -> list[int]:
    genome: list[int] = []
    node = root
    default_action = 0
    while len(genome) < int(steps):
        best_action = None
        best_mean = -float("inf")
        for action, visits in node.action_N.items():
            if int(visits) <= 0:
                continue
            mean = float(node.action_Q.get(int(action), 0.0)) / float(visits)
            if mean > best_mean:
                best_mean = mean
                best_action = int(action)
        if best_action is None:
            break
        genome.append(int(best_action))
        child = node.children.get(int(best_action))
        if child is None:
            break
        node = child
    while len(genome) < int(steps):
        genome.append(int(default_action % max(1, n_actions)))
    return genome


def run_mcts_search(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    reward_ctx: su.RewardContext,
    prompt: str,
    metadata: dict[str, Any] | None,
    seed: int,
    h: int,
    w: int,
    orig_h: int,
    orig_w: int,
    basis_emb: Any,
    basis_indices: list[int],
    weight_profiles: list[dict[str, Any]],
    action_bank: list[tuple[str, int, float]],
    prompt_dir: str,
    save_images: bool,
    *,
    enable_spsa: bool | None = None,
    result_prefix: str = "mcts",
    search_space_name: str = "dynamic_profiles",
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed) + 1777)
    n_actions = int(len(action_bank))
    steps = int(args.steps)
    n_sims = max(1, int(args.mcts_n_sims))
    ucb_c = float(args.mcts_ucb_c)
    log_every = max(1, int(args.mcts_log_every))
    use_spsa = bool(args.enable_prompt_weight_spsa if enable_spsa is None else enable_spsa)
    num_prompt_weights = int(args.num_prompt_weights)
    if num_prompt_weights != len(basis_indices):
        raise RuntimeError(
            f"MCTS prompt-weight search expects {num_prompt_weights} basis prompts, got {len(basis_indices)}. "
            "Adjust --num_prompt_weights or --interp_k/--interp_labels."
        )
    spsa_block_rollouts = max(1, int(args.spsa_block_rollouts))
    spsa_c = float(args.spsa_c)
    spsa_eta = float(args.spsa_eta)
    weight_clip_min = float(args.weight_clip_min)
    weight_clip_max = float(args.weight_clip_max)
    log_clip_min = math.log(max(weight_clip_min, 1e-8))
    log_clip_max = math.log(max(weight_clip_max, weight_clip_min + 1e-8))

    cache: dict[tuple[tuple[float, ...], tuple[int, ...]], dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    spsa_history: list[dict[str, Any]] = []
    eval_calls_total = 0
    search_best: dict[str, Any] | None = None
    theta = np.zeros((num_prompt_weights,), dtype=np.float64)
    current_prompt_weights = _clip_prompt_weights(theta, weight_clip_min, weight_clip_max)

    def evaluate(
        genome: list[int],
        prompt_weights: np.ndarray,
        need_image: bool,
        sim_idx: int,
        phase: str,
    ) -> dict[str, Any]:
        nonlocal eval_calls_total
        repaired = [int(x) % n_actions for x in genome]
        weight_sig = _weight_signature(prompt_weights)
        key = (weight_sig, tuple(repaired))
        eval_calls_total += 1
        if bool(args.mcts_cache):
            cached = cache.get(key)
            if cached is not None and (not need_image or cached.get("image") is not None):
                return cached
        trace = run_stepwise_action_rollout(
            args=args,
            ctx=ctx,
            reward_ctx=reward_ctx,
            prompt=prompt,
            metadata=metadata,
            seed=int(seed),
            h=h,
            w=w,
            orig_h=orig_h,
            orig_w=orig_w,
            basis_emb=basis_emb,
            basis_indices=basis_indices,
            weight_profiles=weight_profiles,
            action_bank=action_bank,
            genome=repaired,
            preview_every=int(args.preview_every),
            prompt_weight_multipliers=prompt_weights,
            save_path=None,
            tag=f"mcts_s{sim_idx}_{phase}",
        )
        payload = {
            "genome": repaired,
            "score": float(trace.final_score),
            "image": trace.final_image if need_image else None,
            "preview_rewards": [float(x) for x in trace.preview_rewards],
            "step_actions": trace.step_actions,
            "prompt_weights": [float(x) for x in prompt_weights.tolist()],
        }
        if bool(args.mcts_cache):
            cache[key] = payload
        return payload

    root = _MCTSNode(step=0, prefix=())

    def maybe_update_search_best(candidate: dict[str, Any], sim_idx: int, phase: str) -> None:
        nonlocal search_best
        if search_best is None or float(candidate["score"]) > float(search_best["score"]):
            search_best = evaluate(
                candidate["genome"],
                np.asarray(candidate["prompt_weights"], dtype=np.float64),
                need_image=save_images,
                sim_idx=sim_idx,
                phase=phase,
            )

    sim = 0
    block_idx = 0
    while sim < n_sims:
        sims_this_block = min(spsa_block_rollouts, n_sims - sim)
        for _ in range(sims_this_block):
            node = root
            root.N += 1
            path: list[tuple[_MCTSNode, int]] = []

            while node.step < steps:
                untried = node.untried_actions(n_actions)
                if len(untried) > 0:
                    action = int(untried[int(rng.integers(0, len(untried)))])
                    child = node.children.get(action)
                    if child is None:
                        child = _MCTSNode(step=node.step + 1, prefix=node.prefix + (action,))
                        node.children[action] = child
                    path.append((node, action))
                    node = child
                    break

                action = max(range(n_actions), key=lambda a: node.ucb(a, ucb_c))
                child = node.children.get(action)
                if child is None:
                    child = _MCTSNode(step=node.step + 1, prefix=node.prefix + (action,))
                    node.children[action] = child
                path.append((node, int(action)))
                node = child

            rollout = [int(x) for x in node.prefix]
            while len(rollout) < steps:
                rollout.append(int(rng.integers(0, n_actions)))
            scored = evaluate(
                rollout,
                current_prompt_weights,
                need_image=False,
                sim_idx=sim,
                phase="rollout",
            )

            maybe_update_search_best(scored, sim, "best")

            score = float(scored["score"])
            for parent, action in path:
                parent.action_N[action] = int(parent.action_N.get(action, 0) + 1)
                parent.action_Q[action] = float(parent.action_Q.get(action, 0.0) + score)
                child = parent.children[action]
                child.N = int(child.N + 1)

            sim += 1
            if (sim % log_every) == 0 or sim == 1:
                top_root = []
                for a, n in root.action_N.items():
                    if int(n) > 0:
                        top_root.append((float(root.action_Q[a]) / float(n), int(a), int(n)))
                top_root.sort(reverse=True)
                top_root = top_root[: max(1, int(args.ga_log_topk))]
                row = {
                    "sim": int(sim),
                    "block": int(block_idx),
                    "mode": "spsa" if use_spsa else "fixed",
                    "best_score": float(search_best["score"]) if search_best is not None else None,
                    "root_visits": int(root.N),
                    "eval_calls_total": int(eval_calls_total),
                    "nfe_total": int(eval_calls_total * steps),
                    "theta": [float(x) for x in theta.tolist()],
                    "prompt_weights": [float(x) for x in current_prompt_weights.tolist()],
                    "root_top": [
                        dict(
                            {
                                "mean_score": float(m),
                                "visits": int(v),
                            },
                            **_action_detail(int(aidx), action_bank, weight_profiles),
                        )
                        for m, aidx, v in top_root
                    ],
                }
                history.append(row)
                print(
                    f"  [mcts:{'spsa' if use_spsa else 'fixed'}] sim={sim:03d}/{n_sims} "
                    f"best={(float(search_best['score']) if search_best is not None else float('-inf')):.4f} "
                    f"w={[float(x) for x in current_prompt_weights.tolist()]} evals={eval_calls_total}"
                )

        if use_spsa:
            delta = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=num_prompt_weights).astype(np.float64)
            theta_before = theta.copy()
            w_before = current_prompt_weights.copy()
            theta_plus = theta_before + spsa_c * delta
            theta_minus = theta_before - spsa_c * delta
            w_plus = _clip_prompt_weights(theta_plus, weight_clip_min, weight_clip_max)
            w_minus = _clip_prompt_weights(theta_minus, weight_clip_min, weight_clip_max)
            eval_genome = _mcts_best_exploit_genome(root, steps, n_actions)

            plus = evaluate(eval_genome, w_plus, need_image=False, sim_idx=sim, phase=f"spsa_plus_b{block_idx}")
            minus = evaluate(eval_genome, w_minus, need_image=False, sim_idx=sim, phase=f"spsa_minus_b{block_idx}")
            maybe_update_search_best(plus, sim, f"spsa_plus_best_b{block_idx}")
            maybe_update_search_best(minus, sim, f"spsa_minus_best_b{block_idx}")

            r_plus = float(plus["score"])
            r_minus = float(minus["score"])
            grad = ((r_plus - r_minus) / (2.0 * spsa_c)) * delta
            theta = np.clip(theta_before + spsa_eta * grad, log_clip_min, log_clip_max)
            current_prompt_weights = _clip_prompt_weights(theta, weight_clip_min, weight_clip_max)
            spsa_history.append(
                {
                    "block": int(block_idx),
                    "after_sim": int(sim),
                    "representative_genome": [int(x) for x in eval_genome],
                    "theta_before": [float(x) for x in theta_before.tolist()],
                    "theta_after": [float(x) for x in theta.tolist()],
                    "w_before": [float(x) for x in w_before.tolist()],
                    "w_after": [float(x) for x in current_prompt_weights.tolist()],
                    "w_plus": [float(x) for x in w_plus.tolist()],
                    "w_minus": [float(x) for x in w_minus.tolist()],
                    "delta": [float(x) for x in delta.tolist()],
                    "R_plus": float(r_plus),
                    "R_minus": float(r_minus),
                    "gradient": [float(x) for x in grad.tolist()],
                    "eval_calls_total": int(eval_calls_total),
                }
            )
            print(
                f"  [mcts-spsa] block={block_idx+1:02d} sims={sim} "
                f"R+={r_plus:.4f} R-={r_minus:.4f} "
                f"w={[float(x) for x in current_prompt_weights.tolist()]}"
            )
        block_idx += 1

    if search_best is None:
        raise RuntimeError("MCTS failed to evaluate any rollout.")

    final_genome = _mcts_best_exploit_genome(root, steps, n_actions)
    final_eval = evaluate(
        final_genome,
        current_prompt_weights,
        need_image=save_images,
        sim_idx=sim,
        phase="final",
    )
    best_genome = [int(x) for x in final_eval["genome"]]
    best_actions = [
        dict({"step": int(i)}, **_action_detail(int(aidx), action_bank, weight_profiles))
        for i, aidx in enumerate(best_genome)
    ]
    best_img_path = None
    if save_images and final_eval.get("image") is not None:
        best_img_path = os.path.join(prompt_dir, f"{result_prefix}_best.png")
        final_eval["image"].save(best_img_path)

    result = {
        "search_space_name": str(search_space_name),
        "mode": "spsa" if use_spsa else "fixed",
        "enable_prompt_weight_spsa": bool(use_spsa),
        "best_score": float(final_eval["score"]),
        "best_genome": best_genome,
        "best_actions": best_actions,
        "best_preview_rewards": [float(x) for x in final_eval.get("preview_rewards", [])],
        "best_step_actions": final_eval.get("step_actions", []),
        "best_image_file": best_img_path,
        "best_prompt_weights": [float(x) for x in current_prompt_weights.tolist()],
        "final_score": float(final_eval["score"]),
        "final_theta": [float(x) for x in theta.tolist()],
        "final_prompt_weights": [float(x) for x in current_prompt_weights.tolist()],
        "search_best_score": float(search_best["score"]),
        "search_best_genome": [int(x) for x in search_best["genome"]],
        "search_best_prompt_weights": [float(x) for x in search_best.get("prompt_weights", [])],
        "spsa_config": {
            "enable_prompt_weight_spsa": bool(use_spsa),
            "num_prompt_weights": int(num_prompt_weights),
            "spsa_block_rollouts": int(spsa_block_rollouts),
            "spsa_c": float(spsa_c),
            "spsa_eta": float(spsa_eta),
            "weight_clip_min": float(weight_clip_min),
            "weight_clip_max": float(weight_clip_max),
        },
        "spsa_history": spsa_history,
        "history": history,
        "eval_calls_total": int(eval_calls_total),
        "nfe_total": int(eval_calls_total * steps),
        "n_actions": int(n_actions),
        "n_profiles": int(len(weight_profiles)),
    }
    save_json(os.path.join(prompt_dir, f"{result_prefix}_action_search.json"), result)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_dir(args.out_dir)

    if args.baseline_cfg is None:
        args.baseline_cfg = float(args.guidance_scale)

    if args.cuda_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(args.cuda_alloc_conf)
        print(f"Set PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

    prompts = load_prompts(args)
    families = _family_list(args)
    t_values = [float(np.clip(v, 0.0, 1.0)) for v in args.interp_values]
    cfg_values: list[float] = []
    for c in args.cfg_scales:
        fc = float(c)
        if fc not in cfg_values:
            cfg_values.append(fc)
    if len(cfg_values) == 0:
        cfg_values = [float(args.guidance_scale)]
    basis_cache = load_basis_cache(args.basis_cache_json)

    t0 = time.perf_counter()
    ctx = su.load_pipeline(args)
    reward_ctx = su.load_reward(args, ctx)
    if callable(getattr(reward_ctx, "before_decode", None)):
        setattr(ctx, "pre_decode_hook", reward_ctx.before_decode)
    neg_embeds, neg_mask = su.load_neg_embed(args, ctx)

    tsv_path = os.path.join(args.out_dir, "slerp_nlerp_unified_scores.tsv")
    mcts_weight_tsv_path = os.path.join(args.out_dir, "mcts_weight_factor_changes.tsv")
    all_rows: list[dict[str, Any]] = []

    with open(tsv_path, "w", encoding="utf-8", newline="") as ftsv, open(
        mcts_weight_tsv_path, "w", encoding="utf-8", newline=""
    ) as fwtsv:
        writer = csv.writer(ftsv, delimiter="\t")
        weight_writer = csv.writer(fwtsv, delimiter="\t")
        writer.writerow(
            [
                "slug",
                "prompt",
                "method",
                "search_space",
                "mode",
                "family",
                "profile",
                "t",
                "cfg",
                "score",
                "delta_vs_baseline",
                "baseline_definition",
                "final_prompt_weights",
                "image_file",
                "genome",
                "weight_log_file",
            ]
        )
        weight_writer.writerow(
            [
                "slug",
                "prompt",
                "method",
                "search_space",
                "mode",
                "event",
                "block",
                "after_sim",
                "R_plus",
                "R_minus",
                "delta_R",
                "w_before",
                "w_after",
                "gradient",
                "final_prompt_weights",
            ]
        )
        def write_score_row(
            *,
            slug: str,
            prompt: str,
            method: str,
            search_space: str,
            mode: str,
            family: str,
            profile: str,
            t: str,
            cfg: str,
            score: float,
            delta_vs_baseline: float,
            baseline_definition: str,
            final_prompt_weights: list[float] | None,
            image_file: str,
            genome: list[int] | None,
            weight_log_file: str = "",
        ) -> None:
            writer.writerow(
                [
                    slug,
                    prompt,
                    method,
                    search_space,
                    mode,
                    family,
                    profile,
                    t,
                    cfg,
                    f"{float(score):.6f}",
                    f"{float(delta_vs_baseline):+.6f}",
                    baseline_definition,
                    json.dumps(final_prompt_weights or []),
                    image_file or "",
                    (json.dumps(genome) if genome is not None else ""),
                    weight_log_file or "",
                ]
            )

        for i, prompt in enumerate(prompts):
            slug = f"p{i:04d}"
            prompt_dir = os.path.join(args.out_dir, slug)
            ensure_dir(prompt_dir)
            save_this = int(args.save_first_k) < 0 or i < int(args.save_first_k)
            weight_json_path = os.path.join(prompt_dir, "mcts_weight_factor_changes.json")

            print("\n" + "=" * 72)
            print(f"[{slug}] {prompt}")
            print("=" * 72)

            basis = build_prompt_basis(args, prompt, basis_cache)
            basis_emb = encode_prompt_basis(args, ctx, basis, neg_embeds, neg_mask, max_seq=256)
            orig_h, orig_w, h, w = resolve_resolution(args, ctx)

            wanted_labels = [str(x) for x in args.interp_labels]
            basis_indices = _pick_basis_indices(
                basis_emb.labels,
                wanted_labels,
                int(args.interp_k),
            )
            basis_labels = [str(basis_emb.labels[i]) for i in basis_indices]
            weight_profiles = _build_weight_profiles(args, basis_labels, t_values)
            action_bank = _action_bank(families, len(weight_profiles), cfg_values)
            variant4_weight_profiles = _build_variant_only_weight_profiles(basis_labels)
            variant4_action_bank = _action_bank(["nlerp"], len(variant4_weight_profiles), cfg_values)
            global_blend_weight_profiles = _build_global_blend_weight_profiles(basis_labels)
            global_blend_action_bank = _action_bank(families, len(global_blend_weight_profiles), cfg_values)
            global_blend_nlerp_action_bank = _action_bank(["nlerp"], len(global_blend_weight_profiles), cfg_values)
            global_blend_slerp_action_bank = _action_bank(["slerp"], len(global_blend_weight_profiles), cfg_values)
            default_prompt_weights = [1.0 for _ in range(len(basis_indices))]
            print(
                f"  blend labels ({len(basis_labels)}): {', '.join(basis_labels)} "
                f"profiles={len(weight_profiles)} actions={len(action_bank)}"
            )

            baseline_img_path = os.path.join(prompt_dir, "baseline_original.png") if (save_this and args.save_images) else None
            old_guidance = float(args.guidance_scale)
            args.guidance_scale = float(args.baseline_cfg)
            try:
                baseline = run_single_prompt_rollout(
                    args=args,
                    ctx=ctx,
                    reward_ctx=reward_ctx,
                    prompt=prompt,
                    metadata=None,
                    seed=int(args.seed),
                    h=h,
                    w=w,
                    orig_h=orig_h,
                    orig_w=orig_w,
                    pe=basis_emb.orig_pe,
                    pm=basis_emb.orig_pm,
                    ue=basis_emb.orig_ue,
                    um=basis_emb.orig_um,
                    preview_every=int(args.preview_every),
                    save_path=baseline_img_path,
                    tag=f"{slug}_baseline",
                )
            finally:
                args.guidance_scale = old_guidance
            baseline_score = float(baseline.final_score)
            baseline_definition = (
                f"original prompt only; no prompt blending/search; "
                f"cfg={float(args.baseline_cfg):.2f}; seed={int(args.seed)}"
            )
            print(f"  baseline={baseline_score:.4f} cfg={float(args.baseline_cfg):.2f}")
            print(f"  baseline_def={baseline_definition}")
            write_score_row(
                slug=slug,
                prompt=prompt,
                method="baseline",
                search_space="original_prompt",
                mode="fixed",
                family="original",
                profile="original_prompt",
                t="",
                cfg=f"{float(args.baseline_cfg):.2f}",
                score=baseline_score,
                delta_vs_baseline=0.0,
                baseline_definition=baseline_definition,
                final_prompt_weights=default_prompt_weights,
                image_file=baseline_img_path or "",
                genome=None,
            )

            sweep_results: list[dict[str, Any]] = []
            if bool(args.run_sweep):
                for action_idx, (fam, profile_idx, cfg) in enumerate(action_bank):
                    profile = weight_profiles[int(profile_idx)]
                    profile_name = str(profile["name"])
                    profile_t = profile.get("t")
                    genome = [int(action_idx) for _ in range(int(args.steps))]
                    img_path = (
                        os.path.join(
                            prompt_dir,
                            f"sweep_{fam}_{profile_name}_cfg{str(f'{cfg:.2f}').replace('.', 'p')}.png",
                        )
                        if (save_this and args.save_images)
                        else None
                    )
                    trace = run_stepwise_action_rollout(
                        args=args,
                        ctx=ctx,
                        reward_ctx=reward_ctx,
                        prompt=prompt,
                        metadata=None,
                        seed=int(args.seed),
                        h=h,
                        w=w,
                        orig_h=orig_h,
                        orig_w=orig_w,
                        basis_emb=basis_emb,
                        basis_indices=basis_indices,
                        weight_profiles=weight_profiles,
                        action_bank=action_bank,
                        genome=genome,
                        preview_every=int(args.preview_every),
                        save_path=img_path,
                        tag=f"{slug}_sweep_{fam}_{profile_name}",
                    )
                    score = float(trace.final_score)
                    delta = float(score - baseline_score)
                    row = {
                        "family": str(fam),
                        "profile_idx": int(profile_idx),
                        "profile": profile_name,
                        "t": float(profile_t) if profile_t is not None else None,
                        "weights": [float(x) for x in profile["weights"]],
                        "cfg": float(cfg),
                        "score": score,
                        "delta_vs_baseline": delta,
                        "image_file": img_path,
                        "genome": genome,
                        "preview_rewards": trace.preview_rewards,
                        "step_actions": trace.step_actions,
                        "image": trace.final_image,
                    }
                    sweep_results.append(row)
                    write_score_row(
                        slug=slug,
                        prompt=prompt,
                        method="sweep",
                        search_space="single_action_profile",
                        mode="fixed",
                        family=str(fam),
                        profile=profile_name,
                        t=(f"{float(profile_t):.2f}" if profile_t is not None else ""),
                        cfg=f"{float(cfg):.2f}",
                        score=score,
                        delta_vs_baseline=delta,
                        baseline_definition=baseline_definition,
                        final_prompt_weights=default_prompt_weights,
                        image_file=img_path or "",
                        genome=genome,
                    )
                    print(
                        f"  sweep {fam:>5} profile={profile_name} cfg={float(cfg):.2f} "
                        f"score={score:.4f} delta={delta:+.4f}"
                    )

            ga_result: dict[str, Any] | None = None
            if bool(args.run_ga):
                ga_result = run_ga_search(
                    args=args,
                    ctx=ctx,
                    reward_ctx=reward_ctx,
                    prompt=prompt,
                    metadata=None,
                    seed=int(args.seed),
                    h=h,
                    w=w,
                    orig_h=orig_h,
                    orig_w=orig_w,
                    basis_emb=basis_emb,
                    basis_indices=basis_indices,
                    weight_profiles=weight_profiles,
                    action_bank=action_bank,
                    prompt_dir=prompt_dir,
                    save_images=(save_this and args.save_images),
                )
                ga_delta = float(ga_result["best_score"] - baseline_score)
                ga_family, ga_profile, ga_t, ga_cfg = _summarize_step_actions(ga_result.get("best_actions", []))
                write_score_row(
                    slug=slug,
                    prompt=prompt,
                    method="ga",
                    search_space="dynamic_profiles",
                    mode="fixed",
                    family=ga_family,
                    profile=ga_profile,
                    t=ga_t,
                    cfg=ga_cfg,
                    score=float(ga_result["best_score"]),
                    delta_vs_baseline=ga_delta,
                    baseline_definition=baseline_definition,
                    final_prompt_weights=default_prompt_weights,
                    image_file=ga_result.get("best_image_file", "") or "",
                    genome=[int(x) for x in ga_result["best_genome"]],
                )
                print(f"  ga_best={float(ga_result['best_score']):.4f} delta={ga_delta:+.4f}")

            mcts_result: dict[str, Any] | None = None
            mcts_fixed_result: dict[str, Any] | None = None
            mcts_spsa_result: dict[str, Any] | None = None
            mcts_variant4_result: dict[str, Any] | None = None
            mcts_globalblend_fixed_result: dict[str, Any] | None = None
            mcts_globalblend_spsa_result: dict[str, Any] | None = None
            mcts_globalblend_nlerp_spsa_result: dict[str, Any] | None = None
            mcts_globalblend_slerp_spsa_result: dict[str, Any] | None = None
            mcts_weight_logs: dict[str, Any] = {}
            if bool(args.run_mcts):
                if bool(args.run_mcts_design_ablation):
                    mcts_variant4_result = run_mcts_search(
                        args=args,
                        ctx=ctx,
                        reward_ctx=reward_ctx,
                        prompt=prompt,
                        metadata=None,
                        seed=int(args.seed),
                        h=h,
                        w=w,
                        orig_h=orig_h,
                        orig_w=orig_w,
                        basis_emb=basis_emb,
                        basis_indices=basis_indices,
                        weight_profiles=variant4_weight_profiles,
                        action_bank=variant4_action_bank,
                        prompt_dir=prompt_dir,
                        save_images=(save_this and args.save_images),
                        enable_spsa=False,
                        result_prefix="mcts_variant4",
                        search_space_name="variant4",
                    )
                    mcts_globalblend_fixed_result = run_mcts_search(
                        args=args,
                        ctx=ctx,
                        reward_ctx=reward_ctx,
                        prompt=prompt,
                        metadata=None,
                        seed=int(args.seed),
                        h=h,
                        w=w,
                        orig_h=orig_h,
                        orig_w=orig_w,
                        basis_emb=basis_emb,
                        basis_indices=basis_indices,
                        weight_profiles=global_blend_weight_profiles,
                        action_bank=global_blend_action_bank,
                        prompt_dir=prompt_dir,
                        save_images=(save_this and args.save_images),
                        enable_spsa=False,
                        result_prefix="mcts_globalblend_fixed",
                        search_space_name="global_blend",
                    )
                    if bool(args.enable_prompt_weight_spsa):
                        mcts_globalblend_spsa_result = run_mcts_search(
                            args=args,
                            ctx=ctx,
                            reward_ctx=reward_ctx,
                            prompt=prompt,
                            metadata=None,
                            seed=int(args.seed),
                            h=h,
                            w=w,
                            orig_h=orig_h,
                            orig_w=orig_w,
                            basis_emb=basis_emb,
                            basis_indices=basis_indices,
                            weight_profiles=global_blend_weight_profiles,
                            action_bank=global_blend_action_bank,
                            prompt_dir=prompt_dir,
                            save_images=(save_this and args.save_images),
                            enable_spsa=True,
                            result_prefix="mcts_globalblend_spsa",
                            search_space_name="global_blend",
                        )
                        if bool(args.run_mcts_family_spsa_ablation):
                            mcts_globalblend_nlerp_spsa_result = run_mcts_search(
                                args=args,
                                ctx=ctx,
                                reward_ctx=reward_ctx,
                                prompt=prompt,
                                metadata=None,
                                seed=int(args.seed),
                                h=h,
                                w=w,
                                orig_h=orig_h,
                                orig_w=orig_w,
                                basis_emb=basis_emb,
                                basis_indices=basis_indices,
                                weight_profiles=global_blend_weight_profiles,
                                action_bank=global_blend_nlerp_action_bank,
                                prompt_dir=prompt_dir,
                                save_images=(save_this and args.save_images),
                                enable_spsa=True,
                                result_prefix="mcts_globalblend_nlerp_spsa",
                                search_space_name="global_blend_nlerp",
                            )
                            mcts_globalblend_slerp_spsa_result = run_mcts_search(
                                args=args,
                                ctx=ctx,
                                reward_ctx=reward_ctx,
                                prompt=prompt,
                                metadata=None,
                                seed=int(args.seed),
                                h=h,
                                w=w,
                                orig_h=orig_h,
                                orig_w=orig_w,
                                basis_emb=basis_emb,
                                basis_indices=basis_indices,
                                weight_profiles=global_blend_weight_profiles,
                                action_bank=global_blend_slerp_action_bank,
                                prompt_dir=prompt_dir,
                                save_images=(save_this and args.save_images),
                                enable_spsa=True,
                                result_prefix="mcts_globalblend_slerp_spsa",
                                search_space_name="global_blend_slerp",
                            )
                    design_arms = [
                        ("mcts_variant4", mcts_variant4_result),
                        ("mcts_globalblend_fixed", mcts_globalblend_fixed_result),
                    ]
                    if mcts_globalblend_spsa_result is not None:
                        design_arms.append(("mcts_globalblend_spsa", mcts_globalblend_spsa_result))
                    if mcts_globalblend_nlerp_spsa_result is not None:
                        design_arms.append(("mcts_globalblend_nlerp_spsa", mcts_globalblend_nlerp_spsa_result))
                    if mcts_globalblend_slerp_spsa_result is not None:
                        design_arms.append(("mcts_globalblend_slerp_spsa", mcts_globalblend_slerp_spsa_result))
                    for method_name, cur_res in design_arms:
                        cur_score = float(cur_res.get("final_score", cur_res["best_score"]))
                        cur_delta = float(cur_score - baseline_score)
                        cur_family, cur_profile, cur_t, cur_cfg = _summarize_step_actions(cur_res.get("best_actions", []))
                        _write_mcts_weight_history_rows(weight_writer, slug, prompt, method_name, cur_res)
                        mcts_weight_logs[method_name] = {
                            "final_prompt_weights": cur_res.get("final_prompt_weights", []),
                            "spsa_history": cur_res.get("spsa_history", []),
                        }
                        write_score_row(
                            slug=slug,
                            prompt=prompt,
                            method=method_name,
                            search_space=str(cur_res.get("search_space_name", "mixed")),
                            mode=str(cur_res.get("mode", "fixed")),
                            family=cur_family,
                            profile=cur_profile,
                            t=cur_t,
                            cfg=cur_cfg,
                            score=cur_score,
                            delta_vs_baseline=cur_delta,
                            baseline_definition=baseline_definition,
                            final_prompt_weights=cur_res.get("final_prompt_weights", cur_res.get("best_prompt_weights", [])),
                            image_file=cur_res.get("best_image_file", "") or "",
                            genome=[int(x) for x in cur_res["best_genome"]],
                            weight_log_file=weight_json_path,
                        )
                    best_arm_name, mcts_result = max(
                        design_arms,
                        key=lambda item: float(item[1].get("final_score", item[1]["best_score"])),
                    )
                    print(
                        "  "
                        f"variant4={float(mcts_variant4_result.get('final_score', mcts_variant4_result['best_score'])):.4f} "
                        f"delta={float(mcts_variant4_result.get('final_score', mcts_variant4_result['best_score']) - baseline_score):+.4f}  "
                        f"global_fixed={float(mcts_globalblend_fixed_result.get('final_score', mcts_globalblend_fixed_result['best_score'])):.4f} "
                        f"delta={float(mcts_globalblend_fixed_result.get('final_score', mcts_globalblend_fixed_result['best_score']) - baseline_score):+.4f}"
                        + (
                            f"  global_spsa={float(mcts_globalblend_spsa_result.get('final_score', mcts_globalblend_spsa_result['best_score'])):.4f} "
                            f"delta={float(mcts_globalblend_spsa_result.get('final_score', mcts_globalblend_spsa_result['best_score']) - baseline_score):+.4f} "
                            f"w={mcts_globalblend_spsa_result.get('final_prompt_weights', mcts_globalblend_spsa_result.get('best_prompt_weights', []))}"
                            if mcts_globalblend_spsa_result is not None
                            else ""
                        )
                        + (
                            f"  nlerp_spsa={float(mcts_globalblend_nlerp_spsa_result.get('final_score', mcts_globalblend_nlerp_spsa_result['best_score'])):.4f} "
                            f"delta={float(mcts_globalblend_nlerp_spsa_result.get('final_score', mcts_globalblend_nlerp_spsa_result['best_score']) - baseline_score):+.4f} "
                            f"w={mcts_globalblend_nlerp_spsa_result.get('final_prompt_weights', mcts_globalblend_nlerp_spsa_result.get('best_prompt_weights', []))}"
                            if mcts_globalblend_nlerp_spsa_result is not None
                            else ""
                        )
                        + (
                            f"  slerp_spsa={float(mcts_globalblend_slerp_spsa_result.get('final_score', mcts_globalblend_slerp_spsa_result['best_score'])):.4f} "
                            f"delta={float(mcts_globalblend_slerp_spsa_result.get('final_score', mcts_globalblend_slerp_spsa_result['best_score']) - baseline_score):+.4f} "
                            f"w={mcts_globalblend_slerp_spsa_result.get('final_prompt_weights', mcts_globalblend_slerp_spsa_result.get('best_prompt_weights', []))}"
                            if mcts_globalblend_slerp_spsa_result is not None
                            else ""
                        )
                        + f"  best_arm={best_arm_name}"
                    )
                elif bool(args.run_mcts_weight_ablation):
                    mcts_fixed_result = run_mcts_search(
                        args=args,
                        ctx=ctx,
                        reward_ctx=reward_ctx,
                        prompt=prompt,
                        metadata=None,
                        seed=int(args.seed),
                        h=h,
                        w=w,
                        orig_h=orig_h,
                        orig_w=orig_w,
                        basis_emb=basis_emb,
                        basis_indices=basis_indices,
                        weight_profiles=weight_profiles,
                        action_bank=action_bank,
                        prompt_dir=prompt_dir,
                        save_images=(save_this and args.save_images),
                        enable_spsa=False,
                        result_prefix="mcts_fixed",
                        search_space_name="dynamic_profiles",
                    )
                    mcts_spsa_result = run_mcts_search(
                        args=args,
                        ctx=ctx,
                        reward_ctx=reward_ctx,
                        prompt=prompt,
                        metadata=None,
                        seed=int(args.seed),
                        h=h,
                        w=w,
                        orig_h=orig_h,
                        orig_w=orig_w,
                        basis_emb=basis_emb,
                        basis_indices=basis_indices,
                        weight_profiles=weight_profiles,
                        action_bank=action_bank,
                        prompt_dir=prompt_dir,
                        save_images=(save_this and args.save_images),
                        enable_spsa=True,
                        result_prefix="mcts_spsa",
                        search_space_name="dynamic_profiles",
                    )
                    mcts_result = mcts_spsa_result
                    for method_name, cur_res in (("mcts_fixed", mcts_fixed_result), ("mcts_spsa", mcts_spsa_result)):
                        cur_score = float(cur_res.get("final_score", cur_res["best_score"]))
                        cur_delta = float(cur_score - baseline_score)
                        cur_family, cur_profile, cur_t, cur_cfg = _summarize_step_actions(cur_res.get("best_actions", []))
                        _write_mcts_weight_history_rows(weight_writer, slug, prompt, method_name, cur_res)
                        mcts_weight_logs[method_name] = {
                            "final_prompt_weights": cur_res.get("final_prompt_weights", []),
                            "spsa_history": cur_res.get("spsa_history", []),
                        }
                        write_score_row(
                            slug=slug,
                            prompt=prompt,
                            method=method_name,
                            search_space=str(cur_res.get("search_space_name", "mixed")),
                            mode=str(cur_res.get("mode", "fixed")),
                            family=cur_family,
                            profile=cur_profile,
                            t=cur_t,
                            cfg=cur_cfg,
                            score=cur_score,
                            delta_vs_baseline=cur_delta,
                            baseline_definition=baseline_definition,
                            final_prompt_weights=cur_res.get("final_prompt_weights", cur_res.get("best_prompt_weights", [])),
                            image_file=cur_res.get("best_image_file", "") or "",
                            genome=[int(x) for x in cur_res["best_genome"]],
                            weight_log_file=weight_json_path,
                        )
                    print(
                        "  "
                        f"mcts_fixed={float(mcts_fixed_result.get('final_score', mcts_fixed_result['best_score'])):.4f} "
                        f"delta={float(mcts_fixed_result.get('final_score', mcts_fixed_result['best_score']) - baseline_score):+.4f} "
                        f"w={mcts_fixed_result.get('final_prompt_weights', mcts_fixed_result.get('best_prompt_weights', []))}  "
                        f"mcts_spsa={float(mcts_spsa_result.get('final_score', mcts_spsa_result['best_score'])):.4f} "
                        f"delta={float(mcts_spsa_result.get('final_score', mcts_spsa_result['best_score']) - baseline_score):+.4f} "
                        f"w={mcts_spsa_result.get('final_prompt_weights', mcts_spsa_result.get('best_prompt_weights', []))}"
                    )
                else:
                    mcts_result = run_mcts_search(
                        args=args,
                        ctx=ctx,
                        reward_ctx=reward_ctx,
                        prompt=prompt,
                        metadata=None,
                        seed=int(args.seed),
                        h=h,
                        w=w,
                        orig_h=orig_h,
                        orig_w=orig_w,
                        basis_emb=basis_emb,
                        basis_indices=basis_indices,
                        weight_profiles=weight_profiles,
                        action_bank=action_bank,
                        prompt_dir=prompt_dir,
                        save_images=(save_this and args.save_images),
                        enable_spsa=bool(args.enable_prompt_weight_spsa),
                        result_prefix="mcts",
                        search_space_name="dynamic_profiles",
                    )
                    mcts_score = float(mcts_result.get("final_score", mcts_result["best_score"]))
                    mcts_delta = float(mcts_score - baseline_score)
                    _write_mcts_weight_history_rows(weight_writer, slug, prompt, "mcts", mcts_result)
                    mcts_weight_logs["mcts"] = {
                        "final_prompt_weights": mcts_result.get("final_prompt_weights", []),
                        "spsa_history": mcts_result.get("spsa_history", []),
                    }
                    mcts_family, mcts_profile, mcts_t, mcts_cfg = _summarize_step_actions(mcts_result.get("best_actions", []))
                    write_score_row(
                        slug=slug,
                        prompt=prompt,
                        method="mcts",
                        search_space=str(mcts_result.get("search_space_name", "dynamic_profiles")),
                        mode=str(mcts_result.get("mode", "fixed")),
                        family=mcts_family,
                        profile=mcts_profile,
                        t=mcts_t,
                        cfg=mcts_cfg,
                        score=mcts_score,
                        delta_vs_baseline=mcts_delta,
                        baseline_definition=baseline_definition,
                        final_prompt_weights=mcts_result.get("final_prompt_weights", mcts_result.get("best_prompt_weights", [])),
                        image_file=mcts_result.get("best_image_file", "") or "",
                        genome=[int(x) for x in mcts_result["best_genome"]],
                        weight_log_file=weight_json_path,
                    )
                    print(
                        f"  mcts_best={mcts_score:.4f} delta={mcts_delta:+.4f} "
                        f"mode={mcts_result.get('mode', 'fixed')} "
                        f"w={mcts_result.get('final_prompt_weights', mcts_result.get('best_prompt_weights', []))}"
                    )

            row_out: dict[str, Any] = {
                "slug": slug,
                "prompt": prompt,
                "basis_labels": list(basis_emb.labels),
                "basis_texts": {str(c.label): str(c.text) for c in basis.candidates},
                "interp_labels_requested": wanted_labels,
                "interp_basis_indices": [int(x) for x in basis_indices],
                "interp_basis_labels": basis_labels,
                "weight_profiles": weight_profiles,
                "action_bank": [dict({"action_idx": int(k)}, **_action_detail(int(k), action_bank, weight_profiles)) for k in range(len(action_bank))],
                "baseline_score": baseline_score,
                "baseline_cfg": float(args.baseline_cfg),
                "baseline_definition": baseline_definition,
                "sweep": [
                    {
                        "family": str(r["family"]),
                        "profile_idx": int(r["profile_idx"]),
                        "profile": str(r["profile"]),
                        "t": float(r["t"]) if r["t"] is not None else None,
                        "weights": [float(x) for x in r["weights"]],
                        "cfg": float(r["cfg"]),
                        "score": float(r["score"]),
                        "delta_vs_baseline": float(r["delta_vs_baseline"]),
                        "image_file": r["image_file"],
                        "genome": [int(x) for x in r["genome"]],
                        "preview_rewards": [float(x) for x in r["preview_rewards"]],
                        "step_actions": r["step_actions"],
                    }
                    for r in sweep_results
                ],
                "ga": ga_result,
                "mcts": mcts_result,
                "mcts_mode": (str(mcts_result.get("mode")) if isinstance(mcts_result, dict) and mcts_result.get("mode") is not None else None),
            }
            if bool(args.run_mcts_design_ablation):
                row_out["mcts_best_arm"] = (
                    str(mcts_result.get("search_space_name", "mcts"))
                    + ":"
                    + str(mcts_result.get("mode", "fixed"))
                    if isinstance(mcts_result, dict)
                    else None
                )
                row_out["mcts_design_action_spaces"] = {
                    "variant4_weight_profiles": variant4_weight_profiles,
                    "variant4_action_bank": [
                        dict({"action_idx": int(k)}, **_action_detail(int(k), variant4_action_bank, variant4_weight_profiles))
                        for k in range(len(variant4_action_bank))
                    ],
                    "global_blend_weight_profiles": global_blend_weight_profiles,
                    "global_blend_action_bank": [
                        dict({"action_idx": int(k)}, **_action_detail(int(k), global_blend_action_bank, global_blend_weight_profiles))
                        for k in range(len(global_blend_action_bank))
                    ],
                    "global_blend_nlerp_action_bank": [
                        dict({"action_idx": int(k)}, **_action_detail(int(k), global_blend_nlerp_action_bank, global_blend_weight_profiles))
                        for k in range(len(global_blend_nlerp_action_bank))
                    ],
                    "global_blend_slerp_action_bank": [
                        dict({"action_idx": int(k)}, **_action_detail(int(k), global_blend_slerp_action_bank, global_blend_weight_profiles))
                        for k in range(len(global_blend_slerp_action_bank))
                    ],
                }
                row_out["mcts_design_ablation"] = {
                    "variant4": mcts_variant4_result,
                    "global_blend_fixed": mcts_globalblend_fixed_result,
                    "global_blend_spsa": mcts_globalblend_spsa_result,
                    "global_blend_nlerp_spsa": mcts_globalblend_nlerp_spsa_result,
                    "global_blend_slerp_spsa": mcts_globalblend_slerp_spsa_result,
                }
            if mcts_fixed_result is not None or mcts_spsa_result is not None:
                row_out["mcts_ablation"] = {
                    "fixed": mcts_fixed_result,
                    "spsa": mcts_spsa_result,
                }
            if len(mcts_weight_logs) > 0:
                save_json(weight_json_path, mcts_weight_logs)
                row_out["mcts_weight_factor_log_file"] = weight_json_path
            if save_this and args.save_images:
                mcts_panel_item = None
                if mcts_result is not None and mcts_result.get("best_image_file"):
                    mcts_img = Image.open(mcts_result["best_image_file"]).convert("RGB")
                    mcts_panel_item = {
                        "image": mcts_img,
                        "score": float(mcts_result.get("final_score", mcts_result["best_score"])),
                    }
                ga_panel_item = None
                if ga_result is not None and ga_result.get("best_image_file"):
                    ga_img = Image.open(ga_result["best_image_file"]).convert("RGB")
                    ga_panel_item = {"image": ga_img, "score": float(ga_result["best_score"])}
                panel_path = os.path.join(prompt_dir, "comparison_panel.png")
                _build_summary_panel(
                    baseline_img=baseline.final_image,
                    baseline_score=baseline_score,
                    sweep_items=sweep_results,
                    mcts_item=mcts_panel_item,
                    ga_item=ga_panel_item,
                    out_path=panel_path,
                )
                row_out["panel_file"] = panel_path

            save_json(os.path.join(prompt_dir, "result.json"), row_out)
            all_rows.append(row_out)

    # Aggregate
    sweep_agg: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        base = float(row["baseline_score"])
        for srow in row.get("sweep", []):
            key = f"{srow['family']}@{srow['profile']}@{float(srow['cfg']):.2f}"
            agg = sweep_agg.setdefault(
                key,
                {
                    "family": srow["family"],
                    "profile": srow["profile"],
                    "t": float(srow["t"]) if srow["t"] is not None else None,
                    "cfg": float(srow["cfg"]),
                    "scores": [],
                    "deltas": [],
                },
            )
            agg["scores"].append(float(srow["score"]))
            agg["deltas"].append(float(srow["score"]) - base)

    aggregate_sweep_rows: list[dict[str, Any]] = []
    for key in sorted(sweep_agg.keys()):
        cur = sweep_agg[key]
        s = np.asarray(cur["scores"], dtype=np.float64)
        d = np.asarray(cur["deltas"], dtype=np.float64)
        aggregate_sweep_rows.append(
            {
                "family": str(cur["family"]),
                "profile": str(cur["profile"]),
                "t": float(cur["t"]) if cur["t"] is not None else None,
                "cfg": float(cur["cfg"]),
                "count": int(s.size),
                "mean_score": float(s.mean()) if s.size else 0.0,
                "mean_delta_vs_baseline": float(d.mean()) if d.size else 0.0,
                "std_delta_vs_baseline": float(d.std()) if d.size else 0.0,
            }
        )

    ga_scores = [float(r["ga"]["best_score"]) for r in all_rows if isinstance(r.get("ga"), dict)]
    ga_deltas = [float(r["ga"]["best_score"]) - float(r["baseline_score"]) for r in all_rows if isinstance(r.get("ga"), dict)]
    mcts_scores = [
        float(r["mcts"].get("final_score", r["mcts"]["best_score"]))
        for r in all_rows
        if isinstance(r.get("mcts"), dict)
    ]
    mcts_deltas = [
        float(r["mcts"].get("final_score", r["mcts"]["best_score"])) - float(r["baseline_score"])
        for r in all_rows
        if isinstance(r.get("mcts"), dict)
    ]
    mcts_fixed_scores = [
        float(r["mcts_ablation"]["fixed"].get("final_score", r["mcts_ablation"]["fixed"]["best_score"]))
        for r in all_rows
        if isinstance(r.get("mcts_ablation"), dict) and isinstance(r["mcts_ablation"].get("fixed"), dict)
    ]
    mcts_fixed_deltas = [
        float(r["mcts_ablation"]["fixed"].get("final_score", r["mcts_ablation"]["fixed"]["best_score"])) - float(r["baseline_score"])
        for r in all_rows
        if isinstance(r.get("mcts_ablation"), dict) and isinstance(r["mcts_ablation"].get("fixed"), dict)
    ]
    mcts_spsa_scores = [
        float(r["mcts_ablation"]["spsa"].get("final_score", r["mcts_ablation"]["spsa"]["best_score"]))
        for r in all_rows
        if isinstance(r.get("mcts_ablation"), dict) and isinstance(r["mcts_ablation"].get("spsa"), dict)
    ]
    mcts_spsa_deltas = [
        float(r["mcts_ablation"]["spsa"].get("final_score", r["mcts_ablation"]["spsa"]["best_score"])) - float(r["baseline_score"])
        for r in all_rows
        if isinstance(r.get("mcts_ablation"), dict) and isinstance(r["mcts_ablation"].get("spsa"), dict)
    ]
    mcts_spsa_gain_over_fixed = [
        float(r["mcts_ablation"]["spsa"].get("final_score", r["mcts_ablation"]["spsa"]["best_score"]))
        - float(r["mcts_ablation"]["fixed"].get("final_score", r["mcts_ablation"]["fixed"]["best_score"]))
        for r in all_rows
        if isinstance(r.get("mcts_ablation"), dict)
        and isinstance(r["mcts_ablation"].get("fixed"), dict)
        and isinstance(r["mcts_ablation"].get("spsa"), dict)
    ]
    mcts_variant4_scores = [
        float(r["mcts_design_ablation"]["variant4"].get("final_score", r["mcts_design_ablation"]["variant4"]["best_score"]))
        for r in all_rows
        if isinstance(r.get("mcts_design_ablation"), dict) and isinstance(r["mcts_design_ablation"].get("variant4"), dict)
    ]
    mcts_variant4_deltas = [
        float(r["mcts_design_ablation"]["variant4"].get("final_score", r["mcts_design_ablation"]["variant4"]["best_score"])) - float(r["baseline_score"])
        for r in all_rows
        if isinstance(r.get("mcts_design_ablation"), dict) and isinstance(r["mcts_design_ablation"].get("variant4"), dict)
    ]
    mcts_globalblend_fixed_scores = [
        float(r["mcts_design_ablation"]["global_blend_fixed"].get("final_score", r["mcts_design_ablation"]["global_blend_fixed"]["best_score"]))
        for r in all_rows
        if isinstance(r.get("mcts_design_ablation"), dict) and isinstance(r["mcts_design_ablation"].get("global_blend_fixed"), dict)
    ]
    mcts_globalblend_fixed_deltas = [
        float(r["mcts_design_ablation"]["global_blend_fixed"].get("final_score", r["mcts_design_ablation"]["global_blend_fixed"]["best_score"])) - float(r["baseline_score"])
        for r in all_rows
        if isinstance(r.get("mcts_design_ablation"), dict) and isinstance(r["mcts_design_ablation"].get("global_blend_fixed"), dict)
    ]
    mcts_globalblend_spsa_scores = [
        float(r["mcts_design_ablation"]["global_blend_spsa"].get("final_score", r["mcts_design_ablation"]["global_blend_spsa"]["best_score"]))
        for r in all_rows
        if isinstance(r.get("mcts_design_ablation"), dict) and isinstance(r["mcts_design_ablation"].get("global_blend_spsa"), dict)
    ]
    mcts_globalblend_spsa_deltas = [
        float(r["mcts_design_ablation"]["global_blend_spsa"].get("final_score", r["mcts_design_ablation"]["global_blend_spsa"]["best_score"])) - float(r["baseline_score"])
        for r in all_rows
        if isinstance(r.get("mcts_design_ablation"), dict) and isinstance(r["mcts_design_ablation"].get("global_blend_spsa"), dict)
    ]
    mcts_globalblend_spsa_gain_over_fixed = [
        float(r["mcts_design_ablation"]["global_blend_spsa"].get("final_score", r["mcts_design_ablation"]["global_blend_spsa"]["best_score"]))
        - float(r["mcts_design_ablation"]["global_blend_fixed"].get("final_score", r["mcts_design_ablation"]["global_blend_fixed"]["best_score"]))
        for r in all_rows
        if isinstance(r.get("mcts_design_ablation"), dict)
        and isinstance(r["mcts_design_ablation"].get("global_blend_fixed"), dict)
        and isinstance(r["mcts_design_ablation"].get("global_blend_spsa"), dict)
    ]
    mcts_globalblend_nlerp_spsa_scores = [
        float(r["mcts_design_ablation"]["global_blend_nlerp_spsa"].get("final_score", r["mcts_design_ablation"]["global_blend_nlerp_spsa"]["best_score"]))
        for r in all_rows
        if isinstance(r.get("mcts_design_ablation"), dict)
        and isinstance(r["mcts_design_ablation"].get("global_blend_nlerp_spsa"), dict)
    ]
    mcts_globalblend_nlerp_spsa_deltas = [
        float(r["mcts_design_ablation"]["global_blend_nlerp_spsa"].get("final_score", r["mcts_design_ablation"]["global_blend_nlerp_spsa"]["best_score"])) - float(r["baseline_score"])
        for r in all_rows
        if isinstance(r.get("mcts_design_ablation"), dict)
        and isinstance(r["mcts_design_ablation"].get("global_blend_nlerp_spsa"), dict)
    ]
    mcts_globalblend_slerp_spsa_scores = [
        float(r["mcts_design_ablation"]["global_blend_slerp_spsa"].get("final_score", r["mcts_design_ablation"]["global_blend_slerp_spsa"]["best_score"]))
        for r in all_rows
        if isinstance(r.get("mcts_design_ablation"), dict)
        and isinstance(r["mcts_design_ablation"].get("global_blend_slerp_spsa"), dict)
    ]
    mcts_globalblend_slerp_spsa_deltas = [
        float(r["mcts_design_ablation"]["global_blend_slerp_spsa"].get("final_score", r["mcts_design_ablation"]["global_blend_slerp_spsa"]["best_score"])) - float(r["baseline_score"])
        for r in all_rows
        if isinstance(r.get("mcts_design_ablation"), dict)
        and isinstance(r["mcts_design_ablation"].get("global_blend_slerp_spsa"), dict)
    ]
    mcts_globalblend_nlerp_spsa_gain_over_fixed = [
        float(r["mcts_design_ablation"]["global_blend_nlerp_spsa"].get("final_score", r["mcts_design_ablation"]["global_blend_nlerp_spsa"]["best_score"]))
        - float(r["mcts_design_ablation"]["global_blend_fixed"].get("final_score", r["mcts_design_ablation"]["global_blend_fixed"]["best_score"]))
        for r in all_rows
        if isinstance(r.get("mcts_design_ablation"), dict)
        and isinstance(r["mcts_design_ablation"].get("global_blend_nlerp_spsa"), dict)
        and isinstance(r["mcts_design_ablation"].get("global_blend_fixed"), dict)
    ]
    mcts_globalblend_slerp_spsa_gain_over_fixed = [
        float(r["mcts_design_ablation"]["global_blend_slerp_spsa"].get("final_score", r["mcts_design_ablation"]["global_blend_slerp_spsa"]["best_score"]))
        - float(r["mcts_design_ablation"]["global_blend_fixed"].get("final_score", r["mcts_design_ablation"]["global_blend_fixed"]["best_score"]))
        for r in all_rows
        if isinstance(r.get("mcts_design_ablation"), dict)
        and isinstance(r["mcts_design_ablation"].get("global_blend_slerp_spsa"), dict)
        and isinstance(r["mcts_design_ablation"].get("global_blend_fixed"), dict)
    ]
    mcts_globalblend_fixed_gain_over_variant4 = [
        float(r["mcts_design_ablation"]["global_blend_fixed"].get("final_score", r["mcts_design_ablation"]["global_blend_fixed"]["best_score"]))
        - float(r["mcts_design_ablation"]["variant4"].get("final_score", r["mcts_design_ablation"]["variant4"]["best_score"]))
        for r in all_rows
        if isinstance(r.get("mcts_design_ablation"), dict)
        and isinstance(r["mcts_design_ablation"].get("global_blend_fixed"), dict)
        and isinstance(r["mcts_design_ablation"].get("variant4"), dict)
    ]
    mcts_globalblend_spsa_gain_over_variant4 = [
        float(r["mcts_design_ablation"]["global_blend_spsa"].get("final_score", r["mcts_design_ablation"]["global_blend_spsa"]["best_score"]))
        - float(r["mcts_design_ablation"]["variant4"].get("final_score", r["mcts_design_ablation"]["variant4"]["best_score"]))
        for r in all_rows
        if isinstance(r.get("mcts_design_ablation"), dict)
        and isinstance(r["mcts_design_ablation"].get("global_blend_spsa"), dict)
        and isinstance(r["mcts_design_ablation"].get("variant4"), dict)
    ]
    baseline_scores = [float(r["baseline_score"]) for r in all_rows]

    summary = {
        "num_prompts": len(all_rows),
        "reward_type": str(args.reward_type),
        "families": families,
        "interp_k": int(args.interp_k),
        "interp_labels_requested": [str(x) for x in args.interp_labels],
        "interp_values": [float(x) for x in t_values],
        "rows": all_rows,
        "aggregate_sweep": aggregate_sweep_rows,
        "aggregate_global": {
            "mean_baseline": float(np.mean(baseline_scores)) if baseline_scores else 0.0,
            "mean_ga_best": float(np.mean(ga_scores)) if ga_scores else None,
            "mean_ga_delta_vs_baseline": float(np.mean(ga_deltas)) if ga_deltas else None,
            "mean_mcts_best": float(np.mean(mcts_scores)) if mcts_scores else None,
            "mean_mcts_delta_vs_baseline": float(np.mean(mcts_deltas)) if mcts_deltas else None,
            "mean_mcts_fixed_best": float(np.mean(mcts_fixed_scores)) if mcts_fixed_scores else None,
            "mean_mcts_fixed_delta_vs_baseline": float(np.mean(mcts_fixed_deltas)) if mcts_fixed_deltas else None,
            "mean_mcts_spsa_best": float(np.mean(mcts_spsa_scores)) if mcts_spsa_scores else None,
            "mean_mcts_spsa_delta_vs_baseline": float(np.mean(mcts_spsa_deltas)) if mcts_spsa_deltas else None,
            "mean_mcts_spsa_gain_over_fixed": float(np.mean(mcts_spsa_gain_over_fixed)) if mcts_spsa_gain_over_fixed else None,
            "mean_mcts_variant4_best": float(np.mean(mcts_variant4_scores)) if mcts_variant4_scores else None,
            "mean_mcts_variant4_delta_vs_baseline": float(np.mean(mcts_variant4_deltas)) if mcts_variant4_deltas else None,
            "mean_mcts_globalblend_fixed_best": float(np.mean(mcts_globalblend_fixed_scores)) if mcts_globalblend_fixed_scores else None,
            "mean_mcts_globalblend_fixed_delta_vs_baseline": float(np.mean(mcts_globalblend_fixed_deltas)) if mcts_globalblend_fixed_deltas else None,
            "mean_mcts_globalblend_spsa_best": float(np.mean(mcts_globalblend_spsa_scores)) if mcts_globalblend_spsa_scores else None,
            "mean_mcts_globalblend_spsa_delta_vs_baseline": float(np.mean(mcts_globalblend_spsa_deltas)) if mcts_globalblend_spsa_deltas else None,
            "mean_mcts_globalblend_spsa_gain_over_fixed": float(np.mean(mcts_globalblend_spsa_gain_over_fixed)) if mcts_globalblend_spsa_gain_over_fixed else None,
            "mean_mcts_globalblend_nlerp_spsa_best": float(np.mean(mcts_globalblend_nlerp_spsa_scores)) if mcts_globalblend_nlerp_spsa_scores else None,
            "mean_mcts_globalblend_nlerp_spsa_delta_vs_baseline": float(np.mean(mcts_globalblend_nlerp_spsa_deltas)) if mcts_globalblend_nlerp_spsa_deltas else None,
            "mean_mcts_globalblend_nlerp_spsa_gain_over_fixed": float(np.mean(mcts_globalblend_nlerp_spsa_gain_over_fixed)) if mcts_globalblend_nlerp_spsa_gain_over_fixed else None,
            "mean_mcts_globalblend_slerp_spsa_best": float(np.mean(mcts_globalblend_slerp_spsa_scores)) if mcts_globalblend_slerp_spsa_scores else None,
            "mean_mcts_globalblend_slerp_spsa_delta_vs_baseline": float(np.mean(mcts_globalblend_slerp_spsa_deltas)) if mcts_globalblend_slerp_spsa_deltas else None,
            "mean_mcts_globalblend_slerp_spsa_gain_over_fixed": float(np.mean(mcts_globalblend_slerp_spsa_gain_over_fixed)) if mcts_globalblend_slerp_spsa_gain_over_fixed else None,
            "mean_mcts_globalblend_fixed_gain_over_variant4": float(np.mean(mcts_globalblend_fixed_gain_over_variant4)) if mcts_globalblend_fixed_gain_over_variant4 else None,
            "mean_mcts_globalblend_spsa_gain_over_variant4": float(np.mean(mcts_globalblend_spsa_gain_over_variant4)) if mcts_globalblend_spsa_gain_over_variant4 else None,
        },
        "scores_tsv": tsv_path,
        "mcts_weight_changes_tsv": mcts_weight_tsv_path,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    save_json(summary_path, summary)
    save_basis_cache(args.basis_cache_json, basis_cache)

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 72)
    print(f"Done. prompts={len(all_rows)} elapsed_sec={elapsed:.2f}")
    print(f"Scores TSV: {tsv_path}")
    print(f"Summary  : {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
