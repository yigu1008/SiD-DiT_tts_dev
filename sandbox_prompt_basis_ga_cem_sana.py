"""
Sandbox: prompt-basis GA + CEM for RL-flavored per-step conditioning (SiD-SANA).

Core method:
  - Build a small Qwen prompt basis (balanced/subject/composition/texture).
  - Outer GA searches discrete structure:
      * prompt subset (basis indices)
      * global blend family (nlerp/slerp)
  - Inner CEM optimizes continuous weight policy:
      w_t = softmax(b + a * u_t), where u_t is scheduler progress.
  - Rollout reward is final fitness.

Baselines:
  1) original prompt only
  2) one-shot Qwen (balanced candidate)
  3) fixed equal-weight blending
  4) naive equal-weight blend ablation (nlerp vs slerp, optional)
  5) GA(prompt structure) + CEM(weights)
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import numpy as np

import sampling_unified as su
from eval_and_log import ensure_dir, make_score_panel, save_json, to_jsonable
from ga_prompt_search import PromptGASearchResult, run_ga_prompt_search
from prompt_basis import (
    BASIS_LABELS,
    PromptBasis,
    build_prompt_basis,
    encode_prompt_basis,
    load_basis_cache,
    save_basis_cache,
)
from rollout_runner import resolve_resolution, run_dynamic_rollout, run_single_prompt_rollout
from weight_policy import WeightParams


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sandbox prompt-basis GA+CEM controller (SiD-SANA).")

    # I/O
    p.add_argument("--prompt", type=str, default="a studio portrait of an elderly woman smiling, soft window light, 85mm lens, photorealistic")
    p.add_argument("--prompt_file", type=str, default=None)
    p.add_argument("--max_prompts", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="./sandbox_prompt_basis_out")
    p.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_first_k", type=int, default=10)
    p.add_argument("--basis_cache_json", type=str, default=None)

    # Model/pipeline
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

    # Reward
    p.add_argument("--reward_type", choices=["imagereward", "auto", "unifiedreward", "unified", "hpsv2", "blend"], default="imagereward")
    p.add_argument("--reward_device", type=str, default="cpu")
    p.add_argument("--reward_model", type=str, default="CodeGoat24/UnifiedReward-2.0-qwen3vl-4b")
    p.add_argument("--unifiedreward_model", type=str, default=None)
    p.add_argument("--image_reward_model", type=str, default="ImageReward-v1.0")
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

    # Rollout logging
    p.add_argument("--preview_every", type=int, default=1)

    # Fixed baseline blend
    p.add_argument("--basis_k", type=int, default=3, help="How many prompts are selected in each basis subset.")
    p.add_argument("--fixed_family", choices=["nlerp", "slerp"], default="nlerp")
    p.add_argument("--run_naive_blend_ablation", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--naive_blend_k", type=int, default=2, help="Prompt count for naive nlerp/slerp equal-weight test.")

    # CEM (inner)
    p.add_argument("--cem_iters", type=int, default=4)
    p.add_argument("--cem_population", type=int, default=8)
    p.add_argument("--cem_elite_frac", type=float, default=0.25)
    p.add_argument("--cem_init_std", type=float, default=1.0)
    p.add_argument("--cem_min_std", type=float, default=0.05)
    p.add_argument("--cem_clip", type=float, default=3.0)

    # GA (outer)
    p.add_argument("--ga_population", type=int, default=8)
    p.add_argument("--ga_generations", type=int, default=6)
    p.add_argument("--ga_elites", type=int, default=2)
    p.add_argument("--ga_mutation_prob", type=float, default=0.15)
    p.add_argument("--ga_selection", choices=["rank", "tournament"], default="rank")
    p.add_argument("--ga_rank_pressure", type=float, default=1.7)
    p.add_argument("--ga_tournament_k", type=int, default=3)
    p.add_argument("--ga_log_topk", type=int, default=3)
    p.add_argument("--ga_anchor_family", choices=["nlerp", "slerp"], default="nlerp")
    p.add_argument("--ga_eval_cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--blend_families", nargs="+", default=["nlerp", "slerp"])
    p.add_argument(
        "--hybrid_reward_mode",
        choices=["orig_only", "mixed_shared_image", "paired_rollout"],
        default="orig_only",
        help="Hybrid GA objective mode. paired_rollout is NFE-intensive.",
    )
    p.add_argument("--hybrid_reward_mix_orig", type=float, default=1.0)
    p.add_argument("--hybrid_reward_mix_cond", type=float, default=0.0)
    p.add_argument("--hybrid_reward_mix_delta", type=float, default=0.0, help="Weight for (orig_score - ref_score).")
    p.add_argument(
        "--hybrid_cond_reduce",
        choices=["max", "mean", "min"],
        default="max",
        help="How to reduce per-subset conditioned prompt reward into cond_score.",
    )

    # Placeholders for load_reward(geneval path)
    p.add_argument("--reward_url", type=str, default=None)
    p.add_argument("--geneval_python", type=str, default=None)
    p.add_argument("--geneval_repo", type=str, default=None)
    p.add_argument("--detector_path", type=str, default=None)
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


def _pick_label_index(basis: PromptBasis, label: str) -> int:
    key = str(label).strip().lower()
    for i, c in enumerate(basis.candidates):
        if str(c.label).strip().lower() == key:
            return int(i)
    return 0


def _subset_indices(pool_size: int, basis_k: int) -> list[int]:
    k = max(1, min(int(basis_k), int(pool_size)))
    return list(range(k))


def _rollout_weight_evolution(rollout: Any) -> list[dict[str, Any]]:
    return [
        {
            "step": int(st.step),
            "sigma": float(st.sigma),
            "progress": float(st.progress),
            "blend_family": str(st.blend_family),
            "selected_indices": [int(x) for x in st.selected_indices],
            "selected_labels": [str(x) for x in st.selected_labels],
            "selected_weights": [float(x) for x in st.selected_weights],
            "weights_by_label": {str(k): float(v) for k, v in st.weights_by_label.items()},
            "preview_reward": float(st.preview_reward),
            "delta_reward": float(st.delta_reward),
        }
        for st in rollout.step_traces
    ]


def run_prompt(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    reward_ctx: su.RewardContext,
    neg_embeds: Any,
    neg_mask: Any,
    prompt: str,
    slug: str,
    basis_cache: dict[str, dict[str, str]],
    save_images: bool,
) -> dict[str, Any]:
    prompt_dir = os.path.join(args.out_dir, slug)
    ensure_dir(prompt_dir)

    basis = build_prompt_basis(args, prompt, basis_cache)
    basis_emb = encode_prompt_basis(args, ctx, basis, neg_embeds, neg_mask, max_seq=256)
    orig_h, orig_w, h, w = resolve_resolution(args, ctx)

    # 1) original prompt only
    path_orig = os.path.join(prompt_dir, "baseline_original.png") if save_images and args.save_images else None
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
        save_path=path_orig,
        tag=f"{slug}_baseline_orig",
    )

    # 2) one-shot Qwen rewrite only (balanced)
    idx_balanced = _pick_label_index(basis, "balanced")
    pe_bal, pm_bal = basis_emb.pe_list[idx_balanced]
    path_oneshot = os.path.join(prompt_dir, "oneshot_balanced.png") if save_images and args.save_images else None
    oneshot = run_single_prompt_rollout(
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
        pe=pe_bal,
        pm=pm_bal,
        ue=basis_emb.ue,
        um=basis_emb.um,
        preview_every=int(args.preview_every),
        save_path=path_oneshot,
        tag=f"{slug}_oneshot",
    )

    # 3) fixed equal-weight prompt blending
    fixed_subset = _subset_indices(len(basis_emb.pe_list), int(args.basis_k))
    k = len(fixed_subset)
    theta_fixed = WeightParams(a=np.zeros((k,), dtype=np.float64), b=np.zeros((k,), dtype=np.float64))
    path_fixed = os.path.join(prompt_dir, "fixed_equal_blend.png") if save_images and args.save_images else None
    fixed = run_dynamic_rollout(
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
        basis_indices=fixed_subset,
        blend_family=str(args.fixed_family),
        weight_params=theta_fixed,
        preview_every=int(args.preview_every),
        save_path=path_fixed,
        tag=f"{slug}_fixed",
    )

    naive_blend_payload: dict[str, Any] | None = None
    naive_blend_rollouts: dict[str, Any] = {}
    if bool(args.run_naive_blend_ablation):
        naive_subset = _subset_indices(len(basis_emb.pe_list), int(args.naive_blend_k))
        nk = len(naive_subset)
        theta_naive = WeightParams(a=np.zeros((nk,), dtype=np.float64), b=np.zeros((nk,), dtype=np.float64))
        for fam in ("nlerp", "slerp"):
            naive_path = (
                os.path.join(prompt_dir, f"naive_equal_{fam}.png")
                if save_images and args.save_images
                else None
            )
            naive_roll = run_dynamic_rollout(
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
                basis_indices=naive_subset,
                blend_family=fam,
                weight_params=theta_naive,
                preview_every=int(args.preview_every),
                save_path=naive_path,
                tag=f"{slug}_naive_equal_{fam}",
            )
            naive_blend_rollouts[fam] = naive_roll
        naive_blend_payload = {
            "subset_indices": [int(x) for x in naive_subset],
            "subset_labels": [str(basis_emb.labels[int(x)]) for x in naive_subset],
            "equal_theta": [0.0 for _ in range(2 * nk)],
            "scores": {
                "nlerp": float(naive_blend_rollouts["nlerp"].final_score),
                "slerp": float(naive_blend_rollouts["slerp"].final_score),
                "delta_slerp_minus_nlerp": float(
                    naive_blend_rollouts["slerp"].final_score - naive_blend_rollouts["nlerp"].final_score
                ),
            },
            "nlerp": {
                "preview_rewards": [float(x) for x in naive_blend_rollouts["nlerp"].preview_rewards],
                "step_weight_evolution": _rollout_weight_evolution(naive_blend_rollouts["nlerp"]),
            },
            "slerp": {
                "preview_rewards": [float(x) for x in naive_blend_rollouts["slerp"].preview_rewards],
                "step_weight_evolution": _rollout_weight_evolution(naive_blend_rollouts["slerp"]),
            },
        }
        save_json(os.path.join(prompt_dir, "naive_blend_ablation.json"), naive_blend_payload)

    # 4) GA(prompt subset/family) + CEM(weights)
    path_hybrid = os.path.join(prompt_dir, "hybrid_ga_cem_best.png") if save_images and args.save_images else None
    ga_res: PromptGASearchResult = run_ga_prompt_search(
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
        save_best_image_path=path_hybrid,
    )

    weight_evolution = {
        "baseline_original": _rollout_weight_evolution(baseline),
        "oneshot_balanced": _rollout_weight_evolution(oneshot),
        "fixed_equal_blend": _rollout_weight_evolution(fixed),
        "ga_cem_hybrid_best": _rollout_weight_evolution(ga_res.best.rollout),
        "ga_cem_generation_topk": [
            {
                "generation": int(gen_row.get("generation", 0)),
                "top": [
                    {
                        "rank": int(top_row.get("rank", 0)),
                        "score": float(top_row.get("score", 0.0)),
                        "subset_labels": [str(x) for x in top_row.get("subset_labels", [])],
                        "blend_family": str(top_row.get("blend_family", "")),
                        "theta": [float(x) for x in top_row.get("theta", [])],
                        "step_weight_evolution": top_row.get("step_weight_evolution", []),
                    }
                    for top_row in gen_row.get("top", [])
                ],
            }
            for gen_row in ga_res.history
        ],
        "ga_cem_all_evaluations": to_jsonable(ga_res.evaluation_log),
    }
    weight_path = os.path.join(prompt_dir, "weight_evolution.json")
    save_json(weight_path, weight_evolution)

    if save_images and args.save_images:
        panel_path = os.path.join(prompt_dir, "comparison_panel.png")
        make_score_panel(
            [
                ("orig", baseline.final_image, baseline.final_score),
                ("oneshot", oneshot.final_image, oneshot.final_score),
                ("fixed", fixed.final_image, fixed.final_score),
                ("hybrid", ga_res.best.rollout.final_image, ga_res.best.score),
            ],
            panel_path,
        )

    basis_payload = {c.label: c.text for c in basis.candidates}
    result = {
        "slug": slug,
        "prompt": prompt,
        "basis": basis_payload,
        "scores": {
            "baseline_original": float(baseline.final_score),
            "oneshot_balanced": float(oneshot.final_score),
            "fixed_equal_blend": float(fixed.final_score),
            "ga_cem_hybrid_best": float(ga_res.best.score),
        },
        "naive_blend_ablation": naive_blend_payload,
        "hybrid": {
            "objective": {
                "mode": str(args.hybrid_reward_mode),
                "mix_orig": float(args.hybrid_reward_mix_orig),
                "mix_cond": float(args.hybrid_reward_mix_cond),
                "mix_delta": float(args.hybrid_reward_mix_delta),
                "cond_reduce": str(args.hybrid_cond_reduce),
            },
            "best_score": float(ga_res.best.score),
            "best_orig_score": float(ga_res.best.orig_score),
            "best_cond_score": float(ga_res.best.cond_score),
            "best_ref_score": None if ga_res.best.ref_score is None else float(ga_res.best.ref_score),
            "best_genome": [int(x) for x in ga_res.best.genome],
            "best_subset_indices": [int(x) for x in ga_res.best.subset_indices],
            "best_subset_labels": list(ga_res.best.subset_labels),
            "best_blend_family": str(ga_res.best.blend_family),
            "best_theta": [float(x) for x in ga_res.best.theta],
            "eval_calls_total": int(ga_res.eval_calls_total),
            "nfe_total": int(ga_res.nfe_total),
            "wallclock_total_sec": float(ga_res.wallclock_total_sec),
            "history": to_jsonable(ga_res.history),
            "evaluation_log": to_jsonable(ga_res.evaluation_log),
            "best_rollout": to_jsonable(ga_res.best.rollout),
            "weight_evolution_file": weight_path,
        },
    }
    if naive_blend_payload is not None:
        result["scores"]["naive_equal_nlerp"] = float(naive_blend_rollouts["nlerp"].final_score)
        result["scores"]["naive_equal_slerp"] = float(naive_blend_rollouts["slerp"].final_score)
        result["scores"]["naive_delta_slerp_minus_nlerp"] = float(
            naive_blend_rollouts["slerp"].final_score - naive_blend_rollouts["nlerp"].final_score
        )
    save_json(os.path.join(prompt_dir, "result.json"), result)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_dir(args.out_dir)

    if args.cuda_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(args.cuda_alloc_conf)
        print(f"Set PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

    prompts = load_prompts(args)
    basis_cache = load_basis_cache(args.basis_cache_json)

    t0 = time.perf_counter()
    ctx = su.load_pipeline(args)
    reward_ctx = su.load_reward(args, ctx)
    if callable(getattr(reward_ctx, "before_decode", None)):
        setattr(ctx, "pre_decode_hook", reward_ctx.before_decode)
    neg_embeds, neg_mask = su.load_neg_embed(args, ctx)

    all_rows: list[dict[str, Any]] = []
    for i, prompt in enumerate(prompts):
        slug = f"p{i:04d}"
        save_this = int(args.save_first_k) < 0 or i < int(args.save_first_k)
        print("\n" + "=" * 72)
        print(f"[{slug}] {prompt}")
        print("=" * 72)
        row = run_prompt(
            args=args,
            ctx=ctx,
            reward_ctx=reward_ctx,
            neg_embeds=neg_embeds,
            neg_mask=neg_mask,
            prompt=prompt,
            slug=slug,
            basis_cache=basis_cache,
            save_images=save_this,
        )
        scores = row["scores"]
        print(
            "  scores: "
            f"orig={scores['baseline_original']:.4f} "
            f"oneshot={scores['oneshot_balanced']:.4f} "
            f"fixed={scores['fixed_equal_blend']:.4f} "
            f"hybrid={scores['ga_cem_hybrid_best']:.4f}"
        )
        h = row["hybrid"]
        ref_txt = "None" if h["best_ref_score"] is None else f"{float(h['best_ref_score']):.4f}"
        print(
            "  hybrid-objective: "
            f"mode={h['objective']['mode']} "
            f"orig={h['best_orig_score']:.4f} "
            f"cond={h['best_cond_score']:.4f} "
            f"ref={ref_txt}"
        )
        if "naive_equal_nlerp" in scores and "naive_equal_slerp" in scores:
            print(
                "  naive-blend: "
                f"nlerp={scores['naive_equal_nlerp']:.4f} "
                f"slerp={scores['naive_equal_slerp']:.4f} "
                f"delta={scores['naive_delta_slerp_minus_nlerp']:+.4f}"
            )
        all_rows.append(row)

    save_basis_cache(args.basis_cache_json, basis_cache)
    summary_path = os.path.join(args.out_dir, "sandbox_prompt_basis_summary.json")
    save_json(summary_path, all_rows)
    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 72)
    print(f"Done. prompts={len(all_rows)} elapsed_sec={elapsed:.2f}")
    print(f"Summary: {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
