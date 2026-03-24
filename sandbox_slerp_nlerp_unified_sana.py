"""
SANA SLERP/NLERP sandbox with UnifiedReward defaults.

This script focuses on two things:
  1) fixed interpolation sweep over (family, t)
  2) optional GA search over per-step (family, t) actions

GA here is intentionally simple (discrete action genomes) and reuses the
same rank/tournament selection style used elsewhere in this repo.
"""

from __future__ import annotations

import argparse
import csv
import json
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
        description="SANA SLERP/NLERP test with UnifiedReward defaults and optional GA action search."
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
        choices=["imagereward", "auto", "unifiedreward", "unified", "pickscore", "hpsv2", "blend"],
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

    # Interp setup
    p.add_argument("--interp_labels", nargs=2, default=["balanced", "subject"], help="Two basis labels to interpolate.")
    p.add_argument("--interp_values", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--families", nargs="+", default=["nlerp", "slerp"])
    p.add_argument("--preview_every", type=int, default=-1, help="Set -1 to disable preview decodes for speed.")

    # Which evaluations to run
    p.add_argument("--run_sweep", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run_ga", action=argparse.BooleanOptionalAction, default=True)

    # GA over per-step discrete actions (action bank = families x interp_values)
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


def _pick_pair_indices(labels: list[str], want_a: str, want_b: str) -> tuple[int, int]:
    key_to_idx = {str(lbl).strip().lower(): i for i, lbl in enumerate(labels)}
    a = key_to_idx.get(str(want_a).strip().lower(), -1)
    b = key_to_idx.get(str(want_b).strip().lower(), -1)
    if a < 0:
        a = 0
    if b < 0:
        b = 1 if len(labels) > 1 else 0
    if a == b and len(labels) > 1:
        b = 1 if a == 0 else 0
    return int(a), int(b)


def _family_list(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    for fam in args.families:
        f = str(fam).strip().lower()
        if f in {"nlerp", "slerp"} and f not in out:
            out.append(f)
    if len(out) == 0:
        out = ["nlerp", "slerp"]
    return out


def _action_bank(families: list[str], t_values: list[float]) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for fam in families:
        for t in t_values:
            out.append((str(fam), float(t)))
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
    ga_item: dict[str, Any] | None,
    out_path: str,
) -> None:
    # Keep panel compact: baseline + top 4 sweep + ga best (if present).
    ranked = sorted(sweep_items, key=lambda x: float(x["score"]), reverse=True)
    ranked = ranked[:4]
    entries: list[tuple[str, Image.Image, float, float]] = [("baseline", baseline_img, baseline_score, 0.0)]
    for row in ranked:
        label = f"{row['family']} t={float(row['t']):.2f}"
        entries.append((label, row["image"], float(row["score"]), float(row["score"]) - baseline_score))
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
    pair_indices: list[int],
    action_bank: list[tuple[str, float]],
    genome: list[int],
    preview_every: int,
    save_path: str | None = None,
    tag: str = "stepwise",
) -> ActionRollout:
    if len(pair_indices) != 2:
        raise ValueError(f"Expected two interpolation indices, got {pair_indices}")
    if len(action_bank) <= 0:
        raise ValueError("action_bank must be non-empty.")

    pe_bank = [basis_emb.pe_list[int(i)][0] for i in pair_indices]
    pm_bank = [basis_emb.pe_list[int(i)][1] for i in pair_indices]

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
        fam, tval = action_bank[a_idx]
        weights_t = torch.tensor(
            [1.0 - float(tval), float(tval)],
            dtype=pe_bank[0].dtype,
            device=ctx.device,
        )
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
            float(args.guidance_scale),
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
                "t": float(tval),
                "weights": [float(1.0 - tval), float(tval)],
                "selected_indices": [int(pair_indices[i]) for i in blend.selected_indices],
                "selected_labels": [str([basis_emb.labels[j] for j in pair_indices][i]) for i in blend.selected_indices],
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
    pair_indices: list[int],
    action_bank: list[tuple[str, float]],
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
            pair_indices=pair_indices,
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
                        {
                            "step": int(i),
                            "action_idx": int(aidx),
                            "family": str(action_bank[aidx][0]),
                            "t": float(action_bank[aidx][1]),
                        }
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
        {"step": int(i), "action_idx": int(aidx), "family": str(action_bank[aidx][0]), "t": float(action_bank[aidx][1])}
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_dir(args.out_dir)

    if args.cuda_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(args.cuda_alloc_conf)
        print(f"Set PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

    prompts = load_prompts(args)
    families = _family_list(args)
    t_values = [float(np.clip(v, 0.0, 1.0)) for v in args.interp_values]
    action_bank = _action_bank(families, t_values)
    basis_cache = load_basis_cache(args.basis_cache_json)

    t0 = time.perf_counter()
    ctx = su.load_pipeline(args)
    reward_ctx = su.load_reward(args, ctx)
    if callable(getattr(reward_ctx, "before_decode", None)):
        setattr(ctx, "pre_decode_hook", reward_ctx.before_decode)
    neg_embeds, neg_mask = su.load_neg_embed(args, ctx)

    tsv_path = os.path.join(args.out_dir, "slerp_nlerp_unified_scores.tsv")
    all_rows: list[dict[str, Any]] = []

    with open(tsv_path, "w", encoding="utf-8", newline="") as ftsv:
        writer = csv.writer(ftsv, delimiter="\t")
        writer.writerow(
            [
                "slug",
                "prompt",
                "method",
                "family",
                "t",
                "score",
                "delta_vs_baseline",
                "image_file",
                "genome",
            ]
        )

        for i, prompt in enumerate(prompts):
            slug = f"p{i:04d}"
            prompt_dir = os.path.join(args.out_dir, slug)
            ensure_dir(prompt_dir)
            save_this = int(args.save_first_k) < 0 or i < int(args.save_first_k)

            print("\n" + "=" * 72)
            print(f"[{slug}] {prompt}")
            print("=" * 72)

            basis = build_prompt_basis(args, prompt, basis_cache)
            basis_emb = encode_prompt_basis(args, ctx, basis, neg_embeds, neg_mask, max_seq=256)
            orig_h, orig_w, h, w = resolve_resolution(args, ctx)

            li0, li1 = _pick_pair_indices(
                basis_emb.labels,
                str(args.interp_labels[0]),
                str(args.interp_labels[1]),
            )
            pair_indices = [int(li0), int(li1)]
            pair_labels = [basis_emb.labels[li0], basis_emb.labels[li1]]
            print(f"  pair: {pair_labels[0]} -> {pair_labels[1]}")

            baseline_img_path = os.path.join(prompt_dir, "baseline_original.png") if (save_this and args.save_images) else None
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
            baseline_score = float(baseline.final_score)
            print(f"  baseline={baseline_score:.4f}")
            writer.writerow([slug, prompt, "baseline", "baseline", "baseline", f"{baseline_score:.6f}", "+0.000000", baseline_img_path or "", ""])

            sweep_results: list[dict[str, Any]] = []
            if bool(args.run_sweep):
                for action_idx, (fam, tval) in enumerate(action_bank):
                    genome = [int(action_idx) for _ in range(int(args.steps))]
                    img_path = (
                        os.path.join(prompt_dir, f"sweep_{fam}_t{str(f'{tval:.2f}').replace('.', 'p')}.png")
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
                        pair_indices=pair_indices,
                        action_bank=action_bank,
                        genome=genome,
                        preview_every=int(args.preview_every),
                        save_path=img_path,
                        tag=f"{slug}_sweep_{fam}_{tval:.2f}",
                    )
                    score = float(trace.final_score)
                    delta = float(score - baseline_score)
                    row = {
                        "family": str(fam),
                        "t": float(tval),
                        "score": score,
                        "delta_vs_baseline": delta,
                        "image_file": img_path,
                        "genome": genome,
                        "preview_rewards": trace.preview_rewards,
                        "step_actions": trace.step_actions,
                        "image": trace.final_image,
                    }
                    sweep_results.append(row)
                    writer.writerow(
                        [
                            slug,
                            prompt,
                            "sweep",
                            str(fam),
                            f"{float(tval):.2f}",
                            f"{score:.6f}",
                            f"{delta:+.6f}",
                            img_path or "",
                            json.dumps(genome),
                        ]
                    )
                    print(f"  sweep {fam:>5} t={float(tval):.2f} score={score:.4f} delta={delta:+.4f}")

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
                    pair_indices=pair_indices,
                    action_bank=action_bank,
                    prompt_dir=prompt_dir,
                    save_images=(save_this and args.save_images),
                )
                ga_delta = float(ga_result["best_score"] - baseline_score)
                writer.writerow(
                    [
                        slug,
                        prompt,
                        "ga",
                        "mixed",
                        "mixed",
                        f"{float(ga_result['best_score']):.6f}",
                        f"{ga_delta:+.6f}",
                        ga_result.get("best_image_file", "") or "",
                        json.dumps(ga_result["best_genome"]),
                    ]
                )
                print(f"  ga_best={float(ga_result['best_score']):.4f} delta={ga_delta:+.4f}")

            row_out: dict[str, Any] = {
                "slug": slug,
                "prompt": prompt,
                "basis_labels": list(basis_emb.labels),
                "basis_texts": {str(c.label): str(c.text) for c in basis.candidates},
                "interp_pair_labels": pair_labels,
                "interp_pair_indices": pair_indices,
                "action_bank": [{"action_idx": int(k), "family": str(v[0]), "t": float(v[1])} for k, v in enumerate(action_bank)],
                "baseline_score": baseline_score,
                "sweep": [
                    {
                        "family": str(r["family"]),
                        "t": float(r["t"]),
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
            }
            if save_this and args.save_images:
                ga_panel_item = None
                if ga_result is not None and ga_result.get("best_image_file"):
                    ga_img = Image.open(ga_result["best_image_file"]).convert("RGB")
                    ga_panel_item = {"image": ga_img, "score": float(ga_result["best_score"])}
                panel_path = os.path.join(prompt_dir, "comparison_panel.png")
                _build_summary_panel(
                    baseline_img=baseline.final_image,
                    baseline_score=baseline_score,
                    sweep_items=sweep_results,
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
            key = f"{srow['family']}@{float(srow['t']):.2f}"
            agg = sweep_agg.setdefault(
                key,
                {"family": srow["family"], "t": float(srow["t"]), "scores": [], "deltas": []},
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
                "t": float(cur["t"]),
                "count": int(s.size),
                "mean_score": float(s.mean()) if s.size else 0.0,
                "mean_delta_vs_baseline": float(d.mean()) if d.size else 0.0,
                "std_delta_vs_baseline": float(d.std()) if d.size else 0.0,
            }
        )

    ga_scores = [float(r["ga"]["best_score"]) for r in all_rows if isinstance(r.get("ga"), dict)]
    baseline_scores = [float(r["baseline_score"]) for r in all_rows]

    summary = {
        "num_prompts": len(all_rows),
        "reward_type": str(args.reward_type),
        "families": families,
        "interp_values": [float(x) for x in t_values],
        "rows": all_rows,
        "aggregate_sweep": aggregate_sweep_rows,
        "aggregate_global": {
            "mean_baseline": float(np.mean(baseline_scores)) if baseline_scores else 0.0,
            "mean_ga_best": float(np.mean(ga_scores)) if ga_scores else None,
            "mean_ga_delta_vs_baseline": (
                float(np.mean(np.asarray(ga_scores) - np.asarray(baseline_scores[: len(ga_scores)])))
                if ga_scores
                else None
            ),
        },
        "scores_tsv": tsv_path,
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
