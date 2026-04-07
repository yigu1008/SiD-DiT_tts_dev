from __future__ import annotations

import argparse
import faulthandler
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from sampling_unified_sd35 import (
    _apply_backend_defaults,
    encode_variants,
    generate_variants,
    load_pipeline,
    load_reward_model,
    run_baseline,
    run_baseline_batch,
    run_beam,
    run_bon,
    run_ga,
    run_greedy,
    run_greedy_batch,
    run_mcts,
    run_smc,
    save_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP multi-GPU SD3.5 evaluation (base/greedy/mcts/ga/smc).")
    parser.add_argument(
        "--backend",
        choices=["sid", "senseflow_large", "senseflow_medium"],
        default=None,
        help="Convenience shortcut. Sets --model_id, --transformer_id, --sigmas together.",
    )
    parser.add_argument("--model_id", default=os.environ.get("MODEL_ID"))
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--transformer_id", default=os.environ.get("TRANSFORMER_ID"),
                        help="HuggingFace repo for the transformer (e.g. domiso/SenseFlow).")
    parser.add_argument("--transformer_subfolder", default=None,
                        help="Subfolder within --transformer_id (e.g. SenseFlow-SD35L/transformer).")
    parser.add_argument("--prompt_file", required=True, help="Prompt txt file (one prompt per line).")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1, help="Exclusive end index; -1 means all.")
    parser.add_argument("--out_dir", default="./sd35_ddp_out")

    parser.add_argument("--modes", nargs="+", choices=["base", "greedy", "mcts", "ga", "smc", "bon", "beam"], default=["base", "greedy", "mcts", "ga"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_per_prompt", action="store_true", help="Use seed + prompt_index for each prompt.")

    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--sigmas",
        nargs="+",
        type=float,
        default=None,
        help="Explicit sigma schedule (overrides --backend default).",
    )
    parser.add_argument("--cfg_scales", nargs="+", type=float, default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5])
    parser.add_argument("--baseline_cfg", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--time_scale", type=float, default=1000.0)

    parser.add_argument("--n_variants", type=int, default=3)
    parser.add_argument("--no_qwen", action="store_true")
    parser.add_argument("--qwen_id", default="Qwen/Qwen3-4B")
    parser.add_argument("--qwen_python", default="python3")
    parser.add_argument("--qwen_dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--qwen_timeout_sec", type=float, default=240.0)
    parser.add_argument("--rewrites_file", default=None)
    parser.add_argument("--max_sequence_length", type=int, default=256)

    parser.add_argument("--n_sims", type=int, default=50)
    parser.add_argument("--ucb_c", type=float, default=1.41)
    parser.add_argument("--mcts_interp_family", choices=["none", "slerp", "nlerp"], default="none",
                        help="Embedding interpolation family for MCTS action space expansion.")
    parser.add_argument("--mcts_n_interp", type=int, default=1,
                        help="Number of interpolation points inserted between each adjacent variant pair.")
    parser.add_argument("--smc_k", type=int, default=8)
    parser.add_argument("--smc_gamma", type=float, default=0.10)
    parser.add_argument("--ess_threshold", type=float, default=0.5)
    parser.add_argument("--resample_start_frac", type=float, default=0.3)
    parser.add_argument("--smc_cfg_scale", type=float, default=1.25)
    parser.add_argument("--smc_variant_idx", type=int, default=0)
    parser.add_argument("--bon_n", type=int, default=16, help="Number of candidates for Best-of-N search.")
    parser.add_argument("--beam_width", type=int, default=4, help="Number of beams to keep per step in beam search.")
    parser.add_argument(
        "--correction_strengths",
        nargs="+",
        type=float,
        default=[0.0],
        help="Reward-gradient correction strength values included as actions (like --cfg_scales). "
             "[0.0] disables correction. E.g. --correction_strengths 0.0 0.5 1.0.",
    )
    parser.add_argument(
        "--x0_sampler",
        action="store_true",
        default=False,
        help="Treat transformer output as a direct x0 prediction instead of flow/velocity. "
             "Set automatically for senseflow backends.",
    )
    parser.add_argument("--ga_population", type=int, default=24)
    parser.add_argument("--ga_generations", type=int, default=8)
    parser.add_argument("--ga_elites", type=int, default=3)
    parser.add_argument("--ga_mutation_prob", type=float, default=0.10)
    parser.add_argument("--ga_tournament_k", type=int, default=3)
    parser.add_argument("--ga_crossover", choices=["uniform", "one_point"], default="uniform")
    parser.add_argument("--ga_selection", choices=["rank", "tournament"], default="rank")
    parser.add_argument("--ga_rank_pressure", type=float, default=1.7)
    parser.add_argument("--ga_log_topk", type=int, default=3)
    parser.add_argument("--ga_eval_batch", type=int, default=1, help="Batched GA rollout eval size (>=1).")
    parser.add_argument("--ga_phase_constraints", action="store_true")
    parser.add_argument(
        "--reward_model",
        default="CodeGoat24/UnifiedReward-qwen-7b",
        help="Legacy alias for UnifiedReward model id (kept for compatibility).",
    )
    parser.add_argument(
        "--unifiedreward_model",
        default=None,
        help="UnifiedReward model id override. Defaults to --reward_model when unset.",
    )
    parser.add_argument(
        "--image_reward_model",
        default="ImageReward-v1.0",
        help="ImageReward model id/checkpoint name.",
    )
    parser.add_argument(
        "--pickscore_model",
        default="yuvalkirstain/PickScore_v1",
        help="PickScore model id.",
    )
    parser.add_argument(
        "--reward_backend",
        choices=["auto", "unifiedreward", "unified", "imagereward", "pickscore", "hpsv3", "hpsv2", "blend", "all"],
        default="unifiedreward",
    )
    parser.add_argument(
        "--reward_weights",
        nargs=2,
        type=float,
        default=[1.0, 1.0],
        help="Blend backend weights: imagereward hps(v2/v3)",
    )
    parser.add_argument("--reward_api_base", default=None, help="Optional OpenAI-compatible API base for UnifiedReward.")
    parser.add_argument("--reward_api_key", default="unifiedreward")
    parser.add_argument("--reward_api_model", default="UnifiedReward-7b-v1.5")
    parser.add_argument("--reward_max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--reward_prompt_mode",
        choices=["standard", "strict"],
        default="standard",
        help="UnifiedReward prompt template mode.",
    )

    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--save_best_images", action="store_true",
                        help="Save only the final best image per method (minimal set for eval).")
    parser.add_argument("--save_variants", action="store_true")
    parser.add_argument(
        "--rewrite_check_topk",
        type=int,
        default=5,
        help="Save up to K prompt-rewrite examples for quick inspection.",
    )
    parser.add_argument(
        "--rank_log_wait_sec",
        type=int,
        default=7200,
        help="Rank0 wait timeout for all rank logs before aggregation (seconds).",
    )
    parser.add_argument(
        "--rank_log_poll_sec",
        type=float,
        default=2.0,
        help="Rank0 polling interval while waiting for rank logs.",
    )
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        default=None,
        help="Number of prompts to process through the transformer simultaneously. "
             "Defaults to backend setting (1 for sid/CFG, 2 for senseflow). "
             "Applied to baseline and greedy; MCTS/GA/SMC remain per-prompt.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the transformer for faster repeated inference.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip prompts already in the rank log (default: on). Use --no_resume to disable.",
    )
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default=None,
        help="Pipeline/transformer dtype. Defaults to backend setting (float16 for sid, bfloat16 for senseflow).",
    )
    return _apply_backend_defaults(parser.parse_args())


def _resolve_file(path_str: str, label: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(path)


def _resolve_optional_file(path_str: str | None, label: str) -> str | None:
    if path_str is None:
        return None
    return _resolve_file(path_str, label)


def _resolve_out_dir(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.prompt_file = _resolve_file(args.prompt_file, "prompt_file")
    args.ckpt = _resolve_optional_file(args.ckpt, "ckpt")
    args.rewrites_file = _resolve_optional_file(args.rewrites_file, "rewrites_file")
    args.out_dir = _resolve_out_dir(args.out_dir)
    return args


def init_runtime() -> Tuple[int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    # IMPORTANT:
    # This script does not use distributed collectives for model execution.
    # We only use torchrun's rank env vars for prompt sharding. Initializing a
    # NCCL process group here can cause hard-to-debug collective timeouts when
    # one rank fails. Keep ranks independent.
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, distributed


def wait_for_rank_logs(log_dir: str, world_size: int, timeout_sec: int, poll_sec: float) -> None:
    if world_size <= 1:
        return
    deadline = time.time() + float(max(1, timeout_sec))
    expected = [os.path.join(log_dir, f"rank_{r:03d}.jsonl") for r in range(world_size)]
    while time.time() < deadline:
        missing = [p for p in expected if not os.path.exists(p)]
        if not missing:
            return
        time.sleep(max(0.1, float(poll_sec)))
    missing = [p for p in expected if not os.path.exists(p)]
    raise RuntimeError(
        "Timed out waiting for rank logs before aggregation. "
        f"missing={len(missing)} files, examples={missing[:3]}"
    )


def load_prompt_range(prompt_file: str, start_index: int, end_index: int) -> List[Tuple[int, str]]:
    prompts = [line.strip() for line in open(prompt_file, encoding="utf-8") if line.strip()]
    if end_index == -1:
        end_index = len(prompts)
    start = max(0, start_index)
    end = min(len(prompts), end_index)
    selected = []
    for global_idx in range(start, end):
        selected.append((global_idx, prompts[global_idx]))
    return selected


def shard(entries: List[Tuple[int, str]], rank: int, world_size: int) -> List[Tuple[int, str]]:
    return [entry for i, entry in enumerate(entries) if i % world_size == rank]


def aggregate_logs(log_dir: str, out_dir: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(log_dir)):
        if not name.endswith(".jsonl") or name.endswith("_rewrite_examples.jsonl"):
            continue
        with open(os.path.join(log_dir, name), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    by_mode = defaultdict(list)
    by_prompt = defaultdict(dict)
    for row in rows:
        mode = row["mode"]
        by_mode[mode].append(row)
        by_prompt[row["prompt_index"]][mode] = row

    mode_stats: Dict[str, Dict[str, Any]] = {}
    for mode, mode_rows in by_mode.items():
        deltas = [r.get("delta_vs_base", 0.0) for r in mode_rows]
        scores = [r["score"] for r in mode_rows]
        nfes = [r["nfe"] for r in mode_rows if "nfe" in r]
        mode_stats[mode] = {
            "count": len(mode_rows),
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "mean_delta_vs_base": float(np.mean(deltas)) if deltas else 0.0,
            "std_delta_vs_base": float(np.std(deltas)) if deltas else 0.0,
            "mean_nfe": float(np.mean(nfes)) if nfes else None,
        }

    best_mode_counts = defaultdict(int)
    for _, mode_map in by_prompt.items():
        best_mode = max(mode_map.items(), key=lambda kv: kv[1]["score"])[0]
        best_mode_counts[best_mode] += 1

    aggregate = {
        "num_rows": len(rows),
        "num_prompts": len(by_prompt),
        "mode_stats": mode_stats,
        "best_mode_counts": dict(best_mode_counts),
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "aggregate_summary.json"), "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    txt = []
    txt.append("Mode\tCount\tMeanScore\tMeanDeltaVsBase\tStdDeltaVsBase\tMeanNFE")
    for mode in sorted(mode_stats):
        s = mode_stats[mode]
        nfe_str = f"{s['mean_nfe']:.1f}" if s.get("mean_nfe") is not None else "n/a"
        txt.append(
            f"{mode}\t{s['count']}\t{s['mean_score']:.6f}\t"
            f"{s['mean_delta_vs_base']:+.6f}\t{s['std_delta_vs_base']:.6f}\t{nfe_str}"
        )
    txt.append("")
    txt.append("Best mode counts:")
    for mode, c in sorted(best_mode_counts.items()):
        txt.append(f"{mode}: {c}")
    with open(os.path.join(out_dir, "aggregate_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt) + "\n")
    return aggregate


def aggregate_rewrite_examples(log_dir: str, out_dir: str, topk: int) -> str | None:
    topk = max(0, int(topk))
    if topk <= 0:
        return None

    rows: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(log_dir)):
        if not name.endswith("_rewrite_examples.jsonl"):
            continue
        path = os.path.join(log_dir, name)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

    if not rows:
        return None

    rows.sort(key=lambda r: (int(r.get("prompt_index", 10**9)), int(r.get("rank", 10**9))))
    selected = rows[:topk]

    payload = {
        "topk": topk,
        "total_candidates": len(rows),
        "examples": selected,
    }
    out_path = os.path.join(out_dir, f"rewrite_examples_top{topk}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def _chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _write_row(f: Any, row: Dict[str, Any]) -> None:
    """Write a single JSON row to the open log file and flush immediately."""
    f.write(json.dumps(row) + "\n")
    f.flush()


def _load_done_indices(rank_log: str) -> set:
    """Return prompt_indices already written to rank_log (for resume)."""
    done: set = set()
    if not os.path.exists(rank_log):
        return done
    with open(rank_log, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "prompt_index" in row and "mode" in row:
                    done.add(int(row["prompt_index"]))
            except json.JSONDecodeError:
                pass
    return done


def main() -> None:
    faulthandler.enable(all_threads=True)
    args = normalize_paths(parse_args())
    rank, world_size, local_rank, distributed = init_runtime()

    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.join(args.out_dir, "logs")
    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(log_dir, exist_ok=True)
    if args.save_images or args.save_best_images:
        os.makedirs(img_dir, exist_ok=True)

    all_entries = load_prompt_range(args.prompt_file, args.start_index, args.end_index)
    my_entries = shard(all_entries, rank, world_size)

    # Resume: skip prompts already written to the rank log.
    rank_log = os.path.join(log_dir, f"rank_{rank:03d}.jsonl")
    if args.resume:
        done_indices = _load_done_indices(rank_log)
        if done_indices:
            before = len(my_entries)
            my_entries = [(i, p) for i, p in my_entries if i not in done_indices]
            print(f"[rank {rank}] resume: skipped {before - len(my_entries)} done prompts, {len(my_entries)} remaining")

    if rank == 0:
        print(
            f"DDP launch: world_size={world_size} prompts_total={len(all_entries)} "
            f"range=[{args.start_index},{args.end_index if args.end_index!=-1 else 'end'})"
        )
        print(
            f"Runtime cfg: size={args.width}x{args.height} "
            f"{'sigmas=' + str(args.sigmas) if args.sigmas else 'steps=' + str(args.steps)} "
            f"modes={','.join(args.modes)} reward={args.reward_backend} "
            f"qwen={'off' if args.no_qwen else 'on'}"
        )
    print(f"[rank {rank}] local_rank={local_rank} assigned_prompts={len(my_entries)}")

    try:
        rewrite_cache: Dict[str, List[str]] = {}
        if args.rewrites_file and os.path.exists(args.rewrites_file):
            rewrite_cache = json.load(open(args.rewrites_file, encoding="utf-8"))

        ctx = load_pipeline(args)

        if args.compile:
            print(f"[rank {rank}] torch.compile transformer (mode=reduce-overhead)...")
            ctx.pipe.transformer = torch.compile(
                ctx.pipe.transformer, mode="reduce-overhead", fullgraph=False
            )

        reward_model = load_reward_model(args, ctx.device)

        # Open rank log in append mode — rows are written immediately after each prompt.
        rank_log_f = open(rank_log, "a", encoding="utf-8")
        total_rows_written = 0
        rewrite_rows: List[Dict[str, Any]] = []
        gbs = int(args.gen_batch_size)

        for batch_entries in _chunks(my_entries, gbs):
            t0_batch = time.time()
            batch_prompts = [p for _, p in batch_entries]
            batch_indices = [i for i, _ in batch_entries]
            batch_seeds = [args.seed + i if args.seed_per_prompt else args.seed for i in batch_indices]

            # --- variants + embeddings for every prompt in the batch ---
            batch_variants: List[List[str]] = []
            batch_embs = []
            for (prompt_index, prompt), seed in zip(batch_entries, batch_seeds):
                slug = f"p{prompt_index:05d}"
                variants = generate_variants(args, prompt, rewrite_cache)
                if len(rewrite_rows) < max(0, int(args.rewrite_check_topk)):
                    changed = [v for v in variants[1:] if str(v).strip() != str(prompt).strip()]
                    rewrite_rows.append({
                        "rank": int(rank), "prompt_index": int(prompt_index), "slug": slug,
                        "original": prompt, "variants": [str(v) for v in variants],
                        "changed_count": int(len(changed)), "from_cache": bool(prompt in rewrite_cache),
                    })
                if args.save_variants:
                    with open(os.path.join(args.out_dir, f"{slug}_variants.txt"), "w", encoding="utf-8") as f:
                        for vi, text in enumerate(variants):
                            f.write(f"v{vi}: {text}\n")
                batch_variants.append(variants)
                batch_embs.append(encode_variants(ctx, variants, max_sequence_length=args.max_sequence_length))

            # --- batched baseline ---
            ctx.nfe = 0
            if gbs > 1 and len(batch_entries) > 1:
                batch_base = run_baseline_batch(
                    args, ctx, batch_embs, reward_model, batch_prompts, batch_seeds,
                    cfg_scale=float(args.baseline_cfg),
                )
            else:
                base_img_0, base_score_0 = run_baseline(
                    args, ctx, batch_embs[0], reward_model, batch_prompts[0], batch_seeds[0],
                    cfg_scale=float(args.baseline_cfg),
                )
                batch_base = [(base_img_0, base_score_0)]
            nfe_base = ctx.nfe // len(batch_entries)  # per-prompt NFE

            # --- batched greedy ---
            batch_greedy = None
            nfe_greedy = 0
            if "greedy" in args.modes:
                ctx.nfe = 0
                if gbs > 1 and len(batch_entries) > 1:
                    batch_greedy = run_greedy_batch(
                        args, ctx, batch_embs, reward_model, batch_prompts, batch_variants, batch_seeds,
                    )
                else:
                    batch_greedy = [run_greedy(
                        args, ctx, batch_embs[0], reward_model, batch_prompts[0], batch_variants[0], batch_seeds[0],
                    )]
                nfe_greedy = ctx.nfe // len(batch_entries)

            # --- per-prompt: save, log, run MCTS/GA/SMC ---
            n_steps = len(args.sigmas) if args.sigmas else args.steps
            for bi, ((prompt_index, prompt), seed, variants, emb) in enumerate(
                zip(batch_entries, batch_seeds, batch_variants, batch_embs)
            ):
                slug = f"p{prompt_index:05d}"
                print(f"[rank {rank}] {slug} seed={seed}")
                base_img, base_score = batch_base[bi]

                if args.save_images or args.save_best_images:
                    base_img.save(os.path.join(img_dir, f"{slug}_base.png"))

                if "base" in args.modes:
                    _write_row(rank_log_f, {
                        "prompt_index": prompt_index, "prompt": prompt, "seed": seed,
                        "mode": "base", "score": float(base_score), "delta_vs_base": 0.0,
                        "baseline_cfg": float(args.baseline_cfg),
                        "nfe": int(nfe_base),
                        "schedule": [[0, float(args.baseline_cfg)] for _ in range(n_steps)],
                    })
                    total_rows_written += 1

                if "greedy" in args.modes:
                    greedy = batch_greedy[bi]
                    if args.save_images or args.save_best_images:
                        greedy.image.save(os.path.join(img_dir, f"{slug}_greedy.png"))
                    if args.save_images:
                        save_comparison(
                            os.path.join(img_dir, f"{slug}_greedy_comp.png"),
                            base_img, greedy.image, base_score, greedy.score, greedy.actions,
                        )
                    _write_row(rank_log_f, {
                        "prompt_index": prompt_index, "prompt": prompt, "seed": seed,
                        "mode": "greedy", "score": float(greedy.score),
                        "delta_vs_base": float(greedy.score - base_score),
                        "baseline_score": float(base_score),
                        "nfe": int(nfe_greedy),
                        "actions": [[int(v), float(c), float(r)] for v, c, r in greedy.actions],
                    })
                    total_rows_written += 1

                if "mcts" in args.modes:
                    ctx.nfe = 0
                    mcts = run_mcts(args, ctx, emb, reward_model, prompt, variants, seed)
                    if args.save_images or args.save_best_images:
                        mcts.image.save(os.path.join(img_dir, f"{slug}_mcts.png"))
                    if args.save_images:
                        save_comparison(
                            os.path.join(img_dir, f"{slug}_mcts_comp.png"),
                            base_img, mcts.image, base_score, mcts.score, mcts.actions,
                        )
                    _write_row(rank_log_f, {
                        "prompt_index": prompt_index, "prompt": prompt, "seed": seed,
                        "mode": "mcts", "score": float(mcts.score),
                        "delta_vs_base": float(mcts.score - base_score),
                        "baseline_score": float(base_score),
                        "nfe": int(ctx.nfe),
                        "actions": [[int(v), float(c), float(r)] for v, c, r in mcts.actions],
                    })
                    total_rows_written += 1

                if "ga" in args.modes:
                    ctx.nfe = 0
                    ga = run_ga(
                        args, ctx, emb, reward_model, prompt, variants, seed,
                        log_dir=os.path.join(args.out_dir, "ga_logs"),
                        log_prefix=slug,
                    )
                    if args.save_images or args.save_best_images:
                        ga.image.save(os.path.join(img_dir, f"{slug}_ga.png"))
                    if args.save_images:
                        save_comparison(
                            os.path.join(img_dir, f"{slug}_ga_comp.png"),
                            base_img, ga.image, base_score, ga.score, ga.actions,
                        )
                    _write_row(rank_log_f, {
                        "prompt_index": prompt_index, "prompt": prompt, "seed": seed,
                        "mode": "ga", "score": float(ga.score),
                        "delta_vs_base": float(ga.score - base_score),
                        "baseline_score": float(base_score),
                        "nfe": int(ctx.nfe),
                        "actions": [[int(v), float(c), float(r)] for v, c, r in ga.actions],
                    })
                    total_rows_written += 1

                if "smc" in args.modes:
                    ctx.nfe = 0
                    smc = run_smc(args, ctx, emb, reward_model, prompt, variants, seed)
                    if args.save_images or args.save_best_images:
                        smc.image.save(os.path.join(img_dir, f"{slug}_smc.png"))
                    if args.save_images:
                        save_comparison(
                            os.path.join(img_dir, f"{slug}_smc_comp.png"),
                            base_img, smc.image, base_score, smc.score, smc.actions,
                        )
                    _write_row(rank_log_f, {
                        "prompt_index": prompt_index, "prompt": prompt, "seed": seed,
                        "mode": "smc", "score": float(smc.score),
                        "delta_vs_base": float(smc.score - base_score),
                        "baseline_score": float(base_score),
                        "nfe": int(ctx.nfe),
                        "actions": [[int(v), float(c), float(r)] for v, c, r in smc.actions],
                        "search_diagnostics": smc.diagnostics,
                    })
                    total_rows_written += 1

                if "bon" in args.modes:
                    ctx.nfe = 0
                    bon = run_bon(args, ctx, emb, reward_model, prompt, seed)
                    if args.save_images or args.save_best_images:
                        bon.image.save(os.path.join(img_dir, f"{slug}_bon.png"))
                    if args.save_images:
                        save_comparison(
                            os.path.join(img_dir, f"{slug}_bon_comp.png"),
                            base_img, bon.image, base_score, bon.score, bon.actions,
                        )
                    _write_row(rank_log_f, {
                        "prompt_index": prompt_index, "prompt": prompt, "seed": seed,
                        "mode": "bon", "score": float(bon.score),
                        "delta_vs_base": float(bon.score - base_score),
                        "baseline_score": float(base_score),
                        "nfe": int(ctx.nfe),
                        "actions": [[int(v), float(c), float(r)] for v, c, r in bon.actions],
                        "search_diagnostics": bon.diagnostics,
                    })
                    total_rows_written += 1

                if "beam" in args.modes:
                    ctx.nfe = 0
                    beam = run_beam(args, ctx, emb, reward_model, prompt, variants, seed)
                    if args.save_images or args.save_best_images:
                        beam.image.save(os.path.join(img_dir, f"{slug}_beam.png"))
                    if args.save_images:
                        save_comparison(
                            os.path.join(img_dir, f"{slug}_beam_comp.png"),
                            base_img, beam.image, base_score, beam.score, beam.actions,
                        )
                    _write_row(rank_log_f, {
                        "prompt_index": prompt_index, "prompt": prompt, "seed": seed,
                        "mode": "beam", "score": float(beam.score),
                        "delta_vs_base": float(beam.score - base_score),
                        "baseline_score": float(base_score),
                        "nfe": int(ctx.nfe),
                        "actions": [[int(v), float(c), float(r)] for v, c, r in beam.actions],
                        "search_diagnostics": beam.diagnostics,
                    })
                    total_rows_written += 1


            dt = time.time() - t0_batch
            print(
                f"[rank {rank}] batch={[f'p{i:05d}' for i in batch_indices]} "
                f"base={[f'{batch_base[bi][1]:.4f}' for bi in range(len(batch_entries))]} "
                f"elapsed={dt:.1f}s"
            )

        rank_log_f.close()
        rewrite_log = os.path.join(log_dir, f"rank_{rank:03d}_rewrite_examples.jsonl")
        with open(rewrite_log, "w", encoding="utf-8") as f:
            for row in rewrite_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(os.path.join(log_dir, f"rank_{rank:03d}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "rank": rank,
                    "world_size": world_size,
                    "assigned_prompts": len(my_entries),
                    "rows": total_rows_written,
                    "rewrite_rows": len(rewrite_rows),
                },
                f,
                indent=2,
            )

        if rank == 0:
            wait_for_rank_logs(
                log_dir=log_dir,
                world_size=world_size,
                timeout_sec=int(args.rank_log_wait_sec),
                poll_sec=float(args.rank_log_poll_sec),
            )
            aggregate = aggregate_logs(log_dir=log_dir, out_dir=args.out_dir)
            print("Aggregate summary:")
            print(json.dumps(aggregate, indent=2))
            rewrite_summary = aggregate_rewrite_examples(
                log_dir=log_dir,
                out_dir=args.out_dir,
                topk=int(args.rewrite_check_topk),
            )
            if rewrite_summary:
                print(f"Rewrite examples: {rewrite_summary}")
    except Exception as exc:
        err_path = os.path.join(log_dir, f"rank_{rank:03d}_error.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(f"rank={rank} local_rank={local_rank} world_size={world_size}\n")
            f.write(f"error_type={type(exc).__name__}\n")
            f.write(f"error={exc}\n")
        print(f"[rank {rank}] ERROR: {type(exc).__name__}: {exc}")
        print(f"[rank {rank}] wrote error log: {err_path}")
        try:
            rank_log_f.close()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
