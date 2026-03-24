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
    encode_variants,
    generate_variants,
    load_pipeline,
    load_reward_model,
    run_baseline,
    run_ga,
    run_greedy,
    run_mcts,
    run_smc,
    save_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP multi-GPU SD3.5 evaluation (base/greedy/mcts/ga/smc).")
    parser.add_argument("--model_id", default="YGu1998/SiD-DiT-SD3.5-large")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--prompt_file", required=True, help="Prompt txt file (one prompt per line).")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1, help="Exclusive end index; -1 means all.")
    parser.add_argument("--out_dir", default="./sd35_ddp_out")

    parser.add_argument("--modes", nargs="+", choices=["base", "greedy", "mcts", "ga", "smc"], default=["base", "greedy", "mcts", "ga"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_per_prompt", action="store_true", help="Use seed + prompt_index for each prompt.")

    parser.add_argument("--steps", type=int, default=4)
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
    parser.add_argument("--rewrites_file", default=None)
    parser.add_argument("--max_sequence_length", type=int, default=256)

    parser.add_argument("--n_sims", type=int, default=50)
    parser.add_argument("--ucb_c", type=float, default=1.41)
    parser.add_argument("--smc_k", type=int, default=8)
    parser.add_argument("--smc_gamma", type=float, default=0.10)
    parser.add_argument("--ess_threshold", type=float, default=0.5)
    parser.add_argument("--resample_start_frac", type=float, default=0.3)
    parser.add_argument("--smc_cfg_scale", type=float, default=1.25)
    parser.add_argument("--smc_variant_idx", type=int, default=0)
    parser.add_argument("--ga_population", type=int, default=24)
    parser.add_argument("--ga_generations", type=int, default=12)
    parser.add_argument("--ga_elites", type=int, default=3)
    parser.add_argument("--ga_mutation_prob", type=float, default=0.10)
    parser.add_argument("--ga_tournament_k", type=int, default=3)
    parser.add_argument("--ga_crossover", choices=["uniform", "one_point"], default="uniform")
    parser.add_argument("--ga_selection", choices=["rank", "tournament"], default="rank")
    parser.add_argument("--ga_rank_pressure", type=float, default=1.7)
    parser.add_argument("--ga_log_topk", type=int, default=3)
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
        "--reward_backend",
        choices=["auto", "unifiedreward", "unified", "imagereward", "hpsv2", "blend"],
        default="unifiedreward",
    )
    parser.add_argument(
        "--reward_weights",
        nargs=2,
        type=float,
        default=[1.0, 1.0],
        help="Blend backend weights: imagereward hpsv2",
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
    parser.add_argument("--save_variants", action="store_true")
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
    return parser.parse_args()


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
        if not name.endswith(".jsonl"):
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
        mode_stats[mode] = {
            "count": len(mode_rows),
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "mean_delta_vs_base": float(np.mean(deltas)) if deltas else 0.0,
            "std_delta_vs_base": float(np.std(deltas)) if deltas else 0.0,
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
    txt.append("Mode\tCount\tMeanScore\tMeanDeltaVsBase\tStdDeltaVsBase")
    for mode in sorted(mode_stats):
        s = mode_stats[mode]
        txt.append(
            f"{mode}\t{s['count']}\t{s['mean_score']:.6f}\t"
            f"{s['mean_delta_vs_base']:+.6f}\t{s['std_delta_vs_base']:.6f}"
        )
    txt.append("")
    txt.append("Best mode counts:")
    for mode, c in sorted(best_mode_counts.items()):
        txt.append(f"{mode}: {c}")
    with open(os.path.join(out_dir, "aggregate_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt) + "\n")
    return aggregate


def main() -> None:
    faulthandler.enable(all_threads=True)
    args = normalize_paths(parse_args())
    rank, world_size, local_rank, distributed = init_runtime()

    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.join(args.out_dir, "logs")
    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(log_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(img_dir, exist_ok=True)

    all_entries = load_prompt_range(args.prompt_file, args.start_index, args.end_index)
    my_entries = shard(all_entries, rank, world_size)

    if rank == 0:
        print(
            f"DDP launch: world_size={world_size} prompts_total={len(all_entries)} "
            f"range=[{args.start_index},{args.end_index if args.end_index!=-1 else 'end'})"
        )
    print(f"[rank {rank}] local_rank={local_rank} assigned_prompts={len(my_entries)}")

    try:
        rewrite_cache: Dict[str, List[str]] = {}
        if args.rewrites_file and os.path.exists(args.rewrites_file):
            rewrite_cache = json.load(open(args.rewrites_file, encoding="utf-8"))

        ctx = load_pipeline(args)
        reward_model = load_reward_model(args, ctx.device)

        rank_rows: List[Dict[str, Any]] = []
        for prompt_index, prompt in my_entries:
            slug = f"p{prompt_index:05d}"
            seed = args.seed + prompt_index if args.seed_per_prompt else args.seed
            print(f"[rank {rank}] {slug} seed={seed}")

            variants = generate_variants(args, prompt, rewrite_cache)
            if args.save_variants:
                with open(os.path.join(args.out_dir, f"{slug}_variants.txt"), "w", encoding="utf-8") as f:
                    for vi, text in enumerate(variants):
                        f.write(f"v{vi}: {text}\n")

            emb = encode_variants(ctx, variants, max_sequence_length=args.max_sequence_length)
            base_img, base_score = run_baseline(
                args,
                ctx,
                emb,
                reward_model,
                prompt,
                seed,
                cfg_scale=float(args.baseline_cfg),
            )

            if args.save_images:
                base_img.save(os.path.join(img_dir, f"{slug}_base.png"))

            if "base" in args.modes:
                rank_rows.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "seed": seed,
                        "mode": "base",
                        "score": float(base_score),
                        "delta_vs_base": 0.0,
                        "baseline_cfg": float(args.baseline_cfg),
                        "schedule": [[0, float(args.baseline_cfg)] for _ in range(args.steps)],
                    }
                )

            if "greedy" in args.modes:
                greedy = run_greedy(args, ctx, emb, reward_model, prompt, variants, seed)
                if args.save_images:
                    greedy.image.save(os.path.join(img_dir, f"{slug}_greedy.png"))
                    save_comparison(
                        os.path.join(img_dir, f"{slug}_greedy_comp.png"),
                        base_img,
                        greedy.image,
                        base_score,
                        greedy.score,
                        greedy.actions,
                    )
                rank_rows.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "seed": seed,
                        "mode": "greedy",
                        "score": float(greedy.score),
                        "delta_vs_base": float(greedy.score - base_score),
                        "baseline_score": float(base_score),
                        "actions": [[int(v), float(c)] for v, c in greedy.actions],
                    }
                )

            if "mcts" in args.modes:
                mcts = run_mcts(args, ctx, emb, reward_model, prompt, variants, seed)
                if args.save_images:
                    mcts.image.save(os.path.join(img_dir, f"{slug}_mcts.png"))
                    save_comparison(
                        os.path.join(img_dir, f"{slug}_mcts_comp.png"),
                        base_img,
                        mcts.image,
                        base_score,
                        mcts.score,
                        mcts.actions,
                    )
                rank_rows.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "seed": seed,
                        "mode": "mcts",
                        "score": float(mcts.score),
                        "delta_vs_base": float(mcts.score - base_score),
                        "baseline_score": float(base_score),
                        "actions": [[int(v), float(c)] for v, c in mcts.actions],
                    }
                )

            if "ga" in args.modes:
                ga = run_ga(
                    args,
                    ctx,
                    emb,
                    reward_model,
                    prompt,
                    variants,
                    seed,
                    log_dir=os.path.join(args.out_dir, "ga_logs"),
                    log_prefix=slug,
                )
                if args.save_images:
                    ga.image.save(os.path.join(img_dir, f"{slug}_ga.png"))
                    save_comparison(
                        os.path.join(img_dir, f"{slug}_ga_comp.png"),
                        base_img,
                        ga.image,
                        base_score,
                        ga.score,
                        ga.actions,
                    )
                rank_rows.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "seed": seed,
                        "mode": "ga",
                        "score": float(ga.score),
                        "delta_vs_base": float(ga.score - base_score),
                        "baseline_score": float(base_score),
                        "actions": [[int(v), float(c)] for v, c in ga.actions],
                    }
                )

            if "smc" in args.modes:
                smc = run_smc(args, ctx, emb, reward_model, prompt, variants, seed)
                if args.save_images:
                    smc.image.save(os.path.join(img_dir, f"{slug}_smc.png"))
                    save_comparison(
                        os.path.join(img_dir, f"{slug}_smc_comp.png"),
                        base_img,
                        smc.image,
                        base_score,
                        smc.score,
                        smc.actions,
                    )
                rank_rows.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "seed": seed,
                        "mode": "smc",
                        "score": float(smc.score),
                        "delta_vs_base": float(smc.score - base_score),
                        "baseline_score": float(base_score),
                        "actions": [[int(v), float(c)] for v, c in smc.actions],
                        "search_diagnostics": smc.diagnostics,
                    }
                )

        rank_log = os.path.join(log_dir, f"rank_{rank:03d}.jsonl")
        with open(rank_log, "w", encoding="utf-8") as f:
            for row in rank_rows:
                f.write(json.dumps(row) + "\n")
        with open(os.path.join(log_dir, f"rank_{rank:03d}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "rank": rank,
                    "world_size": world_size,
                    "assigned_prompts": len(my_entries),
                    "rows": len(rank_rows),
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
    except Exception as exc:
        err_path = os.path.join(log_dir, f"rank_{rank:03d}_error.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(f"rank={rank} local_rank={local_rank} world_size={world_size}\n")
            f.write(f"error_type={type(exc).__name__}\n")
            f.write(f"error={exc}\n")
        print(f"[rank {rank}] ERROR: {type(exc).__name__}: {exc}")
        print(f"[rank {rank}] wrote error log: {err_path}")
        raise


if __name__ == "__main__":
    main()
