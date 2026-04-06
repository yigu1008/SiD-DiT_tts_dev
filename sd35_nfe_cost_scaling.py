#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sampling_unified_sd35 as su
from sampling_unified_sd35_lookahead_reweighting import run_mcts_lookahead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NFE-cost scaling benchmark: lookahead MCTS with reward correction vs SMC, "
            "then plot ImageReward vs reward-reweighted NFE."
        )
    )
    parser.add_argument("--prompt_file", default=str(Path(__file__).resolve().parent / "hpsv2_subset.txt"))
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=5)
    parser.add_argument("--sim_costs", nargs="+", type=int, default=[5, 10, 20, 35, 50])
    parser.add_argument("--out_dir", default="./sd35_nfe_cost_scaling_out")

    parser.add_argument("--backend", choices=["sid", "senseflow_large", "senseflow_medium"], default="sid")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--cfg_scales", nargs="+", type=float, default=[1.0, 1.25, 1.5, 1.75, 2.0])
    parser.add_argument("--baseline_cfg", type=float, default=1.0)
    parser.add_argument("--correction_strengths", nargs="+", type=float, default=[1.0])
    parser.add_argument("--n_variants", type=int, default=3)
    parser.add_argument("--use_qwen", action="store_true")
    parser.add_argument("--rewrites_file", default=None)

    parser.add_argument(
        "--reward_backend",
        choices=["imagereward", "unifiedreward", "unified", "pickscore", "blend", "auto"],
        default="imagereward",
    )
    parser.add_argument("--reward_model", default="CodeGoat24/UnifiedReward-qwen-7b")
    parser.add_argument("--unifiedreward_model", default=None)
    parser.add_argument("--image_reward_model", default="ImageReward-v1.0")
    parser.add_argument("--pickscore_model", default="yuvalkirstain/PickScore_v1")
    parser.add_argument("--reward_weights", nargs=2, type=float, default=[1.0, 1.0])
    parser.add_argument("--reward_api_base", default=None)
    parser.add_argument("--reward_api_key", default="unifiedreward")
    parser.add_argument("--reward_api_model", default="UnifiedReward-7b-v1.5")
    parser.add_argument("--reward_max_new_tokens", type=int, default=512)
    parser.add_argument("--reward_prompt_mode", choices=["standard", "strict"], default="standard")

    parser.add_argument(
        "--lookahead_mode",
        choices=[
            "instrumentation",
            "rollout_prior",
            "tree_prior",
            "rollout_tree_prior",
            "adaptive_cfg_width",
        ],
        default="rollout_tree_prior",
    )
    parser.add_argument(
        "--lookahead_u_t_def",
        choices=["latent_delta_rms", "latent_rms", "dx_rms"],
        default="latent_delta_rms",
    )
    parser.add_argument("--lookahead_tau", type=float, default=0.35)
    parser.add_argument("--lookahead_c_puct", type=float, default=1.20)
    parser.add_argument("--lookahead_u_ref", type=float, default=0.0)
    parser.add_argument("--lookahead_w_cfg", type=float, default=1.0)
    parser.add_argument("--lookahead_w_variant", type=float, default=0.25)
    parser.add_argument("--lookahead_w_cs", type=float, default=0.10)
    parser.add_argument("--lookahead_w_q", type=float, default=0.20)
    parser.add_argument("--lookahead_w_explore", type=float, default=0.05)
    parser.add_argument("--lookahead_cfg_width_min", type=int, default=3)
    parser.add_argument("--lookahead_cfg_width_max", type=int, default=7)
    parser.add_argument("--lookahead_cfg_anchor_count", type=int, default=2)
    parser.add_argument("--lookahead_min_visits_for_center", type=int, default=3)
    parser.add_argument("--lookahead_log_action_topk", type=int, default=12)

    parser.add_argument("--smc_gamma", type=float, default=0.10)
    parser.add_argument("--ess_threshold", type=float, default=0.5)
    parser.add_argument("--resample_start_frac", type=float, default=0.3)
    parser.add_argument("--smc_cfg_scale", type=float, default=1.25)
    parser.add_argument("--smc_variant_idx", type=int, default=0)

    parser.add_argument(
        "--reward_nfe_weight",
        type=float,
        default=1.0,
        help="reward-reweighted NFE = nfe_transformer + reward_nfe_weight * nfe_correction",
    )
    return parser.parse_args()


def build_sampling_args(cfg: argparse.Namespace) -> argparse.Namespace:
    argv = [
        "--backend",
        str(cfg.backend),
        "--steps",
        str(int(cfg.steps)),
        "--width",
        str(int(cfg.width)),
        "--height",
        str(int(cfg.height)),
        "--seed",
        str(int(cfg.seed)),
        "--n_variants",
        str(int(cfg.n_variants)),
        "--cfg_scales",
        *[str(float(v)) for v in cfg.cfg_scales],
        "--baseline_cfg",
        str(float(cfg.baseline_cfg)),
        "--correction_strengths",
        *[str(float(v)) for v in cfg.correction_strengths],
        "--reward_backend",
        str(cfg.reward_backend),
        "--reward_model",
        str(cfg.reward_model),
        "--unifiedreward_model",
        str(cfg.unifiedreward_model or cfg.reward_model),
        "--image_reward_model",
        str(cfg.image_reward_model),
        "--pickscore_model",
        str(cfg.pickscore_model),
        "--reward_weights",
        str(float(cfg.reward_weights[0])),
        str(float(cfg.reward_weights[1])),
        "--reward_api_key",
        str(cfg.reward_api_key),
        "--reward_api_model",
        str(cfg.reward_api_model),
        "--reward_max_new_tokens",
        str(int(cfg.reward_max_new_tokens)),
        "--reward_prompt_mode",
        str(cfg.reward_prompt_mode),
    ]
    if cfg.reward_api_base:
        argv.extend(["--reward_api_base", str(cfg.reward_api_base)])
    if not bool(cfg.use_qwen):
        argv.append("--no_qwen")
    if cfg.rewrites_file:
        argv.extend(["--rewrites_file", str(Path(cfg.rewrites_file).expanduser().resolve())])

    args = su.parse_args(argv)
    args.out_dir = str(Path(cfg.out_dir).expanduser().resolve())
    os.makedirs(args.out_dir, exist_ok=True)

    args.lookahead_mode = str(cfg.lookahead_mode)
    args.lookahead_u_t_def = str(cfg.lookahead_u_t_def)
    args.lookahead_tau = float(cfg.lookahead_tau)
    args.lookahead_c_puct = float(cfg.lookahead_c_puct)
    args.lookahead_u_ref = float(cfg.lookahead_u_ref)
    args.lookahead_w_cfg = float(cfg.lookahead_w_cfg)
    args.lookahead_w_variant = float(cfg.lookahead_w_variant)
    args.lookahead_w_cs = float(cfg.lookahead_w_cs)
    args.lookahead_w_q = float(cfg.lookahead_w_q)
    args.lookahead_w_explore = float(cfg.lookahead_w_explore)
    args.lookahead_cfg_width_min = int(cfg.lookahead_cfg_width_min)
    args.lookahead_cfg_width_max = int(cfg.lookahead_cfg_width_max)
    args.lookahead_cfg_anchor_count = int(cfg.lookahead_cfg_anchor_count)
    args.lookahead_min_visits_for_center = int(cfg.lookahead_min_visits_for_center)
    args.lookahead_log_action_topk = int(cfg.lookahead_log_action_topk)

    args.smc_gamma = float(cfg.smc_gamma)
    args.ess_threshold = float(cfg.ess_threshold)
    args.resample_start_frac = float(cfg.resample_start_frac)
    args.smc_cfg_scale = float(cfg.smc_cfg_scale)
    args.smc_variant_idx = int(cfg.smc_variant_idx)
    return args


def load_prompt_subset(prompt_file: str, start_index: int, num_prompts: int) -> list[tuple[int, str]]:
    path = Path(prompt_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"prompt_file not found: {path}")
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    start = max(0, int(start_index))
    end = min(len(prompts), start + max(1, int(num_prompts)))
    if start >= end:
        raise RuntimeError(f"Empty prompt subset: start={start} end={end} total={len(prompts)}")
    return [(idx, prompts[idx]) for idx in range(start, end)]


def reward_reweighted_nfe(nfe_t: int, nfe_r: int, weight: float) -> float:
    return float(float(nfe_t) + (float(weight) * float(nfe_r)))


def run_benchmark(cfg: argparse.Namespace) -> dict[str, Any]:
    run_args = build_sampling_args(cfg)
    prompts = load_prompt_subset(cfg.prompt_file, cfg.start_index, cfg.num_prompts)
    sim_costs = sorted({int(v) for v in cfg.sim_costs if int(v) > 0})
    if len(sim_costs) < 5:
        raise RuntimeError("Please provide at least 5 positive SIM costs via --sim_costs.")

    rewrite_cache: dict[str, list[str]] = {}
    if run_args.rewrites_file and os.path.exists(run_args.rewrites_file):
        with open(run_args.rewrites_file, encoding="utf-8") as f:
            rewrite_cache = json.load(f)
        print(f"[nfe-scaling] loaded rewrite cache: {run_args.rewrites_file} ({len(rewrite_cache)} prompts)")

    print(f"[nfe-scaling] loading pipeline backend={run_args.backend} reward_backend={run_args.reward_backend}")
    ctx = su.load_pipeline(run_args)
    reward_model = su.load_reward_model(run_args, ctx.device)

    records: list[dict[str, Any]] = []
    for p_order, (p_idx, prompt) in enumerate(prompts):
        print(f"\n[{p_order + 1}/{len(prompts)}] p{p_idx:03d}: {prompt}")
        variants = su.generate_variants(run_args, prompt, rewrite_cache)
        emb = su.encode_variants(ctx, variants)
        p_seed = int(run_args.seed) + int(p_idx)

        for sim_cost in sim_costs:
            mcts_args = copy.deepcopy(run_args)
            mcts_args.n_sims = int(sim_cost)
            ctx.nfe = 0
            ctx.correction_nfe = 0
            mcts = run_mcts_lookahead(mcts_args, ctx, emb, reward_model, prompt, variants, p_seed)
            mcts_nfe_t = int(ctx.nfe)
            mcts_nfe_r = int(ctx.correction_nfe)
            mcts_nfe_rw = reward_reweighted_nfe(mcts_nfe_t, mcts_nfe_r, cfg.reward_nfe_weight)
            records.append(
                {
                    "method": "lookahead_mcts_rc",
                    "prompt_index": int(p_idx),
                    "seed": int(p_seed),
                    "sim_cost": int(sim_cost),
                    "imagereward": float(mcts.score),
                    "nfe_transformer": int(mcts_nfe_t),
                    "nfe_correction": int(mcts_nfe_r),
                    "nfe_reward_reweighted": float(mcts_nfe_rw),
                    "lookahead_mode": str(mcts_args.lookahead_mode),
                }
            )
            print(
                f"  mcts_rc cost={sim_cost:>3d} IR={mcts.score:+.4f} "
                f"NFE(T={mcts_nfe_t},R={mcts_nfe_r},RW={mcts_nfe_rw:.1f})"
            )

            smc_args = copy.deepcopy(run_args)
            smc_args.smc_k = int(sim_cost)
            smc_args.smc_variant_idx = int(max(0, min(len(emb.cond_text) - 1, int(smc_args.smc_variant_idx))))
            ctx.nfe = 0
            ctx.correction_nfe = 0
            smc = su.run_smc(smc_args, ctx, emb, reward_model, prompt, variants, p_seed)
            smc_nfe_t = int(ctx.nfe)
            smc_nfe_r = int(ctx.correction_nfe)
            smc_nfe_rw = reward_reweighted_nfe(smc_nfe_t, smc_nfe_r, cfg.reward_nfe_weight)
            records.append(
                {
                    "method": "smc",
                    "prompt_index": int(p_idx),
                    "seed": int(p_seed),
                    "sim_cost": int(sim_cost),
                    "imagereward": float(smc.score),
                    "nfe_transformer": int(smc_nfe_t),
                    "nfe_correction": int(smc_nfe_r),
                    "nfe_reward_reweighted": float(smc_nfe_rw),
                    "smc_k": int(smc_args.smc_k),
                }
            )
            print(
                f"  smc     cost={sim_cost:>3d} IR={smc.score:+.4f} "
                f"NFE(T={smc_nfe_t},R={smc_nfe_r},RW={smc_nfe_rw:.1f})"
            )

    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_key[(str(row["method"]), int(row["sim_cost"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (method, sim_cost), rows in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ir = np.asarray([float(r["imagereward"]) for r in rows], dtype=np.float64)
        nfe_t = np.asarray([float(r["nfe_transformer"]) for r in rows], dtype=np.float64)
        nfe_r = np.asarray([float(r["nfe_correction"]) for r in rows], dtype=np.float64)
        nfe_rw = np.asarray([float(r["nfe_reward_reweighted"]) for r in rows], dtype=np.float64)
        summary_rows.append(
            {
                "method": str(method),
                "sim_cost": int(sim_cost),
                "count": int(len(rows)),
                "mean_imagereward": float(ir.mean()) if ir.size else 0.0,
                "std_imagereward": float(ir.std()) if ir.size else 0.0,
                "mean_nfe_transformer": float(nfe_t.mean()) if nfe_t.size else 0.0,
                "mean_nfe_correction": float(nfe_r.mean()) if nfe_r.size else 0.0,
                "mean_nfe_reward_reweighted": float(nfe_rw.mean()) if nfe_rw.size else 0.0,
            }
        )

    return {
        "config": {
            "prompt_file": str(Path(cfg.prompt_file).expanduser().resolve()),
            "start_index": int(cfg.start_index),
            "num_prompts": int(cfg.num_prompts),
            "sim_costs": [int(v) for v in sim_costs],
            "reward_nfe_weight": float(cfg.reward_nfe_weight),
            "lookahead_mode": str(cfg.lookahead_mode),
            "correction_strengths": [float(v) for v in cfg.correction_strengths],
        },
        "records": records,
        "summary": summary_rows,
    }


def write_outputs(payload: dict[str, Any], out_dir: str) -> dict[str, str]:
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "nfe_cost_scaling_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    tsv_path = out_path / "nfe_cost_scaling_summary.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write(
            "method\tsim_cost\tcount\tmean_imagereward\tstd_imagereward\t"
            "mean_nfe_transformer\tmean_nfe_correction\tmean_nfe_reward_reweighted\n"
        )
        for row in payload["summary"]:
            f.write(
                f"{row['method']}\t{int(row['sim_cost'])}\t{int(row['count'])}\t"
                f"{float(row['mean_imagereward']):.6f}\t{float(row['std_imagereward']):.6f}\t"
                f"{float(row['mean_nfe_transformer']):.2f}\t{float(row['mean_nfe_correction']):.2f}\t"
                f"{float(row['mean_nfe_reward_reweighted']):.2f}\n"
            )

    fig_path = out_path / "imagereward_vs_nfe_reward_reweighted.png"
    plt.figure(figsize=(8.0, 5.8))
    summary = payload["summary"]
    for method, color, marker in [
        ("lookahead_mcts_rc", "#1f77b4", "o"),
        ("smc", "#d62728", "s"),
    ]:
        rows = [r for r in summary if r["method"] == method]
        rows = sorted(rows, key=lambda r: float(r["mean_nfe_reward_reweighted"]))
        if not rows:
            continue
        xs = [float(r["mean_nfe_reward_reweighted"]) for r in rows]
        ys = [float(r["mean_imagereward"]) for r in rows]
        plt.plot(xs, ys, marker=marker, color=color, linewidth=2.0, markersize=6, label=method)
        for r in rows:
            plt.annotate(
                f"c{int(r['sim_cost'])}",
                (float(r["mean_nfe_reward_reweighted"]), float(r["mean_imagereward"])),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color=color,
            )

    w = float(payload["config"]["reward_nfe_weight"])
    plt.xlabel(f"Reward-Reweighted NFE (T + {w:g}*R)")
    plt.ylabel("ImageReward (mean over prompts)")
    plt.title("NFE Cost Scaling: Lookahead MCTS + Reward Correction vs SMC")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    return {
        "json": str(json_path),
        "tsv": str(tsv_path),
        "plot_png": str(fig_path),
    }


def main() -> None:
    cfg = parse_args()
    payload = run_benchmark(cfg)
    paths = write_outputs(payload, cfg.out_dir)

    print("\nNFE cost scaling summary")
    print(f"{'method':<18} {'cost':>6} {'IR(mean)':>10} {'NFE_RW(mean)':>14}")
    print("-" * 54)
    for row in payload["summary"]:
        print(
            f"{row['method']:<18} {int(row['sim_cost']):>6d} "
            f"{float(row['mean_imagereward']):>+10.4f} {float(row['mean_nfe_reward_reweighted']):>14.2f}"
        )
    print("-" * 54)
    print(f"JSON: {paths['json']}")
    print(f"TSV:  {paths['tsv']}")
    print(f"PLOT: {paths['plot_png']}")


if __name__ == "__main__":
    main()
