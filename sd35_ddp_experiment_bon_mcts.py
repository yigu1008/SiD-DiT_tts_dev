from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

import mcts_hybrid_ut_dt
import mcts_improved
import sd35_ddp_experiment as base
from sampling_unified_sd35 import encode_variants as _encode_variants_with_neg
from sampling_unified_sd35_lookahead_reweighting import run_mcts_lookahead


def _build_neg_emb_bank(args: argparse.Namespace, ctx: Any, variants: list[str], default_emb: Any) -> list[Any]:
    """Pre-encode one EmbeddingContext per negative prompt in the bank.

    Slot 0 is always ``default_emb`` (the empty-negative encoding already done
    by the outer pipeline) — saves one redundant encode pass when the bank
    starts with the empty string.
    """
    raw_bank = getattr(args, "bon_mcts_neg_bank", None) or [""]
    bank: list[Any] = []
    max_seq = int(getattr(args, "max_sequence_length", 256) or 256)
    for neg in raw_bank:
        s = str(neg or "")
        if s == "" and len(bank) == 0:
            bank.append(default_emb)
            continue
        bank.append(_encode_variants_with_neg(ctx, variants, max_sequence_length=max_seq, negative_prompt=s))
    return bank


def _build_sigma_perturb_bank(args: argparse.Namespace) -> list[tuple[float, list[float] | None]]:
    """Pre-build perturbed sigma schedules from --bon_mcts_sigma_perturb_bank.

    Returns list of (delta, perturbed_sigmas). delta is for diagnostics.
    perturbed_sigmas is None for delta==0.0 (means: use args.sigmas as-is).
    """
    raw = getattr(args, "bon_mcts_sigma_perturb_bank", None) or [0.0]
    base_sigmas = getattr(args, "sigmas", None)
    if base_sigmas is None:
        # Linear schedule will be built at sampling time — perturbation only
        # supported when explicit sigmas exist.
        return [(0.0, None)]
    sig = [float(s) for s in base_sigmas]
    sigma_max = max(sig) if sig else 1.0
    out: list[tuple[float, list[float] | None]] = []
    for delta in raw:
        d = float(delta)
        if d == 0.0:
            out.append((0.0, None))
            continue
        perturbed = [max(0.0, min(sigma_max, s + d * sigma_max)) for s in sig]
        # Re-sort descending (sigma_next < sigma_current), in case perturbation flipped order.
        perturbed = sorted(perturbed, reverse=True)
        out.append((d, perturbed))
    return out


def _parse_bon_mcts_flags(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--bon_mcts_n_seeds",
        type=int,
        default=8,
        help="Number of seeds in BoN prescreen before MCTS refinement.",
    )
    parser.add_argument(
        "--bon_mcts_topk",
        type=int,
        default=2,
        help="Top-K prescreen seeds to refine with MCTS.",
    )
    parser.add_argument(
        "--bon_mcts_seed_stride",
        type=int,
        default=1,
        help="Stride between candidate seeds in prescreen pool.",
    )
    parser.add_argument(
        "--bon_mcts_seed_offset",
        type=int,
        default=0,
        help="Offset applied to the prompt seed before BoN seed generation.",
    )
    parser.add_argument(
        "--bon_mcts_sim_alloc",
        choices=["split", "full"],
        default="split",
        help="split: divide n_sims across top-K seeds, full: run full n_sims for each top-K seed.",
    )
    parser.add_argument(
        "--bon_mcts_min_sims",
        type=int,
        default=8,
        help="Lower bound for per-seed MCTS sims when sim_alloc=split.",
    )
    parser.add_argument(
        "--bon_mcts_prescreen_cfg",
        type=float,
        default=None,
        help="CFG used in BoN prescreen baseline scoring. Defaults to --baseline_cfg.",
    )
    parser.add_argument(
        "--bon_mcts_refine_method",
        choices=["ours_tree", "mcts", "mcts_improved", "hybrid_ut_dt"],
        default="ours_tree",
        help="Refinement tree-search after BoN prescreen: "
             "ours_tree (full lookahead module) | "
             "mcts (vanilla UCB1) | "
             "mcts_improved (UCB1-Tuned + reward norm + x0-pred bootstrap) | "
             "hybrid_ut_dt (U_t/D_t latent priors + reward norm + x0-pred bootstrap).",
    )
    parser.add_argument(
        "--bon_mcts_neg_bank",
        nargs="+",
        default=None,
        help="Negative-prompt bank for the prescreen axis. Each entry produces "
             "a distinct uncond-text encoding; the prescreen pool fans out over "
             "(seed × neg_idx × sigma_perturb_idx). Default = single empty "
             "negative (original behavior). Example: \"\" \"low quality, blurry\".",
    )
    parser.add_argument(
        "--bon_mcts_sigma_perturb_bank",
        type=float,
        nargs="+",
        default=None,
        help="Sigma-schedule perturbation bank (additive, in [-1,1] of sigma_max). "
             "Each entry produces a perturbed copy of args.sigmas used for that "
             "prescreen rollout: sigma_i ← clip(sigma_i + delta * sigma_max, 0, sigma_max). "
             "Default = [0.0] (original schedule). Example: -0.05 0.0 0.05.",
    )
    parser.add_argument(
        "--mcts_step_reward_alpha",
        type=float,
        default=0.0,
        help="SoP-style per-step reward backup mix factor in [0,1]. "
             "0=terminal-only (default), 1=per-step-only.  "
             "Mixed backup: alpha·avg_step_reward + (1-alpha)·terminal_reward. "
             "Per-step reward = decode x̂₀ + score at every MCTS key step "
             "(adds ~1 reward call per key step per rollout).",
    )
    parser.add_argument(
        "--mcts_step_reward_progress_weight",
        type=int,
        default=1,
        choices=[0, 1],
        help="1=weight step rewards by progress (later steps trusted more), "
             "0=uniform.  Only used when mcts_step_reward_alpha > 0.",
    )
    # Surface flag groups for both improved variants.
    mcts_improved.add_mcts_improved_args(parser)
    mcts_hybrid_ut_dt.add_mcts_hybrid_args(parser)
    parser.add_argument(
        "--lookahead_mode",
        choices=[
            "standard",
            "instrumentation",
            "rollout_prior",
            "tree_prior",
            "rollout_tree_prior",
            "rollout_tree_prior_adaptive_cfg",
            "adaptive_cfg_width",
        ],
        default="rollout_tree_prior_adaptive_cfg",
    )
    parser.add_argument(
        "--lookahead_u_t_def",
        choices=["latent_delta_rms", "latent_rms", "dx_rms"],
        default="latent_delta_rms",
    )
    parser.add_argument("--lookahead_tau", type=float, default=0.35)
    parser.add_argument("--lookahead_c_puct", type=float, default=1.20)
    parser.add_argument("--lookahead_u_ref", type=float, default=0.0)
    parser.add_argument("--lookahead_d_ref", type=float, default=0.0)
    parser.add_argument("--lookahead_ref_percentile", type=float, default=75.0)
    parser.add_argument("--lookahead_prior_tau", type=float, default=0.35)
    parser.add_argument("--lookahead_w_cfg", type=float, default=1.0)
    parser.add_argument("--lookahead_w_variant", type=float, default=0.25)
    parser.add_argument("--lookahead_w_cs", type=float, default=0.10)
    parser.add_argument("--lookahead_w_q", type=float, default=0.20)
    parser.add_argument("--lookahead_w_explore", type=float, default=0.05)
    parser.add_argument("--lookahead_prior_mode", choices=["heuristic", "signal", "progress_prompt", "trajectory_feedback"], default="heuristic")
    parser.add_argument("--lookahead_w_progress", type=float, default=1.0)
    parser.add_argument("--lookahead_w_prompt", type=float, default=1.0)
    parser.add_argument("--lookahead_w_update", type=float, default=1.0)
    parser.add_argument("--lookahead_w_cond", type=float, default=1.0)
    parser.add_argument("--lookahead_use_stepwise_refs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lookahead_cfg_width_min", type=int, default=3)
    parser.add_argument("--lookahead_cfg_width_max", type=int, default=7)
    parser.add_argument("--lookahead_cfg_anchor_count", type=int, default=2)
    parser.add_argument("--lookahead_min_visits_for_center", type=int, default=3)
    parser.add_argument("--lookahead_log_action_topk", type=int, default=12)
    return parser.parse_known_args(argv)


def _make_patched_parse_args(
    original_parse_args: Callable[[], argparse.Namespace],
) -> Callable[[], argparse.Namespace]:
    def _patched_parse_args() -> argparse.Namespace:
        bon_mcts_args, remaining = _parse_bon_mcts_flags(sys.argv[1:])
        original_argv = sys.argv
        sys.argv = [original_argv[0]] + remaining
        try:
            args = original_parse_args()
        finally:
            sys.argv = original_argv
        for key, value in vars(bon_mcts_args).items():
            setattr(args, key, value)
        return args

    return _patched_parse_args


def _compute_sim_budgets(args: argparse.Namespace, topk: int) -> list[int]:
    topk = max(1, int(topk))
    mode = str(getattr(args, "bon_mcts_sim_alloc", "split")).strip().lower()
    total = max(1, int(getattr(args, "n_sims", 1)))
    if mode == "full":
        return [int(total)] * int(topk)

    min_sims = max(1, int(getattr(args, "bon_mcts_min_sims", 1)))
    base = max(1, int(total) // int(topk))
    budgets = [int(base)] * int(topk)
    rem = int(total) - int(base) * int(topk)
    for i in range(max(0, rem)):
        budgets[i] += 1
    budgets = [max(int(min_sims), int(b)) for b in budgets]
    return budgets


def _run_bon_mcts(
    args: argparse.Namespace,
    ctx: Any,
    emb: Any,
    reward_model: Any,
    prompt: str,
    variants: list[str],
    seed: int,
    original_run_mcts: Callable[..., Any],
    lookahead_run_mcts: Callable[..., Any],
    improved_run_mcts: Callable[..., Any],
    hybrid_run_mcts: Callable[..., Any],
) -> Any:
    base_seed = int(seed)
    n_seeds = max(1, int(getattr(args, "bon_mcts_n_seeds", 8)))
    topk = max(1, min(int(getattr(args, "bon_mcts_topk", 2)), n_seeds))
    stride = max(1, int(getattr(args, "bon_mcts_seed_stride", 1)))
    offset = int(getattr(args, "bon_mcts_seed_offset", 0))
    prescreen_cfg_raw = getattr(args, "bon_mcts_prescreen_cfg", None)
    prescreen_cfg = float(args.baseline_cfg if prescreen_cfg_raw is None else prescreen_cfg_raw)
    refine_method = str(getattr(args, "bon_mcts_refine_method", "ours_tree")).strip().lower()

    seed_pool = [int(base_seed + offset + i * stride) for i in range(n_seeds)]

    # Build prescreen axes for negative-prompt + sigma-perturb branching.
    neg_emb_bank = _build_neg_emb_bank(args, ctx, variants, emb)
    sigma_bank = _build_sigma_perturb_bank(args)
    neg_active = len(neg_emb_bank) > 1
    sigma_active = len(sigma_bank) > 1 and any(d != 0.0 for d, _ in sigma_bank)

    # Cartesian fan-out: each prescreen item is (seed, neg_idx, sigma_idx).
    # We do NOT multiply seeds × neg × sigma blindly — that would n×-balloon the
    # prescreen.  Instead we keep total prescreens ≈ n_seeds by *cycling* seeds
    # through the (neg × sigma) grid, mod n_seeds.  This holds wallclock fixed
    # while spreading exploration across new axes.
    n_negs = len(neg_emb_bank)
    n_sigs = len(sigma_bank)
    n_prescreens = max(n_seeds, n_negs * n_sigs)
    prescreen_items: list[tuple[int, int, int]] = []
    for k in range(n_prescreens):
        seed_i = seed_pool[k % n_seeds]
        neg_idx = k % n_negs
        sig_idx = (k // n_negs) % n_sigs
        prescreen_items.append((int(seed_i), int(neg_idx), int(sig_idx)))

    print(
        "  bon_mcts: "
        f"prescreen_n={n_prescreens} topk={topk} "
        f"prescreen_cfg={prescreen_cfg:.2f} sim_alloc={args.bon_mcts_sim_alloc} "
        f"refine={refine_method}"
        + (f" neg_bank={n_negs}" if neg_active else "")
        + (f" sigma_bank={n_sigs}" if sigma_active else "")
    )

    prescreen_rows: list[dict[str, Any]] = []
    orig_sigmas = getattr(args, "sigmas", None)
    for i, (seed_i, neg_idx, sig_idx) in enumerate(prescreen_items):
        emb_use = neg_emb_bank[neg_idx]
        sig_delta, sig_perturbed = sigma_bank[sig_idx]
        # Temporarily swap args.sigmas if a perturbed schedule applies.
        if sig_perturbed is not None:
            args.sigmas = list(sig_perturbed)
        else:
            args.sigmas = orig_sigmas
        try:
            _img, score = base.run_baseline(
                args,
                ctx,
                emb_use,
                reward_model,
                prompt,
                int(seed_i),
                cfg_scale=float(prescreen_cfg),
            )
        finally:
            args.sigmas = orig_sigmas
        prescreen_rows.append(
            {
                "seed": int(seed_i),
                "neg_idx": int(neg_idx),
                "sigma_idx": int(sig_idx),
                "sigma_delta": float(sig_delta),
                "prescreen_score": float(score),
            }
        )
        if i == 0 or (i + 1) % 10 == 0 or (i + 1) == len(prescreen_items):
            best_so_far = max(float(r["prescreen_score"]) for r in prescreen_rows)
            print(f"    prescreen {i + 1:3d}/{len(prescreen_items)} best={best_so_far:.4f}")

    prescreen_rows.sort(key=lambda row: float(row["prescreen_score"]), reverse=True)
    selected = prescreen_rows[:topk]
    budgets = _compute_sim_budgets(args, len(selected))
    print(f"    refine_budgets={budgets} total_refine_sims={sum(int(x) for x in budgets)}")

    best_result = None
    best_seed = int(selected[0]["seed"])
    best_neg_idx = int(selected[0]["neg_idx"])
    best_sigma_idx = int(selected[0]["sigma_idx"])
    mcts_rows: list[dict[str, Any]] = []
    for idx, (row, budget) in enumerate(zip(selected, budgets)):
        run_args = argparse.Namespace(**vars(args))
        run_args.n_sims = int(max(1, budget))
        if hasattr(run_args, "mcts_n_seeds"):
            run_args.mcts_n_seeds = 1
        seed_i = int(row["seed"])
        neg_idx_i = int(row["neg_idx"])
        sig_idx_i = int(row["sigma_idx"])
        emb_use = neg_emb_bank[neg_idx_i]
        _, sig_perturbed = sigma_bank[sig_idx_i]
        if sig_perturbed is not None:
            run_args.sigmas = list(sig_perturbed)
        if refine_method == "mcts":
            result = original_run_mcts(run_args, ctx, emb_use, reward_model, prompt, variants, seed_i)
        elif refine_method == "mcts_improved":
            result = improved_run_mcts(run_args, ctx, emb_use, reward_model, prompt, variants, seed_i)
        elif refine_method == "hybrid_ut_dt":
            result = hybrid_run_mcts(run_args, ctx, emb_use, reward_model, prompt, variants, seed_i)
        else:  # ours_tree
            result = lookahead_run_mcts(run_args, ctx, emb_use, reward_model, prompt, variants, seed_i)
        mcts_rows.append(
            {
                "rank": int(idx + 1),
                "seed": int(seed_i),
                "neg_idx": int(neg_idx_i),
                "sigma_idx": int(sig_idx_i),
                "sigma_delta": float(row["sigma_delta"]),
                "prescreen_score": float(row["prescreen_score"]),
                "n_sims_used": int(run_args.n_sims),
                "tree_search_score": float(result.score),
            }
        )
        print(
            f"    refine {idx + 1}/{len(selected)} seed={seed_i} "
            f"n_sims={run_args.n_sims} score={float(result.score):.4f}"
        )
        if best_result is None or float(result.score) > float(best_result.score):
            best_result = result
            best_seed = int(seed_i)
            best_neg_idx = int(neg_idx_i)
            best_sigma_idx = int(sig_idx_i)

    if best_result is None:
        raise RuntimeError("BoN+MCTS produced no result.")

    diagnostics = dict(getattr(best_result, "diagnostics", {}) or {})
    diagnostics["bon_mcts"] = {
        "enabled": True,
        "base_seed": int(base_seed),
        "prescreen_cfg": float(prescreen_cfg),
        "prescreen_n": int(n_seeds),
        "topk": int(topk),
        "seed_stride": int(stride),
        "seed_offset": int(offset),
        "refine_method": str(refine_method),
        "sim_alloc": str(getattr(args, "bon_mcts_sim_alloc", "split")),
        "sim_budgets": [int(x) for x in budgets],
        "winner_seed": int(best_seed),
        "winner_neg_idx": int(best_neg_idx),
        "winner_sigma_idx": int(best_sigma_idx),
        "neg_bank": list(getattr(args, "bon_mcts_neg_bank", None) or [""]),
        "sigma_perturb_bank": [float(d) for d, _ in sigma_bank],
        "prescreen_ranked": prescreen_rows,
        "tree_refine": mcts_rows,
    }
    if refine_method != "mcts":
        diagnostics["bon_mcts"]["lookahead"] = {
            "lookahead_mode": str(getattr(args, "lookahead_mode", "rollout_tree_prior_adaptive_cfg")),
            "lookahead_u_t_def": str(getattr(args, "lookahead_u_t_def", "latent_delta_rms")),
            "lookahead_tau": float(getattr(args, "lookahead_tau", 0.35)),
            "lookahead_c_puct": float(getattr(args, "lookahead_c_puct", 1.20)),
        }
    # Keep the concrete result object produced by the active backend's MCTS
    # implementation; just attach merged diagnostics.
    best_result.diagnostics = diagnostics
    return best_result


def main() -> None:
    original_parse_args = base.parse_args
    original_run_mcts = base.run_mcts

    def _patched_run_mcts(args, ctx, emb, reward_model, prompt, variants, seed):
        return _run_bon_mcts(
            args=args,
            ctx=ctx,
            emb=emb,
            reward_model=reward_model,
            prompt=prompt,
            variants=variants,
            seed=seed,
            original_run_mcts=original_run_mcts,
            lookahead_run_mcts=run_mcts_lookahead,
            improved_run_mcts=mcts_improved.run_mcts_improved,
            hybrid_run_mcts=mcts_hybrid_ut_dt.run_mcts_hybrid_ut_dt,
        )

    base.parse_args = _make_patched_parse_args(original_parse_args)
    base.run_mcts = _patched_run_mcts
    try:
        base.main()
    finally:
        base.parse_args = original_parse_args
        base.run_mcts = original_run_mcts


if __name__ == "__main__":
    main()
