#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

import sampling_flux_unified as base


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
        "--bon_mcts_prescreen_guidance",
        type=float,
        default=None,
        help="Guidance used in BoN prescreen baseline scoring. Defaults to --baseline_guidance_scale.",
    )
    parser.add_argument(
        "--bon_mcts_prescreen_cfg",
        type=float,
        default=None,
        help="Alias for --bon_mcts_prescreen_guidance (for config compatibility).",
    )
    return parser.parse_known_args(argv)


def _rewrite_search_method(argv: list[str]) -> list[str]:
    out = list(argv)
    for i, tok in enumerate(out):
        if tok == "--search_method" and i + 1 < len(out) and out[i + 1] == "bon_mcts":
            out[i + 1] = "mcts"
        elif tok.startswith("--search_method=") and tok.split("=", 1)[1] == "bon_mcts":
            out[i] = "--search_method=mcts"
    return out


def _compute_sim_budgets(args: argparse.Namespace, topk: int) -> list[int]:
    topk = max(1, int(topk))
    mode = str(getattr(args, "bon_mcts_sim_alloc", "split")).strip().lower()
    total = max(1, int(getattr(args, "n_sims", 1)))
    if mode == "full":
        return [int(total)] * int(topk)

    min_sims = max(1, int(getattr(args, "bon_mcts_min_sims", 1)))
    base_budget = max(1, int(total) // int(topk))
    budgets = [int(base_budget)] * int(topk)
    rem = int(total) - int(base_budget) * int(topk)
    for i in range(max(0, rem)):
        budgets[i] += 1
    budgets = [max(int(min_sims), int(v)) for v in budgets]
    return budgets


def _make_patched_parse_args(
    original_parse_args: Callable[[list[str] | None], argparse.Namespace],
) -> Callable[[list[str] | None], argparse.Namespace]:
    def _patched_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
        source = list(argv) if argv is not None else []
        if argv is None:
            import sys

            source = list(sys.argv[1:])
        bon_mcts_args, remaining = _parse_bon_mcts_flags(source)
        normalized = _rewrite_search_method(remaining)
        args = original_parse_args(normalized)
        for key, value in vars(bon_mcts_args).items():
            setattr(args, key, value)
        return args

    return _patched_parse_args


def _run_bon_mcts(
    args: argparse.Namespace,
    ctx: Any,
    reward_model: Any,
    prompt: str,
    embeds: list[Any],
    guidance_bank: list[float],
    seed: int,
    original_run_mcts: Callable[..., Any],
) -> Any:
    base_seed = int(seed)
    n_seeds = max(1, int(getattr(args, "bon_mcts_n_seeds", 8)))
    topk = max(1, min(int(getattr(args, "bon_mcts_topk", 2)), n_seeds))
    stride = max(1, int(getattr(args, "bon_mcts_seed_stride", 1)))
    offset = int(getattr(args, "bon_mcts_seed_offset", 0))
    prescreen_guidance = getattr(args, "bon_mcts_prescreen_guidance", None)
    if prescreen_guidance is None:
        prescreen_guidance = getattr(args, "bon_mcts_prescreen_cfg", None)
    if prescreen_guidance is None:
        prescreen_guidance = float(getattr(args, "baseline_guidance_scale", guidance_bank[0]))
    prescreen_guidance = float(prescreen_guidance)

    seed_pool = [int(base_seed + offset + i * stride) for i in range(n_seeds)]
    print(
        "  bon_mcts: "
        f"prescreen_n={n_seeds} topk={topk} "
        f"prescreen_guidance={prescreen_guidance:.3f} sim_alloc={args.bon_mcts_sim_alloc}"
    )

    prescreen_rows: list[dict[str, float | int]] = []
    for i, seed_i in enumerate(seed_pool):
        baseline_actions = [(0, float(prescreen_guidance)) for _ in range(int(args.steps))]
        result = base.run_action_sequence(
            args=args,
            ctx=ctx,
            reward_model=reward_model,
            prompt=prompt,
            embeds=embeds,
            seed=int(seed_i),
            actions=baseline_actions,
        )
        prescreen_rows.append({"seed": int(seed_i), "prescreen_score": float(result.score)})
        if i == 0 or (i + 1) % 10 == 0 or (i + 1) == len(seed_pool):
            best_so_far = max(float(r["prescreen_score"]) for r in prescreen_rows)
            print(f"    prescreen {i + 1:3d}/{len(seed_pool)} best={best_so_far:.4f}")

    prescreen_rows.sort(key=lambda row: float(row["prescreen_score"]), reverse=True)
    selected = prescreen_rows[:topk]
    budgets = _compute_sim_budgets(args, len(selected))
    print(f"    refine_budgets={budgets} total_refine_sims={sum(int(x) for x in budgets)}")

    best_result = None
    best_seed = int(selected[0]["seed"])
    mcts_rows: list[dict[str, Any]] = []
    for idx, (row, budget) in enumerate(zip(selected, budgets)):
        run_args = argparse.Namespace(**vars(args))
        run_args.n_sims = int(max(1, budget))
        seed_i = int(row["seed"])
        result = original_run_mcts(run_args, ctx, reward_model, prompt, embeds, guidance_bank, seed_i)
        mcts_rows.append(
            {
                "rank": int(idx + 1),
                "seed": int(seed_i),
                "prescreen_score": float(row["prescreen_score"]),
                "n_sims_used": int(run_args.n_sims),
                "mcts_score": float(result.score),
            }
        )
        print(
            f"    refine {idx + 1}/{len(selected)} seed={seed_i} "
            f"n_sims={run_args.n_sims} mcts={float(result.score):.4f}"
        )
        if best_result is None or float(result.score) > float(best_result.score):
            best_result = result
            best_seed = int(seed_i)

    if best_result is None:
        raise RuntimeError("BoN+MCTS produced no result.")

    diagnostics = dict(getattr(best_result, "diagnostics", {}) or {})
    diagnostics["bon_mcts"] = {
        "enabled": True,
        "base_seed": int(base_seed),
        "prescreen_guidance": float(prescreen_guidance),
        "prescreen_n": int(n_seeds),
        "topk": int(topk),
        "seed_stride": int(stride),
        "seed_offset": int(offset),
        "sim_alloc": str(getattr(args, "bon_mcts_sim_alloc", "split")),
        "sim_budgets": [int(x) for x in budgets],
        "winner_seed": int(best_seed),
        "prescreen_ranked": prescreen_rows,
        "mcts_refine": mcts_rows,
    }
    best_result.diagnostics = diagnostics
    return best_result


def main(argv: list[str] | None = None) -> None:
    original_parse_args = base.parse_args
    original_run_mcts = base.run_mcts

    def _patched_run_mcts(args, ctx, reward_model, prompt, embeds, guidance_bank, seed):
        return _run_bon_mcts(
            args=args,
            ctx=ctx,
            reward_model=reward_model,
            prompt=prompt,
            embeds=embeds,
            guidance_bank=guidance_bank,
            seed=seed,
            original_run_mcts=original_run_mcts,
        )

    base.parse_args = _make_patched_parse_args(original_parse_args)
    base.run_mcts = _patched_run_mcts
    try:
        base.main(argv)
    finally:
        base.parse_args = original_parse_args
        base.run_mcts = original_run_mcts


if __name__ == "__main__":
    main()
