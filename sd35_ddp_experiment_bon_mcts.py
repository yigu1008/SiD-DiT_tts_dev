from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

import sd35_ddp_experiment as base


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
) -> Any:
    base_seed = int(seed)
    n_seeds = max(1, int(getattr(args, "bon_mcts_n_seeds", 8)))
    topk = max(1, min(int(getattr(args, "bon_mcts_topk", 2)), n_seeds))
    stride = max(1, int(getattr(args, "bon_mcts_seed_stride", 1)))
    offset = int(getattr(args, "bon_mcts_seed_offset", 0))
    prescreen_cfg_raw = getattr(args, "bon_mcts_prescreen_cfg", None)
    prescreen_cfg = float(args.baseline_cfg if prescreen_cfg_raw is None else prescreen_cfg_raw)

    seed_pool = [int(base_seed + offset + i * stride) for i in range(n_seeds)]
    print(
        "  bon_mcts: "
        f"prescreen_n={n_seeds} topk={topk} "
        f"prescreen_cfg={prescreen_cfg:.2f} sim_alloc={args.bon_mcts_sim_alloc}"
    )

    prescreen_rows: list[dict[str, float | int]] = []
    for i, seed_i in enumerate(seed_pool):
        _img, score = base.run_baseline(
            args,
            ctx,
            emb,
            reward_model,
            prompt,
            int(seed_i),
            cfg_scale=float(prescreen_cfg),
        )
        prescreen_rows.append(
            {
                "seed": int(seed_i),
                "prescreen_score": float(score),
            }
        )
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
        if hasattr(run_args, "mcts_n_seeds"):
            run_args.mcts_n_seeds = 1
        seed_i = int(row["seed"])
        result = original_run_mcts(run_args, ctx, emb, reward_model, prompt, variants, seed_i)
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
        "prescreen_cfg": float(prescreen_cfg),
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
    return base.SearchResult(
        image=best_result.image,
        score=float(best_result.score),
        actions=[(int(v), float(c), float(r)) for v, c, r in best_result.actions],
        diagnostics=diagnostics,
    )


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
