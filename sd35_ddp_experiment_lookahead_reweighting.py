from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import sd35_ddp_experiment as base
from sampling_unified_sd35_lookahead_reweighting import run_mcts_lookahead


def _parse_lookahead_flags(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
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
        default="rollout_prior",
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
    parser.add_argument("--lookahead_log_action_topk", type=int, default=-1)

    # Key-step branching (shared with dynamic_cfg variant)
    parser.add_argument("--mcts_key_steps", default="",
                        help="Comma-separated key step indices for MCTS branching.")
    parser.add_argument("--mcts_key_step_count", type=int, default=0,
                        help="Auto-compute N evenly-spaced key steps. 0 = branch at every step.")
    return parser.parse_known_args(argv)


def _make_patched_parse_args(
    original_parse_args: Callable[[], argparse.Namespace],
) -> Callable[[], argparse.Namespace]:
    def _patched_parse_args() -> argparse.Namespace:
        lookahead_args, remaining = _parse_lookahead_flags(sys.argv[1:])
        original_argv = sys.argv
        sys.argv = [original_argv[0]] + remaining
        try:
            args = original_parse_args()
        finally:
            sys.argv = original_argv

        # sd35_ddp_experiment parser does not define --x0_sampler, but
        # sampling_unified_sd35 run_baseline/run_mcts expects args.x0_sampler.
        if not hasattr(args, "x0_sampler"):
            setattr(args, "x0_sampler", False)

        for key, value in vars(lookahead_args).items():
            setattr(args, key, value)
        return args

    return _patched_parse_args


def main() -> None:
    original_parse_args = base.parse_args
    original_run_mcts = base.run_mcts

    def _patched_run_mcts(args, ctx, emb, reward_model, prompt, variants, seed):
        mode = str(getattr(args, "lookahead_mode", "rollout_prior")).strip().lower()
        if mode == "standard":
            return original_run_mcts(args, ctx, emb, reward_model, prompt, variants, seed)
        return run_mcts_lookahead(args, ctx, emb, reward_model, prompt, variants, seed)

    base.parse_args = _make_patched_parse_args(original_parse_args)
    base.run_mcts = _patched_run_mcts
    try:
        base.main()
    finally:
        base.parse_args = original_parse_args
        base.run_mcts = original_run_mcts


if __name__ == "__main__":
    main()
