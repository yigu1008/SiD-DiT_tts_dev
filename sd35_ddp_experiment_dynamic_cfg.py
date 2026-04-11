from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import sd35_ddp_experiment as base
from sampling_unified_sd35_dynamic_cfg import run_mcts_dynamic_cfg


def _parse_dynamic_cfg_flags(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mcts_cfg_mode", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--mcts_cfg_root_bank", nargs="+", type=float, default=[1.0, 1.5, 2.0])
    parser.add_argument("--mcts_cfg_anchors", nargs="+", type=float, default=[1.0, 2.0])
    parser.add_argument("--mcts_cfg_step_anchor_count", type=int, default=2)
    parser.add_argument("--mcts_cfg_min_parent_visits", type=int, default=3)
    parser.add_argument("--mcts_cfg_round_ndigits", type=int, default=6)
    parser.add_argument("--mcts_cfg_log_action_topk", type=int, default=12)
    parser.add_argument("--mcts_key_mode", choices=["all", "manual", "count", "stride"], default="count")
    parser.add_argument("--mcts_key_step_stride", type=int, default=0)
    parser.add_argument("--mcts_key_default_count", type=int, default=2)
    return parser.parse_known_args(argv)


def _make_patched_parse_args(
    original_parse_args: Callable[[], argparse.Namespace],
) -> Callable[[], argparse.Namespace]:
    def _patched_parse_args() -> argparse.Namespace:
        dynamic_args, remaining = _parse_dynamic_cfg_flags(sys.argv[1:])
        original_argv = sys.argv
        sys.argv = [original_argv[0]] + remaining
        try:
            args = original_parse_args()
        finally:
            sys.argv = original_argv
        # sd35_ddp_experiment parser does not define --x0_sampler, but
        # sampling_unified_sd35.run_baseline/run_mcts expects args.x0_sampler.
        # SenseFlow paths may set it in backend defaults; sid needs a safe fallback.
        if not hasattr(args, "x0_sampler"):
            setattr(args, "x0_sampler", False)
        for key, value in vars(dynamic_args).items():
            setattr(args, key, value)
        return args

    return _patched_parse_args


def main() -> None:
    original_parse_args = base.parse_args
    original_run_mcts = base.run_mcts
    base.parse_args = _make_patched_parse_args(original_parse_args)
    base.run_mcts = run_mcts_dynamic_cfg
    try:
        base.main()
    finally:
        base.parse_args = original_parse_args
        base.run_mcts = original_run_mcts


if __name__ == "__main__":
    main()
