"""DDP runner for SD3.5 MCTS-improved sampling.

Patches sd35_ddp_experiment so ``--modes mcts`` routes through
:func:`mcts_improved.run_mcts_improved` (UCB1-Tuned + reward normalization +
x0-pred bootstrap + step-dependent rollout policy) instead of the original
``run_mcts``.

The original ``run_mcts`` is left intact in ``sampling_unified_sd35`` for
direct A/B comparison.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import mcts_improved
import sampling_unified_sd35 as su
import sd35_ddp_experiment as base


def _parse_mcts_improved_flags(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    mcts_improved.add_mcts_improved_args(parser)
    return parser.parse_known_args(argv)


def _make_patched_parse_args(
    original_parse_args: Callable[[], argparse.Namespace],
) -> Callable[[], argparse.Namespace]:
    def _patched_parse_args() -> argparse.Namespace:
        mcts_args, remaining = _parse_mcts_improved_flags(sys.argv[1:])
        original_argv = sys.argv
        sys.argv = [original_argv[0]] + remaining
        try:
            args = original_parse_args()
        finally:
            sys.argv = original_argv
        for k, v in vars(mcts_args).items():
            setattr(args, k, v)
        return args
    return _patched_parse_args


def main() -> None:
    original_parse_args = base.parse_args
    # Rebind in BOTH modules: `sampling_unified_sd35` is where the function
    # lives canonically, but `sd35_ddp_experiment` does
    # `from sampling_unified_sd35 import run_mcts` at import time and holds
    # its own reference. Both bindings need to be swapped, otherwise the
    # call site in sd35_ddp_experiment uses the original.
    original_su_run_mcts = su.run_mcts
    original_base_run_mcts = getattr(base, "run_mcts", None)

    base.parse_args = _make_patched_parse_args(original_parse_args)
    su.run_mcts = mcts_improved.run_mcts_improved
    if original_base_run_mcts is not None:
        base.run_mcts = mcts_improved.run_mcts_improved

    try:
        base.main()
    finally:
        base.parse_args = original_parse_args
        su.run_mcts = original_su_run_mcts
        if original_base_run_mcts is not None:
            base.run_mcts = original_base_run_mcts


if __name__ == "__main__":
    main()
