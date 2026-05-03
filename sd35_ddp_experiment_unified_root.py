"""DDP runner for SD3.5 unified-root MCTS sampling.

Patches sd35_ddp_experiment so ``--modes mcts`` routes through
:func:`mcts_unified_root.run_mcts_unified_root` (single tree with seed pool
at the root layer) instead of the original ``run_mcts``.

The original ``run_mcts`` is left intact in ``sampling_unified_sd35`` for
direct A/B comparison against bon_mcts.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import mcts_unified_root
import sampling_unified_sd35 as su
import sd35_ddp_experiment as base


def _parse_unified_root_flags(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    mcts_unified_root.add_unified_root_args(parser)
    return parser.parse_known_args(argv)


def _make_patched_parse_args(
    original_parse_args: Callable[[], argparse.Namespace],
) -> Callable[[], argparse.Namespace]:
    def _patched_parse_args() -> argparse.Namespace:
        unified_args, remaining = _parse_unified_root_flags(sys.argv[1:])
        original_argv = sys.argv
        sys.argv = [original_argv[0]] + remaining
        try:
            args = original_parse_args()
        finally:
            sys.argv = original_argv
        for k, v in vars(unified_args).items():
            setattr(args, k, v)
        return args
    return _patched_parse_args


def main() -> None:
    original_parse_args = base.parse_args
    # Rebind in BOTH modules — sd35_ddp_experiment imports run_mcts at module
    # load time, so it holds its own reference. Same fix as the dynamic_cfg_x0
    # wrapper.
    original_su_run_mcts = su.run_mcts
    original_base_run_mcts = getattr(base, "run_mcts", None)

    base.parse_args = _make_patched_parse_args(original_parse_args)
    su.run_mcts = mcts_unified_root.run_mcts_unified_root
    if original_base_run_mcts is not None:
        base.run_mcts = mcts_unified_root.run_mcts_unified_root

    try:
        base.main()
    finally:
        base.parse_args = original_parse_args
        su.run_mcts = original_su_run_mcts
        if original_base_run_mcts is not None:
            base.run_mcts = original_base_run_mcts


if __name__ == "__main__":
    main()
