"""DDP runner for SD3.5 dynamic-CFG-x0 sampling.

Patches sd35_ddp_experiment so methods="baseline" routes through
sampling_unified_sd35_dynamic_cfg_x0._run_baseline_dynamic_cfg_x0
when --dynamic_cfg_x0 is set.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import dynamic_cfg_x0 as dcx
import sampling_unified_sd35 as su
import sd35_ddp_experiment as base
from sampling_unified_sd35_dynamic_cfg_x0 import _run_baseline_dynamic_cfg_x0


def _parse_dyncfg_x0_flags(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    dcx.add_dynamic_cfg_x0_args(parser)
    return parser.parse_known_args(argv)


def _make_patched_parse_args(
    original_parse_args: Callable[[], argparse.Namespace],
) -> Callable[[], argparse.Namespace]:
    def _patched_parse_args() -> argparse.Namespace:
        dyn_args, remaining = _parse_dyncfg_x0_flags(sys.argv[1:])
        original_argv = sys.argv
        sys.argv = [original_argv[0]] + remaining
        try:
            args = original_parse_args()
        finally:
            sys.argv = original_argv
        if not hasattr(args, "x0_sampler"):
            setattr(args, "x0_sampler", False)
        for k, v in vars(dyn_args).items():
            setattr(args, k, v)
        return args
    return _patched_parse_args


def main() -> None:
    original_parse_args = base.parse_args
    original_run_baseline = su.run_baseline
    base.parse_args = _make_patched_parse_args(original_parse_args)
    su.run_baseline = _run_baseline_dynamic_cfg_x0
    try:
        base.main()
    finally:
        base.parse_args = original_parse_args
        su.run_baseline = original_run_baseline


if __name__ == "__main__":
    main()
