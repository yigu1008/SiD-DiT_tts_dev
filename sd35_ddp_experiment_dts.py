"""DDP runner for Diffusion Tree Sampling (DTS) and Diffusion Tree Search
(DTS*).

Reproduces Jain et al., arXiv:2506.20701. Reuses the prompt loop and DDP
plumbing from `sd35_ddp_experiment` by monkey-patching `run_mcts` to call
either `run_dts` or `run_dts_star` from `sampling_unified_sd35_dts`. Invoke
with `--modes mcts --dts_method dts` (or `dts_star`).

Image suffix and log mode_key remain `mcts` so the existing eval pipeline
(`evaluate_best_images_multi_reward.py --method mcts`) finds the outputs.
The per-method directory `${RUN_DIR}/${method}` keeps DTS and DTS* separate.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import sd35_ddp_experiment as base
from sampling_unified_sd35_dts import run_dts, run_dts_star


def _parse_dts_flags(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dts_method",
        choices=["dts", "dts_star"],
        default="dts",
        help="dts: Boltzmann selection (sampling). dts_star: argmax + UCB (search).",
    )
    parser.add_argument(
        "--dts_m_iter",
        type=int,
        default=64,
        help="Number of tree iterations (selection→expansion→rollout→backup) per prompt.",
    )
    parser.add_argument(
        "--dts_lambda",
        type=float,
        default=1.0,
        help="Inverse temperature for soft-Bellman backup and Boltzmann selection.",
    )
    parser.add_argument(
        "--dts_pw_c",
        type=float,
        default=1.0,
        help="Progressive widening constant C in B(N) = ceil(C * N^alpha).",
    )
    parser.add_argument(
        "--dts_pw_alpha",
        type=float,
        default=0.5,
        help="Progressive widening exponent alpha in B(N) = ceil(C * N^alpha).",
    )
    parser.add_argument(
        "--dts_c_uct",
        type=float,
        default=1.0,
        help="UCB exploration constant for DTS* selection.",
    )
    parser.add_argument(
        "--dts_sde_noise_scale",
        type=float,
        default=0.0,
        help=(
            "Std-dev of Gaussian noise injected after each Euler step. "
            "Required >0 to enable per-step branching for sd35_base (Euler ODE). "
            "Ignored for SiD/SenseFlow (re-noising provides natural stochasticity)."
        ),
    )
    parser.add_argument(
        "--dts_cfg_bank",
        type=str,
        default="",
        help=(
            "Optional whitespace/comma-separated CFG bank to sample from per "
            "expansion (e.g. '1.0 2.0 3.0'). Empty → use baseline_cfg only. "
            "Each rollout draws one cfg uniformly and applies it to every "
            "denoiser step in that rollout."
        ),
    )
    return parser.parse_known_args(argv)


def _make_patched_parse_args(
    original_parse_args: Callable[[], argparse.Namespace],
) -> Callable[[], argparse.Namespace]:
    def _patched_parse_args() -> argparse.Namespace:
        dts_args, remaining = _parse_dts_flags(sys.argv[1:])
        original_argv = sys.argv
        sys.argv = [original_argv[0]] + remaining
        try:
            args = original_parse_args()
        finally:
            sys.argv = original_argv
        if not hasattr(args, "x0_sampler"):
            setattr(args, "x0_sampler", False)
        for key, value in vars(dts_args).items():
            setattr(args, key, value)
        return args

    return _patched_parse_args


def main() -> None:
    original_parse_args = base.parse_args
    original_run_mcts = base.run_mcts

    def _patched_run_mcts(args, ctx, emb, reward_model, prompt, variants, seed):
        method = str(getattr(args, "dts_method", "dts")).strip().lower()
        if method == "dts_star":
            return run_dts_star(args, ctx, emb, reward_model, prompt, seed)
        return run_dts(args, ctx, emb, reward_model, prompt, seed)

    base.parse_args = _make_patched_parse_args(original_parse_args)
    base.run_mcts = _patched_run_mcts
    try:
        base.main()
    finally:
        base.parse_args = original_parse_args
        base.run_mcts = original_run_mcts


if __name__ == "__main__":
    main()
