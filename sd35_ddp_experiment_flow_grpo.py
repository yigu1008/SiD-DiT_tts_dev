from __future__ import annotations

import os
import sys

import sd35_ddp_experiment_bon_mcts as bon_mcts


_SD35_BASE_CFG_SCALES = ["3.5", "4.0", "4.5", "5.0", "5.5", "6.0", "7.0"]


def _has_flag(argv: list[str], name: str) -> bool:
    flag = f"--{name}"
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def _inject_flow_grpo_defaults(argv: list[str]) -> list[str]:
    out = list(argv)
    if not _has_flag(out[1:], "backend"):
        out.extend(["--backend", "sd35_base"])
    if not _has_flag(out[1:], "steps"):
        out.extend(["--steps", "28"])
    if not _has_flag(out[1:], "cfg_scales"):
        out.extend(["--cfg_scales", *_SD35_BASE_CFG_SCALES])
    if not _has_flag(out[1:], "baseline_cfg"):
        out.extend(["--baseline_cfg", "4.5"])
    if not _has_flag(out[1:], "smc_cfg_scale"):
        out.extend(["--smc_cfg_scale", "4.5"])
    if not _has_flag(out[1:], "correction_strengths"):
        out.extend(["--correction_strengths", "0.0"])

    flow_grpo_ckpt = os.environ.get("FLOW_GRPO_CKPT", "").strip()
    if flow_grpo_ckpt and not _has_flag(out[1:], "ckpt"):
        out.extend(["--ckpt", flow_grpo_ckpt])
    flow_grpo_lora = os.environ.get("FLOW_GRPO_LORA_PATH", "").strip() or os.environ.get("SD35_LORA_PATH", "").strip()
    if flow_grpo_lora and not _has_flag(out[1:], "lora_path"):
        out.extend(["--lora_path", flow_grpo_lora])
    flow_grpo_lora_scale = os.environ.get("FLOW_GRPO_LORA_SCALE", "").strip() or os.environ.get("SD35_LORA_SCALE", "").strip()
    if flow_grpo_lora_scale and not _has_flag(out[1:], "lora_scale"):
        out.extend(["--lora_scale", flow_grpo_lora_scale])
    return out


def main() -> None:
    original_argv = sys.argv
    sys.argv = _inject_flow_grpo_defaults(original_argv)
    try:
        bon_mcts.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
