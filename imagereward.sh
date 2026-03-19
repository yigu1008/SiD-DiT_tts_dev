#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

"${PYTHON_BIN}" "${SCRIPT_DIR}/sampling_unified.py" \
  --search_method greedy \
  --reward_type imagereward \
  --reward_device same \
  --sana_no_fp32_attn \
  --decode_device auto \
  --decode_cpu_if_free_below_gb 20 \
  --no-resolution_binning \
  --min_free_gb 16 \
  --prompt_file "${SCRIPT_DIR}/prompts.txt" \
  --n_variants 3 \
  --cfg_scales 1.0 1.25 1.5 1.75 2.0 2.25 2.5 \
  --steps 4 \
  --seed 42 \
  --out_dir ./imagereward_greedy_out \
  "$@"
