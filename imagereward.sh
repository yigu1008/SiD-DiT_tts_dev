#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

"${PYTHON_BIN}" "${SCRIPT_DIR}/sampling_unified.py" \
  --search_method greedy \
  --reward_type imagereward \
  --prompt_file "${SCRIPT_DIR}/prompts.txt" \
  --n_variants 3 \
  --cfg_scales 1.0 1.25 1.5 1.75 2.0 2.25 2.5 \
  --steps 4 \
  --seed 42 \
  --out_dir ./imagereward_greedy_out \
  "$@"
