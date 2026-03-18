#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

"${PYTHON_BIN}" "${SCRIPT_DIR}/sampling_unified_sd35.py" \
  --search_method greedy \
  --prompt_file "${SCRIPT_DIR}/prompts.txt" \
  --n_variants 3 \
  --cfg_scales 1.0 1.25 1.5 1.75 2.0 2.25 2.5 \
  --steps 4 \
  --seed 42 \
  --width 1024 \
  --height 1024 \
  --out_dir ./imagereward_sd35_greedy_out \
  "$@"
