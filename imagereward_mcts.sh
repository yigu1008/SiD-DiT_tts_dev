#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

"${PYTHON_BIN}" "${SCRIPT_DIR}/sampling_unified.py" \
  --search_method mcts \
  --reward_type imagereward \
  --prompt_file "${SCRIPT_DIR}/prompts.txt" \
  --n_variants 3 \
  --cfg_scales 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0 4.25 4.5 4.75 5.0 \
  --steps 4 \
  --n_sims 50 \
  --ucb_c 1.41 \
  --seed 42 \
  --out_dir ./imagereward_mcts_out \
  "$@"
