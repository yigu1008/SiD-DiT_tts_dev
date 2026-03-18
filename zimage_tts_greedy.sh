#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

"${PYTHON_BIN}" "${SCRIPT_DIR}/zimage_tts.py" \
  --search_method greedy \
  --model Tongyi-MAI/Z-Image-Turbo \
  --prompt_file "${SCRIPT_DIR}/prompts.txt" \
  --steps 9 \
  --cfg_scales 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \
  --n_variants 3 \
  --seed 42 \
  --outdir ./zimage_tts_greedy_out \
  "$@"
