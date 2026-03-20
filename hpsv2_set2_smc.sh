#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

PROMPT_FILE="${PROMPT_FILE:-/data/ygu/hpsv2_prompts.txt}"
OUT_DIR="${OUT_DIR:-/data/ygu/imagereward_smc_hpsv2_set2}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found: ${PROMPT_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/smc_baseline.py" \
  --model_id "Efficient-Large-Model/Sana_600M_512px_diffusers" \
  --device cuda \
  --reward_backend imagereward \
  --prompt_file "${PROMPT_FILE}" \
  --n_prompts -1 \
  --n_seeds 1 \
  --seed 42 \
  --no-shuffle_prompts \
  --methods baseline smc \
  --cfg_scale 1.5 \
  --denoise_steps 4 \
  --smc_k 12 \
  --smc_gamma 0.1 \
  --ess_threshold 0.5 \
  --resample_start_frac 0.3 \
  --img_size 512 \
  --save_first_k 10 \
  --output_dir "${OUT_DIR}" \
  "$@"
