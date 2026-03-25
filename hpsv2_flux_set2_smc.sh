#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

DEFAULT_PROMPT_FILE="${SCRIPT_DIR}/hpsv2_subset.txt"
if [[ -f "${DEFAULT_PROMPT_FILE}" ]]; then
  PROMPT_FILE="${PROMPT_FILE:-${DEFAULT_PROMPT_FILE}}"
else
  PROMPT_FILE="${PROMPT_FILE:-/data/ygu/hpsv2_prompts.txt}"
fi
OUT_DIR="${OUT_DIR:-/data/ygu/flux_imagereward_smc_hpsv2_set2}"
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-schnell}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found: ${PROMPT_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/sampling_flux_unified.py" \
  --search_method smc \
  --model_id "${MODEL_ID}" \
  --reward_backend imagereward \
  --reward_device cpu \
  --dtype bf16 \
  --device cuda \
  --auto_select_gpu \
  --offload_text_encoder_after_encode \
  --decode_device auto \
  --decode_cpu_if_free_below_gb 16 \
  --empty_cache_after_decode \
  --prompt_file "${PROMPT_FILE}" \
  --n_prompts -1 \
  --n_samples 1 \
  --steps 4 \
  --width 512 \
  --height 512 \
  --seed 42 \
  --baseline_guidance_scale 1.0 \
  --smc_k 12 \
  --smc_gamma 0.10 \
  --ess_threshold 0.5 \
  --resample_start_frac 0.3 \
  --smc_guidance_scale 1.25 \
  --smc_chunk 4 \
  --save_first_k 10 \
  --out_dir "${OUT_DIR}" \
  "$@"
