#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

PROMPT_FILE="${PROMPT_FILE:-/data/ygu/hpsv2_prompts.txt}"
OUT_DIR="${OUT_DIR:-/data/ygu/flux_imagereward_ga_hpsv2_set1}"
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-schnell}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found: ${PROMPT_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

GA_EVAL_LOG_ARGS=()
if [[ "${GA_LOG_EVALS:-0}" == "1" ]]; then
  GA_EVAL_LOG_ARGS+=(--ga_log_evals)
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/sampling_flux_unified.py" \
  --search_method ga \
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
  --ga_population 24 \
  --ga_generations 8 \
  --ga_elites 3 \
  --ga_mutation_prob 0.10 \
  --ga_tournament_k 3 \
  --ga_selection rank \
  --ga_rank_pressure 1.7 \
  --ga_crossover uniform \
  --ga_init_mode random \
  --ga_log_topk 3 \
  --ga_phase_constraints \
  --ga_guidance_scales 1.0 1.25 1.5 1.75 2.0 2.25 2.5 \
  "${GA_EVAL_LOG_ARGS[@]}" \
  --save_first_k 10 \
  --out_dir "${OUT_DIR}" \
  "$@"
