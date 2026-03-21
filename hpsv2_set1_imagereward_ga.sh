#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

PROMPT_FILE="${PROMPT_FILE:-/data/ygu/hpsv2_prompts.txt}"
OUT_DIR="${OUT_DIR:-/data/ygu/imagereward_ga_hpsv2_set1}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found: ${PROMPT_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

GA_EVAL_LOG_ARGS=()
if [[ "${GA_LOG_EVALS:-0}" == "1" ]]; then
  GA_EVAL_LOG_ARGS+=(--ga_log_evals)
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/sampling_unified.py" \
  --search_method ga \
  --reward_type imagereward \
  --dtype bf16 \
  --reward_device cpu \
  --sana_no_fp32_attn \
  --offload_text_encoder_after_encode \
  --decode_device cuda \
  --empty_cache_after_decode \
  --no-resolution_binning \
  --min_free_gb 16 \
  --no-ga_run_baselines \
  --prompt_file "${PROMPT_FILE}" \
  --steps 4 \
  --seed 42 \
  --time_scale 1000.0 \
  --n_samples 1 \
  --ga_population 48 \
  --ga_generations 30 \
  --ga_elites 4 \
  --ga_mutation_prob 0.15 \
  --ga_tournament_k 4 \
  --ga_selection rank \
  --ga_rank_pressure 1.7 \
  --ga_crossover uniform \
  --ga_init_mode random \
  --ga_log_topk 5 \
  --ga_random_trials 128 \
  --ga_phase_constraints \
  --ga_cfg_scales 1.0 1.25 1.5 \
  --baseline_noise_mode fresh \
  --ga_noise_modes fresh fixed \
  "${GA_EVAL_LOG_ARGS[@]}" \
  --save_first_k 10 \
  --out_dir "${OUT_DIR}" \
  "$@"
