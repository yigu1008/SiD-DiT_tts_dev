#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

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
  --prompt_file "${SCRIPT_DIR}/prompts.txt" \
  --steps 4 \
  --seed 42 \
  --time_scale 1000.0 \
  --n_samples 1 \
  --ga_population 48 \
  --ga_generations 30 \
  --ga_elites 4 \
  --ga_mutation_prob 0.15 \
  --ga_tournament_k 4 \
  --ga_crossover uniform \
  --ga_init_mode random \
  --ga_log_topk 5 \
  --ga_random_trials 128 \
  --ga_phase_constraints \
  --ga_cfg_scales 1.0 1.25 1.5 \
  "${GA_EVAL_LOG_ARGS[@]}" \
  --out_dir ./imagereward_ga_out \
  "$@"
