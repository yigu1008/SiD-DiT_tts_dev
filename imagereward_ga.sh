#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

"${PYTHON_BIN}" "${SCRIPT_DIR}/sampling_unified.py" \
  --search_method ga \
  --reward_type imagereward \
  --dtype bf16 \
  --reward_device cpu \
  --sana_no_fp32_attn \
  --decode_device auto \
  --decode_cpu_dtype fp32 \
  --decode_cpu_if_free_below_gb 20 \
  --no-resolution_binning \
  --min_free_gb 16 \
  --no-ga_run_baselines \
  --prompt_file "${SCRIPT_DIR}/prompts.txt" \
  --steps 4 \
  --seed 42 \
  --n_samples 1 \
  --ga_population 24 \
  --ga_generations 12 \
  --ga_elites 3 \
  --ga_mutation_prob 0.10 \
  --ga_tournament_k 3 \
  --ga_crossover uniform \
  --ga_log_topk 3 \
  --ga_random_trials 32 \
  --ga_phase_constraints \
  --ga_cfg_scales 1.0 1.25 1.5 \
  --out_dir ./imagereward_ga_out \
  "$@"
