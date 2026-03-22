#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

"${PYTHON_BIN}" "${SCRIPT_DIR}/sampling_unified_sd35.py" \
  --search_method ga \
  --reward_backend imagereward \
  --image_reward_model ImageReward-v1.0 \
  --prompt_file "${SCRIPT_DIR}/prompts.txt" \
  --n_variants 3 \
  --cfg_scales 1.0 1.25 1.5 1.75 2.0 2.25 2.5 \
  --steps 4 \
  --seed 42 \
  --width 512 \
  --height 512 \
  --ga_population 24 \
  --ga_generations 12 \
  --ga_elites 3 \
  --ga_mutation_prob 0.10 \
  --ga_tournament_k 3 \
  --ga_selection rank \
  --ga_rank_pressure 1.7 \
  --ga_crossover uniform \
  --ga_log_topk 3 \
  --ga_phase_constraints \
  --out_dir ./imagereward_sd35_ga_out
