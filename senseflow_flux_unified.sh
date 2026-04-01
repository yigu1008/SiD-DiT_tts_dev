#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

FLUX_BACKEND="${FLUX_BACKEND:-senseflow_flux}"
SEARCH_METHOD="${SEARCH_METHOD:-ga}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/hpsv2_subset.txt}"
OUT_DIR="${OUT_DIR:-./senseflow_flux_out}"
CFG_SCALES="${CFG_SCALES:-0.0}"
GA_GUIDANCE_SCALES="${GA_GUIDANCE_SCALES:-0.0}"
FLUX_SIGMAS="${FLUX_SIGMAS:-1.0 0.75}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/sampling_flux_unified.py"
  --backend "${FLUX_BACKEND}"
  --search_method "${SEARCH_METHOD}"
  --reward_backend "${REWARD_BACKEND:-imagereward}"
  --image_reward_model "${IMAGE_REWARD_MODEL:-ImageReward-v1.0}"
  --prompt_file "${PROMPT_FILE}"
  --n_prompts "${N_PROMPTS:--1}"
  --steps "${STEPS:-4}"
  --seed "${SEED:-42}"
  --width "${WIDTH:-512}"
  --height "${HEIGHT:-512}"
  --cfg_scales ${CFG_SCALES}
  --ga_guidance_scales ${GA_GUIDANCE_SCALES}
  --baseline_guidance_scale "${BASELINE_GUIDANCE_SCALE:-0.0}"
  --ga_population "${GA_POPULATION:-24}"
  --ga_generations "${GA_GENERATIONS:-8}"
  --ga_elites "${GA_ELITES:-3}"
  --ga_mutation_prob "${GA_MUTATION_PROB:-0.10}"
  --ga_tournament_k "${GA_TOURNAMENT_K:-3}"
  --ga_selection "${GA_SELECTION:-rank}"
  --ga_rank_pressure "${GA_RANK_PRESSURE:-1.7}"
  --ga_crossover "${GA_CROSSOVER:-uniform}"
  --ga_log_topk "${GA_LOG_TOPK:-3}"
  --out_dir "${OUT_DIR}"
)

if [[ -n "${FLUX_SIGMAS}" ]]; then
  cmd+=(--sigmas ${FLUX_SIGMAS})
fi

"${cmd[@]}" "$@"

