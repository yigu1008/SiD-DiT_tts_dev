#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

SD35_BACKEND="${SD35_BACKEND:-senseflow_large}"
SEARCH_METHOD="${SEARCH_METHOD:-ga}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/hpsv2_subset.txt}"
OUT_DIR="${OUT_DIR:-./senseflow_sd35_out}"
CFG_SCALES="${CFG_SCALES:-0.0}"
SD35_SIGMAS="${SD35_SIGMAS:-}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/sampling_unified_sd35.py"
  --backend "${SD35_BACKEND}"
  --search_method "${SEARCH_METHOD}"
  --reward_backend "${REWARD_BACKEND:-imagereward}"
  --image_reward_model "${IMAGE_REWARD_MODEL:-ImageReward-v1.0}"
  --prompt_file "${PROMPT_FILE}"
  --steps "${STEPS:-4}"
  --seed "${SEED:-42}"
  --width "${WIDTH:-1024}"
  --height "${HEIGHT:-1024}"
  --cfg_scales ${CFG_SCALES}
  --baseline_cfg "${BASELINE_CFG:-0.0}"
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

if [[ -n "${SD35_SIGMAS}" ]]; then
  cmd+=(--sigmas ${SD35_SIGMAS})
fi

"${cmd[@]}" "$@"

