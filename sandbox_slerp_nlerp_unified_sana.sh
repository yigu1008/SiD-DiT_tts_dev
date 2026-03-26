#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

PROMPT_STYLE="${PROMPT_STYLE:-all}"
PROMPT_DIR="${PROMPT_DIR:-/data/ygu}"
DEFAULT_PROMPT_FILE="${SCRIPT_DIR}/hpsv2_subset.txt"
if [[ -f "${DEFAULT_PROMPT_FILE}" ]]; then
  PROMPT_FILE="${PROMPT_FILE:-${DEFAULT_PROMPT_FILE}}"
else
  PROMPT_FILE="${PROMPT_FILE:-${PROMPT_DIR}/hpsv2_prompts.txt}"
fi
OUT_DIR="${OUT_DIR:-/data/ygu/sandbox_slerp_nlerp_unified_sana}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "[sandbox-slerp-nlerp] prompt file not found, exporting HPSv2 prompts first ..."
  OUT_DIR="${PROMPT_DIR}" STYLE="${PROMPT_STYLE}" bash "${SCRIPT_DIR}/get_hpsv2_prompts.sh"
fi

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found after export: ${PROMPT_FILE}" >&2
  exit 1
fi

ensure_hpsv3_runtime() {
  local backend_lc
  backend_lc="$(echo "${REWARD_TYPE:-unifiedreward}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${backend_lc}" != "hpsv3" && "${backend_lc}" != "auto" ]]; then
    return 0
  fi
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import hpsv3
import omegaconf
import hydra
print(getattr(hpsv3, "__file__", "ok"), getattr(omegaconf, "__version__", "ok"), getattr(hydra, "__version__", "ok"))
PY
  then
    return 0
  fi
  echo "[deps] HPSv3 runtime deps missing. Installing with install_reward_deps.sh ..."
  PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/install_reward_deps.sh"
}

ensure_hpsv3_runtime

reward_args=(
  --reward_type "${REWARD_TYPE:-unifiedreward}"
  --reward_device "${REWARD_DEVICE:-cpu}"
  --reward_model "${REWARD_MODEL:-CodeGoat24/UnifiedReward-qwen-7b}"
  --image_reward_model "${IMAGE_REWARD_MODEL:-ImageReward-v1.0}"
  --pickscore_model "${PICKSCORE_MODEL:-yuvalkirstain/PickScore_v1}"
  --reward_max_new_tokens "${REWARD_MAX_NEW_TOKENS:-512}"
  --reward_prompt_mode "${REWARD_PROMPT_MODE:-standard}"
)

if [[ -n "${UNIFIEDREWARD_MODEL:-}" ]]; then
  reward_args+=(--unifiedreward_model "${UNIFIEDREWARD_MODEL}")
fi
if [[ -n "${REWARD_API_BASE:-}" ]]; then
  reward_args+=(--reward_api_base "${REWARD_API_BASE}")
fi
if [[ -n "${REWARD_API_KEY:-}" ]]; then
  reward_args+=(--reward_api_key "${REWARD_API_KEY}")
fi
if [[ -n "${REWARD_API_MODEL:-}" ]]; then
  reward_args+=(--reward_api_model "${REWARD_API_MODEL}")
fi
reward_weights_str="${REWARD_WEIGHTS:-1.0 1.0}"
if [[ -n "${reward_weights_str}" ]]; then
  # shellcheck disable=SC2206
  reward_weights_arr=(${reward_weights_str})
  if [[ "${#reward_weights_arr[@]}" -eq 2 ]]; then
    reward_args+=(--reward_weights "${reward_weights_arr[0]}" "${reward_weights_arr[1]}")
  fi
fi

algo_args=()
if [[ "${RUN_SWEEP:-1}" == "0" ]]; then
  algo_args+=(--no-run_sweep)
fi
if [[ "${RUN_GA:-1}" == "0" ]]; then
  algo_args+=(--no-run_ga)
fi
if [[ "${RUN_MCTS:-1}" == "0" ]]; then
  algo_args+=(--no-run_mcts)
fi

mix_args=()
if [[ -n "${MIX_WEIGHT_VECTORS:-}" ]]; then
  # shellcheck disable=SC2206
  mix_vec_arr=(${MIX_WEIGHT_VECTORS})
  if [[ "${#mix_vec_arr[@]}" -gt 0 ]]; then
    mix_args+=(--mix_weight_vectors "${mix_vec_arr[@]}")
  fi
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/sandbox_slerp_nlerp_unified_sana.py" \
  --prompt_file "${PROMPT_FILE}" \
  --max_prompts "${MAX_PROMPTS:-0}" \
  --out_dir "${OUT_DIR}" \
  --model_id "${MODEL_ID:-YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow}" \
  --dtype "${DTYPE:-bf16}" \
  --steps "${STEPS:-4}" \
  --width "${WIDTH:-512}" \
  --height "${HEIGHT:-512}" \
  --seed "${SEED:-42}" \
  --guidance_scale "${GUIDANCE_SCALE:-1.0}" \
  --baseline_cfg "${BASELINE_CFG:-1.0}" \
  --cfg_scales ${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5} \
  "${reward_args[@]}" \
  --interp_labels ${INTERP_LABELS:-balanced subject composition texture} \
  --interp_k "${INTERP_K:-4}" \
  --interp_values ${INTERP_VALUES:-0.0 0.25 0.5 0.75 1.0} \
  "${mix_args[@]}" \
  --families ${FAMILIES:-nlerp slerp} \
  --preview_every "${PREVIEW_EVERY:--1}" \
  --ga_population "${GA_POPULATION:-24}" \
  --ga_generations "${GA_GENERATIONS:-8}" \
  --ga_elites "${GA_ELITES:-3}" \
  --ga_mutation_prob "${GA_MUTATION_PROB:-0.10}" \
  --ga_tournament_k "${GA_TOURNAMENT_K:-3}" \
  --ga_selection "${GA_SELECTION:-rank}" \
  --ga_rank_pressure "${GA_RANK_PRESSURE:-1.7}" \
  --ga_crossover "${GA_CROSSOVER:-uniform}" \
  --ga_log_topk "${GA_LOG_TOPK:-3}" \
  --mcts_n_sims "${MCTS_N_SIMS:-50}" \
  --mcts_ucb_c "${MCTS_UCB_C:-1.41}" \
  --mcts_log_every "${MCTS_LOG_EVERY:-10}" \
  "${algo_args[@]}" \
  --save_first_k "${SAVE_FIRST_K:-10}" \
  --save_images \
  "$@"
