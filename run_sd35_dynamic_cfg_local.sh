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
OUT_DIR="${OUT_DIR:-./sd35_dynamic_cfg_local_out}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "[sd35-dynamic-cfg] prompt file not found, exporting HPSv2 prompts first ..."
  OUT_DIR="${PROMPT_DIR}" STYLE="${PROMPT_STYLE}" bash "${SCRIPT_DIR}/get_hpsv2_prompts.sh"
fi

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found after export: ${PROMPT_FILE}" >&2
  exit 1
fi

CFG_SCALES_STR="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0}"
read -r -a CFG_SCALES_ARR <<< "${CFG_SCALES_STR}"
if [[ "${#CFG_SCALES_ARR[@]}" -eq 0 ]]; then
  echo "Error: CFG_SCALES is empty." >&2
  exit 1
fi

CORRECTION_STRENGTHS_STR="${CORRECTION_STRENGTHS:-0.0}"
read -r -a CORRECTION_STRENGTHS_ARR <<< "${CORRECTION_STRENGTHS_STR}"
if [[ "${#CORRECTION_STRENGTHS_ARR[@]}" -eq 0 ]]; then
  CORRECTION_STRENGTHS_ARR=(0.0)
fi

MCTS_CFG_ROOT_BANK_STR="${MCTS_CFG_ROOT_BANK:-1.0 1.5 2.0}"
read -r -a MCTS_CFG_ROOT_BANK_ARR <<< "${MCTS_CFG_ROOT_BANK_STR}"
MCTS_CFG_ANCHORS_STR="${MCTS_CFG_ANCHORS:-1.0 2.0}"
read -r -a MCTS_CFG_ANCHORS_ARR <<< "${MCTS_CFG_ANCHORS_STR}"

CFG_ONLY="${CFG_ONLY:-1}"
N_VARIANTS="${N_VARIANTS:-0}"
USE_QWEN="${USE_QWEN:-0}"
if [[ "${CFG_ONLY}" == "1" ]]; then
  N_VARIANTS=0
  USE_QWEN=0
  CORRECTION_STRENGTHS_ARR=(0.0)
fi

qwen_args=()
if [[ "${USE_QWEN}" == "1" ]]; then
  qwen_args+=(--qwen_id "${QWEN_ID:-Qwen/Qwen3-4B}")
  qwen_args+=(--qwen_dtype "${QWEN_DTYPE:-bfloat16}")
  qwen_args+=(--qwen_timeout_sec "${QWEN_TIMEOUT_SEC:-240}")
else
  qwen_args+=(--no_qwen)
fi

extra_reward_args=()
if [[ -n "${REWARD_API_BASE:-}" ]]; then
  extra_reward_args+=(--reward_api_base "${REWARD_API_BASE}")
fi

extra_args=()
if [[ -n "${SIGMAS:-}" ]]; then
  # shellcheck disable=SC2206
  sigmas_arr=(${SIGMAS})
  if [[ "${#sigmas_arr[@]}" -gt 0 ]]; then
    extra_args+=(--sigmas "${sigmas_arr[@]}")
  fi
fi

reward_weights_str="${REWARD_WEIGHTS:-1.0 1.0}"
# shellcheck disable=SC2206
reward_weights_arr=(${reward_weights_str})
if [[ "${#reward_weights_arr[@]}" -ne 2 ]]; then
  echo "Error: REWARD_WEIGHTS must contain exactly 2 values." >&2
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/sampling_unified_sd35_dynamic_cfg.py" \
  --search_method mcts \
  --backend "${SD35_BACKEND:-sid}" \
  --prompt_file "${PROMPT_FILE}" \
  --steps "${STEPS:-4}" \
  --n_variants "${N_VARIANTS}" \
  --cfg_scales "${CFG_SCALES_ARR[@]}" \
  --baseline_cfg "${BASELINE_CFG:-1.0}" \
  --correction_strengths "${CORRECTION_STRENGTHS_ARR[@]}" \
  --n_sims "${N_SIMS:-50}" \
  --ucb_c "${UCB_C:-1.41}" \
  --seed "${SEED:-42}" \
  --width "${WIDTH:-1024}" \
  --height "${HEIGHT:-1024}" \
  --reward_backend "${REWARD_BACKEND:-imagereward}" \
  --reward_model "${REWARD_MODEL:-CodeGoat24/UnifiedReward-qwen-7b}" \
  --unifiedreward_model "${UNIFIEDREWARD_MODEL:-CodeGoat24/UnifiedReward-qwen-7b}" \
  --image_reward_model "${IMAGE_REWARD_MODEL:-ImageReward-v1.0}" \
  --pickscore_model "${PICKSCORE_MODEL:-yuvalkirstain/PickScore_v1}" \
  --reward_weights "${reward_weights_arr[0]}" "${reward_weights_arr[1]}" \
  --reward_api_key "${REWARD_API_KEY:-unifiedreward}" \
  --reward_api_model "${REWARD_API_MODEL:-UnifiedReward-7b-v1.5}" \
  --reward_max_new_tokens "${REWARD_MAX_NEW_TOKENS:-512}" \
  --reward_prompt_mode "${REWARD_PROMPT_MODE:-standard}" \
  "${extra_reward_args[@]}" \
  "${qwen_args[@]}" \
  --mcts_cfg_mode "${MCTS_CFG_MODE:-adaptive}" \
  --mcts_cfg_root_bank "${MCTS_CFG_ROOT_BANK_ARR[@]}" \
  --mcts_cfg_anchors "${MCTS_CFG_ANCHORS_ARR[@]}" \
  --mcts_cfg_step_anchor_count "${MCTS_CFG_STEP_ANCHOR_COUNT:-2}" \
  --mcts_cfg_min_parent_visits "${MCTS_CFG_MIN_PARENT_VISITS:-3}" \
  --mcts_cfg_round_ndigits "${MCTS_CFG_ROUND_NDIGITS:-6}" \
  --mcts_cfg_log_action_topk "${MCTS_CFG_LOG_ACTION_TOPK:-12}" \
  "${extra_args[@]}" \
  --out_dir "${OUT_DIR}" \
  "$@"
