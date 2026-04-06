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

START_INDEX="${START_INDEX:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
OUT_DIR="${OUT_DIR:-./sd35_nfe_cost_scaling_out}"
SIM_COSTS="${SIM_COSTS:-5 10 20 35 50}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "[nfe-scaling] prompt file not found, exporting HPSv2 prompts first ..."
  OUT_DIR="${PROMPT_DIR}" STYLE="${PROMPT_STYLE}" bash "${SCRIPT_DIR}/get_hpsv2_prompts.sh"
fi
if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found after export: ${PROMPT_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

CFG_SCALES_STR="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0}"
read -r -a CFG_SCALES_ARR <<< "${CFG_SCALES_STR}"
if [[ "${#CFG_SCALES_ARR[@]}" -eq 0 ]]; then
  echo "Error: CFG_SCALES is empty." >&2
  exit 1
fi

CORRECTION_STRENGTHS_STR="${CORRECTION_STRENGTHS:-1.0}"
read -r -a CORRECTION_STRENGTHS_ARR <<< "${CORRECTION_STRENGTHS_STR}"
if [[ "${#CORRECTION_STRENGTHS_ARR[@]}" -eq 0 ]]; then
  echo "Error: CORRECTION_STRENGTHS is empty." >&2
  exit 1
fi

read -r -a SIM_COSTS_ARR <<< "${SIM_COSTS}"
if [[ "${#SIM_COSTS_ARR[@]}" -lt 5 ]]; then
  echo "Error: SIM_COSTS must contain at least 5 values." >&2
  exit 1
fi

REWARD_WEIGHTS_STR="${REWARD_WEIGHTS:-1.0 1.0}"
read -r -a REWARD_WEIGHTS_ARR <<< "${REWARD_WEIGHTS_STR}"
if [[ "${#REWARD_WEIGHTS_ARR[@]}" -ne 2 ]]; then
  echo "Error: REWARD_WEIGHTS must contain exactly 2 values." >&2
  exit 1
fi

use_qwen_arg=()
if [[ "${USE_QWEN:-0}" == "1" ]]; then
  use_qwen_arg+=(--use_qwen)
fi

rewrites_arg=()
if [[ -n "${REWRITES_FILE:-}" ]]; then
  rewrites_arg+=(--rewrites_file "${REWRITES_FILE}")
fi

reward_api_base_arg=()
if [[ -n "${REWARD_API_BASE:-}" ]]; then
  reward_api_base_arg+=(--reward_api_base "${REWARD_API_BASE}")
fi

echo "[nfe-scaling] prompt_file=${PROMPT_FILE}"
echo "[nfe-scaling] range=[${START_INDEX},$((START_INDEX + NUM_PROMPTS)))"
echo "[nfe-scaling] sim_costs=[${SIM_COSTS_ARR[*]}]"
echo "[nfe-scaling] out_dir=${OUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/sd35_nfe_cost_scaling.py" \
  --prompt_file "${PROMPT_FILE}" \
  --start_index "${START_INDEX}" \
  --num_prompts "${NUM_PROMPTS}" \
  --sim_costs "${SIM_COSTS_ARR[@]}" \
  --out_dir "${OUT_DIR}" \
  --backend "${SD35_BACKEND:-sid}" \
  --steps "${STEPS:-4}" \
  --width "${WIDTH:-1024}" \
  --height "${HEIGHT:-1024}" \
  --seed "${SEED:-42}" \
  --cfg_scales "${CFG_SCALES_ARR[@]}" \
  --baseline_cfg "${BASELINE_CFG:-1.0}" \
  --correction_strengths "${CORRECTION_STRENGTHS_ARR[@]}" \
  --n_variants "${N_VARIANTS:-3}" \
  --reward_backend "${REWARD_BACKEND:-imagereward}" \
  --reward_model "${REWARD_MODEL:-CodeGoat24/UnifiedReward-qwen-7b}" \
  --unifiedreward_model "${UNIFIEDREWARD_MODEL:-CodeGoat24/UnifiedReward-qwen-7b}" \
  --image_reward_model "${IMAGE_REWARD_MODEL:-ImageReward-v1.0}" \
  --pickscore_model "${PICKSCORE_MODEL:-yuvalkirstain/PickScore_v1}" \
  --reward_weights "${REWARD_WEIGHTS_ARR[0]}" "${REWARD_WEIGHTS_ARR[1]}" \
  --reward_api_key "${REWARD_API_KEY:-unifiedreward}" \
  --reward_api_model "${REWARD_API_MODEL:-UnifiedReward-7b-v1.5}" \
  --reward_max_new_tokens "${REWARD_MAX_NEW_TOKENS:-512}" \
  --reward_prompt_mode "${REWARD_PROMPT_MODE:-standard}" \
  "${reward_api_base_arg[@]}" \
  "${use_qwen_arg[@]}" \
  "${rewrites_arg[@]}" \
  --lookahead_mode "${LOOKAHEAD_MODE:-rollout_tree_prior}" \
  --lookahead_u_t_def "${LOOKAHEAD_U_T_DEF:-latent_delta_rms}" \
  --lookahead_tau "${LOOKAHEAD_TAU:-0.35}" \
  --lookahead_c_puct "${LOOKAHEAD_C_PUCT:-1.20}" \
  --lookahead_u_ref "${LOOKAHEAD_U_REF:-0.0}" \
  --lookahead_w_cfg "${LOOKAHEAD_W_CFG:-1.0}" \
  --lookahead_w_variant "${LOOKAHEAD_W_VARIANT:-0.25}" \
  --lookahead_w_cs "${LOOKAHEAD_W_CS:-0.10}" \
  --lookahead_w_q "${LOOKAHEAD_W_Q:-0.20}" \
  --lookahead_w_explore "${LOOKAHEAD_W_EXPLORE:-0.05}" \
  --lookahead_cfg_width_min "${LOOKAHEAD_CFG_WIDTH_MIN:-3}" \
  --lookahead_cfg_width_max "${LOOKAHEAD_CFG_WIDTH_MAX:-7}" \
  --lookahead_cfg_anchor_count "${LOOKAHEAD_CFG_ANCHOR_COUNT:-2}" \
  --lookahead_min_visits_for_center "${LOOKAHEAD_MIN_VISITS_FOR_CENTER:-3}" \
  --lookahead_log_action_topk "${LOOKAHEAD_LOG_ACTION_TOPK:-12}" \
  --smc_gamma "${SMC_GAMMA:-0.10}" \
  --ess_threshold "${ESS_THRESHOLD:-0.5}" \
  --resample_start_frac "${RESAMPLE_START_FRAC:-0.3}" \
  --smc_cfg_scale "${SMC_CFG_SCALE:-1.25}" \
  --smc_variant_idx "${SMC_VARIANT_IDX:-0}" \
  --reward_nfe_weight "${REWARD_NFE_WEIGHT:-1.0}" \
  "$@"
