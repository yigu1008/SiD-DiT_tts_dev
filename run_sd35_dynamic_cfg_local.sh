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
START_INDEX="${START_INDEX:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-0}"
END_INDEX="${END_INDEX:-}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "[sd35-dynamic-cfg] prompt file not found, exporting HPSv2 prompts first ..."
  OUT_DIR="${PROMPT_DIR}" STYLE="${PROMPT_STYLE}" bash "${SCRIPT_DIR}/get_hpsv2_prompts.sh"
fi

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found after export: ${PROMPT_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

# Optional prompt range slicing for local runs.
PROMPT_FILE_RUN="${PROMPT_FILE}"
if [[ -n "${END_INDEX}" || "${START_INDEX}" != "0" || "${NUM_PROMPTS}" != "0" ]]; then
  # Prefer NUM_PROMPTS over END_INDEX so one-shot runs are predictable even if
  # END_INDEX is exported in the shell from earlier experiments.
  if (( NUM_PROMPTS > 0 )); then
    if [[ -n "${END_INDEX}" ]]; then
      echo "[sd35-dynamic-cfg] both NUM_PROMPTS and END_INDEX are set; using NUM_PROMPTS (END_INDEX ignored)."
    fi
    SLICE_END="$(( START_INDEX + NUM_PROMPTS ))"
  elif [[ -n "${END_INDEX}" ]]; then
    SLICE_END="${END_INDEX}"
  else
    SLICE_END="-1"
  fi
  SLICE_FILE="${OUT_DIR}/prompts_slice_${START_INDEX}_${SLICE_END}.txt"
  "${PYTHON_BIN}" - <<'PY' "${PROMPT_FILE}" "${SLICE_FILE}" "${START_INDEX}" "${SLICE_END}"
import sys
src, dst, start, end = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(src, encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]
total = len(prompts)
if end < 0:
    end = total
start = max(0, min(start, total))
end = max(start, min(end, total))
with open(dst, "w", encoding="utf-8") as f:
    for p in prompts[start:end]:
        f.write(p + "\n")
print(f"[sd35-dynamic-cfg] prompt slice: src={src} range=[{start},{end}) n={end-start} -> {dst}")
PY
  PROMPT_FILE_RUN="${SLICE_FILE}"
fi

SD35_BACKEND_VAL="${SD35_BACKEND:-sd35_base}"
if [[ "${SD35_BACKEND_VAL}" == "sid" ]]; then
  CFG_SCALES_STR="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5}"
  BASELINE_CFG_VAL="${BASELINE_CFG:-1.0}"
  MCTS_CFG_ROOT_BANK_STR="${MCTS_CFG_ROOT_BANK:-1.0 1.5 2.0 2.5}"
  MCTS_CFG_ANCHORS_STR="${MCTS_CFG_ANCHORS:-1.0 2.0}"
else
  CFG_SCALES_STR="${CFG_SCALES:-3.5 4.0 4.5 5.0 5.5 6.0 7.0}"
  BASELINE_CFG_VAL="${BASELINE_CFG:-4.5}"
  MCTS_CFG_ROOT_BANK_STR="${MCTS_CFG_ROOT_BANK:-4.0 4.5 5.0 5.5}"
  MCTS_CFG_ANCHORS_STR="${MCTS_CFG_ANCHORS:-3.5 7.0}"
fi
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

read -r -a MCTS_CFG_ROOT_BANK_ARR <<< "${MCTS_CFG_ROOT_BANK_STR}"
read -r -a MCTS_CFG_ANCHORS_ARR <<< "${MCTS_CFG_ANCHORS_STR}"
MCTS_KEY_MODE="${MCTS_KEY_MODE:-count}"
MCTS_KEY_STEPS="${MCTS_KEY_STEPS:-}"
MCTS_KEY_STEP_COUNT="${MCTS_KEY_STEP_COUNT:-2}"
MCTS_KEY_STEP_STRIDE="${MCTS_KEY_STEP_STRIDE:-0}"
MCTS_KEY_DEFAULT_COUNT="${MCTS_KEY_DEFAULT_COUNT:-2}"
MCTS_FRESH_NOISE_STEPS="${MCTS_FRESH_NOISE_STEPS:-}"
MCTS_FRESH_NOISE_SAMPLES="${MCTS_FRESH_NOISE_SAMPLES:-1}"
MCTS_FRESH_NOISE_SCALE="${MCTS_FRESH_NOISE_SCALE:-1.0}"
MCTS_FRESH_NOISE_KEY_STEPS="${MCTS_FRESH_NOISE_KEY_STEPS:-0}"

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

key_args=(
  --mcts_key_mode "${MCTS_KEY_MODE}"
  --mcts_key_step_count "${MCTS_KEY_STEP_COUNT}"
  --mcts_key_step_stride "${MCTS_KEY_STEP_STRIDE}"
  --mcts_key_default_count "${MCTS_KEY_DEFAULT_COUNT}"
)
if [[ -n "${MCTS_KEY_STEPS}" ]]; then
  key_args+=(--mcts_key_steps "${MCTS_KEY_STEPS}")
fi

fresh_noise_args=(
  --mcts_fresh_noise_steps "${MCTS_FRESH_NOISE_STEPS}"
  --mcts_fresh_noise_samples "${MCTS_FRESH_NOISE_SAMPLES}"
  --mcts_fresh_noise_scale "${MCTS_FRESH_NOISE_SCALE}"
)
if [[ "${MCTS_FRESH_NOISE_KEY_STEPS}" == "1" ]]; then
  fresh_noise_args+=(--mcts_fresh_noise_key_steps)
fi

reward_weights_str="${REWARD_WEIGHTS:-1.0 1.0}"
# shellcheck disable=SC2206
reward_weights_arr=(${reward_weights_str})
if [[ "${#reward_weights_arr[@]}" -ne 2 ]]; then
  echo "Error: REWARD_WEIGHTS must contain exactly 2 values." >&2
  exit 1
fi

echo "[sd35-dynamic-cfg] prompt_file=${PROMPT_FILE_RUN}"
echo "[sd35-dynamic-cfg] out_dir=${OUT_DIR}"
echo "[sd35-dynamic-cfg] cfg_mode=${MCTS_CFG_MODE:-adaptive}"
echo "[sd35-dynamic-cfg] backend=${SD35_BACKEND_VAL}"
echo "[sd35-dynamic-cfg] cfg_scales=[${CFG_SCALES_ARR[*]}] baseline_cfg=${BASELINE_CFG_VAL}"
echo "[sd35-dynamic-cfg] root_bank=[${MCTS_CFG_ROOT_BANK_ARR[*]}] anchors=[${MCTS_CFG_ANCHORS_ARR[*]}] step_anchor_count=${MCTS_CFG_STEP_ANCHOR_COUNT:-2}"
echo "[sd35-dynamic-cfg] key_mode=${MCTS_KEY_MODE} key_steps='${MCTS_KEY_STEPS}' key_step_count=${MCTS_KEY_STEP_COUNT} key_step_stride=${MCTS_KEY_STEP_STRIDE}"
echo "[sd35-dynamic-cfg] fresh_noise_steps='${MCTS_FRESH_NOISE_STEPS}' samples=${MCTS_FRESH_NOISE_SAMPLES} scale=${MCTS_FRESH_NOISE_SCALE} key_steps=${MCTS_FRESH_NOISE_KEY_STEPS}"
echo "[sd35-dynamic-cfg] cfg_only=${CFG_ONLY} n_variants=${N_VARIANTS} use_qwen=${USE_QWEN}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/sampling_unified_sd35_dynamic_cfg.py" \
  --search_method mcts \
  --backend "${SD35_BACKEND_VAL}" \
  --prompt_file "${PROMPT_FILE_RUN}" \
  --steps "${STEPS:-28}" \
  --n_variants "${N_VARIANTS}" \
  --cfg_scales "${CFG_SCALES_ARR[@]}" \
  --baseline_cfg "${BASELINE_CFG_VAL}" \
  --correction_strengths "${CORRECTION_STRENGTHS_ARR[@]}" \
  --n_sims "${N_SIMS:-60}" \
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
  "${key_args[@]}" \
  "${fresh_noise_args[@]}" \
  "${extra_args[@]}" \
  --out_dir "${OUT_DIR}" \
  "$@"
