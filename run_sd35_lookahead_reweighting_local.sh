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
OUT_DIR="${OUT_DIR:-./sd35_lookahead_reweighting_local_out}"
START_INDEX="${START_INDEX:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
END_INDEX="${END_INDEX:-}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "[sd35-lookahead] prompt file not found, exporting HPSv2 prompts first ..."
  OUT_DIR="${PROMPT_DIR}" STYLE="${PROMPT_STYLE}" bash "${SCRIPT_DIR}/get_hpsv2_prompts.sh"
fi
if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found after export: ${PROMPT_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

PROMPT_FILE_RUN="${PROMPT_FILE}"
if [[ -n "${END_INDEX}" || "${START_INDEX}" != "0" || "${NUM_PROMPTS}" != "0" ]]; then
  # Prefer NUM_PROMPTS over END_INDEX so quick runs stay stable even if
  # END_INDEX is still exported in the shell.
  if (( NUM_PROMPTS > 0 )); then
    if [[ -n "${END_INDEX}" ]]; then
      echo "[sd35-lookahead] both NUM_PROMPTS and END_INDEX are set; using NUM_PROMPTS (END_INDEX ignored)."
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
print(f"[sd35-lookahead] prompt slice: src={src} range=[{start},{end}) n={end-start} -> {dst}")
PY
  PROMPT_FILE_RUN="${SLICE_FILE}"
fi

CFG_SCALES_STR="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5}"
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

CFG_ONLY="${CFG_ONLY:-0}"
N_VARIANTS="${N_VARIANTS:-4}"
USE_QWEN="${USE_QWEN:-1}"
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

LOOKAHEAD_MODE="${LOOKAHEAD_MODE:-rollout_prior}"
LOOKAHEAD_RUN_ABLATIONS="${LOOKAHEAD_RUN_ABLATIONS:-1}"
LOOKAHEAD_ABLATION_SET="${LOOKAHEAD_ABLATION_SET:-A B C D E F}"
LOOKAHEAD_INCLUDE_STANDARD="${LOOKAHEAD_INCLUDE_STANDARD:-1}"
LOOKAHEAD_U_T_DEF="${LOOKAHEAD_U_T_DEF:-latent_delta_rms}"
LOOKAHEAD_U_T_DEFS_FOR_F="${LOOKAHEAD_U_T_DEFS_FOR_F:-latent_delta_rms latent_rms dx_rms}"

lookahead_args=(
  --lookahead_mode "${LOOKAHEAD_MODE}"
  --lookahead_u_t_def "${LOOKAHEAD_U_T_DEF}"
  --lookahead_tau "${LOOKAHEAD_TAU:-0.35}"
  --lookahead_c_puct "${LOOKAHEAD_C_PUCT:-1.20}"
  --lookahead_u_ref "${LOOKAHEAD_U_REF:-0.0}"
  --lookahead_w_cfg "${LOOKAHEAD_W_CFG:-1.0}"
  --lookahead_w_variant "${LOOKAHEAD_W_VARIANT:-0.25}"
  --lookahead_w_cs "${LOOKAHEAD_W_CS:-0.10}"
  --lookahead_w_q "${LOOKAHEAD_W_Q:-0.20}"
  --lookahead_w_explore "${LOOKAHEAD_W_EXPLORE:-0.05}"
  --lookahead_cfg_width_min "${LOOKAHEAD_CFG_WIDTH_MIN:-3}"
  --lookahead_cfg_width_max "${LOOKAHEAD_CFG_WIDTH_MAX:-7}"
  --lookahead_cfg_anchor_count "${LOOKAHEAD_CFG_ANCHOR_COUNT:-2}"
  --lookahead_min_visits_for_center "${LOOKAHEAD_MIN_VISITS_FOR_CENTER:-3}"
  --lookahead_log_action_topk "${LOOKAHEAD_LOG_ACTION_TOPK:-12}"
)

if [[ "${LOOKAHEAD_INCLUDE_STANDARD}" == "1" ]]; then
  lookahead_args+=(--lookahead_include_standard)
else
  lookahead_args+=(--no-lookahead_include_standard)
fi

if [[ "${LOOKAHEAD_RUN_ABLATIONS}" == "1" ]]; then
  # shellcheck disable=SC2206
  ablation_arr=(${LOOKAHEAD_ABLATION_SET})
  # shellcheck disable=SC2206
  u_defs_arr=(${LOOKAHEAD_U_T_DEFS_FOR_F})
  lookahead_args+=(--lookahead_run_ablations)
  if [[ "${#ablation_arr[@]}" -gt 0 ]]; then
    lookahead_args+=(--lookahead_ablation_set "${ablation_arr[@]}")
  fi
  if [[ "${#u_defs_arr[@]}" -gt 0 ]]; then
    lookahead_args+=(--lookahead_u_t_defs_for_f "${u_defs_arr[@]}")
  fi
fi

echo "[sd35-lookahead] prompt_file=${PROMPT_FILE_RUN}"
echo "[sd35-lookahead] out_dir=${OUT_DIR}"
echo "[sd35-lookahead] cfg_only=${CFG_ONLY} n_variants=${N_VARIANTS} use_qwen=${USE_QWEN}"
echo "[sd35-lookahead] cfg_scales=[${CFG_SCALES_ARR[*]}] baseline_cfg=${BASELINE_CFG:-1.0}"
echo "[sd35-lookahead] mode=${LOOKAHEAD_MODE} run_ablations=${LOOKAHEAD_RUN_ABLATIONS} ablations=[${LOOKAHEAD_ABLATION_SET}]"

"${PYTHON_BIN}" "${SCRIPT_DIR}/sampling_unified_sd35_lookahead_reweighting.py" \
  --search_method mcts \
  --backend "${SD35_BACKEND:-sid}" \
  --prompt_file "${PROMPT_FILE_RUN}" \
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
  "${lookahead_args[@]}" \
  "${extra_args[@]}" \
  --out_dir "${OUT_DIR}" \
  "$@"
