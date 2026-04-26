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

OUT_DIR="${OUT_DIR:-./sd35_base_bon_mcts_local_out}"
START_INDEX="${START_INDEX:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
END_INDEX="${END_INDEX:-}"
NUM_GPUS="${NUM_GPUS:-1}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "[sd35-base-bon-mcts] prompt file not found, exporting HPSv2 prompts first ..."
  OUT_DIR="${PROMPT_DIR}" STYLE="${PROMPT_STYLE}" bash "${SCRIPT_DIR}/get_hpsv2_prompts.sh"
fi
if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found after export: ${PROMPT_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

PROMPT_FILE_RUN="${PROMPT_FILE}"
if [[ -n "${END_INDEX}" || "${START_INDEX}" != "0" || "${NUM_PROMPTS}" != "0" ]]; then
  if (( NUM_PROMPTS > 0 )); then
    if [[ -n "${END_INDEX}" ]]; then
      echo "[sd35-base-bon-mcts] both NUM_PROMPTS and END_INDEX are set; using NUM_PROMPTS (END_INDEX ignored)."
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
print(f"[sd35-base-bon-mcts] prompt slice: src={src} range=[{start},{end}) n={end-start} -> {dst}")
PY
  PROMPT_FILE_RUN="${SLICE_FILE}"
fi

SD35_BACKEND_VAL="${SD35_BACKEND:-sd35_base}"
CFG_SCALES_STR="${CFG_SCALES:-3.5 4.0 4.5 5.0 5.5 6.0 7.0}"
BASELINE_CFG_VAL="${BASELINE_CFG:-4.5}"
DEFAULT_STEPS=28
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

N_VARIANTS="${N_VARIANTS:-4}"
USE_QWEN="${USE_QWEN:-1}"
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

if (( NUM_GPUS > 1 )); then
  launch_cmd=("${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_GPUS}" "${SCRIPT_DIR}/sd35_ddp_experiment_bon_mcts_sd35base.py")
else
  launch_cmd=("${PYTHON_BIN}" "${SCRIPT_DIR}/sd35_ddp_experiment_bon_mcts_sd35base.py")
fi

echo "[sd35-base-bon-mcts] prompt_file=${PROMPT_FILE_RUN}"
echo "[sd35-base-bon-mcts] out_dir=${OUT_DIR}"
echo "[sd35-base-bon-mcts] backend=${SD35_BACKEND_VAL} steps=${STEPS:-${DEFAULT_STEPS}} num_gpus=${NUM_GPUS}"
echo "[sd35-base-bon-mcts] cfg_scales=[${CFG_SCALES_ARR[*]}] baseline_cfg=${BASELINE_CFG_VAL}"
echo "[sd35-base-bon-mcts] bon_mcts: n_seeds=${BON_MCTS_N_SEEDS:-12} topk=${BON_MCTS_TOPK:-3} sim_alloc=${BON_MCTS_SIM_ALLOC:-split}"
echo "[sd35-base-bon-mcts] refine=${BON_MCTS_REFINE_METHOD:-ours_tree} lookahead_mode=${LOOKAHEAD_METHOD_MODE:-rollout_tree_prior_adaptive_cfg}"

cmd=(
  "${launch_cmd[@]}"
  --backend "${SD35_BACKEND_VAL}"
  --prompt_file "${PROMPT_FILE_RUN}"
  --out_dir "${OUT_DIR}"
  --modes base mcts
  --steps "${STEPS:-${DEFAULT_STEPS}}"
  --cfg_scales "${CFG_SCALES_ARR[@]}"
  --baseline_cfg "${BASELINE_CFG_VAL}"
  --n_variants "${N_VARIANTS}"
  --correction_strengths "${CORRECTION_STRENGTHS_ARR[@]}"
  --n_sims "${N_SIMS:-120}"
  --ucb_c "${UCB_C:-1.41}"
  --seed "${SEED:-42}"
  --width "${WIDTH:-1024}"
  --height "${HEIGHT:-1024}"
  --reward_backend "${REWARD_BACKEND:-imagereward}"
  --reward_model "${REWARD_MODEL:-CodeGoat24/UnifiedReward-qwen-7b}"
  --unifiedreward_model "${UNIFIEDREWARD_MODEL:-CodeGoat24/UnifiedReward-qwen-7b}"
  --image_reward_model "${IMAGE_REWARD_MODEL:-ImageReward-v1.0}"
  --pickscore_model "${PICKSCORE_MODEL:-yuvalkirstain/PickScore_v1}"
  --reward_api_key "${REWARD_API_KEY:-unifiedreward}"
  --reward_api_model "${REWARD_API_MODEL:-UnifiedReward-7b-v1.5}"
  --reward_max_new_tokens "${REWARD_MAX_NEW_TOKENS:-512}"
  --reward_prompt_mode "${REWARD_PROMPT_MODE:-standard}"
  --bon_mcts_n_seeds "${BON_MCTS_N_SEEDS:-12}"
  --bon_mcts_topk "${BON_MCTS_TOPK:-3}"
  --bon_mcts_seed_stride "${BON_MCTS_SEED_STRIDE:-1}"
  --bon_mcts_seed_offset "${BON_MCTS_SEED_OFFSET:-0}"
  --bon_mcts_sim_alloc "${BON_MCTS_SIM_ALLOC:-split}"
  --bon_mcts_min_sims "${BON_MCTS_MIN_SIMS:-16}"
  --bon_mcts_refine_method "${BON_MCTS_REFINE_METHOD:-ours_tree}"
  --lookahead_mode "${LOOKAHEAD_METHOD_MODE:-rollout_tree_prior_adaptive_cfg}"
  --lookahead_u_t_def "${LOOKAHEAD_U_T_DEF:-latent_delta_rms}"
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
if [[ ${#extra_reward_args[@]} -gt 0 ]]; then
  cmd+=("${extra_reward_args[@]}")
fi
if [[ ${#qwen_args[@]} -gt 0 ]]; then
  cmd+=("${qwen_args[@]}")
fi
if [[ ${#extra_args[@]} -gt 0 ]]; then
  cmd+=("${extra_args[@]}")
fi
if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

"${cmd[@]}"
