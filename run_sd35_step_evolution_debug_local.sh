#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUT_DIR="${OUT_DIR:-./sd35_step_evolution_debug_out}"
PROMPT_INDEX="${PROMPT_INDEX:-0}"

DEFAULT_PROMPT_FILE="${SCRIPT_DIR}/hpsv2_subset.txt"
PROMPT_FILE="${PROMPT_FILE:-}"
if [[ -z "${PROMPT_FILE}" && -f "${DEFAULT_PROMPT_FILE}" ]]; then
  PROMPT_FILE="${DEFAULT_PROMPT_FILE}"
fi
PROMPT="${PROMPT:-a cinematic portrait of a woman in soft rim light, 85mm, ultra detailed}"

METHODS_STR="${METHODS:-mcts smc}"
# shellcheck disable=SC2206
METHODS_ARR=(${METHODS_STR})
if [[ "${#METHODS_ARR[@]}" -eq 0 ]]; then
  METHODS_ARR=(mcts smc)
fi

NO_QWEN="${NO_QWEN:-1}"
QWEN_ARGS=()
if [[ "${NO_QWEN}" == "1" ]]; then
  QWEN_ARGS+=(--no_qwen)
else
  QWEN_ARGS+=(--qwen_id "${QWEN_ID:-Qwen/Qwen3-4B}")
  QWEN_ARGS+=(--qwen_dtype "${QWEN_DTYPE:-bfloat16}")
  QWEN_ARGS+=(--qwen_timeout_sec "${QWEN_TIMEOUT_SEC:-240}")
fi

mkdir -p "${OUT_DIR}"

echo "[sd35-step-debug] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[sd35-step-debug] out_dir=${OUT_DIR}"
echo "[sd35-step-debug] methods=${METHODS_ARR[*]}"
if [[ -n "${PROMPT_FILE}" ]]; then
  echo "[sd35-step-debug] prompt_file=${PROMPT_FILE} prompt_index=${PROMPT_INDEX}"
else
  echo "[sd35-step-debug] prompt='${PROMPT}'"
fi

ARGS=(
  --backend sd35_base
  --reward_backend "${REWARD_BACKEND:-imagereward}"
  --image_reward_model "${IMAGE_REWARD_MODEL:-ImageReward-v1.0}"
  --pickscore_model "${PICKSCORE_MODEL:-yuvalkirstain/PickScore_v1}"
  --reward_model "${REWARD_MODEL:-CodeGoat24/UnifiedReward-qwen-7b}"
  --unifiedreward_model "${UNIFIEDREWARD_MODEL:-CodeGoat24/UnifiedReward-qwen-7b}"
  --reward_api_key "${REWARD_API_KEY:-unifiedreward}"
  --reward_api_model "${REWARD_API_MODEL:-UnifiedReward-7b-v1.5}"
  --reward_max_new_tokens "${REWARD_MAX_NEW_TOKENS:-512}"
  --reward_prompt_mode "${REWARD_PROMPT_MODE:-standard}"
  --steps "${STEPS:-28}"
  --cfg_scales ${CFG_SCALES:-3.5 4.0 4.5 5.0 5.5 6.0 7.0}
  --baseline_cfg "${BASELINE_CFG:-4.5}"
  --seed "${SEED:-42}"
  --width "${WIDTH:-1024}"
  --height "${HEIGHT:-1024}"
  --n_variants "${N_VARIANTS:-4}"
  --n_sims "${N_SIMS:-50}"
  --ucb_c "${UCB_C:-1.41}"
  --smc_k "${SMC_K:-8}"
  --smc_gamma "${SMC_GAMMA:-0.10}"
  --ess_threshold "${ESS_THRESHOLD:-0.5}"
  --resample_start_frac "${RESAMPLE_START_FRAC:-0.3}"
  --smc_cfg_scale "${SMC_CFG_SCALE:-4.5}"
  --smc_variant_idx "${SMC_VARIANT_IDX:-0}"
  --methods "${METHODS_ARR[@]}"
  --prompt_index "${PROMPT_INDEX}"
  --debug_out "${OUT_DIR}"
  --make_gif
  --save_latent_steps
  --save_x0_steps
)

if [[ "${SAVE_SMC_PARTICLE_GRIDS:-0}" == "1" ]]; then
  ARGS+=(--save_smc_particle_grids)
else
  ARGS+=(--no-save_smc_particle_grids)
fi

if [[ -n "${PROMPT_FILE}" ]]; then
  ARGS+=(--prompt_file "${PROMPT_FILE}")
else
  ARGS+=(--prompt "${PROMPT}")
fi

if [[ -n "${REWRITES_FILE:-}" ]]; then
  ARGS+=(--rewrites_file "${REWRITES_FILE}")
fi
if [[ -n "${REWARD_API_BASE:-}" ]]; then
  ARGS+=(--reward_api_base "${REWARD_API_BASE}")
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/sd35_step_evolution_debug.py" \
  "${ARGS[@]}" \
  "${QWEN_ARGS[@]}" \
  "$@"
