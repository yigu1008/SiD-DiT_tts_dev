#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

NUM_GPUS="${NUM_GPUS:-$("${PYTHON_BIN}" - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)}"

PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/parti_prompts.txt}"
OUT_DIR="${OUT_DIR:-./sd35_ddp_out}"
REWARD_BACKEND="${REWARD_BACKEND:-unifiedreward}"
REWARD_MODEL="${REWARD_MODEL:-CodeGoat24/UnifiedReward-qwen-7b}"
UNIFIEDREWARD_MODEL="${UNIFIEDREWARD_MODEL:-${REWARD_MODEL}}"
IMAGE_REWARD_MODEL="${IMAGE_REWARD_MODEL:-ImageReward-v1.0}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-1.0 1.0}"
REWARD_API_BASE="${REWARD_API_BASE:-}"
REWARD_API_KEY="${REWARD_API_KEY:-unifiedreward}"
REWARD_API_MODEL="${REWARD_API_MODEL:-UnifiedReward-7b-v1.5}"
REWARD_MAX_NEW_TOKENS="${REWARD_MAX_NEW_TOKENS:-512}"
REWARD_PROMPT_MODE="${REWARD_PROMPT_MODE:-standard}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE does not exist: ${PROMPT_FILE}" >&2
  exit 1
fi
mkdir -p "${OUT_DIR}"

PROMPT_FILE_ABS="$("${PYTHON_BIN}" - <<'PY' "${PROMPT_FILE}"
import pathlib,sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PY
)"
OUT_DIR_ABS="$("${PYTHON_BIN}" - <<'PY' "${OUT_DIR}"
import pathlib,sys
p = pathlib.Path(sys.argv[1]).expanduser().resolve()
p.mkdir(parents=True, exist_ok=True)
print(p)
PY
)"

EXTRA_ARGS=()
if [[ "${NO_QWEN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no_qwen)
fi
if [[ "${SEED_PER_PROMPT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--seed_per_prompt)
fi
if [[ "${SAVE_IMAGES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save_images)
fi
if [[ "${SAVE_VARIANTS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save_variants)
fi
if [[ -n "${REWRITES_FILE:-}" ]]; then
  EXTRA_ARGS+=(--rewrites_file "${REWRITES_FILE}")
fi
if [[ -n "${REWARD_API_BASE}" ]]; then
  EXTRA_ARGS+=(--reward_api_base "${REWARD_API_BASE}")
fi

torchrun --standalone --nproc_per_node "${NUM_GPUS}" "${SCRIPT_DIR}/sd35_ddp_experiment.py" \
  --prompt_file "${PROMPT_FILE_ABS}" \
  --start_index "${START_INDEX:-0}" \
  --end_index "${END_INDEX:--1}" \
  --modes ${MODES:-base greedy mcts ga smc} \
  --cfg_scales ${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5} \
  --baseline_cfg "${BASELINE_CFG:-1.0}" \
  --steps "${STEPS:-4}" \
  --n_variants "${N_VARIANTS:-3}" \
  --n_sims "${N_SIMS:-50}" \
  --ucb_c "${UCB_C:-1.41}" \
  --smc_k "${SMC_K:-8}" \
  --smc_gamma "${SMC_GAMMA:-0.10}" \
  --ess_threshold "${ESS_THRESHOLD:-0.5}" \
  --resample_start_frac "${RESAMPLE_START_FRAC:-0.3}" \
  --smc_cfg_scale "${SMC_CFG_SCALE:-1.25}" \
  --smc_variant_idx "${SMC_VARIANT_IDX:-0}" \
  --reward_backend "${REWARD_BACKEND}" \
  --reward_model "${REWARD_MODEL}" \
  --unifiedreward_model "${UNIFIEDREWARD_MODEL}" \
  --image_reward_model "${IMAGE_REWARD_MODEL}" \
  --reward_weights ${REWARD_WEIGHTS} \
  --reward_api_key "${REWARD_API_KEY}" \
  --reward_api_model "${REWARD_API_MODEL}" \
  --reward_max_new_tokens "${REWARD_MAX_NEW_TOKENS}" \
  --reward_prompt_mode "${REWARD_PROMPT_MODE}" \
  --seed "${SEED:-42}" \
  --out_dir "${OUT_DIR_ABS}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_sd35_ddp.py" --log_dir "${OUT_DIR_ABS}/logs" --out_dir "${OUT_DIR_ABS}"
