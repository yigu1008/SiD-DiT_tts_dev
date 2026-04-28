#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

DEFAULT_PROMPT_FILE="${SCRIPT_DIR}/hpsv2_subset.txt"
if [[ -f "${DEFAULT_PROMPT_FILE}" ]]; then
  export PROMPT_FILE="${PROMPT_FILE:-${DEFAULT_PROMPT_FILE}}"
fi

export OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/sd35_multireward_ir_hpsv3_pick_local_out}"
export METHODS="${METHODS:-baseline bon_mcts}"
export RUN_SANA=0
export RUN_SD35=1
export RUN_FLUX=0
export SD35_BACKEND="${SD35_BACKEND:-sid}"

# Search-time reward: equal-weight mean of available backends.
export REWARD_BACKEND="${REWARD_BACKEND:-all}"
export REWARD_DEVICE="${REWARD_DEVICE:-cuda}"

# Post-eval reward breakdown.
export EVAL_BEST_IMAGES="${EVAL_BEST_IMAGES:-1}"
export EVAL_BACKENDS="${EVAL_BACKENDS:-imagereward hpsv3 pickscore}"
export EVAL_REWARD_DEVICE="${EVAL_REWARD_DEVICE:-cuda}"
export EVAL_ALLOW_MISSING_BACKENDS="${EVAL_ALLOW_MISSING_BACKENDS:-0}"

# Strongly recommended for hpsv3 compatibility isolation.
export USE_REWARD_SERVER="${USE_REWARD_SERVER:-1}"
export REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5100}"
export REWARD_SERVER_BACKENDS="${REWARD_SERVER_BACKENDS:-imagereward hpsv3 pickscore}"
export REWARD_ENV_NAME="${REWARD_ENV_NAME:-reward}"
export REWARD_ENV_CONDA_BASE="${REWARD_ENV_CONDA_BASE:-/opt/conda}"

# HPSv3 implementation choice (auto|official|imscore). imscore is typically most stable.
export SID_HPSV3_IMPL="${SID_HPSV3_IMPL:-imscore}"

# Keep artifacts for multi-backend post-eval.
export SAVE_IMAGES="${SAVE_IMAGES:-0}"
export SAVE_BEST_IMAGES="${SAVE_BEST_IMAGES:-1}"

echo "[sd35-multireward]"
echo "  prompt_file=${PROMPT_FILE:-<unset>}"
echo "  out_root=${OUT_ROOT}"
echo "  methods=${METHODS}"
echo "  backend=${SD35_BACKEND}"
echo "  reward_backend=${REWARD_BACKEND}"
echo "  eval_backends=${EVAL_BACKENDS}"
echo "  reward_server=${USE_REWARD_SERVER} backends=${REWARD_SERVER_BACKENDS}"

bash "${SCRIPT_DIR}/hpsv2_sd35_flux_allalgos.sh" "$@"
