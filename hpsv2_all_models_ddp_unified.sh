#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

REWARD_BACKEND="${REWARD_BACKEND:-unifiedreward}"
REWARD_TYPE="${REWARD_TYPE:-unifiedreward}"
REWARD_MODEL="${REWARD_MODEL:-CodeGoat24/UnifiedReward-2.0-qwen3vl-4b}"
UNIFIEDREWARD_MODEL="${UNIFIEDREWARD_MODEL:-${REWARD_MODEL}}"
REWARD_PROMPT_MODE="${REWARD_PROMPT_MODE:-standard}"
REWARD_MAX_NEW_TOKENS="${REWARD_MAX_NEW_TOKENS:-512}"
REWARD_DEVICE="${REWARD_DEVICE:-cpu}"

export REWARD_BACKEND
export REWARD_TYPE
export REWARD_MODEL
export UNIFIEDREWARD_MODEL
export REWARD_PROMPT_MODE
export REWARD_MAX_NEW_TOKENS
export REWARD_DEVICE

"${SCRIPT_DIR}/hpsv2_all_models_ddp_suite.sh" "$@"
