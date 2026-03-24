#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

REWARD_BACKEND="${REWARD_BACKEND:-unifiedreward}"
REWARD_MODEL="${REWARD_MODEL:-CodeGoat24/UnifiedReward-qwen-7b}"
UNIFIEDREWARD_MODEL="${UNIFIEDREWARD_MODEL:-${REWARD_MODEL}}"
REWARD_PROMPT_MODE="${REWARD_PROMPT_MODE:-standard}"
REWARD_MAX_NEW_TOKENS="${REWARD_MAX_NEW_TOKENS:-512}"

export REWARD_BACKEND
export REWARD_MODEL
export UNIFIEDREWARD_MODEL
export REWARD_PROMPT_MODE
export REWARD_MAX_NEW_TOKENS

"${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh" "$@"
