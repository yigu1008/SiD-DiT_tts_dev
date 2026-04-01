#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

# Two-model entrypoint: SD3.5 + FLUX, all algorithms.
export RUN_SANA="${RUN_SANA:-0}"
export RUN_SD35="${RUN_SD35:-1}"
export RUN_FLUX="${RUN_FLUX:-1}"
export METHODS="${METHODS:-baseline greedy mcts ga smc}"
export USE_QWEN="${USE_QWEN:-1}"
export SD35_BACKEND="${SD35_BACKEND:-sid}"
export FLUX_BACKEND="${FLUX_BACKEND:-flux}"

bash "${SCRIPT_DIR}/hpsv2_all_models_ddp_suite.sh" "$@"
