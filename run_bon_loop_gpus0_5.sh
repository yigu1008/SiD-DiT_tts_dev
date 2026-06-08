#!/usr/bin/env bash
# Loop bon_schedule comparison across 3 backends on GPUs 0,1,2,3,4,5.
# Designed to live alongside an MCTS keystep ablation on GPUs 6,7.
# No env-var prefix needed -- just `bash run_bon_loop_gpus0_5.sh`.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPUS=0,1,2,3,4,5
export TOTAL_GPUS=6
export N_PROMPTS="${N_PROMPTS:-200}"
export BON_N="${BON_N:-32}"
export CFG_SCALES="${CFG_SCALES:-1.0 1.25 1.5 1.75}"
export USE_QWEN="${USE_QWEN:-1}"
export N_VARIANTS="${N_VARIANTS:-3}"
export BACKENDS="${BACKENDS:-sid senseflow_large flux_schnell}"

exec bash "${SCRIPT_DIR}/run_bon_loop_gpus2_5.sh"
