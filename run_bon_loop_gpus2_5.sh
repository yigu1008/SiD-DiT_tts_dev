#!/usr/bin/env bash
# Loop bon_schedule comparison across 3 backends on GPUs 2,3,4,5.
# Designed to live alongside an MCTS keystep ablation on GPUs 6,7.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Knobs (override via env)
GPUS="${GPUS:-2,3,4,5}"
TOTAL_GPUS_LOCAL="${TOTAL_GPUS:-4}"
N_PROMPTS_LOCAL="${N_PROMPTS:-200}"
BON_N_LOCAL="${BON_N:-32}"
CFG_SCALES_LOCAL="${CFG_SCALES:-1.0 1.25 1.5 1.75}"
USE_QWEN_LOCAL="${USE_QWEN:-1}"
N_VARIANTS_LOCAL="${N_VARIANTS:-3}"
BACKENDS_LOCAL="${BACKENDS:-sid senseflow_large flux_schnell}"

for backend in ${BACKENDS_LOCAL}; do
    echo "================================================================"
    echo "[bon-loop] starting backend=${backend}"
    echo "================================================================"
    CUDA_VISIBLE_DEVICES="${GPUS}" \
    TOTAL_GPUS="${TOTAL_GPUS_LOCAL}" \
    BACKEND="${backend}" \
    N_PROMPTS="${N_PROMPTS_LOCAL}" \
    BON_N="${BON_N_LOCAL}" \
    CFG_SCALES="${CFG_SCALES_LOCAL}" \
    USE_QWEN="${USE_QWEN_LOCAL}" \
    N_VARIANTS="${N_VARIANTS_LOCAL}" \
      bash "${SCRIPT_DIR}/run_bon_schedule_comparison.sh" \
      2>&1 | tee "/tmp/bonsched_${backend}.log"
    echo "[bon-loop] finished backend=${backend}"
done
