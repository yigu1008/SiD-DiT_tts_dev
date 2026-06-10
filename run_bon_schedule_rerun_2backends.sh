#!/usr/bin/env bash
# Re-run bon_schedule on the 2 backends we don't have yet (senseflow_large,
# flux_schnell).  The sid run already produced +0.384 IR over baseline on 200
# prompts — this script replicates that win on the other two distilled backends
# so we have 3-backend coverage for the paper.
#
# Lean: only baseline + bon_schedule per backend (skips bon and bon_actdiff_cfg
# which crashed early in the previous loop attempt and aren't needed for the
# headline result).
#
# Just run:
#   bash run_bon_schedule_rerun_2backends.sh
# Background:
#   nohup bash run_bon_schedule_rerun_2backends.sh > /tmp/bonsched_rerun.out 2>&1 & disown

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GPUs 0-5 (6 total: 1 reward + 5 sampling).  Override with GPUS=... TOTAL_GPUS=...
export GPUS="${GPUS:-0,1,2,3,4,5}"
export TOTAL_GPUS="${TOTAL_GPUS:-6}"
export N_PROMPTS="${N_PROMPTS:-200}"
export BON_N="${BON_N:-32}"
export CFG_SCALES="${CFG_SCALES:-1.0 1.25 1.5 1.75}"
export USE_QWEN="${USE_QWEN:-0}"
export N_VARIANTS="${N_VARIANTS:-1}"

# Backends remaining (sid already done).  Override BACKENDS to re-include sid.
BACKENDS="${BACKENDS:-senseflow_large flux_schnell}"
# Only baseline + bon_schedule — the lean, focused rerun.
export METHODS_RUN="${METHODS_RUN:-baseline bon_schedule}"

for backend in ${BACKENDS}; do
    echo "================================================================"
    echo "[bon-sched-rerun] starting backend=${backend}"
    echo "================================================================"
    CUDA_VISIBLE_DEVICES="${GPUS}" \
    TOTAL_GPUS="${TOTAL_GPUS}" \
    BACKEND="${backend}" \
    N_PROMPTS="${N_PROMPTS}" \
    BON_N="${BON_N}" \
    CFG_SCALES="${CFG_SCALES}" \
    USE_QWEN="${USE_QWEN}" \
    N_VARIANTS="${N_VARIANTS}" \
    METHODS_RUN="${METHODS_RUN}" \
      bash "${SCRIPT_DIR}/run_bon_schedule_comparison.sh" \
      2>&1 | tee "/tmp/bonsched_rerun_${backend}.log"
    rc=${PIPESTATUS[0]}
    echo "[bon-sched-rerun] finished backend=${backend}  rc=${rc}"
done
