#!/usr/bin/env bash
# One-button launcher for the user's 200-prompt actdiff_grid_sid run.
# All knobs baked in -- just `bash run_my_actdiff_grid.sh` and walk away.
#
# This wraps run_actdiff_grid_sid_a6000.sh with pinned settings so future
# adjustments only happen here, not scattered across env exports.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Pinned configuration (edit these if you need to change anything) ─────
export N_PROMPTS=200
# ./prompts.txt next to this script
export PROMPT_FILE="${SCRIPT_DIR}/prompts.txt"
export SEARCH_REWARD=imagereward
export EVAL_BACKENDS=imagereward
export BACKEND=sid
export SEED=42
export TOTAL_GPUS=8
# Slim mode: don't write per-prompt images, variants files, step traces, etc.
# Only the rank_*.jsonl scores + summary.tsv are kept.  Saves ~50 GB on a
# 200-prompt x 11-method grid.
export SLIM_MODE=1

# Method list (full 11-method grid).  Subset by commenting out unwanted ones.
export METHODS="baseline bon bon_actdiff_cfg bon_actdiff_full sop sop_actdiff_cfg sop_actdiff_full smc smc_actdiff_cfg smc_actdiff_full bon_mcts"

# Where to write outputs
export RUN_ROOT="/data/ygu/runs/my_actdiff_grid_$(date +%Y%m%d_%H%M%S)"

# ── Pre-flight cleanup ────────────────────────────────────────────────────
pkill -9 -f _heartbeat.sh   2>/dev/null || true
pkill -9 -f reward_server.py 2>/dev/null || true
pkill -9 -f sd35_ddp_experiment 2>/dev/null || true
pkill -9 -f torchrun        2>/dev/null || true
sleep 5

# ── Auto-fetch HPSv2+DrawBench if prompt file missing or too short ───────
# So this script is truly one-button: no manual prompt prep needed.
_need_fetch=0
if [[ ! -s "${PROMPT_FILE}" ]]; then
    _need_fetch=1
    echo "[fetch] PROMPT_FILE missing; will fetch HPSv2+DrawBench"
else
    _n_avail=$(grep -c . "${PROMPT_FILE}" 2>/dev/null || echo 0)
    if [[ "${_n_avail}" -lt "${N_PROMPTS}" ]]; then
        _need_fetch=1
        echo "[fetch] PROMPT_FILE has ${_n_avail} < ${N_PROMPTS}; will re-fetch"
    fi
fi

if [[ "${_need_fetch}" == "1" ]]; then
    PYTHONNOUSERSITE=1 python3 "${SCRIPT_DIR}/fetch_hpsv2.py" \
        --out_file "${PROMPT_FILE}" \
        --n_prompts "${N_PROMPTS}" \
        --shuffle --seed "${SEED}" \
        || { echo "[FATAL] HPSv2 fetch failed" >&2; exit 1; }
fi

_n_avail=$(grep -c . "${PROMPT_FILE}" 2>/dev/null || echo 0)
echo "[fetch] ${PROMPT_FILE} has ${_n_avail} non-empty lines"
if [[ "${_n_avail}" -lt "${N_PROMPTS}" ]]; then
    echo "[WARN] still short; will use the first ${_n_avail}" >&2
    export N_PROMPTS="${_n_avail}"
fi

echo "================================================================"
echo "MY ACTDIFF GRID RUN"
echo "  N_PROMPTS      = ${N_PROMPTS}"
echo "  PROMPT_FILE    = ${PROMPT_FILE}  (${_n_avail} lines)"
echo "  BACKEND        = ${BACKEND}"
echo "  SEED           = ${SEED}"
echo "  SEARCH_REWARD  = ${SEARCH_REWARD}"
echo "  EVAL_BACKENDS  = ${EVAL_BACKENDS}"
echo "  TOTAL_GPUS     = ${TOTAL_GPUS}     (GPU 0 reward, GPUs 1..N-1 sampling)"
echo "  METHODS        = ${METHODS}"
echo "  RUN_ROOT       = ${RUN_ROOT}"
echo "================================================================"
echo "ETA: ~20-22 hours on 8 GPUs"
echo
sleep 3   # give user a chance to Ctrl-C if something looks wrong

# ── Hand off to the actual runner ─────────────────────────────────────────
bash "${SCRIPT_DIR}/run_actdiff_grid_sid_a6000.sh" 2>&1 | tee "${RUN_ROOT}_console.log"
