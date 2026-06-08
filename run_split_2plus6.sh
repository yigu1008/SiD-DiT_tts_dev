#!/usr/bin/env bash
# Split 8 GPUs:
#   GPUs 0,1  → MCTS ablation on sd3.5_base       (run_mcts_ablation_sd35base.sh)
#   GPUs 2..7 → BoN schedule continuation testing (run_bon_schedule_comparison.sh)
#
# Each slice has its own reward server (GPU = cuda:0 within its visible set).
#
# Just run:
#   bash run_split_2plus6.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MCTS_GPUS="${MCTS_GPUS:-0,1}"
BON_GPUS="${BON_GPUS:-2,3,4,5,6,7}"

# Pre-flight cleanup
pkill -9 -f "_heartbeat.sh"   2>/dev/null
pkill -9 -f reward_server.py  2>/dev/null
pkill -9 -f sd35_ddp_experiment 2>/dev/null
pkill -9 -f torchrun          2>/dev/null
pkill -9 -f sampling_unified  2>/dev/null
sleep 10
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

STAMP="$(date +%Y%m%d_%H%M%S)"

echo "================================================================"
echo "SPLIT LAUNCH @ ${STAMP}"
echo "  GPUs ${MCTS_GPUS}  → MCTS ablation on sd3.5_base"
echo "  GPUs ${BON_GPUS}  → BoN schedule continuation"
echo "================================================================"

# ── Job A: MCTS ablation on sd3.5_base, 2 GPUs ───────────────────────────
MCTS_LOG="/tmp/mcts_abl_${STAMP}.out"
MCTS_OUT="/data/ygu/runs/mcts_ablation_sd35base_${STAMP}"
nohup env \
    CUDA_VISIBLE_DEVICES="${MCTS_GPUS}" \
    TOTAL_GPUS=2 \
    BACKEND=sd35_base \
    N_PROMPTS="${MCTS_N_PROMPTS:-5}" \
    N_SIMS="${MCTS_N_SIMS:-120}" \
    SEED=42 \
    METHODS="${MCTS_METHODS:-baseline bon bon_mcts_singleseed bon_mcts_static_cfg bon_mcts_adaptive_cfg bon_mcts_rewrite_only bon_mcts_full}" \
    OUT_ROOT="${MCTS_OUT}" \
    SLIM_MODE=1 \
    bash "${SCRIPT_DIR}/run_mcts_ablation_sd35base.sh" > "${MCTS_LOG}" 2>&1 &
MCTS_PID=$!
disown ${MCTS_PID}
echo "[A] MCTS-ablation PID=${MCTS_PID}  log=${MCTS_LOG}"

sleep 15

# ── Job B: BoN schedule comparison, 6 GPUs ───────────────────────────────
BON_LOG="/tmp/bon_sched_${STAMP}.out"
BON_OUT="/data/ygu/runs/bon_schedule_${STAMP}"
nohup env \
    CUDA_VISIBLE_DEVICES="${BON_GPUS}" \
    TOTAL_GPUS=6 \
    BACKEND="${BON_BACKEND:-sid}" \
    N_PROMPTS="${BON_N_PROMPTS:-200}" \
    BON_N="${BON_N_BUDGET:-32}" \
    CFG_SCALES="${BON_CFG_SCALES:-1.0 1.25 1.5 1.75}" \
    USE_QWEN="${BON_USE_QWEN:-1}" \
    N_VARIANTS="${BON_N_VARIANTS:-3}" \
    OUT_ROOT="${BON_OUT}" \
    SLIM_MODE=1 \
    bash "${SCRIPT_DIR}/run_bon_schedule_comparison.sh" > "${BON_LOG}" 2>&1 &
BON_PID=$!
disown ${BON_PID}
echo "[B] BoN-schedule  PID=${BON_PID}  log=${BON_LOG}"

echo
echo "Monitor:"
echo "  tail -f ${MCTS_LOG}"
echo "  tail -f ${BON_LOG}"
echo
echo "Outputs:"
echo "  ${MCTS_OUT}/"
echo "  ${BON_OUT}/"
echo
echo "Kill both:"
echo "  pkill -P ${MCTS_PID}; kill ${MCTS_PID}"
echo "  pkill -P ${BON_PID};  kill ${BON_PID}"
