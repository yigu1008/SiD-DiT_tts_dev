#!/usr/bin/env bash
# Split the 8 local GPUs across two parallel experiments:
#   GPUs 0-3 : limited-noise comparison (run_limited_noise.sh)
#   GPUs 4-7 : scaled-up fresh-vs-fixed noise ablation (run_noise_ablation_a6000.sh)
#
# Each job sees 4 GPUs as cuda:0..3 (CUDA_VISIBLE_DEVICES masks the rest).
# Within each 4-GPU slice the cluster pattern still holds: cuda:0 = reward,
# cuda:1..3 = sampling.
#
# Just run:
#   bash run_split_8gpu.sh
# Knobs:
#   LIMITED_N_PROMPTS=200    LIMITED_BON_N=8
#   ABL_N_PROMPTS=100        ABL_N_SIMS=30

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Slice config ─────────────────────────────────────────────────────────
LIMITED_GPUS="${LIMITED_GPUS:-0,1,2,3}"
LIMITED_N_PROMPTS="${LIMITED_N_PROMPTS:-200}"
LIMITED_BON_N="${LIMITED_BON_N:-8}"

ABL_GPUS="${ABL_GPUS:-4,5,6,7}"
ABL_N_PROMPTS="${ABL_N_PROMPTS:-100}"
ABL_N_SIMS="${ABL_N_SIMS:-30}"

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p /tmp /data/ygu/runs 2>/dev/null || true

# Pre-flight cleanup -- kill leftover servers/heartbeats from prior runs
pkill -9 -f reward_server.py 2>/dev/null || true
pkill -9 -f _heartbeat.sh 2>/dev/null || true
pkill -9 -f sd35_ddp_experiment 2>/dev/null || true
pkill -9 -f torchrun 2>/dev/null || true
sleep 5

echo "================================================================"
echo "SPLIT 8-GPU LAUNCH @ ${STAMP}"
echo "  GPUs ${LIMITED_GPUS}  -> limited-noise (BON_N=${LIMITED_BON_N}, N=${LIMITED_N_PROMPTS})"
echo "  GPUs ${ABL_GPUS}      -> noise ablation (N=${ABL_N_PROMPTS}, sims=${ABL_N_SIMS})"
echo "================================================================"

# ── Job A: limited-noise on GPUs 0-3 ─────────────────────────────────────
LIMITED_LOG="/tmp/limited_noise_${STAMP}.out"
nohup env \
    CUDA_VISIBLE_DEVICES="${LIMITED_GPUS}" \
    TOTAL_GPUS=4 \
    N_PROMPTS="${LIMITED_N_PROMPTS}" \
    BON_N="${LIMITED_BON_N}" \
    OUT_ROOT="/data/ygu/runs/limited_noise_N${LIMITED_BON_N}_${STAMP}" \
    bash "${SCRIPT_DIR}/run_limited_noise.sh" \
    > "${LIMITED_LOG}" 2>&1 &
LIMITED_PID=$!
disown ${LIMITED_PID}
echo "[A] limited-noise PID=${LIMITED_PID}  log=${LIMITED_LOG}"

# Stagger by 10s to avoid simultaneous HF cache races
sleep 10

# ── Job B: noise ablation on GPUs 4-7 ────────────────────────────────────
ABL_LOG="/tmp/noise_ablation_${STAMP}.out"
nohup env \
    CUDA_VISIBLE_DEVICES="${ABL_GPUS}" \
    TOTAL_GPUS=4 \
    N_PROMPTS="${ABL_N_PROMPTS}" \
    N_SIMS="${ABL_N_SIMS}" \
    OUT_ROOT="/data/ygu/runs/noise_ablation_${STAMP}" \
    bash "${SCRIPT_DIR}/run_noise_ablation_a6000.sh" \
    > "${ABL_LOG}" 2>&1 &
ABL_PID=$!
disown ${ABL_PID}
echo "[B] noise-ablation PID=${ABL_PID}  log=${ABL_LOG}"

echo
echo "Monitor:"
echo "  tail -f ${LIMITED_LOG}"
echo "  tail -f ${ABL_LOG}"
echo
echo "Outputs:"
echo "  /data/ygu/runs/limited_noise_N${LIMITED_BON_N}_${STAMP}/"
echo "  /data/ygu/runs/noise_ablation_${STAMP}/"
echo
echo "Kill both:"
echo "  pkill -P ${LIMITED_PID}; kill ${LIMITED_PID}"
echo "  pkill -P ${ABL_PID}; kill ${ABL_PID}"
