#!/usr/bin/env bash
# Launches the limited-noise comparison on GPUs 0-3 (reward on cuda:0,
# sampling on cuda:1-3).  Defaults pinned; edit the top of this file
# to change them.
#
# Just run:
#   bash run_limited_noise_gpus0_3.sh
# Backgrounded:
#   nohup bash run_limited_noise_gpus0_3.sh > /tmp/lim.out 2>&1 & disown

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Pinned config ────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOTAL_GPUS=4
export N_PROMPTS=200
export BON_N=4
export SEED=42
export BACKEND=sid
export SEARCH_REWARD=imagereward
export OUT_ROOT="/data/ygu/runs/limited_noise_N${BON_N}_$(date +%Y%m%d_%H%M%S)"

# Pre-flight (kill leftovers from prior runs on these GPUs only)
pkill -9 -f "reward_server.py" 2>/dev/null || true
pkill -9 -f "_heartbeat.sh" 2>/dev/null || true
sleep 3

echo "================================================================"
echo "LIMITED-NOISE on GPUs 0,1,2,3"
echo "  BON_N=${BON_N}   N_PROMPTS=${N_PROMPTS}"
echo "  OUT_ROOT=${OUT_ROOT}"
echo "================================================================"

bash "${SCRIPT_DIR}/run_limited_noise.sh"
