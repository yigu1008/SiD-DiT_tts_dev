#!/usr/bin/env bash
# Launches the noise ablation (fresh-vs-fixed) on GPUs 4-7 (reward on
# cuda:0 == physical GPU 4, sampling on cuda:1-3 == physical GPUs 5,6,7).
# Defaults pinned; edit the top to change.
#
# Just run:
#   bash run_noise_ablation_gpus4_7.sh
# Backgrounded:
#   nohup bash run_noise_ablation_gpus4_7.sh > /tmp/abl.out 2>&1 & disown

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Pinned config ────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TOTAL_GPUS=4
export N_PROMPTS=100
export N_SIMS=30
export SEED=42
export BACKEND=sid
export OUT_ROOT="/data/ygu/runs/noise_ablation_$(date +%Y%m%d_%H%M%S)"

# Pre-flight
pkill -9 -f "reward_server.py" 2>/dev/null || true
pkill -9 -f "_heartbeat.sh" 2>/dev/null || true
sleep 3

echo "================================================================"
echo "NOISE ABLATION on GPUs 4,5,6,7"
echo "  N_PROMPTS=${N_PROMPTS}   N_SIMS=${N_SIMS}"
echo "  Conditions: fixed (MCTS_FRESH_ROLLOUT_NOISE=0) vs fresh (=1)"
echo "  OUT_ROOT=${OUT_ROOT}"
echo "================================================================"

bash "${SCRIPT_DIR}/run_noise_ablation_a6000.sh"
