#!/usr/bin/env bash
# Idiot-friendly cherry-pick: sd3.5_base × ImageReward.
# Methods (small budgets, matching budget across methods):
#   bon         (BON_N=2)
#   bon_mcts    (BON_MCTS_TOPK=4, n_seeds=8, n_sims=30 default)
#   fksteering  (SMC_K=2)
# Plus baseline as reference.
#
# Just: bash run_cherry_pick_sd35base_ir.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VISIBLE_DEVICES_REWARD="${CUDA_VISIBLE_DEVICES_REWARD:-4}"
CUDA_VISIBLE_DEVICES_SAMPLE="${CUDA_VISIBLE_DEVICES_SAMPLE:-5,6,7}"
RUN_ROOT="${RUN_ROOT:-/data/ygu/cherry_pick}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5118}"
PYTHON_BIN="${PYTHON_BIN:-python}"
N_PROMPTS="${N_PROMPTS:-20}"
N_WINNERS="${N_WINNERS:-8}"
SEEDS="${SEEDS:-42}"

# Method-specific budgets:
export BON_N="${BON_N:-2}"                      # bon: 2 samples
export SMC_K="${SMC_K:-2}"                      # fksteering: 2 particles
export BON_MCTS_TOPK="${BON_MCTS_TOPK:-4}"      # bon_mcts: refine top-4 trees
export BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS:-8}"
export N_SIMS="${N_SIMS:-30}"

# Larger MCTS noise exploration on top of bumped topk.
export MCTS_FRESH_NOISE_STEPS="${MCTS_FRESH_NOISE_STEPS:-all}"
export MCTS_FRESH_NOISE_SAMPLES="${MCTS_FRESH_NOISE_SAMPLES:-3}"
export MCTS_FRESH_NOISE_SCALE="${MCTS_FRESH_NOISE_SCALE:-1.5}"
export MCTS_FRESH_NOISE_KEY_STEPS="${MCTS_FRESH_NOISE_KEY_STEPS:-1}"

# DTS not in this method list — but if you add it later, sd35_base needs SDE
# noise to actually branch under Euler.
export DTS_SDE_NOISE_SCALE="${DTS_SDE_NOISE_SCALE:-0.1}"

mkdir -p "${RUN_ROOT}"
SERVER_LOG="${RUN_ROOT}/reward_server.log"
REWARD_SERVER_URL="http://localhost:${REWARD_SERVER_PORT}"

# ── 1. Reward server: reuse if healthy, boot otherwise ─────────────────────
if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then
    echo "[sd35base-ir] reusing reward server at ${REWARD_SERVER_URL}"
else
    echo "[sd35base-ir] booting reward server on GPU(s) ${CUDA_VISIBLE_DEVICES_REWARD} (port ${REWARD_SERVER_PORT})"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_REWARD}" \
      "${PYTHON_BIN}" "${SCRIPT_DIR}/reward_server.py" \
        --port "${REWARD_SERVER_PORT}" --device cuda:0 \
        --backends hpsv3 imagereward \
        --image_reward_model ImageReward-v1.0 \
        > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    trap 'kill "${SERVER_PID}" >/dev/null 2>&1 || true' EXIT
    HEALTH_OK=0
    for i in $(seq 1 100); do
        if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then HEALTH_OK=1; break; fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[sd35base-ir] FATAL reward server died early; tail of log:" >&2
            tail -n 80 "${SERVER_LOG}" >&2; exit 1
        fi
        if (( i % 10 == 0 )); then echo "[sd35base-ir] waiting... (${i}s)"; fi
        sleep 3
    done
    [[ "${HEALTH_OK}" == "1" ]] || { echo "[sd35base-ir] FATAL server not healthy"; tail -n 80 "${SERVER_LOG}" >&2; exit 1; }
    echo "[sd35base-ir] reward server READY"
fi

# ── 2. Run sd35_base × imagereward ─────────────────────────────────────────
echo
echo "================================================================"
echo "[sd35base-ir] sd35_base × imagereward"
echo "  bon BON_N=${BON_N}  fksteering SMC_K=${SMC_K}"
echo "  bon_mcts TOPK=${BON_MCTS_TOPK} N_SEEDS=${BON_MCTS_N_SEEDS} N_SIMS=${N_SIMS}"
echo "================================================================"
env \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_SAMPLE}" \
    REWARD_SERVER_URL="${REWARD_SERVER_URL}" \
    RUN_ROOT="${RUN_ROOT}" \
    BACKEND=sd35_base \
    SEARCH_REWARD=imagereward \
    METHODS="baseline bon fksteering bon_mcts" \
    SEEDS="${SEEDS}" \
    N_PROMPTS="${N_PROMPTS}" \
    BON_N="${BON_N}" \
    SMC_K="${SMC_K}" \
    BON_MCTS_TOPK="${BON_MCTS_TOPK}" \
    BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS}" \
    N_SIMS="${N_SIMS}" \
    N_WINNERS="${N_WINNERS}" \
    MCTS_FRESH_NOISE_STEPS="${MCTS_FRESH_NOISE_STEPS}" \
    MCTS_FRESH_NOISE_SAMPLES="${MCTS_FRESH_NOISE_SAMPLES}" \
    MCTS_FRESH_NOISE_SCALE="${MCTS_FRESH_NOISE_SCALE}" \
    MCTS_FRESH_NOISE_KEY_STEPS="${MCTS_FRESH_NOISE_KEY_STEPS}" \
    DTS_SDE_NOISE_SCALE="${DTS_SDE_NOISE_SCALE}" \
    ${SHUFFLE_ID+SHUFFLE_ID="${SHUFFLE_ID}"} \
    bash "${SCRIPT_DIR}/run_cherry_pick_a6000.sh"

# ── 3. Summary ─────────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "[sd35base-ir] DONE."
latest=$(ls -td "${RUN_ROOT}/sd35_base_imagereward"* 2>/dev/null | head -1)
[[ -n "${latest}" ]] && echo "  Output: ${latest}/_winners/"
