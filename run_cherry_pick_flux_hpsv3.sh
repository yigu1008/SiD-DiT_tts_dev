#!/usr/bin/env bash
# Idiot-friendly cherry-pick: flux_schnell × HPSv3 only.
# Auto-detects/boots reward server, auto-picks a fresh prompt subset (next vN),
# runs baseline + fksteering + dts_star + bon_mcts.
#
# Just: bash run_cherry_pick_flux_hpsv3.sh
#
# Override knobs (env vars):
#   N_PROMPTS=20      (default)
#   N_WINNERS=8       (default)
#   N_SIMS=30         (bon_mcts sims; default)
#   BON_MCTS_TOPK=2   (default)
#   BON_MCTS_N_SEEDS=16
#   METHODS="baseline fksteering dts_star bon_mcts"
#   SHUFFLE_ID=v2     (pin a specific subset; default = auto-next-free vN)
#   RUN_ROOT=/data/ygu/cherry_pick (default)
#   CUDA_VISIBLE_DEVICES_REWARD=4
#   CUDA_VISIBLE_DEVICES_SAMPLE=5,6,7
#   REWARD_SERVER_PORT=5118

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VISIBLE_DEVICES_REWARD="${CUDA_VISIBLE_DEVICES_REWARD:-4}"
CUDA_VISIBLE_DEVICES_SAMPLE="${CUDA_VISIBLE_DEVICES_SAMPLE:-5,6,7}"
RUN_ROOT="${RUN_ROOT:-/data/ygu/cherry_pick}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5118}"
PYTHON_BIN="${PYTHON_BIN:-python}"
N_PROMPTS="${N_PROMPTS:-20}"
N_WINNERS="${N_WINNERS:-8}"
N_SIMS="${N_SIMS:-30}"
BON_MCTS_TOPK="${BON_MCTS_TOPK:-2}"
BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS:-16}"
METHODS="${METHODS:-baseline fksteering dts_star bon_mcts}"
SEEDS="${SEEDS:-42}"

mkdir -p "${RUN_ROOT}"
SERVER_LOG="${RUN_ROOT}/reward_server.log"
REWARD_SERVER_URL="http://localhost:${REWARD_SERVER_PORT}"

# ── 1. Reward server: reuse if healthy, boot otherwise ─────────────────────
if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then
    echo "[flux-hpsv3] reusing reward server at ${REWARD_SERVER_URL}"
else
    echo "[flux-hpsv3] booting reward server on GPU(s) ${CUDA_VISIBLE_DEVICES_REWARD} (port ${REWARD_SERVER_PORT})"
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
            echo "[flux-hpsv3] FATAL reward server died early; tail of log:" >&2
            tail -n 80 "${SERVER_LOG}" >&2
            exit 1
        fi
        if (( i % 10 == 0 )); then echo "[flux-hpsv3] waiting for server health... (${i}s)"; fi
        sleep 3
    done
    [[ "${HEALTH_OK}" == "1" ]] || { echo "[flux-hpsv3] FATAL server not healthy"; tail -n 80 "${SERVER_LOG}" >&2; exit 1; }
    echo "[flux-hpsv3] reward server READY"
fi

# ── 2. Run flux_schnell × hpsv3 (auto-shuffle subset via SHUFFLE_ID) ───────
echo
echo "================================================================"
echo "[flux-hpsv3] flux_schnell × hpsv3"
echo "================================================================"
env \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_SAMPLE}" \
    REWARD_SERVER_URL="${REWARD_SERVER_URL}" \
    RUN_ROOT="${RUN_ROOT}" \
    BACKEND=flux_schnell \
    SEARCH_REWARD=hpsv3 \
    METHODS="${METHODS}" \
    SEEDS="${SEEDS}" \
    N_PROMPTS="${N_PROMPTS}" \
    N_SIMS="${N_SIMS}" \
    BON_MCTS_TOPK="${BON_MCTS_TOPK}" \
    BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS}" \
    N_WINNERS="${N_WINNERS}" \
    ${SHUFFLE_ID+SHUFFLE_ID="${SHUFFLE_ID}"} \
    bash "${SCRIPT_DIR}/run_cherry_pick_a6000.sh"

# ── 3. Summary ─────────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "[flux-hpsv3] DONE."
echo "  Output: ${RUN_ROOT}/flux_schnell_hpsv3*/_winners/winners.json"
ls -d "${RUN_ROOT}"/flux_schnell_hpsv3* 2>/dev/null | tail -5
