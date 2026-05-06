#!/usr/bin/env bash
# Cherry-pick on flux_schnell × {hpsv3, imagereward}, single seed (42).
# Mirrors run_cherry_pick_5configs.sh but narrowed to flux only.
#
# Layout:
#   GPU 4         → reward server (HPSv3 + ImageReward)
#   GPUs 5,6,7    → sampling (DDP, 3 ranks)
#
# Each (backend, reward) pair samples a non-overlapping prompt subset.
# Bumped bon_mcts knobs are NOT applied here — defaults match the cluster
# parity profile (n_sims=30, topk=2, n_seeds=8). Override at submit time.
#
# Override knobs:
#   CUDA_VISIBLE_DEVICES_REWARD=4
#   CUDA_VISIBLE_DEVICES_SAMPLE=5,6,7
#   RUN_ROOT=/data/ygu/cherry_pick
#   SEEDS="42"
#   N_PROMPTS=30
#   N_WINNERS=12
#   N_SIMS=30 BON_MCTS_TOPK=2 BON_MCTS_N_SEEDS=8
#   REWARD_SERVER_PORT=5118
#   SKIP_SERVER=1   ← if a reward server is already up at REWARD_SERVER_URL
#   REWARD_SERVER_URL=http://localhost:5118  (used when SKIP_SERVER=1)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VISIBLE_DEVICES_REWARD="${CUDA_VISIBLE_DEVICES_REWARD:-4}"
CUDA_VISIBLE_DEVICES_SAMPLE="${CUDA_VISIBLE_DEVICES_SAMPLE:-5,6,7}"
RUN_ROOT="${RUN_ROOT:-/data/ygu/cherry_pick}"
SEEDS="${SEEDS:-42}"
N_PROMPTS="${N_PROMPTS:-30}"
N_WINNERS="${N_WINNERS:-12}"
N_SIMS="${N_SIMS:-30}"
BON_MCTS_TOPK="${BON_MCTS_TOPK:-2}"
BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS:-8}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5118}"
SKIP_SERVER="${SKIP_SERVER:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${RUN_ROOT}"

# ── 1. Reward server (skip if SKIP_SERVER=1 and one is already healthy) ────
if [[ "${SKIP_SERVER}" != "1" ]]; then
    SERVER_LOG="${RUN_ROOT}/reward_server.log"
    echo "[flux] booting reward server on GPU(s) ${CUDA_VISIBLE_DEVICES_REWARD} (port ${REWARD_SERVER_PORT})"
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
        if curl -s "http://localhost:${REWARD_SERVER_PORT}/health" >/dev/null 2>&1; then
            HEALTH_OK=1; break
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[flux] FATAL reward server died early; tail of log:" >&2
            tail -n 80 "${SERVER_LOG}" >&2
            exit 1
        fi
        sleep 3
    done
    [[ "${HEALTH_OK}" == "1" ]] || { echo "[flux] FATAL server not healthy"; tail -n 80 "${SERVER_LOG}" >&2; exit 1; }
    echo "[flux] reward server READY"
    REWARD_SERVER_URL="http://localhost:${REWARD_SERVER_PORT}"
else
    echo "[flux] skipping server boot (SKIP_SERVER=1); using REWARD_SERVER_URL=${REWARD_SERVER_URL:-http://localhost:${REWARD_SERVER_PORT}}"
    REWARD_SERVER_URL="${REWARD_SERVER_URL:-http://localhost:${REWARD_SERVER_PORT}}"
    if ! curl -s "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then
        echo "[flux] FATAL: REWARD_SERVER_URL=${REWARD_SERVER_URL} not healthy"
        exit 1
    fi
fi

# ── 2. Run flux × {hpsv3, imagereward} ─────────────────────────────────────
COMMON_ENV=(
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_SAMPLE}"
    REWARD_SERVER_URL="${REWARD_SERVER_URL}"
    RUN_ROOT="${RUN_ROOT}"
    SEEDS="${SEEDS}"
    N_PROMPTS="${N_PROMPTS}"
    N_SIMS="${N_SIMS}"
    BON_MCTS_TOPK="${BON_MCTS_TOPK}"
    BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS}"
    N_WINNERS="${N_WINNERS}"
)

failed=()
for reward in hpsv3 imagereward; do
    echo
    echo "================================================================"
    echo "[flux] === flux_schnell × ${reward} ==="
    echo "================================================================"
    if env "${COMMON_ENV[@]}" BACKEND=flux_schnell SEARCH_REWARD="${reward}" \
            bash "${SCRIPT_DIR}/run_cherry_pick_a6000.sh"; then
        echo "[flux] OK flux_schnell_${reward}"
    else
        rc=$?
        echo "[flux] FAIL flux_schnell_${reward} rc=${rc}" >&2
        failed+=("flux_schnell_${reward}")
    fi
done

# ── 3. Summary ─────────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "[flux] DONE."
echo "  RUN_ROOT: ${RUN_ROOT}"
for reward in hpsv3 imagereward; do
    winners="${RUN_ROOT}/flux_schnell_${reward}/_winners/winners.json"
    if [[ -f "${winners}" ]]; then
        echo "  OK   flux_schnell_${reward} → ${winners}"
    else
        echo "  MISS flux_schnell_${reward} (no winners.json)"
    fi
done
if (( ${#failed[@]} > 0 )); then
    echo "[flux] WARN failures: ${failed[*]}"
    exit 1
fi
