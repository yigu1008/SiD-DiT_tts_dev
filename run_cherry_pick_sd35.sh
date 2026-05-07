#!/usr/bin/env bash
# Idiot-friendly cherry-pick: SD3.5 backends × {ImageReward, HPSv3}.
# Backends: sid + senseflow_large (4-step distilled).
#   (sd35_base 28-step is excluded by default — too slow on a single A6000;
#    add via BACKENDS env override if you really want it.)
#
# Auto-detects/boots reward server, auto-picks a fresh prompt subset
# (next-free vN), runs baseline + fksteering + dts_star + bon_mcts.
#
# bon_mcts is bumped with **larger fresh-noise exploration** so the MCTS
# tree branches on noise as well as cfg/variant:
#   MCTS_FRESH_NOISE_STEPS=all     (every step gets noise candidates)
#   MCTS_FRESH_NOISE_SAMPLES=3     (3 noise candidates per step)
#   MCTS_FRESH_NOISE_SCALE=1.5     (~1.5× scale of additive noise)
#
# Just: bash run_cherry_pick_sd35.sh
#
# Override knobs:
#   N_PROMPTS=20      (default)
#   N_WINNERS=8       (default)
#   N_SIMS=30
#   BON_MCTS_TOPK=2
#   BON_MCTS_N_SEEDS=16
#   BACKENDS="sid senseflow_large"     # add sd35_base if you want
#   REWARDS="hpsv3 imagereward"         # narrow if you want
#   SHUFFLE_ID=v2     (pin a specific subset; default = next-free vN)

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
BACKENDS="${BACKENDS:-sid senseflow_large}"
REWARDS="${REWARDS:-hpsv3 imagereward}"

# ── Bumped MCTS noise exploration ──────────────────────────────────────────
# These propagate through run_cherry_pick_a6000.sh → suite → sampler.
export MCTS_FRESH_NOISE_STEPS="${MCTS_FRESH_NOISE_STEPS:-all}"
export MCTS_FRESH_NOISE_SAMPLES="${MCTS_FRESH_NOISE_SAMPLES:-3}"
export MCTS_FRESH_NOISE_SCALE="${MCTS_FRESH_NOISE_SCALE:-1.5}"
export MCTS_FRESH_NOISE_KEY_STEPS="${MCTS_FRESH_NOISE_KEY_STEPS:-1}"

mkdir -p "${RUN_ROOT}"
SERVER_LOG="${RUN_ROOT}/reward_server.log"
REWARD_SERVER_URL="http://localhost:${REWARD_SERVER_PORT}"

# ── 1. Reward server: reuse if healthy, boot otherwise ─────────────────────
if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then
    echo "[sd35] reusing reward server at ${REWARD_SERVER_URL}"
else
    echo "[sd35] booting reward server on GPU(s) ${CUDA_VISIBLE_DEVICES_REWARD} (port ${REWARD_SERVER_PORT})"
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
            echo "[sd35] FATAL reward server died early; tail of log:" >&2
            tail -n 80 "${SERVER_LOG}" >&2
            exit 1
        fi
        if (( i % 10 == 0 )); then echo "[sd35] waiting for server health... (${i}s)"; fi
        sleep 3
    done
    [[ "${HEALTH_OK}" == "1" ]] || { echo "[sd35] FATAL server not healthy"; tail -n 80 "${SERVER_LOG}" >&2; exit 1; }
    echo "[sd35] reward server READY"
fi

# ── 2. Run all (backend, reward) configs sequentially ──────────────────────
COMMON_ENV=(
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_SAMPLE}"
    REWARD_SERVER_URL="${REWARD_SERVER_URL}"
    RUN_ROOT="${RUN_ROOT}"
    METHODS="${METHODS}"
    SEEDS="${SEEDS}"
    N_PROMPTS="${N_PROMPTS}"
    N_SIMS="${N_SIMS}"
    BON_MCTS_TOPK="${BON_MCTS_TOPK}"
    BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS}"
    N_WINNERS="${N_WINNERS}"
    MCTS_FRESH_NOISE_STEPS="${MCTS_FRESH_NOISE_STEPS}"
    MCTS_FRESH_NOISE_SAMPLES="${MCTS_FRESH_NOISE_SAMPLES}"
    MCTS_FRESH_NOISE_SCALE="${MCTS_FRESH_NOISE_SCALE}"
    MCTS_FRESH_NOISE_KEY_STEPS="${MCTS_FRESH_NOISE_KEY_STEPS}"
)

failed=()
for backend in ${BACKENDS}; do
    for reward in ${REWARDS}; do
        echo
        echo "================================================================"
        echo "[sd35] === ${backend} × ${reward} ==="
        echo "================================================================"
        if env "${COMMON_ENV[@]}" BACKEND="${backend}" SEARCH_REWARD="${reward}" \
               ${SHUFFLE_ID+SHUFFLE_ID="${SHUFFLE_ID}"} \
               bash "${SCRIPT_DIR}/run_cherry_pick_a6000.sh"; then
            echo "[sd35] OK ${backend}_${reward}"
        else
            rc=$?
            echo "[sd35] FAIL ${backend}_${reward} rc=${rc}" >&2
            failed+=("${backend}_${reward}")
        fi
    done
done

# ── 3. Summary ─────────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "[sd35] DONE."
echo "  RUN_ROOT: ${RUN_ROOT}"
for backend in ${BACKENDS}; do
    for reward in ${REWARDS}; do
        # Find the most recent matching dir (vN suffix may be auto-picked).
        latest=$(ls -td "${RUN_ROOT}/${backend}_${reward}"* 2>/dev/null | head -1)
        if [[ -n "${latest}" && -f "${latest}/_winners/winners.json" ]]; then
            echo "  OK   ${backend}_${reward} → ${latest}/_winners/"
        else
            echo "  MISS ${backend}_${reward}"
        fi
    done
done
if (( ${#failed[@]} > 0 )); then
    echo "[sd35] WARN failures: ${failed[*]}"
    exit 1
fi
