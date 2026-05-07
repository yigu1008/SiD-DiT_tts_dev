#!/usr/bin/env bash
# Local-A6000 cherry-pick across 5 (backend, search_reward) configs:
#   sid + hpsv3
#   senseflow_large + hpsv3
#   senseflow_large + imagereward
#   flux_schnell    + hpsv3
#   flux_schnell    + imagereward
#
# - One reward server boot on GPU 4 (HPSv3 + ImageReward).
# - Sampling on GPUs 5,6,7 (DDP, 3 ranks).
# - Each (backend, reward) pair gets a NON-OVERLAPPING prompt subset
#   (cherry_pick_prompts.py --tag <reward> → unique seed-offset).
# - Single seed (42), 30 prompts per config, top-12 winners kept.
# - Bumped bon_mcts compute (n_sims=60, topk=4, n_seeds=16) — fksteering /
#   dts_star / baseline use suite defaults.
#
# Override at submit time:
#   CUDA_VISIBLE_DEVICES_REWARD=4
#   CUDA_VISIBLE_DEVICES_SAMPLE=5,6,7
#   RUN_ROOT=/data/ygu/cherry_pick
#   SEEDS="42"
#   N_PROMPTS=30
#   N_WINNERS=12
#   REWARD_SERVER_PORT=5118

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Knobs (overridable) ────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES_REWARD="${CUDA_VISIBLE_DEVICES_REWARD:-4}"
CUDA_VISIBLE_DEVICES_SAMPLE="${CUDA_VISIBLE_DEVICES_SAMPLE:-5,6,7}"
RUN_ROOT="${RUN_ROOT:-/data/ygu/cherry_pick}"
SEEDS="${SEEDS:-42}"
N_PROMPTS="${N_PROMPTS:-30}"
N_WINNERS="${N_WINNERS:-12}"
N_SIMS="${N_SIMS:-30}"
BON_MCTS_TOPK="${BON_MCTS_TOPK:-2}"
BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS:-16}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5118}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${RUN_ROOT}"
SERVER_LOG="${RUN_ROOT}/reward_server.log"

# ── 1. Boot reward server on the dedicated GPU ─────────────────────────────
echo "[run5] booting reward server on GPU(s) ${CUDA_VISIBLE_DEVICES_REWARD} (port ${REWARD_SERVER_PORT})"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_REWARD}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/reward_server.py" \
    --port "${REWARD_SERVER_PORT}" --device cuda:0 \
    --backends hpsv3 imagereward \
    --image_reward_model ImageReward-v1.0 \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" >/dev/null 2>&1 || true' EXIT

# ── Wait for /health (up to 5 min) ─────────────────────────────────────────
HEALTH_OK=0
for i in $(seq 1 100); do
    if curl -s "http://localhost:${REWARD_SERVER_PORT}/health" >/dev/null 2>&1; then
        HEALTH_OK=1; break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[run5] FATAL reward server died early; tail of log:"
        tail -n 80 "${SERVER_LOG}" >&2
        exit 1
    fi
    sleep 3
done
if [[ "${HEALTH_OK}" != "1" ]]; then
    echo "[run5] FATAL reward server failed to become healthy"
    tail -n 80 "${SERVER_LOG}" >&2
    exit 1
fi
echo "[run5] reward server READY at http://localhost:${REWARD_SERVER_PORT}"

# ── 2. Run the 5 configs sequentially ──────────────────────────────────────
COMMON_ENV=(
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_SAMPLE}"
    REWARD_SERVER_URL="http://localhost:${REWARD_SERVER_PORT}"
    RUN_ROOT="${RUN_ROOT}"
    SEEDS="${SEEDS}"
    N_PROMPTS="${N_PROMPTS}"
    N_SIMS="${N_SIMS}"
    BON_MCTS_TOPK="${BON_MCTS_TOPK}"
    BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS}"
    N_WINNERS="${N_WINNERS}"
)

CONFIGS=(
    "sid             hpsv3"
    "senseflow_large hpsv3"
    "senseflow_large imagereward"
    "flux_schnell    hpsv3"
    "flux_schnell    imagereward"
)

failed=()
for line in "${CONFIGS[@]}"; do
    backend="$(awk '{print $1}' <<<"${line}")"
    reward="$(awk '{print $2}' <<<"${line}")"
    echo
    echo "================================================================"
    echo "[run5] === ${backend} × ${reward} ==="
    echo "================================================================"
    if env "${COMMON_ENV[@]}" BACKEND="${backend}" SEARCH_REWARD="${reward}" \
            bash "${SCRIPT_DIR}/run_cherry_pick_a6000.sh"; then
        echo "[run5] OK ${backend}_${reward}"
    else
        rc=$?
        echo "[run5] FAIL ${backend}_${reward} rc=${rc}" >&2
        failed+=("${backend}_${reward}")
    fi
done

# ── 3. Summary ─────────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "[run5] DONE."
echo "  RUN_ROOT: ${RUN_ROOT}"
for line in "${CONFIGS[@]}"; do
    backend="$(awk '{print $1}' <<<"${line}")"
    reward="$(awk '{print $2}' <<<"${line}")"
    winners="${RUN_ROOT}/${backend}_${reward}/_winners/winners.json"
    if [[ -f "${winners}" ]]; then
        echo "  OK   ${backend}_${reward} → ${winners}"
    else
        echo "  MISS ${backend}_${reward} (no winners.json)"
    fi
done
if (( ${#failed[@]} > 0 )); then
    echo "[run5] WARN failures: ${failed[*]}"
    exit 1
fi
