#!/usr/bin/env bash
# Multi-GPU helpers for 8-GPU local node (reward server on GPU 0,
# DDP sampling on GPUs 1..N-1).  Sourced by the 8-GPU launcher scripts.

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# ── Boot reward server on GPU 0 ───────────────────────────────────────────
# args:  $1 server_log_path   $2 (optional) reward_backends_server
# exports: REWARD_SERVER_URL, REWARD_SERVER_PID
mgpu_boot_reward_server() {
    local server_log="$1"
    local reward_backends="${2:-imagereward}"
    local port="${REWARD_SERVER_PORT:-$((5400 + RANDOM % 200))}"
    REWARD_SERVER_PORT="${port}"
    REWARD_SERVER_URL="http://localhost:${port}"
    export REWARD_SERVER_PORT REWARD_SERVER_URL

    if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then
        echo "[mgpu] reusing reward server at ${REWARD_SERVER_URL}"
        REWARD_SERVER_PID=""
        return 0
    fi

    echo "[mgpu] booting reward server on GPU 0, port ${port}, backends=${reward_backends}"
    mkdir -p "$(dirname "${server_log}")"
    CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
    TOKENIZERS_PARALLELISM=false \
      "${PYTHON_BIN:-python}" -u "${SCRIPT_DIR}/reward_server.py" \
        --port "${port}" --device cuda:0 \
        --backends ${reward_backends} \
        --image_reward_model ImageReward-v1.0 \
        > "${server_log}" 2>&1 &
    REWARD_SERVER_PID=$!
    export REWARD_SERVER_PID
    # Health check (up to 5 min for HPSv3 + ImageReward to load)
    for i in $(seq 1 100); do
        if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then break; fi
        if ! kill -0 "${REWARD_SERVER_PID}" 2>/dev/null; then
            echo "[FATAL] reward server died -- last 80 lines of ${server_log}:"
            tail -n 80 "${server_log}"
            return 1
        fi
        sleep 3
    done
    if ! curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then
        echo "[FATAL] reward server did not become healthy"
        tail -n 80 "${server_log}"
        return 1
    fi
    echo "[mgpu] reward server healthy at ${REWARD_SERVER_URL} (pid=${REWARD_SERVER_PID})"
}

mgpu_kill_reward_server() {
    if [[ -n "${REWARD_SERVER_PID:-}" ]]; then
        kill "${REWARD_SERVER_PID}" 2>/dev/null || true
        wait "${REWARD_SERVER_PID}" 2>/dev/null || true
        REWARD_SERVER_PID=""
    fi
}

# ── Multi-GPU sampling setup ─────────────────────────────────────────────
# args: $1 total_gpus (default 8). Reserves GPU 0 for reward, rest for sampling.
mgpu_setup_sampling_gpus() {
    local total="${1:-${TOTAL_GPUS:-8}}"
    local sampling=$((total - 1))
    export TOTAL_GPUS="${total}"
    export NUM_GPUS="${sampling}"
    # The suite reads CUDA_VISIBLE_DEVICES at torchrun time -- give it 1..N-1
    export CUDA_VISIBLE_DEVICES_SAMPLE="$(seq -s, 1 "${sampling}")"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_SAMPLE}"
    echo "[mgpu] total=${total}  reward=GPU 0  sampling=GPUs ${CUDA_VISIBLE_DEVICES_SAMPLE}  NUM_GPUS=${NUM_GPUS}"
}
