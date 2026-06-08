#!/usr/bin/env bash
# Multi-GPU helpers for 8-GPU local node (reward server on GPU 0,
# DDP sampling on GPUs 1..N-1).  Sourced by the 8-GPU launcher scripts.

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
# Force CUDA to enumerate GPUs by PCI bus id so cuda:N matches `nvidia-smi`'s
# "GPU N".  Without this, CUDA uses FASTEST_FIRST which can permute the
# mapping (e.g. cuda:0 != physical GPU 0).  Setting it BEFORE any CUDA env
# is read ensures CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 truly selects physical
# GPUs 2-7 as reported by nvidia-smi.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ── Boot reward server on the FIRST GPU of the parent's visible slice ────
# args:  $1 server_log_path   $2 (optional) reward_backends_server
# exports: REWARD_SERVER_URL, REWARD_SERVER_PID
#
# Picks the first physical GPU index from CUDA_VISIBLE_DEVICES.  E.g. with
# parent CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 the reward server lands on
# physical GPU 2 (not GPU 0).
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

    # First physical GPU index from the parent's visible list.
    local visible="${CUDA_VISIBLE_DEVICES:-0}"
    local reward_gpu="${visible%%,*}"   # e.g. "2,3,4,5,6,7" -> "2"

    echo "[mgpu] booting reward server on physical GPU ${reward_gpu} (port ${port}), backends=${reward_backends}"
    mkdir -p "$(dirname "${server_log}")"
    CUDA_VISIBLE_DEVICES="${reward_gpu}" PYTHONNOUSERSITE=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
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
    # Parse the parent's CUDA_VISIBLE_DEVICES into a list, drop the first
    # (which is owned by the reward server), assign the rest to sampling.
    local visible="${CUDA_VISIBLE_DEVICES:-0}"
    local sampling_list="${visible#*,}"     # everything after the first comma
    [[ "${sampling_list}" == "${visible}" ]] && sampling_list="${visible}"  # single-GPU fallback
    export CUDA_VISIBLE_DEVICES_SAMPLE="${sampling_list}"
    export CUDA_VISIBLE_DEVICES="${sampling_list}"
    local reward_gpu="${visible%%,*}"
    echo "[mgpu] total=${total}  reward=GPU ${reward_gpu}  sampling=GPUs ${sampling_list}  NUM_GPUS=${NUM_GPUS}"
    echo "[mgpu] CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-(unset)}  (PCI_BUS_ID makes cuda:N == nvidia-smi GPU N)"
    # Show which physical GPUs nvidia-smi will report as in-use.  Helpful when
    # debugging "why is the process on GPU 0,1 when I set CUDA_VISIBLE_DEVICES=2-7".
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "[mgpu] physical GPUs visible (after parent CUDA_VISIBLE_DEVICES filter):"
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -16 | sed 's/^/[mgpu]   /'
    fi
}
