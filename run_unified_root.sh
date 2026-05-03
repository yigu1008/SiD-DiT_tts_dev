#!/usr/bin/env bash
# Single-bash driver for unified_root MCTS evaluation across backends.
# Mirrors run_cherry_pick.sh's seed-loop pattern but runs ONLY the
# `unified_root` method, on the same prompts (sampled from HPSv2+DrawBench).
#
# Required env:
#   RUN_ROOT           - output dir parent
#   REWARD_SERVER_URL  - shared reward server URL
#
# Optional env:
#   BACKENDS           (default "sid senseflow_large sd35_base")
#   N_PROMPTS          (default 100)
#   SEEDS              (default "42 43 44 45")
#   N_SIMS             (default 30)
#   PYTHON_BIN         (default python)
#   FAIL_FAST          (default 0)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "unified-root"

: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${REWARD_SERVER_URL:?REWARD_SERVER_URL must be set}"

BACKENDS="${BACKENDS:-sid senseflow_large sd35_base}"
N_PROMPTS="${N_PROMPTS:-100}"
SEEDS="${SEEDS:-42 43 44 45}"
N_SIMS="${N_SIMS:-30}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FAIL_FAST="${FAIL_FAST:-0}"

PROMPTS_DIR="${RUN_ROOT}/_prompts"
mkdir -p "${PROMPTS_DIR}"

# Shared run knobs.
export METHODS="unified_root"
export START_INDEX=0
export END_INDEX="${N_PROMPTS}"
export N_VARIANTS=1
export USE_QWEN=0
export PRECOMPUTE_REWRITES=0
export REWARDS_OVERWRITE=0
export CORRECTION_STRENGTHS="0.0"
export SAVE_BEST_IMAGES=1
export SAVE_IMAGES=0
export SAVE_VARIANTS=0
export EVAL_BACKENDS="imagereward hpsv3"
export REWARD_BACKEND="${REWARD_BACKEND:-imagereward}"
export REWARD_BACKENDS="imagereward hpsv3"

_run_one_backend() {
    local backend="$1"
    local prompt_file="${PROMPTS_DIR}/backend_${backend}.txt"

    # Sample prompts (HF online for this call).
    if [[ ! -f "${prompt_file}" ]]; then
        echo "[unified-root] sampling prompts → ${prompt_file}"
        env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
            "${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_prompts.py" \
            --n_prompts "${N_PROMPTS}" \
            --out_dir "${PROMPTS_DIR}" \
            --backends "${backend}"
    else
        echo "[unified-root] reusing ${prompt_file}"
    fi

    export PROMPT_FILE="${prompt_file}"
    export SD35_BACKEND="${backend}"
    unset FLUX_BACKEND || true

    # The unified_root algorithm uses N_SIMS as its total sim budget.
    export N_SIMS

    local seed_failed=()
    for seed in ${SEEDS}; do
        local seed_root="${RUN_ROOT}/${backend}/seed${seed}"
        mkdir -p "${seed_root}"
        echo
        echo "==== [unified-root] backend=${backend} seed=${seed} N_SIMS=${N_SIMS} → ${seed_root} ===="
        if SEED="${seed}" OUT_ROOT="${seed_root}" \
           bash "${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"; then
            echo "[unified-root] OK ${backend} seed=${seed}"
        else
            local rc=$?
            echo "[unified-root] FAIL ${backend} seed=${seed} rc=${rc}" >&2
            seed_failed+=("${seed}")
            if [[ "${FAIL_FAST}" == "1" ]]; then return "${rc}"; fi
        fi
    done

    if (( ${#seed_failed[@]} > 0 )); then
        echo "[unified-root] WARN ${backend} seed failures: ${seed_failed[*]}"
        return 1
    fi
    return 0
}

echo "[unified-root] backends: ${BACKENDS}"
echo "[unified-root] seeds:    ${SEEDS}"
echo "[unified-root] N_SIMS:   ${N_SIMS}"

backend_failed=()
for backend in ${BACKENDS}; do
    if _run_one_backend "${backend}"; then
        echo "[unified-root] DONE OK ${backend}"
    else
        rc=$?
        echo "[unified-root] FAIL ${backend} rc=${rc}" >&2
        backend_failed+=("${backend}")
        if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
    fi
done

if (( ${#backend_failed[@]} > 0 )); then
    echo "[unified-root] DONE with backend failures: ${backend_failed[*]}"
    exit 1
fi
echo "[unified-root] DONE all backends OK"
