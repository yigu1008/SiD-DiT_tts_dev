#!/usr/bin/env bash
# A6000 end-to-end: bon_mcts run + tree/image/log visualization in one script.
#
# Stage A: run bon_mcts (default: sid, 5 prompts) -- generates rank_*.jsonl
#          with lookahead_node_logs for visualization.
# Stage B: render decision trees + per-step x_0 images + text logs.
#
# Single-GPU friendly (A6000 48GB).  Reward server boots on GPU 0 alongside
# sampling.  ImageReward is light enough to coexist with SD3.5L on 48GB.
#
# Usage:
#   bash run_actdiff_render_a6000.sh
# Override:
#   BACKEND=senseflow_large N_PROMPTS=10 bash run_actdiff_render_a6000.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "actdiff-render-a6000"
export PYTHONUNBUFFERED=1

# ── Env ────────────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
BACKEND="${BACKEND:-sid}"
N_PROMPTS="${N_PROMPTS:-5}"
SEED="${SEED:-42}"
SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"

RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/figures/actdiff_render_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${RUN_ROOT}/viz}"
mkdir -p "${RUN_ROOT}" "${OUT_ROOT}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5318}"
REWARD_SERVER_URL="${REWARD_SERVER_URL:-http://localhost:${REWARD_SERVER_PORT}}"

# A6000 memory knobs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OFFLOAD_TEXT_ENCODER_AFTER_ENCODE="${OFFLOAD_TEXT_ENCODER_AFTER_ENCODE:-1}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"

echo "================================================================"
echo "actdiff-render-a6000"
echo "  BACKEND        = ${BACKEND}"
echo "  N_PROMPTS      = ${N_PROMPTS}"
echo "  SEED           = ${SEED}"
echo "  SEARCH_REWARD  = ${SEARCH_REWARD}"
echo "  RUN_ROOT       = ${RUN_ROOT}"
echo "  OUT_ROOT       = ${OUT_ROOT}"
echo "  CUDA_DEVICE    = ${CUDA_DEVICE}"
echo "================================================================"

# ── Reward server (single-GPU shares with sampling — ImageReward is light) ─
SERVER_LOG="${RUN_ROOT}/reward_server.log"
if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then
    echo "[a6000] reusing reward server at ${REWARD_SERVER_URL}"
else
    echo "[a6000] booting ImageReward server (shares GPU ${CUDA_DEVICE})"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
      "${PYTHON_BIN}" "${SCRIPT_DIR}/reward_server.py" \
        --port "${REWARD_SERVER_PORT}" --device cuda:0 \
        --backends imagereward --image_reward_model ImageReward-v1.0 \
        > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    trap 'kill "${SERVER_PID}" >/dev/null 2>&1 || true' EXIT
    for i in $(seq 1 60); do
        if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then break; fi
        kill -0 "${SERVER_PID}" 2>/dev/null || { echo "[FATAL] server died"; tail -n 50 "${SERVER_LOG}"; exit 1; }
        sleep 3
    done
fi
export REWARD_SERVER_URL

# ── Sample prompts ─────────────────────────────────────────────────────────
# If caller set PROMPT_FILE in the env (e.g. pointing at dpg_bench_prompts.txt),
# honor it; otherwise auto-generate via cherry_pick_prompts.py.
PROMPT_FILE_OVERRIDE="${PROMPT_FILE:-}"
PROMPTS_DIR="${RUN_ROOT}/_prompts"
mkdir -p "${PROMPTS_DIR}"
if [[ -n "${PROMPT_FILE_OVERRIDE}" && -f "${PROMPT_FILE_OVERRIDE}" ]]; then
    PROMPT_FILE="${PROMPT_FILE_OVERRIDE}"
    echo "[a6000] using user-provided PROMPT_FILE=${PROMPT_FILE}"
    if [[ "${N_PROMPTS}" -gt 0 ]]; then
        TRUNC="${PROMPTS_DIR}/backend_${BACKEND}.txt"
        head -n "${N_PROMPTS}" "${PROMPT_FILE}" > "${TRUNC}"
        PROMPT_FILE="${TRUNC}"
        echo "[a6000] truncated to first ${N_PROMPTS} prompts -> ${PROMPT_FILE}"
    fi
else
    if [[ -n "${PROMPT_FILE_OVERRIDE}" ]]; then
        echo "[a6000] WARNING: PROMPT_FILE=${PROMPT_FILE_OVERRIDE} does NOT exist on disk -- ignoring and falling back to cherry_pick_prompts.py."
    fi
    PROMPT_FILE="${PROMPTS_DIR}/backend_${BACKEND}.txt"
    if [[ ! -f "${PROMPT_FILE}" ]]; then
        "${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_prompts.py" \
            --n_prompts "${N_PROMPTS}" --out_dir "${PROMPTS_DIR}" \
            --backends "${BACKEND}" || \
        {  # local fallback if HF unreachable
            cp "${SCRIPT_DIR}/hpsv2_subset.txt" "${PROMPT_FILE}" 2>/dev/null || \
                { echo "[FATAL] no prompts available"; exit 1; }
            head -n "${N_PROMPTS}" "${PROMPT_FILE}" > "${PROMPT_FILE}.tmp" && mv "${PROMPT_FILE}.tmp" "${PROMPT_FILE}"
        }
    fi
fi

# ── Stage A: bon_mcts on the chosen backend ────────────────────────────────
case "${BACKEND}" in
    sid|senseflow_large)
        export SD35_BACKEND="${BACKEND}"; unset FLUX_BACKEND || true
        export STEPS=4; export BASELINE_CFG=1.0
        export CFG_SCALES="1.0 1.25 1.5 1.75 2.0 2.25 2.5"
        : "${MCTS_KEY_STEP_COUNT:=4}"
        SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
        ;;
    sd35_base)
        export SD35_BACKEND=sd35_base; unset FLUX_BACKEND || true
        export STEPS=28; export BASELINE_CFG=4.5
        export CFG_SCALES="3.5 4.0 4.5 5.0 5.5 6.0 7.0"
        : "${MCTS_KEY_STEP_COUNT:=8}"
        SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
        ;;
    *) echo "[FATAL] unsupported BACKEND=${BACKEND}"; exit 1 ;;
esac

export METHODS="${METHODS:-bon_mcts}"
export PROMPT_FILE
export START_INDEX=0
export END_INDEX="${N_PROMPTS}"
export SEEDS="${SEED}"
export N_SIMS="${N_SIMS:-30}"         # smaller on A6000 to stay fast
export BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS:-8}"
export BON_MCTS_TOPK="${BON_MCTS_TOPK:-2}"
export BON_MCTS_MIN_SIMS=8
export BON_MCTS_SIM_ALLOC=split
export BON_MCTS_REFINE_METHOD=ours_tree
export LOOKAHEAD_METHOD_MODE=rollout_tree_prior_adaptive_cfg
export N_VARIANTS=1
export USE_QWEN=0
export PRECOMPUTE_REWRITES=0
export CORRECTION_STRENGTHS="0.0"
export UCB_C=1.0
export SAVE_BEST_IMAGES=1
export SAVE_IMAGES=1
export EVAL_BACKENDS="imagereward"
export REWARD_BACKEND="${SEARCH_REWARD}"
export REWARD_TYPE="${SEARCH_REWARD}"
export REWARD_BACKENDS="${SEARCH_REWARD}"
export EVAL_BEST_IMAGES=1
export EVAL_ALLOW_MISSING_BACKENDS=1
export EVAL_REWARD_DEVICE=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export OUT_ROOT="${RUN_ROOT}"
export NUM_GPUS=1

echo
echo "[a6000] STAGE A: running bon_mcts (backend=${BACKEND}, N=${N_PROMPTS})"
bash "${SUITE}" || echo "[a6000] WARN suite exited non-zero; continuing to viz"

# ── Stage B: visualize ──────────────────────────────────────────────────────
echo
echo "[a6000] STAGE B: rendering trees + step images + logs"
export PROMPT_RANGE="0:${N_PROMPTS}"

# Stage B.1: decision trees
"${PYTHON_BIN}" "${SCRIPT_DIR}/render_trees_batch.py" \
    --run_root "${RUN_ROOT}" --method bon_mcts \
    --prompt_range "${PROMPT_RANGE}" \
    --out_dir "${OUT_ROOT}/${BACKEND}" \
    --title_prefix "ActDiff (${BACKEND})" \
    --workers 4 || true

# Stage B.2: per-step x_0 images
"${PYTHON_BIN}" "${SCRIPT_DIR}/replay_winner_step_images.py" \
    --run_root "${RUN_ROOT}" --method bon_mcts --backend "${BACKEND}" \
    --prompt_range "${PROMPT_RANGE}" \
    --out_dir "${OUT_ROOT}/${BACKEND}_step_images" \
    --height 1024 --width 1024 || true

# Stage B.3: text logs
"${PYTHON_BIN}" "${SCRIPT_DIR}/dump_winner_log.py" \
    --run_root "${RUN_ROOT}" --method bon_mcts \
    --prompt_range "${PROMPT_RANGE}" \
    --out_dir "${OUT_ROOT}/${BACKEND}_logs" \
    --combined "${OUT_ROOT}/${BACKEND}_logs/_all.txt" || true

echo
echo "================================================================"
echo "ALL DONE."
echo "  bon_mcts run:    ${RUN_ROOT}/"
echo "  decision trees:  ${OUT_ROOT}/${BACKEND}/*.png"
echo "  step images:     ${OUT_ROOT}/${BACKEND}_step_images/prompt_NNNN/"
echo "  text logs:       ${OUT_ROOT}/${BACKEND}_logs/*.txt"
echo "================================================================"
ls -la "${OUT_ROOT}/${BACKEND}" 2>/dev/null | head -10
