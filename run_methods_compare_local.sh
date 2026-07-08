#!/usr/bin/env bash
# Local driver for a QUALITATIVE method comparison on SD3.5 (sd35_base):
#   base (baseline) | DAS (smc) | FK-Steering (fksteering) | MCTS/ours (bon_mcts)
# Runs all four methods on the SAME prompts/seed with images saved, reusing the
# reward-host/sampler GPU split from run_synergy_local.sh. No Qwen rewrites --
# every method searches from the ORIGINAL prompt (fair comparison).
#
# Usage:
#   CUDA_VISIBLE_DEVICES_REWARD=2 CUDA_VISIBLE_DEVICES_SAMPLE=3 \
#   RUN_ROOT=/data/ygu/methods_cmp/sd35 PROMPT_SETS="$PWD/prompts_portraits.txt" \
#   bash run_methods_compare_local.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES_REWARD="${CUDA_VISIBLE_DEVICES_REWARD:-2}"
CUDA_VISIBLE_DEVICES_SAMPLE="${CUDA_VISIBLE_DEVICES_SAMPLE:-3}"
RUN_ROOT="${RUN_ROOT:-/data/ygu/methods_cmp/run_$(date +%Y%m%d_%H%M%S)}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5119}"
BACKEND="${BACKEND:-sd35_base}"
SEARCH_REWARD="${SEARCH_REWARD:-composite_3}"
SEEDS="${SEEDS:-42}"
N_SIMS="${N_SIMS:-30}"
METHODS="${METHODS:-baseline das smc fksteering bon_mcts}"
PROMPT_SETS="${PROMPT_SETS:-${SCRIPT_DIR}/prompts_portraits.txt}"

mkdir -p "${RUN_ROOT}/_prompts"
SERVER_LOG="${RUN_ROOT}/reward_server.log"
REWARD_SERVER_URL="${REWARD_SERVER_URL:-http://localhost:${REWARD_SERVER_PORT}}"

# ── 1. Reward server (composite_3 = hpsv3 + imagereward + pickscore) ──────────
if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" | grep -q hpsv3; then
  echo "[methods-cmp] reusing reward server at ${REWARD_SERVER_URL}"
else
  echo "[methods-cmp] booting reward server (hpsv3 imagereward pickscore) on GPU ${CUDA_VISIBLE_DEVICES_REWARD}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_REWARD}" "${PYTHON_BIN}" "${SCRIPT_DIR}/reward_server.py" \
    --port "${REWARD_SERVER_PORT}" --device cuda:0 \
    --backends hpsv3 imagereward pickscore \
    --image_reward_model ImageReward-v1.0 --pickscore_model yuvalkirstain/PickScore_v1 \
    > "${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  trap 'kill "${SERVER_PID}" >/dev/null 2>&1 || true' EXIT
  for _ in $(seq 1 200); do
    curl -s --max-time 3 "${REWARD_SERVER_URL}/health" | grep -q hpsv3 && break
    kill -0 "${SERVER_PID}" 2>/dev/null || { echo "FATAL reward server died"; tail -n 80 "${SERVER_LOG}" >&2; exit 1; }
    sleep 3
  done
  curl -s --max-time 3 "${REWARD_SERVER_URL}/health" | grep -q hpsv3 \
    || { echo "FATAL reward server up but not serving hpsv3 (GPU ${CUDA_VISIBLE_DEVICES_REWARD} OOM?). See ${SERVER_LOG}"; exit 1; }
fi
echo "[methods-cmp] reward server OK: $(curl -s "${REWARD_SERVER_URL}/health")"

# ── 2. Prompts (original, no rewrites — fair method comparison) ───────────────
cat ${PROMPT_SETS} > "${RUN_ROOT}/_prompts/backend_${BACKEND}.txt"
N_PROMPTS="$(grep -c . "${RUN_ROOT}/_prompts/backend_${BACKEND}.txt")"
echo "[methods-cmp] prompts=${N_PROMPTS} backend=${BACKEND} methods='${METHODS}' seeds='${SEEDS}' n_sims=${N_SIMS}"

# ── 3. Shared env, then one composite pass (all 4 methods, images saved) ──────
export REWARD_SERVER_URL REWARD_API_BASE="${REWARD_SERVER_URL}"
export RUN_ROOT BACKENDS="${BACKEND}" N_PROMPTS SEEDS PYTHON_BIN
export SEARCH_REWARD REWARD_BACKEND="${SEARCH_REWARD}" REWARD_TYPE="${SEARCH_REWARD}" REWARD_BACKENDS="${SEARCH_REWARD}"
export EVAL_BACKENDS="imagereward hpsv3 pickscore hpsv2" EVAL_ALLOW_MISSING_BACKENDS=1
export USE_QWEN=0            # method comparison: original prompt, no rewrite axis
export SAVE_IMAGES=1 SAVE_BEST_IMAGES=1 EVAL_BEST_IMAGES=1
export LIGHTWEIGHT_SD35_BASE=0   # honor our METHODS on sd35_base (no lightweight swap)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_SAMPLE}"
export NUM_GPUS="$(echo "${CUDA_VISIBLE_DEVICES_SAMPLE}" | tr ',' '\n' | grep -c .)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[methods-cmp] launching suite: METHODS='${METHODS}' N_SIMS=${N_SIMS} SAVE_IMAGES=1 LIGHTWEIGHT_SD35_BASE=0"
METHODS="${METHODS}" N_SIMS="${N_SIMS}" bash "${SCRIPT_DIR}/hpsv2_composite_all_backends.sh" \
  || echo "[methods-cmp] suite run FAILED (check logs under ${RUN_ROOT})"

# ── 4. Point at the run dir (contains the per-method subdirs) for the montage ─
RUN_DIR="$(ls -dt "${RUN_ROOT}/${BACKEND}"/seed*/run_* 2>/dev/null | head -1 || true)"
echo "[methods-cmp] DONE -> ${RUN_ROOT}"
if [[ -n "${RUN_DIR}" ]]; then
  echo "[methods-cmp] method dirs under: ${RUN_DIR}"
  echo "[methods-cmp] montage:  ${PYTHON_BIN} ${SCRIPT_DIR}/plot_methods_compare.py --run_dir '${RUN_DIR}' --seed ${SEEDS%% *}"
else
  echo "[methods-cmp] could not auto-locate run dir; look under ${RUN_ROOT}/${BACKEND}/seed*/run_*"
fi
