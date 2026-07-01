#!/usr/bin/env bash
# Local driver for the fixed (cfg x prompt-variant) reward-rectangle grid.
# Splits reward host and sampler onto SEPARATE GPUs (like the suite / synergy
# driver): reward server on CUDA_VISIBLE_DEVICES_REWARD, grid sampler on
# CUDA_VISIBLE_DEVICES_SAMPLE. On a shared node, point both at FREE GPUs
# (check: nvidia-smi --query-gpu=index,memory.free --format=csv).
#
# Usage:
#   CUDA_VISIBLE_DEVICES_REWARD=2 CUDA_VISIBLE_DEVICES_SAMPLE=3 \
#   bash run_cfg_prompt_grid_local.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Config (override via env) ───────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES_REWARD="${CUDA_VISIBLE_DEVICES_REWARD:-2}"   # GPU for reward server (FREE)
CUDA_VISIBLE_DEVICES_SAMPLE="${CUDA_VISIBLE_DEVICES_SAMPLE:-3}"   # GPU for the grid sampler (FREE)
RUN_ROOT="${RUN_ROOT:-/data/ygu/cfg_prompt_grid/run_$(date +%Y%m%d_%H%M%S)}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5119}"
BACKEND="${BACKEND:-sid}"
STEPS="${STEPS:-4}"
CFG_SCALES="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5}"
BASELINE_CFG="${BASELINE_CFG:-1.0}"
N_VARIANTS="${N_VARIANTS:-3}"
SEARCH_REWARD="${SEARCH_REWARD:-composite_3}"
GRID_SEEDS="${GRID_SEEDS:-42}"
GRID_END="${GRID_END:-}"                          # limit #prompts (empty = all)
GRID_SAVE_IMAGES="${GRID_SAVE_IMAGES:-0}"         # 1 -> also dump each cell image
PROMPT_SETS="${PROMPT_SETS:-${SCRIPT_DIR}/prompts_qual_exp1.txt ${SCRIPT_DIR}/prompts_qual_exp2.txt}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen3-4B}"

mkdir -p "${RUN_ROOT}/_prompts"
SERVER_LOG="${RUN_ROOT}/reward_server.log"
REWARD_SERVER_URL="${REWARD_SERVER_URL:-http://localhost:${REWARD_SERVER_PORT}}"

# ── 1. Reward host on its OWN GPU (hpsv3 + imagereward + pickscore) ──────────
if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" | grep -q hpsv3; then
  echo "[grid-local] reusing reward server at ${REWARD_SERVER_URL} (serves hpsv3)"
else
  echo "[grid-local] booting reward server on GPU ${CUDA_VISIBLE_DEVICES_REWARD} (hpsv3 imagereward pickscore)"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_REWARD}" "${PYTHON_BIN}" "${SCRIPT_DIR}/reward_server.py" \
    --port "${REWARD_SERVER_PORT}" --device cuda:0 \
    --backends hpsv3 imagereward pickscore \
    --image_reward_model ImageReward-v1.0 --pickscore_model yuvalkirstain/PickScore_v1 \
    > "${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  trap 'kill "${SERVER_PID}" >/dev/null 2>&1 || true' EXIT
  for _ in $(seq 1 200); do
    if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" | grep -q hpsv3; then break; fi
    kill -0 "${SERVER_PID}" 2>/dev/null || { echo "FATAL reward server died"; tail -n 100 "${SERVER_LOG}" >&2; exit 1; }
    sleep 3
  done
  if ! curl -s --max-time 3 "${REWARD_SERVER_URL}/health" | grep -q hpsv3; then
    echo "FATAL reward server up but not serving hpsv3 (likely GPU ${CUDA_VISIBLE_DEVICES_REWARD} OOM). Check ${SERVER_LOG}" >&2
    tail -n 60 "${SERVER_LOG}" >&2; exit 1
  fi
fi
echo "[grid-local] reward server OK: $(curl -s "${REWARD_SERVER_URL}/health")"

# ── 2. Prompts + Qwen rewrites (on the sampler GPU) ─────────────────────────
PROMPT_FILE="${RUN_ROOT}/_prompts/backend_${BACKEND}.txt"
[[ -f "${PROMPT_FILE}" ]] || cat ${PROMPT_SETS} > "${PROMPT_FILE}"
REWRITES_FILE="${RUN_ROOT}/_prompts/rewrites_qwen.json"
if [[ ! -f "${REWRITES_FILE}" ]]; then
  echo "[grid-local] pre-computing Qwen rewrites (n_variants=${N_VARIANTS}) -> ${REWRITES_FILE}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_SAMPLE%%,*}" "${PYTHON_BIN}" "${SCRIPT_DIR}/precompute_sd35_rewrites.py" \
    --prompt_file "${PROMPT_FILE}" --rewrites_file "${REWRITES_FILE}" \
    --n_variants "${N_VARIANTS}" --qwen_id "${QWEN_ID}" --device cuda:0 \
    || { echo "[grid-local] rewrite precompute failed"; exit 1; }
fi

# ── 3. Grid sampler on the sampler GPU, scoring via the reward server ────────
echo "[grid-local] running grid on GPU ${CUDA_VISIBLE_DEVICES_SAMPLE} -> ${RUN_ROOT}"
REWARD_SERVER_URL="${REWARD_SERVER_URL}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_SAMPLE}" \
GRID_SEEDS="${GRID_SEEDS}" GRID_END="${GRID_END}" GRID_SAVE_IMAGES="${GRID_SAVE_IMAGES}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_cfg_prompt_grid.py" \
  --backend "${BACKEND}" --steps "${STEPS}" \
  --prompt_file "${PROMPT_FILE}" --rewrites_file "${REWRITES_FILE}" \
  --n_variants "${N_VARIANTS}" --cfg_scales ${CFG_SCALES} --baseline_cfg "${BASELINE_CFG}" \
  --reward_backend "${SEARCH_REWARD}" --out_dir "${RUN_ROOT}"

# ── 4. Reward-rectangle plot ────────────────────────────────────────────────
"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_cfg_prompt_grid.py" \
  --grid_csv "${RUN_ROOT}/cfg_prompt_grid.csv" --baseline_cfg "${BASELINE_CFG}" \
  --rank_by_synergy --top "${PLOT_TOP:-12}" --out "${RUN_ROOT}/cfg_prompt_rectangles.png" \
  || echo "[grid-local] plot skipped"

echo "[grid-local] DONE -> ${RUN_ROOT}"
echo "[grid-local]   grid csv : ${RUN_ROOT}/cfg_prompt_grid.csv"
echo "[grid-local]   rectangles: ${RUN_ROOT}/cfg_prompt_rectangles.png"
