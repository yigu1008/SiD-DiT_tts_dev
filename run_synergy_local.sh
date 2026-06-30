#!/usr/bin/env bash
# Local driver for the bon_mcts CFG x prompt 2x2 synergy experiment
# (local equivalent of amlt/synergy_tts_scaling.yaml).
#
#   backend  : sid          reward : composite_3 (1/3 IR + 1/3 HPSv3 + 1/3 PickScore)
#   prompts  : both qual sets (prompts_qual_exp1.txt + prompts_qual_exp2.txt)
#   seeds    : 5 per prompt
#   cells    : bon_mcts_static_cfg(base) / adaptive_cfg(+cfg) /
#              rewrite_only(+prompt) / full(both)
#   sims     : full = 60, base + single-axis = 30 (compute-matched 60 vs 30+30)
#
# PREREQUISITES (local): a working python env with the sampling + reward deps,
# the sid (SD3.5-SiD) weights cached, and GPUs. Qwen is used for the +prompt
# rewrites (USE_QWEN=1). Edit the GPU / path vars below for your machine.
#
# Usage:
#   CUDA_VISIBLE_DEVICES_REWARD=0 CUDA_VISIBLE_DEVICES_SAMPLE=1,2,3 \
#   RUN_ROOT=$HOME/synergy_local bash run_synergy_local.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Local config (override via env) ─────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES_REWARD="${CUDA_VISIBLE_DEVICES_REWARD:-0}"      # GPU for the reward server
CUDA_VISIBLE_DEVICES_SAMPLE="${CUDA_VISIBLE_DEVICES_SAMPLE:-1,2,3}" # GPU(s) for sampling
RUN_ROOT="${RUN_ROOT:-$HOME/synergy_local/run_$(date +%Y%m%d_%H%M%S)}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5119}"
BACKEND="${BACKEND:-sid}"
SEARCH_REWARD="${SEARCH_REWARD:-composite_3}"
SEEDS="${SEEDS:-42 43 44 45 46}"
N_SIMS_FULL="${N_SIMS_FULL:-60}"
N_SIMS_SINGLE="${N_SIMS_SINGLE:-30}"
PROMPT_SETS="${PROMPT_SETS:-${SCRIPT_DIR}/prompts_qual_exp1.txt ${SCRIPT_DIR}/prompts_qual_exp2.txt}"

mkdir -p "${RUN_ROOT}/_prompts"
SERVER_LOG="${RUN_ROOT}/reward_server.log"
REWARD_SERVER_URL="${REWARD_SERVER_URL:-http://localhost:${REWARD_SERVER_PORT}}"

# ── 1. Reward server: hpsv3 + imagereward + pickscore (composite_3 needs all) ─
if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then
  echo "[synergy-local] reusing reward server at ${REWARD_SERVER_URL}"
else
  echo "[synergy-local] booting reward server (hpsv3 imagereward pickscore) on GPU ${CUDA_VISIBLE_DEVICES_REWARD}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_REWARD}" "${PYTHON_BIN}" "${SCRIPT_DIR}/reward_server.py" \
    --port "${REWARD_SERVER_PORT}" --device cuda:0 \
    --backends hpsv3 imagereward pickscore \
    --image_reward_model ImageReward-v1.0 --pickscore_model yuvalkirstain/PickScore_v1 \
    > "${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  trap 'kill "${SERVER_PID}" >/dev/null 2>&1 || true' EXIT
  for _ in $(seq 1 200); do
    curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1 && break
    kill -0 "${SERVER_PID}" 2>/dev/null || { echo "FATAL reward server died"; tail -n 80 "${SERVER_LOG}" >&2; exit 1; }
    sleep 3
  done
  curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1 || { echo "FATAL reward server not healthy"; exit 1; }
fi

# ── 2. Prompts: both qualitative sets, pre-placed so the orchestrator uses them
cat ${PROMPT_SETS} > "${RUN_ROOT}/_prompts/backend_${BACKEND}.txt"
N_PROMPTS="$(grep -c . "${RUN_ROOT}/_prompts/backend_${BACKEND}.txt")"
echo "[synergy-local] prompts=${N_PROMPTS} backend=${BACKEND} reward=${SEARCH_REWARD} seeds='${SEEDS}'"

# ── 3. Shared env, then two passes into one RUN_ROOT ────────────────────────
export REWARD_SERVER_URL REWARD_API_BASE="${REWARD_SERVER_URL}"
export RUN_ROOT BACKENDS="${BACKEND}" N_PROMPTS SEEDS PYTHON_BIN
export SEARCH_REWARD REWARD_BACKEND="${SEARCH_REWARD}" REWARD_TYPE="${SEARCH_REWARD}" REWARD_BACKENDS="${SEARCH_REWARD}"
export EVAL_BACKENDS="imagereward hpsv3 pickscore hpsv2" EVAL_ALLOW_MISSING_BACKENDS=1
export USE_QWEN=1 PRECOMPUTE_REWRITES=1 SYNERGY_N_VARIANTS=3
export CUDA_VISIBLE_DEVICES_SAMPLE
export NUM_GPUS="$(echo "${CUDA_VISIBLE_DEVICES_SAMPLE}" | tr ',' '\n' | grep -c .)"

echo "[synergy-local] === pass A: full (both axes), N_SIMS=${N_SIMS_FULL} ==="
METHODS="bon_mcts_full" N_SIMS="${N_SIMS_FULL}" bash "${SCRIPT_DIR}/hpsv2_composite_all_backends.sh" \
  || echo "[synergy-local] pass A FAILED"
echo "[synergy-local] === pass B: base + single-axis, N_SIMS=${N_SIMS_SINGLE} ==="
METHODS="bon_mcts_static_cfg bon_mcts_adaptive_cfg bon_mcts_rewrite_only" N_SIMS="${N_SIMS_SINGLE}" \
  bash "${SCRIPT_DIR}/hpsv2_composite_all_backends.sh" || echo "[synergy-local] pass B FAILED"

# ── 4. Summary + synergy plots (bars + qualitative image strips) ────────────
SUM="${RUN_ROOT}/synergy-${BACKEND}-summary.tsv"
"${PYTHON_BIN}" "${SCRIPT_DIR}/rebuild_summary.py" --run_root "${RUN_ROOT}/${BACKEND}" --layout all_method --out "${SUM}" \
  || { echo "[synergy-local] rebuild_summary skipped"; SUM=""; }
if [[ -n "${SUM}" ]]; then
  for ex in 0 1 2 3 4; do
    "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_synergy_2x2.py" --summary "${SUM}" --metric eval_hpsv3 \
      --cell_base bon_mcts_static_cfg --cell_cfg bon_mcts_adaptive_cfg \
      --cell_prompt bon_mcts_rewrite_only --cell_both bon_mcts_full \
      --example_strip "${ex}" --run_root "${RUN_ROOT}/${BACKEND}" \
      --out_png "${RUN_ROOT}/synergy-${BACKEND}-ex-${ex}.png" \
      --title "synergy ${BACKEND} (composite_3)" || echo "[synergy-local] plot ex${ex} skipped"
  done
fi
echo "[synergy-local] DONE -> ${RUN_ROOT}"
