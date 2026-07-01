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
# Usage (one shot):
#   CUDA_VISIBLE_DEVICES_REWARD=0 CUDA_VISIBLE_DEVICES_SAMPLE=1,2,3 \
#   bash run_synergy_local.sh
#
# Usage (two-step: prepare prompts+rewrites first, then run) -- use a FIXED
# RUN_ROOT so step 2 reuses step 1's artifacts:
#   RUN_ROOT=/data/ygu/synergy_local/exp1 PREPARE_ONLY=1 bash run_synergy_local.sh   # prep only
#   RUN_ROOT=/data/ygu/synergy_local/exp1 CUDA_VISIBLE_DEVICES_SAMPLE=1,2,3 bash run_synergy_local.sh  # run
#
# Outputs default under /data/ygu/synergy_local/run_<timestamp> (override RUN_ROOT).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Local config (override via env) ─────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES_REWARD="${CUDA_VISIBLE_DEVICES_REWARD:-0}"      # GPU for the reward server
CUDA_VISIBLE_DEVICES_SAMPLE="${CUDA_VISIBLE_DEVICES_SAMPLE:-1,2,3}" # GPU(s) for sampling
RUN_ROOT="${RUN_ROOT:-/data/ygu/synergy_local/run_$(date +%Y%m%d_%H%M%S)}"
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
if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "[synergy-local] PREPARE_ONLY=1 — skipping reward server (prep = prompts + Qwen rewrites only)"
elif curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then
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

# ── 2b. Pre-rewrite with Qwen ONCE (deterministic) so the +prompt cells reuse a
#        cached rewrites file instead of re-running Qwen per cell/seed. ─────────
export SYNERGY_REWRITES_FILE="${RUN_ROOT}/_prompts/rewrites_qwen.json"
if [[ ! -f "${SYNERGY_REWRITES_FILE}" ]]; then
  echo "[synergy-local] pre-computing Qwen rewrites (n_variants=3) -> ${SYNERGY_REWRITES_FILE}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_SAMPLE%%,*}" "${PYTHON_BIN}" "${SCRIPT_DIR}/precompute_sd35_rewrites.py" \
    --prompt_file "${RUN_ROOT}/_prompts/backend_${BACKEND}.txt" \
    --rewrites_file "${SYNERGY_REWRITES_FILE}" \
    --n_variants 3 --qwen_id "${QWEN_ID:-Qwen/Qwen2.5-3B-Instruct}" --device cuda:0 \
    || { echo "[synergy-local] FATAL: Qwen rewrite precompute failed -> the +prompt axis would be empty and the synergy invalid. Fix Qwen (transformers version / QWEN_ID) and retry."; exit 1; }
fi
if [[ ! -s "${SYNERGY_REWRITES_FILE}" ]]; then
  echo "[synergy-local] FATAL: ${SYNERGY_REWRITES_FILE} missing/empty after precompute — no prompt rewrites."; exit 1
fi

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "[synergy-local] PREPARE_ONLY done. Prep artifacts under ${RUN_ROOT}/_prompts:"
  echo "[synergy-local]   prompts : ${RUN_ROOT}/_prompts/backend_${BACKEND}.txt (${N_PROMPTS})"
  echo "[synergy-local]   rewrites: ${SYNERGY_REWRITES_FILE:-<none>}"
  echo "[synergy-local] Now run the search reusing them:  RUN_ROOT=${RUN_ROOT} bash run_synergy_local.sh"
  exit 0
fi

# ── 3. Shared env, then two passes into one RUN_ROOT ────────────────────────
export REWARD_SERVER_URL REWARD_API_BASE="${REWARD_SERVER_URL}"
export RUN_ROOT BACKENDS="${BACKEND}" N_PROMPTS SEEDS PYTHON_BIN
export SEARCH_REWARD REWARD_BACKEND="${SEARCH_REWARD}" REWARD_TYPE="${SEARCH_REWARD}" REWARD_BACKENDS="${SEARCH_REWARD}"
export EVAL_BACKENDS="imagereward hpsv3 pickscore hpsv2" EVAL_ALLOW_MISSING_BACKENDS=1
export USE_QWEN=1 PRECOMPUTE_REWRITES=1 SYNERGY_N_VARIANTS=3
export CUDA_VISIBLE_DEVICES_SAMPLE
# Pin sampling to the sampling GPUs so it does NOT land on the reward server's
# GPU (the reward server was launched pinned to CUDA_VISIBLE_DEVICES_REWARD).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_SAMPLE}"
export NUM_GPUS="$(echo "${CUDA_VISIBLE_DEVICES_SAMPLE}" | tr ',' '\n' | grep -c .)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Ablation cells first (cheaper @30 sims) so base/+cfg/+prompt appear early,
# then the expensive both-axes cell @60 last. Toggle either pass:
#   SKIP_FULL=1       -> run only the ablation cells (base/+cfg/+prompt)
#   SKIP_ABLATIONS=1  -> run only bon_mcts_full
if [[ "${SKIP_ABLATIONS:-0}" != "1" ]]; then
  echo "[synergy-local] === pass 1: ablations base + single-axis, N_SIMS=${N_SIMS_SINGLE} ==="
  METHODS="bon_mcts_static_cfg bon_mcts_adaptive_cfg bon_mcts_rewrite_only" N_SIMS="${N_SIMS_SINGLE}" \
    bash "${SCRIPT_DIR}/hpsv2_composite_all_backends.sh" || echo "[synergy-local] pass 1 (ablations) FAILED"
else
  echo "[synergy-local] SKIP_ABLATIONS=1 — skipping ablation cells"
fi
if [[ "${SKIP_FULL:-0}" != "1" ]]; then
  echo "[synergy-local] === pass 2: full (both axes), N_SIMS=${N_SIMS_FULL} ==="
  METHODS="bon_mcts_full" N_SIMS="${N_SIMS_FULL}" bash "${SCRIPT_DIR}/hpsv2_composite_all_backends.sh" \
    || echo "[synergy-local] pass 2 (full) FAILED"
else
  echo "[synergy-local] SKIP_FULL=1 — skipping bon_mcts_full (ablations only)"
fi

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

# ── 5. Per-prompt synergy montage: base / +cfg / +prompt / both side by side,
#       so you can SEE what each axis adds and what is lost when one is removed.
"${PYTHON_BIN}" "${SCRIPT_DIR}/make_synergy_montage.py" \
  --run_root "${RUN_ROOT}/${BACKEND}" --seed "${MONTAGE_SEED:-42}" \
  --rank_by_synergy --top "${MONTAGE_TOP:-12}" \
  --out "${RUN_ROOT}/synergy-${BACKEND}-montage.png" || echo "[synergy-local] montage skipped"

echo "[synergy-local] DONE -> ${RUN_ROOT}"
echo "[synergy-local]   montage : ${RUN_ROOT}/synergy-${BACKEND}-montage.png"
echo "[synergy-local]   bars    : ${RUN_ROOT}/synergy-${BACKEND}-ex-*.png"
