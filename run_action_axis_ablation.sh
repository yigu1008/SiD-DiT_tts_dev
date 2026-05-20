#!/usr/bin/env bash
# Ablation: do negative-prompt branching and sigma-perturbation help bon_mcts?
#
# Single-pipeline-load mode: METHODS contains 4 bon_mcts variants + baseline,
# so the SD3.5 pipeline / T5 encoders / VAE / transformer are loaded ONCE
# and reused across all 5 method runs.  This saves ~30-60s × 3 pipeline reloads
# vs. the previous "4 separate suite invocations" version.
#
# Methods (mapped in hpsv2_sd35_sid_ddp_suite.sh:953-988):
#   baseline       : no-search reference (single CFG sample)
#   bon_mcts       : current ActDiff (seed-only prescreen)              — base
#   bon_mcts_neg   : prescreen fans out over a 2-entry negative bank
#   bon_mcts_sigma : prescreen fans out over a 3-entry sigma-perturb bank
#   bon_mcts_axes  : merged — both axes fan out (cartesian, cycled mod n_seeds)
#
# Output layout (single OUT_ROOT):
#   <RUN_ROOT>/run_*/{baseline,bon_mcts,bon_mcts_neg,bon_mcts_sigma,bon_mcts_axes}/
#     {images, best_images, aggregate_ddp.json, best_images_multi_reward_aggregate.json}
#   <RUN_ROOT>/summary.tsv
#
# Caller env (from amlt yaml OR local):
#   REWARD_SERVER_URL  (preferred; HTTP reward server with imagereward)
#   BACKEND            (default sid)
#   N_PROMPTS          (default 100)
#   SEED               (default 42)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "action-axis-ablation"

CUDA_VISIBLE_DEVICES_REWARD="${CUDA_VISIBLE_DEVICES_REWARD:-4}"
CUDA_VISIBLE_DEVICES_SAMPLE="${CUDA_VISIBLE_DEVICES_SAMPLE:-5,6,7}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-/data/ygu/action_axis_ablation/run_$(date +%Y%m%d_%H%M%S)}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5118}"

BACKEND="${BACKEND:-sid}"
N_PROMPTS="${N_PROMPTS:-100}"
SEED="${SEED:-42}"
SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"

# Bank values (override via env if you want to test different banks).
# Negative bank entries are separated by '||'.  Defaults cover 4 distinct
# failure modes: empty (anchor), quality, anatomy, artifacts.
_DEFAULT_NEG_BANK='||low quality, blurry, lowres, jpeg artifacts||bad anatomy, deformed, mutated, extra limbs||watermark, signature, text, frame, cropped'
BON_MCTS_NEG_BANK_NEG="${BON_MCTS_NEG_BANK_NEG:-${_DEFAULT_NEG_BANK}}"
BON_MCTS_SIGMA_BANK_SIGMA="${BON_MCTS_SIGMA_BANK_SIGMA:--0.05 0.0 0.05}"
BON_MCTS_NEG_BANK_AXES="${BON_MCTS_NEG_BANK_AXES:-${_DEFAULT_NEG_BANK}}"
BON_MCTS_SIGMA_BANK_AXES="${BON_MCTS_SIGMA_BANK_AXES:--0.05 0.0 0.05}"
export BON_MCTS_NEG_BANK_NEG BON_MCTS_SIGMA_BANK_SIGMA BON_MCTS_NEG_BANK_AXES BON_MCTS_SIGMA_BANK_AXES

mkdir -p "${RUN_ROOT}"
SERVER_LOG="${RUN_ROOT}/reward_server.log"
REWARD_SERVER_URL="${REWARD_SERVER_URL:-http://localhost:${REWARD_SERVER_PORT}}"

# ── 1. Reward server (ImageReward only — light, fits next to sampling) ─────
if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then
    echo "[action-axis] reusing reward server at ${REWARD_SERVER_URL}"
else
    echo "[action-axis] booting ImageReward server on GPU ${CUDA_VISIBLE_DEVICES_REWARD}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_REWARD}" \
      "${PYTHON_BIN}" "${SCRIPT_DIR}/reward_server.py" \
        --port "${REWARD_SERVER_PORT}" --device cuda:0 \
        --backends imagereward \
        --image_reward_model ImageReward-v1.0 \
        > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    trap 'kill "${SERVER_PID}" >/dev/null 2>&1 || true' EXIT
    for i in $(seq 1 100); do
        if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then break; fi
        kill -0 "${SERVER_PID}" 2>/dev/null || { echo "FATAL server died"; tail -n 80 "${SERVER_LOG}" >&2; exit 1; }
        sleep 3
    done
    curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1 || { echo "FATAL not healthy"; exit 1; }
fi

# ── 2. Prompt sampling (once, shared across methods) ───────────────────────
PROMPTS_DIR="${RUN_ROOT}/_prompts"
mkdir -p "${PROMPTS_DIR}"
PROMPT_FILE="${PROMPTS_DIR}/backend_${BACKEND}.txt"
if [[ ! -f "${PROMPT_FILE}" ]]; then
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_prompts.py" \
        --n_prompts "${N_PROMPTS}" \
        --out_dir "${PROMPTS_DIR}" \
        --backends "${BACKEND}"
fi

# ── 3. Backend dispatch ────────────────────────────────────────────────────
case "${BACKEND}" in
    sid|senseflow_large)
        export SD35_BACKEND="${BACKEND}"; unset FLUX_BACKEND || true
        export STEPS=4; export BASELINE_CFG=1.0
        export CFG_SCALES="1.0 1.5 2.0 2.5"
        SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
        ;;
    sd35_base)
        export SD35_BACKEND=sd35_base; unset FLUX_BACKEND || true
        export STEPS=28; export BASELINE_CFG=4.5
        export CFG_SCALES="3.5 4.5 5.5 7.0"
        SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
        ;;
    *) echo "[action-axis] ERROR unsupported BACKEND='${BACKEND}'" >&2; exit 1 ;;
esac

# ── 4. Run all 4 bon_mcts variants + baseline in ONE suite invocation ──────
export METHODS="${METHODS:-baseline bon_mcts bon_mcts_neg bon_mcts_sigma bon_mcts_axes}"
export PROMPT_FILE
export START_INDEX=0
export END_INDEX="${N_PROMPTS}"
export SEEDS="${SEED}"
export N_SIMS=30
export BON_MCTS_N_SEEDS=8
export BON_MCTS_TOPK=2
export BON_MCTS_MIN_SIMS=8
export BON_MCTS_SIM_ALLOC=split
export BON_MCTS_REFINE_METHOD=ours_tree
export LOOKAHEAD_METHOD_MODE=rollout_tree_prior_adaptive_cfg
export UCB_C=1.0
export USE_QWEN=0
export N_VARIANTS=1
export CORRECTION_STRENGTHS="0.0"
export SAVE_BEST_IMAGES=1
export SAVE_IMAGES=1
export EVAL_BACKENDS="imagereward hpsv3"
export REWARD_BACKEND="${SEARCH_REWARD}"
export REWARD_TYPE="${SEARCH_REWARD}"
export REWARD_BACKENDS="${SEARCH_REWARD}"
export EVAL_BEST_IMAGES=1
export EVAL_ALLOW_MISSING_BACKENDS=1
export EVAL_REWARD_DEVICE=cuda
export REWARD_SERVER_URL
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_SAMPLE}"
export MCTS_KEY_STEP_COUNT=2
export OUT_ROOT="${RUN_ROOT}"

echo
echo "================================================================"
echo "[action-axis] single-pass run"
echo "  METHODS=${METHODS}"
echo "  NEG bank (neg/axes): ${BON_MCTS_NEG_BANK_NEG}"
echo "  SIGMA bank (sigma/axes): ${BON_MCTS_SIGMA_BANK_SIGMA}"
echo "  → ${RUN_ROOT}"
echo "================================================================"

bash "${SUITE}"

# ── 5. Summarize ───────────────────────────────────────────────────────────
SUMMARY="${RUN_ROOT}/summary.tsv"
printf "method\tneg_bank\tsigma_bank\tmean_search\teval_ir\teval_hpsv3\n" > "${SUMMARY}"

for method in ${METHODS}; do
    neg_show="-"; sig_show="-"
    case "${method}" in
        bon_mcts_neg)   neg_show="${BON_MCTS_NEG_BANK_NEG}" ;;
        bon_mcts_sigma) sig_show="${BON_MCTS_SIGMA_BANK_SIGMA}" ;;
        bon_mcts_axes)  neg_show="${BON_MCTS_NEG_BANK_AXES}"; sig_show="${BON_MCTS_SIGMA_BANK_AXES}" ;;
    esac
    agg=$(find "${RUN_ROOT}" -name 'aggregate_ddp.json' -path "*/${method}/*" 2>/dev/null | head -1)
    ir_eval=$(find "${RUN_ROOT}" -name 'best_images_multi_reward_aggregate.json' -path "*/${method}/*" 2>/dev/null | head -1)
    ms="" eir="" eh=""
    [[ -f "${agg}" ]] && ms=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${agg}')).get('mean_search', ''))")
    [[ -f "${ir_eval}" ]] && {
        eir=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${ir_eval}')); print(d.get('imagereward', {}).get('mean', ''))")
        eh=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${ir_eval}')); print(d.get('hpsv3', {}).get('mean', ''))")
    }
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${method}" "${neg_show}" "${sig_show}" "${ms}" "${eir}" "${eh}" >> "${SUMMARY}"
done

echo
echo "================================================================"
echo "[action-axis] DONE."
echo "  RUN_ROOT: ${RUN_ROOT}"
column -t -s $'\t' "${SUMMARY}" | head -20
