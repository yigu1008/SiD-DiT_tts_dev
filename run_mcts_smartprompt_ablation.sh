#!/usr/bin/env bash
# Focused ablation: cfg-bank=4 + adaptive-cfg + smart prompt rewriting.
#
# Anchors (held fixed across cells unless overridden):
#   N_SIMS=60, TOPK=2, N_SEEDS=8, UCB_C=1.0
#   CFG_SCALES="1.0 1.5 2.0 2.5" (sd35-small) / "3.5 4.5 5.5 7.0" (sd35_base)
#   refine_method=mcts (vanilla), no fresh noise, no Qwen, no interp
#
# Axes ablated:
#   A. Refine + adaptive-CFG schedule
#       cfg_bank4_default            : refine=mcts (anchor; cfg bank used as flat options)
#       cfg_bank4_adaptive_lookahead : refine=ours_tree + adaptive cfg within the 4-value bank
#       cfg_bank4_adaptive_hybrid    : refine=hybrid_ut_dt + adaptive cfg within the 4-value bank
#       cfg_bank4_lookahead_static   : refine=ours_tree but NO adaptive cfg (isolate prior effect)
#   B. Prompt rewriting (3 strategies)
#       no_rewrite                   : N_VARIANTS=1 (original prompt only)
#       stage_heuristic_2level       : 2 deterministic stage suffixes (composition / details)
#       stage_heuristic_3level       : 3 stage suffixes (composition / subject / details)
#       stage_heuristic_4level       : 4 stage suffixes
#       qwen_4b_var3                 : Qwen3-4B free-form rewrites (3 variants)
#       qwen_8b_var3                 : Qwen3-8B free-form rewrites (3 variants)
#   C. Combined
#       combo_adaptive_stage3        : adaptive cfg + 3-level stage suffix
#       combo_adaptive_qwen          : adaptive cfg + Qwen-4B rewrites
#
# Outputs per cell:
#   <RUN_ROOT>/<cell>/run_*/bon_mcts/{images, best_images, aggregate_ddp.json, ...}
#   <RUN_ROOT>/summary.tsv
#
# Just: bash run_mcts_smartprompt_ablation.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "mcts-smartprompt-ablation"

# ── Env defaults ───────────────────────────────────────────────────────────
# ImageReward only — that's what we score MCTS with for this ablation.
CUDA_VISIBLE_DEVICES_REWARD="${CUDA_VISIBLE_DEVICES_REWARD:-4}"
CUDA_VISIBLE_DEVICES_SAMPLE="${CUDA_VISIBLE_DEVICES_SAMPLE:-5,6,7}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-/data/ygu/mcts_smartprompt/run_$(date +%Y%m%d_%H%M%S)}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5118}"

BACKEND="${BACKEND:-sid}"
N_PROMPTS="${N_PROMPTS:-50}"
SEED="${SEED:-42}"
SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"

# Cells: "label  REFINE  LOOKAHEAD_MODE  N_VARIANTS USE_QWEN QWEN_ID STAGE_MODE"
# STAGE_MODE = none / 2level / 3level / 4level (passes to make_stage_rewrites.py)
ALL_CELLS=(
    # ── A. Refine + adaptive CFG (cfg bank 4 baseline) ──────────────────────
    "cfg_bank4_default                mcts         -                                  1  0  Qwen/Qwen3-4B  none"
    "cfg_bank4_adaptive_lookahead     ours_tree    rollout_tree_prior_adaptive_cfg    1  0  Qwen/Qwen3-4B  none"
    "cfg_bank4_lookahead_static       ours_tree    rollout_tree_prior                 1  0  Qwen/Qwen3-4B  none"
    "cfg_bank4_adaptive_hybrid        hybrid_ut_dt rollout_tree_prior_adaptive_cfg    1  0  Qwen/Qwen3-4B  none"
    # ── B. Prompt rewriting (stage-suffix heuristic vs Qwen) ────────────────
    "stage_heuristic_2level           mcts         -                                  2  0  Qwen/Qwen3-4B  2level"
    "stage_heuristic_3level           mcts         -                                  3  0  Qwen/Qwen3-4B  3level"
    "stage_heuristic_4level           mcts         -                                  4  0  Qwen/Qwen3-4B  4level"
    "qwen_4b_var3                     mcts         -                                  3  1  Qwen/Qwen3-4B  none"
    "qwen_8b_var3                     mcts         -                                  3  1  Qwen/Qwen3-8B  none"
    # ── C. Combined: adaptive CFG + smart prompt ───────────────────────────
    "combo_adaptive_stage3            ours_tree    rollout_tree_prior_adaptive_cfg    3  0  Qwen/Qwen3-4B  3level"
    "combo_adaptive_qwen              ours_tree    rollout_tree_prior_adaptive_cfg    3  1  Qwen/Qwen3-4B  none"
    "combo_hybrid_stage3              hybrid_ut_dt rollout_tree_prior_adaptive_cfg    3  0  Qwen/Qwen3-4B  3level"
)
CELLS="${CELLS:-cfg_bank4_default cfg_bank4_adaptive_lookahead cfg_bank4_lookahead_static cfg_bank4_adaptive_hybrid stage_heuristic_2level stage_heuristic_3level stage_heuristic_4level qwen_4b_var3 qwen_8b_var3 combo_adaptive_stage3 combo_adaptive_qwen combo_hybrid_stage3}"

mkdir -p "${RUN_ROOT}"
SERVER_LOG="${RUN_ROOT}/reward_server.log"
REWARD_SERVER_URL="http://localhost:${REWARD_SERVER_PORT}"

# ── 1. Reward server (ImageReward only) ────────────────────────────────────
if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then
    echo "[smartprompt] reusing reward server at ${REWARD_SERVER_URL}"
else
    echo "[smartprompt] booting ImageReward server on GPU ${CUDA_VISIBLE_DEVICES_REWARD}"
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
        kill -0 "${SERVER_PID}" 2>/dev/null || { echo "[smartprompt] FATAL server died"; tail -n 80 "${SERVER_LOG}" >&2; exit 1; }
        sleep 3
    done
    curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1 || { echo "[smartprompt] FATAL not healthy"; exit 1; }
    echo "[smartprompt] reward server READY"
fi

# ── 2. Sample shared prompt subset ─────────────────────────────────────────
PROMPTS_DIR="${RUN_ROOT}/_prompts"
mkdir -p "${PROMPTS_DIR}"
PROMPT_FILE="${PROMPTS_DIR}/backend_${BACKEND}_${SEARCH_REWARD}.txt"
if [[ ! -f "${PROMPT_FILE}" ]]; then
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_prompts.py" \
        --n_prompts "${N_PROMPTS}" \
        --out_dir "${PROMPTS_DIR}" \
        --backends "${BACKEND}" \
        --tag "${SEARCH_REWARD}"
fi

# ── 3. Backend env ─────────────────────────────────────────────────────────
case "${BACKEND}" in
    sid|senseflow_large)
        export SD35_BACKEND="${BACKEND}"; unset FLUX_BACKEND || true
        export STEPS=4; export BASELINE_CFG=1.0
        export CFG_SCALES="1.0 1.5 2.0 2.5"     # ← 4-value bank
        : "${MCTS_KEY_STEP_COUNT:=4}"
        SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
        ;;
    sd35_base)
        export SD35_BACKEND=sd35_base; unset FLUX_BACKEND || true
        export STEPS=28; export BASELINE_CFG=4.5
        export CFG_SCALES="3.5 4.5 5.5 7.0"     # ← 4-value bank
        : "${MCTS_KEY_STEP_COUNT:=8}"
        SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
        echo "[smartprompt] WARN sd35_base is 28-step; each cell ~5-15× longer than sid"
        ;;
    *) echo "[smartprompt] ERROR unknown BACKEND='${BACKEND}'" >&2; exit 1 ;;
esac

# ── 4. Pre-generate heuristic stage rewrites (one file per stage mode) ─────
for mode in 2level 3level 4level; do
    out="${PROMPTS_DIR}/stage_rewrites_${mode}.json"
    if [[ ! -f "${out}" ]]; then
        echo "[smartprompt] generating stage rewrites: mode=${mode}"
        "${PYTHON_BIN}" "${SCRIPT_DIR}/make_stage_rewrites.py" \
            --prompt_file "${PROMPT_FILE}" \
            --out_file "${out}" \
            --mode "${mode}"
    fi
done

# ── 5. Shared run knobs ────────────────────────────────────────────────────
export METHODS="baseline bon_mcts"
export PROMPT_FILE
export START_INDEX=0
export END_INDEX="${N_PROMPTS}"
export SEEDS="${SEED}"
export N_SIMS="${N_SIMS:-60}"
export BON_MCTS_TOPK="${BON_MCTS_TOPK:-4}"
export BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS:-16}"
export BON_MCTS_SIM_ALLOC=split
export BON_MCTS_MIN_SIMS=8
export UCB_C=1.0
export USE_QWEN=0
export PRECOMPUTE_REWRITES=0
export CORRECTION_STRENGTHS="0.0"
export SAVE_BEST_IMAGES=1
export SAVE_IMAGES=1
export SAVE_VARIANTS=0
export EVAL_BACKENDS="imagereward"
export REWARD_BACKEND="${SEARCH_REWARD}"
export REWARD_TYPE="${SEARCH_REWARD}"
export REWARD_BACKENDS="${SEARCH_REWARD}"
export EVAL_BEST_IMAGES=1
export EVAL_ALLOW_MISSING_BACKENDS=1
export EVAL_REWARD_DEVICE=cuda
export REWARD_SERVER_URL
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_SAMPLE}"
export MCTS_INTERP_FAMILY=none
export MCTS_N_INTERP=0
export MCTS_KEY_STEP_COUNT="${MCTS_KEY_STEP_COUNT:-2}"

SUMMARY="${RUN_ROOT}/summary.tsv"
printf "cell\trefine\tlookahead_mode\tn_variants\tuse_qwen\tqwen_id\tstage_mode\tmean_search\teval_ir\n" > "${SUMMARY}"

# ── 6. Run cells ────────────────────────────────────────────────────────────
failed=()
for cell in ${CELLS}; do
    row=""
    for line in "${ALL_CELLS[@]}"; do
        first=$(awk '{print $1}' <<<"${line}")
        if [[ "${first}" == "${cell}" ]]; then row="${line}"; break; fi
    done
    if [[ -z "${row}" ]]; then
        echo "[smartprompt] WARN unknown cell '${cell}'" >&2
        continue
    fi

    read -r _ REFINE LOOKAHEAD_MODE CELL_N_VARIANTS CELL_USE_QWEN CELL_QWEN_ID STAGE_MODE <<<"${row}"
    [[ "${LOOKAHEAD_MODE}" == "-" ]] && LOOKAHEAD_MODE=""

    cell_root="${RUN_ROOT}/${cell}"
    mkdir -p "${cell_root}"

    # Resolve REWRITES_FILE based on stage mode / Qwen mode.
    REWRITES_FILE=""
    if [[ "${STAGE_MODE}" != "none" ]]; then
        REWRITES_FILE="${PROMPTS_DIR}/stage_rewrites_${STAGE_MODE}.json"
    elif [[ "${CELL_USE_QWEN}" == "1" ]]; then
        REWRITES_FILE="${PROMPTS_DIR}/${BACKEND}_${cell}_qwen_rewrites.json"
    fi

    echo
    echo "================================================================"
    echo "[smartprompt] cell=${cell}"
    echo "  refine=${REFINE}  lookahead_mode=${LOOKAHEAD_MODE:-default}"
    echo "  N_VARIANTS=${CELL_N_VARIANTS}  USE_QWEN=${CELL_USE_QWEN}  QWEN_ID=${CELL_QWEN_ID}"
    echo "  STAGE_MODE=${STAGE_MODE}  REWRITES_FILE=${REWRITES_FILE:-<none>}"
    echo "  → ${cell_root}"
    echo "================================================================"

    if env \
       BON_MCTS_REFINE_METHOD="${REFINE}" \
       N_VARIANTS="${CELL_N_VARIANTS}" \
       USE_QWEN="${CELL_USE_QWEN}" \
       PRECOMPUTE_REWRITES="${CELL_USE_QWEN}" \
       QWEN_ID="${CELL_QWEN_ID}" \
       REWRITES_FILE="${REWRITES_FILE}" \
       ${LOOKAHEAD_MODE:+LOOKAHEAD_METHOD_MODE="${LOOKAHEAD_MODE}"} \
       OUT_ROOT="${cell_root}" \
       bash "${SUITE}"; then
        echo "[smartprompt] OK ${cell}"
        agg=$(find "${cell_root}" -name 'aggregate_ddp.json' -path '*/bon_mcts/*' 2>/dev/null | head -1)
        ir_eval=$(find "${cell_root}" -name 'best_images_multi_reward_aggregate.json' -path '*/bon_mcts/*' 2>/dev/null | head -1)
        ms="" eir=""
        [[ -f "${agg}" ]] && ms=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${agg}')).get('mean_search', ''))")
        [[ -f "${ir_eval}" ]] && eir=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${ir_eval}')); print(d.get('imagereward', {}).get('mean', ''))")
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${cell}" "${REFINE}" "${LOOKAHEAD_MODE:-default}" \
            "${CELL_N_VARIANTS}" "${CELL_USE_QWEN}" "${CELL_QWEN_ID}" "${STAGE_MODE}" \
            "${ms}" "${eir}" \
            >> "${SUMMARY}"
    else
        rc=$?
        echo "[smartprompt] FAIL ${cell} rc=${rc}" >&2
        failed+=("${cell}")
    fi
done

echo
echo "================================================================"
echo "[smartprompt] DONE."
echo "  RUN_ROOT: ${RUN_ROOT}"
echo "  Summary:  ${SUMMARY}"
column -t -s $'\t' "${SUMMARY}" | head -40
(( ${#failed[@]} > 0 )) && { echo "[smartprompt] WARN failures: ${failed[*]}"; exit 1; }
