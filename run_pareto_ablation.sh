#!/usr/bin/env bash
# 4-reward composite + Pareto analysis.
#
# All methods run with SEARCH_REWARD=composite_all4 (equal-weight average of
# normalized {imagereward, hpsv3, hpsv2, pickscore}).  Eval runs all 4 raw
# rewards separately so we can plot the 4-D Pareto frontier per method.
#
# Methods (5, one suite invocation):
#   baseline                  : no-search reference
#   bon_mcts_ir_only          : search on imagereward alone (current default)
#   bon_mcts_composite2       : search on composite_hpsv3_ir (the 2-reward composite)
#   bon_mcts_composite4       : search on composite_all4 (the 4-reward composite)
#   bon_mcts_full             : composite4 + dynamic CFG + dynamic prompt (full stack)
#
# Output:
#   <RUN_ROOT>/<method>/...
#   <RUN_ROOT>/summary.tsv
#   <RUN_ROOT>/pareto.png             ← 2×3 grid of pairwise Pareto frontiers
#   <RUN_ROOT>/pareto_summary.csv     ← per-method means + hypervolume
#
# Caller env:
#   REWARD_SERVER_URL  (must serve all 4: imagereward, hpsv3, hpsv2, pickscore)
#   BACKEND            (default sid)
#   N_PROMPTS          (default 100)
#   SEED               (default 42)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "pareto-ablation"

CUDA_VISIBLE_DEVICES_REWARD="${CUDA_VISIBLE_DEVICES_REWARD:-4}"
CUDA_VISIBLE_DEVICES_SAMPLE="${CUDA_VISIBLE_DEVICES_SAMPLE:-5,6,7}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-/data/ygu/pareto_ablation/run_$(date +%Y%m%d_%H%M%S)}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5118}"

BACKEND="${BACKEND:-sid}"
N_PROMPTS="${N_PROMPTS:-100}"
SEED="${SEED:-42}"

mkdir -p "${RUN_ROOT}"
SERVER_LOG="${RUN_ROOT}/reward_server.log"
REWARD_SERVER_URL="${REWARD_SERVER_URL:-http://localhost:${REWARD_SERVER_PORT}}"

# ── 1. Reward server (ALL 4 backends) ──────────────────────────────────────
if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then
    echo "[pareto] reusing reward server at ${REWARD_SERVER_URL}"
else
    echo "[pareto] booting all-4-reward server on GPU ${CUDA_VISIBLE_DEVICES_REWARD}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_REWARD}" \
      "${PYTHON_BIN}" "${SCRIPT_DIR}/reward_server.py" \
        --port "${REWARD_SERVER_PORT}" --device cuda:0 \
        --backends imagereward hpsv3 hpsv2 pickscore \
        --image_reward_model ImageReward-v1.0 \
        --pickscore_model yuvalkirstain/PickScore_v1 \
        > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    trap 'kill "${SERVER_PID}" >/dev/null 2>&1 || true' EXIT
    for i in $(seq 1 180); do
        if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then break; fi
        kill -0 "${SERVER_PID}" 2>/dev/null || { echo "FATAL server died"; tail -n 80 "${SERVER_LOG}" >&2; exit 1; }
        sleep 3
    done
    curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1 || { echo "FATAL not healthy"; exit 1; }
fi

# ── 2. Prompts ─────────────────────────────────────────────────────────────
PROMPTS_DIR="${RUN_ROOT}/_prompts"
mkdir -p "${PROMPTS_DIR}"
PROMPT_FILE="${PROMPTS_DIR}/backend_${BACKEND}.txt"
if [[ ! -f "${PROMPT_FILE}" ]]; then
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_prompts.py" \
        --n_prompts "${N_PROMPTS}" --out_dir "${PROMPTS_DIR}" --backends "${BACKEND}"
fi

# ── 3. Stage rewrites (3-level — for the bon_mcts_full cell) ───────────────
REWRITES_FILE="${PROMPTS_DIR}/stage_rewrites_3level.json"
if [[ ! -f "${REWRITES_FILE}" ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/make_stage_rewrites.py" \
        --prompt_file "${PROMPT_FILE}" --out_file "${REWRITES_FILE}" --mode 3level
fi
export SYNERGY_REWRITES_FILE="${REWRITES_FILE}"
export SYNERGY_N_VARIANTS=3

# ── 4. Backend dispatch ────────────────────────────────────────────────────
case "${BACKEND}" in
    sid|senseflow_large)
        export SD35_BACKEND="${BACKEND}"; unset FLUX_BACKEND || true
        export STEPS=4; export BASELINE_CFG=1.0
        export CFG_SCALES="1.0 1.25 1.5 1.75 2.0 2.25 2.5"
        : "${N_SIMS:=60}"; : "${BON_MCTS_N_SEEDS:=16}"; : "${BON_MCTS_TOPK:=4}"
        SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
        ;;
    sd35_base)
        export SD35_BACKEND=sd35_base; unset FLUX_BACKEND || true
        export STEPS=28; export BASELINE_CFG=4.5
        export CFG_SCALES="3.5 4.0 4.5 5.0 5.5 6.0 7.0"
        : "${N_SIMS:=120}"; : "${BON_MCTS_N_SEEDS:=16}"; : "${BON_MCTS_TOPK:=4}"
        SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
        ;;
    *) echo "[pareto] ERROR unsupported BACKEND='${BACKEND}'" >&2; exit 1 ;;
esac

# ── 5. Per-method search-reward overrides via the suite's per-method block.
#       We piggy-back on existing bon_mcts_* aliases:
#         bon_mcts             → search reward stays at $REWARD_BACKEND
#         bon_mcts_full        → adaptive cfg + rewrites
#       For ir_only / composite2 / composite4, we override REWARD_BACKEND
#       per-method via env wrappers below.
export METHODS="${METHODS:-baseline bon_mcts_ir_only bon_mcts_composite2 bon_mcts_composite4 bon_mcts_full}"

run_one_method () {
    local method="$1"
    local search_reward="$2"
    local cell_root="${RUN_ROOT}/${method}"
    mkdir -p "${cell_root}"
    # Map the alias → underlying suite method
    local suite_method
    case "${method}" in
        baseline)                suite_method="baseline" ;;
        bon_mcts_ir_only)        suite_method="bon_mcts" ;;
        bon_mcts_composite2)     suite_method="bon_mcts" ;;
        bon_mcts_composite4)     suite_method="bon_mcts" ;;
        bon_mcts_full)           suite_method="bon_mcts_full" ;;
        *) echo "[pareto] unknown method '${method}'" >&2; return 1 ;;
    esac
    echo
    echo "================================================================"
    echo "[pareto] method=${method} suite_method=${suite_method} search_reward=${search_reward}"
    echo "  → ${cell_root}"
    echo "================================================================"
    env \
        METHODS="${suite_method}" \
        REWARD_BACKEND="${search_reward}" \
        REWARD_TYPE="${search_reward}" \
        REWARD_BACKENDS="${search_reward}" \
        OUT_ROOT="${cell_root}" \
        bash "${SUITE}"
    # Move the suite's output method dir to our alias name so plotters find it
    if [[ "${suite_method}" != "${method}" ]]; then
        local src
        src=$(find "${cell_root}" -maxdepth 3 -type d -name "${suite_method}" | head -1)
        if [[ -n "${src}" ]]; then
            mv "${src}" "${src%/*}/${method}"
        fi
    fi
}

# ── 6. Shared knobs (the suite reads these per torchrun call) ──────────────
export PROMPT_FILE
export START_INDEX=0
export END_INDEX="${N_PROMPTS}"
export SEEDS="${SEED}"
export N_SIMS="${N_SIMS:-60}"
export BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS:-16}"
export BON_MCTS_TOPK="${BON_MCTS_TOPK:-4}"
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
export EVAL_BACKENDS="imagereward hpsv3 hpsv2 pickscore"
export EVAL_BEST_IMAGES=1
export EVAL_ALLOW_MISSING_BACKENDS=1
export EVAL_REWARD_DEVICE=cuda
export REWARD_SERVER_URL
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_SAMPLE}"
export MCTS_KEY_STEP_COUNT=2

# ── 7. Loop methods ────────────────────────────────────────────────────────
failed=()
for m in ${METHODS}; do
    case "${m}" in
        baseline)              sr="imagereward" ;;            # ignored; baseline doesn't search
        bon_mcts_ir_only)      sr="imagereward" ;;
        bon_mcts_composite2)   sr="composite_hpsv3_ir" ;;
        bon_mcts_composite4)   sr="composite_all4" ;;
        bon_mcts_full)         sr="composite_all4" ;;
    esac
    if run_one_method "${m}" "${sr}"; then
        echo "[pareto] OK ${m}"
    else
        rc=$?
        echo "[pareto] FAIL ${m} rc=${rc}" >&2
        failed+=("${m}")
    fi
done

# ── 8. Summary TSV + Pareto plot ───────────────────────────────────────────
SUMMARY="${RUN_ROOT}/summary.tsv"
printf "method\tsearch_reward\tmean_imagereward\tmean_hpsv3\tmean_hpsv2\tmean_pickscore\n" > "${SUMMARY}"
for m in ${METHODS}; do
    case "${m}" in
        baseline)              sr="-" ;;
        bon_mcts_ir_only)      sr="imagereward" ;;
        bon_mcts_composite2)   sr="composite_hpsv3_ir" ;;
        bon_mcts_composite4)   sr="composite_all4" ;;
        bon_mcts_full)         sr="composite_all4" ;;
    esac
    agg=$(find "${RUN_ROOT}/${m}" -name 'best_images_multi_reward_aggregate.json' 2>/dev/null | head -1)
    ir="" h3="" h2="" ps=""
    if [[ -f "${agg}" ]]; then
        ir=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${agg}')); print(d.get('imagereward', {}).get('mean', ''))")
        h3=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${agg}')); print(d.get('hpsv3', {}).get('mean', ''))")
        h2=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${agg}')); print(d.get('hpsv2', {}).get('mean', ''))")
        ps=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${agg}')); print(d.get('pickscore', {}).get('mean', ''))")
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${m}" "${sr}" "${ir}" "${h3}" "${h2}" "${ps}" >> "${SUMMARY}"
done

echo
echo "================================================================"
echo "[pareto] DONE."
echo "  RUN_ROOT: ${RUN_ROOT}"
column -t -s $'\t' "${SUMMARY}" | head -20

# Render Pareto plot (requires matplotlib)
"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_pareto.py" \
    --run_root "${RUN_ROOT}" \
    --ours bon_mcts_full \
    --out_png "${RUN_ROOT}/pareto.png" \
    --out_csv "${RUN_ROOT}/pareto_summary.csv" || true

(( ${#failed[@]} > 0 )) && { echo "[pareto] WARN failures: ${failed[*]}"; exit 1; }
