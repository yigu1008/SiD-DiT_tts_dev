#!/usr/bin/env bash
# Ablation: do negative-prompt branching and sigma-perturbation help bon_mcts?
#
# Cells (all use the same anchor as the smartprompt ablation winner):
#   base        : bon_mcts unchanged (seed-only prescreen axis)        — reference
#   neg_only    : prescreen fans out over a 2-entry negative bank
#   sigma_only  : prescreen fans out over a 3-entry sigma-perturb bank
#   merged      : prescreen fans out over BOTH banks (cartesian)
#
# We hold total prescreens ≈ n_seeds via cycling — wallclock is comparable to
# the base cell.  Same N_SIMS, same topk, same refine method.
#
# Outputs per cell:
#   <RUN_ROOT>/<cell>/run_*/bon_mcts/{images, best_images, aggregate_ddp.json, ...}
#   <RUN_ROOT>/summary.tsv
#
# Caller env (from amlt yaml OR local):
#   REWARD_SERVER_URL  (preferred; HTTP reward server with imagereward)
#   BACKEND            (default sid)
#   N_PROMPTS          (default 100)
#   SEED               (default 42)
#
# Local:  bash run_action_axis_ablation.sh

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

# Cell config below — NEG_BANK entries are separated by '||'
# (so individual negatives can contain spaces/commas).
# SIGMA bank is whitespace-split floats.
CELLS="${CELLS:-base neg_only sigma_only merged}"

mkdir -p "${RUN_ROOT}"
SERVER_LOG="${RUN_ROOT}/reward_server.log"
REWARD_SERVER_URL="${REWARD_SERVER_URL:-http://localhost:${REWARD_SERVER_PORT}}"

# ── 1. Reward server (ImageReward) ─────────────────────────────────────────
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

# ── 2. Prompt sampling ─────────────────────────────────────────────────────
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

# ── 4. Shared knobs (anchor on the smartprompt winner) ─────────────────────
export METHODS="baseline bon_mcts"
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

SUMMARY="${RUN_ROOT}/summary.tsv"
printf "cell\tneg_bank\tsigma_bank\tmean_search\teval_ir\teval_hpsv3\n" > "${SUMMARY}"

# ── 5. Run cells ───────────────────────────────────────────────────────────
failed=()
for cell in ${CELLS}; do
    NEG_BANK=""
    SIGMA_BANK=""
    case "${cell}" in
        base)
            ;;
        neg_only)
            NEG_BANK="||low quality, blurry, lowres"
            ;;
        sigma_only)
            SIGMA_BANK="-0.05 0.0 0.05"
            ;;
        merged)
            NEG_BANK="||low quality, blurry, lowres"
            SIGMA_BANK="-0.05 0.0 0.05"
            ;;
        *)
            echo "[action-axis] WARN unknown cell '${cell}'" >&2
            continue
            ;;
    esac

    cell_root="${RUN_ROOT}/${cell}"
    mkdir -p "${cell_root}"

    echo
    echo "================================================================"
    echo "[action-axis] cell=${cell}"
    echo "  NEG_BANK=${NEG_BANK:-<empty>}"
    echo "  SIGMA_BANK=${SIGMA_BANK:-<empty>}"
    echo "  → ${cell_root}"
    echo "================================================================"

    if env \
        BON_MCTS_NEG_BANK="${NEG_BANK}" \
        BON_MCTS_SIGMA_PERTURB_BANK="${SIGMA_BANK}" \
        OUT_ROOT="${cell_root}" \
        bash "${SUITE}"; then
        echo "[action-axis] OK ${cell}"
        agg=$(find "${cell_root}" -name 'aggregate_ddp.json' -path '*/bon_mcts/*' 2>/dev/null | head -1)
        ir_eval=$(find "${cell_root}" -name 'best_images_multi_reward_aggregate.json' -path '*/bon_mcts/*' 2>/dev/null | head -1)
        ms="" eir="" eh=""
        [[ -f "${agg}" ]] && ms=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${agg}')).get('mean_search', ''))")
        [[ -f "${ir_eval}" ]] && {
            eir=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${ir_eval}')); print(d.get('imagereward', {}).get('mean', ''))")
            eh=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${ir_eval}')); print(d.get('hpsv3', {}).get('mean', ''))")
        }
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${cell}" "${NEG_BANK:-<empty>}" "${SIGMA_BANK:-<empty>}" \
            "${ms}" "${eir}" "${eh}" \
            >> "${SUMMARY}"
    else
        rc=$?
        echo "[action-axis] FAIL ${cell} rc=${rc}" >&2
        failed+=("${cell}")
    fi
done

echo
echo "================================================================"
echo "[action-axis] DONE."
echo "  RUN_ROOT: ${RUN_ROOT}"
column -t -s $'\t' "${SUMMARY}" | head -20
(( ${#failed[@]} > 0 )) && { echo "[action-axis] WARN failures: ${failed[*]}"; exit 1; }
