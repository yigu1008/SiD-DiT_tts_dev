#!/usr/bin/env bash
# All method-axis ablations in ONE suite invocation (one pipeline load).
#
# Methods covered:
#   baseline                  : no-search reference
#   greedy_prompt             : step-by-step argmax over rewrites (CFG fixed) — #8
#   bon_mcts                  : current ActDiff (Ours) — anchor
#   bon_mcts_static_cfg       : refine=mcts, no adaptive cfg, no rewrite        — 2×2 cell (no/no)
#   bon_mcts_adaptive_cfg     : refine=ours_tree + adaptive cfg, no rewrite     — 2×2 cell (yes/no)
#   bon_mcts_rewrite_only     : refine=mcts, 3-level stage rewrites             — 2×2 cell (no/yes)
#   bon_mcts_full             : ours_tree + adaptive + 3-level rewrites         — 2×2 cell (yes/yes)
#   bon_mcts_neg              : action-axis: negative-prompt branching
#   bon_mcts_sigma            : action-axis: sigma-schedule perturbation
#   bon_mcts_axes             : action-axis: merged neg + sigma
#
# Total: 10 methods, one suite invocation.  Wallclock dominated by the 8
# bon_mcts variants — each is ~equal to anchor bon_mcts wallclock.
#
# After completion:
#   - run_action_axis_ablation can be derived from {baseline, bon_mcts,
#     bon_mcts_neg, bon_mcts_sigma, bon_mcts_axes}
#   - 2×2 synergy plot: python plot_synergy_2x2.py --summary <RUN_ROOT>/summary.tsv
#   - greedy_prompt vs MCTS-over-prompt: bon_mcts_rewrite_only - greedy_prompt
#
# Caller env:
#   REWARD_SERVER_URL  (optional)
#   BACKEND            (default sid)
#   N_PROMPTS          (default 100)
#   SEED               (default 42)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "all-method-ablations"

CUDA_VISIBLE_DEVICES_REWARD="${CUDA_VISIBLE_DEVICES_REWARD:-4}"
CUDA_VISIBLE_DEVICES_SAMPLE="${CUDA_VISIBLE_DEVICES_SAMPLE:-5,6,7}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-/data/ygu/all_method_ablations/run_$(date +%Y%m%d_%H%M%S)}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5118}"

BACKEND="${BACKEND:-sid}"
N_PROMPTS="${N_PROMPTS:-100}"
SEED="${SEED:-42}"
SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"

# Banks (override via env to customize).
_DEFAULT_NEG_BANK='||low quality, blurry, lowres, jpeg artifacts||bad anatomy, deformed, mutated, extra limbs||watermark, signature, text, frame, cropped'
BON_MCTS_NEG_BANK_NEG="${BON_MCTS_NEG_BANK_NEG:-${_DEFAULT_NEG_BANK}}"
BON_MCTS_SIGMA_BANK_SIGMA="${BON_MCTS_SIGMA_BANK_SIGMA:--0.05 0.0 0.05}"
BON_MCTS_NEG_BANK_AXES="${BON_MCTS_NEG_BANK_AXES:-${_DEFAULT_NEG_BANK}}"
BON_MCTS_SIGMA_BANK_AXES="${BON_MCTS_SIGMA_BANK_AXES:--0.05 0.0 0.05}"
export BON_MCTS_NEG_BANK_NEG BON_MCTS_SIGMA_BANK_SIGMA BON_MCTS_NEG_BANK_AXES BON_MCTS_SIGMA_BANK_AXES

mkdir -p "${RUN_ROOT}"
SERVER_LOG="${RUN_ROOT}/reward_server.log"
REWARD_SERVER_URL="${REWARD_SERVER_URL:-http://localhost:${REWARD_SERVER_PORT}}"

# ── 1. Reward server (ImageReward) ─────────────────────────────────────────
if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then
    echo "[all-method] reusing reward server at ${REWARD_SERVER_URL}"
else
    echo "[all-method] booting ImageReward server on GPU ${CUDA_VISIBLE_DEVICES_REWARD}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_REWARD}" \
      "${PYTHON_BIN}" "${SCRIPT_DIR}/reward_server.py" \
        --port "${REWARD_SERVER_PORT}" --device cuda:0 \
        --backends imagereward --image_reward_model ImageReward-v1.0 \
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

# ── 2. Prompts ─────────────────────────────────────────────────────────────
PROMPTS_DIR="${RUN_ROOT}/_prompts"
mkdir -p "${PROMPTS_DIR}"
PROMPT_FILE="${PROMPTS_DIR}/backend_${BACKEND}.txt"
if [[ ! -f "${PROMPT_FILE}" ]]; then
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_prompts.py" \
        --n_prompts "${N_PROMPTS}" --out_dir "${PROMPTS_DIR}" --backends "${BACKEND}"
fi

# ── 3. Stage rewrites (3-level — shared across rewrite-using methods) ──────
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
        : "${MCTS_KEY_STEP_COUNT:=4}"
        SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
        ;;
    sd35_base)
        export SD35_BACKEND=sd35_base; unset FLUX_BACKEND || true
        export STEPS=28; export BASELINE_CFG=4.5
        export CFG_SCALES="3.5 4.0 4.5 5.0 5.5 6.0 7.0"
        : "${N_SIMS:=120}"; : "${BON_MCTS_N_SEEDS:=16}"; : "${BON_MCTS_TOPK:=4}"
        : "${MCTS_KEY_STEP_COUNT:=8}"
        SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
        ;;
    *) echo "[all-method] ERROR unsupported BACKEND='${BACKEND}'" >&2; exit 1 ;;
esac

# ── 5. Shared run knobs ────────────────────────────────────────────────────
export METHODS="${METHODS:-baseline greedy_prompt bon_mcts bon_mcts_static_cfg bon_mcts_adaptive_cfg bon_mcts_rewrite_only bon_mcts_full bon_mcts_neg bon_mcts_sigma bon_mcts_axes}"
export PROMPT_FILE
export START_INDEX=0
export END_INDEX="${N_PROMPTS}"
export SEEDS="${SEED}"
export N_SIMS="${N_SIMS:-60}"
export BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS:-16}"
export BON_MCTS_TOPK="${BON_MCTS_TOPK:-4}"
export BON_MCTS_MIN_SIMS=8
export BON_MCTS_SIM_ALLOC=split
# Outer defaults — per-method aliases override these in the suite case block.
export BON_MCTS_REFINE_METHOD=ours_tree
export LOOKAHEAD_METHOD_MODE=rollout_tree_prior_adaptive_cfg
export N_VARIANTS=1
export USE_QWEN=0
export PRECOMPUTE_REWRITES=0
export CORRECTION_STRENGTHS="0.0"
export UCB_C=1.0
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
export MCTS_KEY_STEP_COUNT="${MCTS_KEY_STEP_COUNT:-2}"
export OUT_ROOT="${RUN_ROOT}"

echo
echo "================================================================"
echo "[all-method] single-pass run"
echo "  METHODS=${METHODS}"
echo "  rewrites: ${REWRITES_FILE}"
echo "  → ${RUN_ROOT}"
echo "================================================================"

bash "${SUITE}" || echo "[warn] suite exited with non-zero — continuing to summary stage"

# ── 6. Summarize ───────────────────────────────────────────────────────────
SUMMARY="${RUN_ROOT}/summary.tsv"
printf "method\tcfg_dynamic\tprompt_dynamic\tneg_bank\tsigma_bank\tmean_search\teval_ir\teval_hpsv3\n" > "${SUMMARY}"

for method in ${METHODS}; do
    cfg_dyn="-"; prm_dyn="-"; neg_show="-"; sig_show="-"
    case "${method}" in
        baseline)                cfg_dyn="-";       prm_dyn="-"        ;;
        greedy_prompt)           cfg_dyn="static";  prm_dyn="greedy"   ;;
        bon_mcts)                cfg_dyn="dynamic"; prm_dyn="static"   ;;
        bon_mcts_static_cfg)     cfg_dyn="static";  prm_dyn="static"   ;;
        bon_mcts_adaptive_cfg)   cfg_dyn="dynamic"; prm_dyn="static"   ;;
        bon_mcts_rewrite_only)   cfg_dyn="static";  prm_dyn="dynamic"  ;;
        bon_mcts_full)           cfg_dyn="dynamic"; prm_dyn="dynamic"  ;;
        bon_mcts_neg)            cfg_dyn="dynamic"; prm_dyn="static"; neg_show="${BON_MCTS_NEG_BANK_NEG}" ;;
        bon_mcts_sigma)          cfg_dyn="dynamic"; prm_dyn="static"; sig_show="${BON_MCTS_SIGMA_BANK_SIGMA}" ;;
        bon_mcts_axes)           cfg_dyn="dynamic"; prm_dyn="static"; neg_show="${BON_MCTS_NEG_BANK_AXES}"; sig_show="${BON_MCTS_SIGMA_BANK_AXES}" ;;
    esac
    agg=$(find "${RUN_ROOT}" -name 'aggregate_ddp.json' -path "*/${method}/*" 2>/dev/null | head -1)
    ir_eval=$(find "${RUN_ROOT}" -name 'best_images_multi_reward_aggregate.json' -path "*/${method}/*" 2>/dev/null | head -1)
    ms="" eir="" eh=""
    [[ -f "${agg}" ]] && ms=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${agg}')).get('mean_search', ''))")
    [[ -f "${ir_eval}" ]] && {
        eir=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${ir_eval}')); print(d.get('imagereward', {}).get('mean', ''))")
        eh=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${ir_eval}')); print(d.get('hpsv3', {}).get('mean', ''))")
    }
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${method}" "${cfg_dyn}" "${prm_dyn}" "${neg_show}" "${sig_show}" \
        "${ms}" "${eir}" "${eh}" >> "${SUMMARY}"
done

echo
echo "================================================================"
echo "[all-method] DONE."
echo "  RUN_ROOT: ${RUN_ROOT}"
column -t -s $'\t' "${SUMMARY}" | head -30
echo
echo "  2×2 synergy plot:"
echo "    python ${SCRIPT_DIR}/plot_synergy_2x2.py --summary ${SUMMARY}"
