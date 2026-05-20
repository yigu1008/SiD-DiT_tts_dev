#!/usr/bin/env bash
# Find the best bon_mcts (ours) config via one-at-a-time ablations.
#
# Anchor: refine=mcts, N_SEEDS=8, TOPK=2, N_SIMS=60, UCB_C=1.0, fresh_noise=off,
# CFG bank size 4, N_VARIANTS=1.
#
# Sweep cells (~20 total, run sequentially):
#   default              : the anchor
#   sims_15 / 30 / 120 / 240
#   topk_1 / 2 / 4
#   seeds_4 / 16 / 32
#   ucb_c_0.5 / 1.41 / 2.0
#   refine_mcts / ours_tree / hybrid_ut_dt
#   noise_on_s1.0 / noise_on_s1.5
#
# Outputs per cell: best_images/ + images/ + aggregate_ddp.json
# Final summary: <RUN_ROOT>/summary.tsv
#
# Just: bash run_mcts_best_ablation.sh
#
# Override knobs:
#   N_PROMPTS=50        (default)
#   SEED=42             (default)
#   BACKEND=sid         (default; also senseflow_large supported)
#   CELLS="default sims_30 sims_60 topk_2 ..."  (subset of the 20 cells)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "mcts-best-ablation"

# ── A6000-friendly env defaults ─────────────────────────────────────────────
CUDA_VISIBLE_DEVICES_REWARD="${CUDA_VISIBLE_DEVICES_REWARD:-4}"
CUDA_VISIBLE_DEVICES_SAMPLE="${CUDA_VISIBLE_DEVICES_SAMPLE:-5,6,7}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-/data/ygu/mcts_best_ablation/run_$(date +%Y%m%d_%H%M%S)}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5118}"

BACKEND="${BACKEND:-sid}"
N_PROMPTS="${N_PROMPTS:-50}"
SEED="${SEED:-42}"
SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"

# Anchor knobs (each cell overrides ONE of these).
ANCHOR_N_SIMS=60
ANCHOR_TOPK=2
ANCHOR_N_SEEDS=8
ANCHOR_UCB_C=1.0
ANCHOR_REFINE=mcts
ANCHOR_NOISE_STEPS=
ANCHOR_NOISE_SAMPLES=1
ANCHOR_NOISE_SCALE=1.0
ANCHOR_NOISE_KEY_STEPS=0

# Cells (each line: "label  N_SIMS TOPK N_SEEDS UCB_C  REFINE  NOISE_STEPS NOISE_SAMPLES NOISE_SCALE NOISE_KEY  N_VARIANTS USE_QWEN QWEN_ID  INTERP_FAMILY N_INTERP  KEY_STEP_COUNT  LOOKAHEAD_MODE")
# LOOKAHEAD_MODE values:
#   '-' (or 'default')                         -- inherit suite default = rollout_tree_prior_adaptive_cfg
#   'rollout_tree_prior'                       -- lookahead prior WITHOUT dynamic CFG
#   'rollout_tree_prior_adaptive_cfg'          -- lookahead prior WITH dynamic CFG schedule
# Only meaningful when REFINE='ours_tree' or 'hybrid_ut_dt'.
# Anchor: 60 sims, topk=2, seeds=8, ucb=1.0, refine=mcts, no fresh noise,
#         N_VARIANTS=1 (no Qwen), no interp, 2 key steps.
ALL_CELLS=(
    # ── Anchor ─────────────────────────────────────────────────────────────
    "default               60  2  8   1.0  mcts          ''   1  1.0  0   1  0  Qwen/Qwen3-4B  none   0  2  -"
    # ── Search-tree breadth/depth ──────────────────────────────────────────
    "topk_1                60  1  8   1.0  mcts          ''   1  1.0  0   1  0  Qwen/Qwen3-4B  none   0  2  -"
    "topk_4                60  4  8   1.0  mcts          ''   1  1.0  0   1  0  Qwen/Qwen3-4B  none   0  2  -"
    "seeds_4               60  2  4   1.0  mcts          ''   1  1.0  0   1  0  Qwen/Qwen3-4B  none   0  2  -"
    "seeds_16              60  2  16  1.0  mcts          ''   1  1.0  0   1  0  Qwen/Qwen3-4B  none   0  2  -"
    "seeds_32              60  2  32  1.0  mcts          ''   1  1.0  0   1  0  Qwen/Qwen3-4B  none   0  2  -"
    # ── Exploration constant ───────────────────────────────────────────────
    "ucb_c_0.5             60  2  8   0.5  mcts          ''   1  1.0  0   1  0  Qwen/Qwen3-4B  none   0  2  -"
    "ucb_c_1.41            60  2  8   1.41 mcts          ''   1  1.0  0   1  0  Qwen/Qwen3-4B  none   0  2  -"
    "ucb_c_2.0             60  2  8   2.0  mcts          ''   1  1.0  0   1  0  Qwen/Qwen3-4B  none   0  2  -"
    # ── Refine algorithm ───────────────────────────────────────────────────
    "refine_ours_tree      60  2  8   1.0  ours_tree     ''   1  1.0  0   1  0  Qwen/Qwen3-4B  none   0  2  rollout_tree_prior_adaptive_cfg"
    "refine_hybrid_ut_dt   60  2  8   1.0  hybrid_ut_dt  ''   1  1.0  0   1  0  Qwen/Qwen3-4B  none   0  2  rollout_tree_prior_adaptive_cfg"
    # ── Dynamic-CFG-schedule isolation (only meaningful for ours_tree / hybrid_ut_dt) ─
    "cfg_lookahead_static  60  2  8   1.0  ours_tree     ''   1  1.0  0   1  0  Qwen/Qwen3-4B  none   0  2  rollout_tree_prior"
    "cfg_lookahead_adaptive 60 2  8   1.0  ours_tree     ''   1  1.0  0   1  0  Qwen/Qwen3-4B  none   0  2  rollout_tree_prior_adaptive_cfg"
    "cfg_hybrid_static     60  2  8   1.0  hybrid_ut_dt  ''   1  1.0  0   1  0  Qwen/Qwen3-4B  none   0  2  rollout_tree_prior"
    "cfg_hybrid_adaptive   60  2  8   1.0  hybrid_ut_dt  ''   1  1.0  0   1  0  Qwen/Qwen3-4B  none   0  2  rollout_tree_prior_adaptive_cfg"
    # ── Fresh-noise exploration ────────────────────────────────────────────
    "noise_on_s1.0         60  2  8   1.0  mcts          all  3  1.0  1   1  0  Qwen/Qwen3-4B  none   0  2  -"
    "noise_on_s1.5         60  2  8   1.0  mcts          all  3  1.5  1   1  0  Qwen/Qwen3-4B  none   0  2  -"
    # ── Smart prompt rewriting (N_VARIANTS + Qwen) ─────────────────────────
    "qwen4b_var3           60  2  8   1.0  mcts          ''   1  1.0  0   3  1  Qwen/Qwen3-4B  none   0  2  -"
    "qwen4b_var5           60  2  8   1.0  mcts          ''   1  1.0  0   5  1  Qwen/Qwen3-4B  none   0  2  -"
    "qwen4b_var7           60  2  8   1.0  mcts          ''   1  1.0  0   7  1  Qwen/Qwen3-4B  none   0  2  -"
    "qwen8b_var3           60  2  8   1.0  mcts          ''   1  1.0  0   3  1  Qwen/Qwen3-8B  none   0  2  -"
    "qwen8b_var5           60  2  8   1.0  mcts          ''   1  1.0  0   5  1  Qwen/Qwen3-8B  none   0  2  -"
    # ── Prompt-embedding interpolation (smooth action space between variants)
    "interp_slerp_n1       60  2  8   1.0  mcts          ''   1  1.0  0   3  1  Qwen/Qwen3-4B  slerp  1  2  -"
    "interp_slerp_n3       60  2  8   1.0  mcts          ''   1  1.0  0   3  1  Qwen/Qwen3-4B  slerp  3  2  -"
    "interp_nlerp_n1       60  2  8   1.0  mcts          ''   1  1.0  0   3  1  Qwen/Qwen3-4B  nlerp  1  2  -"
    # ── Branching schedule (where MCTS expands) ────────────────────────────
    "keystep_1             60  2  8   1.0  mcts          ''   1  1.0  0   1  0  Qwen/Qwen3-4B  none   0  1  -"
    "keystep_3             60  2  8   1.0  mcts          ''   1  1.0  0   1  0  Qwen/Qwen3-4B  none   0  3  -"
    # ── Combined "best-guess" cell (lookahead + dynamic CFG + Qwen + interp + noise) ─
    "combo_kitchen_sink    60  2  16  1.0  hybrid_ut_dt  all  3  1.5  1   5  1  Qwen/Qwen3-4B  slerp  1  2  rollout_tree_prior_adaptive_cfg"
)
CELLS="${CELLS:-default topk_1 topk_4 seeds_4 seeds_16 seeds_32 ucb_c_0.5 ucb_c_1.41 ucb_c_2.0 refine_ours_tree refine_hybrid_ut_dt cfg_lookahead_static cfg_lookahead_adaptive cfg_hybrid_static cfg_hybrid_adaptive noise_on_s1.0 noise_on_s1.5 qwen4b_var3 qwen4b_var5 qwen4b_var7 qwen8b_var3 qwen8b_var5 interp_slerp_n1 interp_slerp_n3 interp_nlerp_n1 keystep_1 keystep_3 combo_kitchen_sink}"

mkdir -p "${RUN_ROOT}"
SERVER_LOG="${RUN_ROOT}/reward_server.log"
REWARD_SERVER_URL="http://localhost:${REWARD_SERVER_PORT}"

# ── 1. Reward server: reuse if healthy, boot otherwise ─────────────────────
if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then
    echo "[mcts-best] reusing reward server at ${REWARD_SERVER_URL}"
else
    echo "[mcts-best] booting reward server on GPU(s) ${CUDA_VISIBLE_DEVICES_REWARD} (port ${REWARD_SERVER_PORT})"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_REWARD}" \
      "${PYTHON_BIN}" "${SCRIPT_DIR}/reward_server.py" \
        --port "${REWARD_SERVER_PORT}" --device cuda:0 \
        --backends hpsv3 imagereward \
        --image_reward_model ImageReward-v1.0 \
        > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    trap 'kill "${SERVER_PID}" >/dev/null 2>&1 || true' EXIT
    OK=0
    for i in $(seq 1 100); do
        if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then OK=1; break; fi
        kill -0 "${SERVER_PID}" 2>/dev/null || { echo "[mcts-best] FATAL server died"; tail -n 80 "${SERVER_LOG}" >&2; exit 1; }
        sleep 3
    done
    [[ "${OK}" == "1" ]] || { echo "[mcts-best] FATAL server not healthy"; tail -n 80 "${SERVER_LOG}" >&2; exit 1; }
    echo "[mcts-best] reward server READY"
fi

# ── 2. Sample one shared prompt subset across all cells ────────────────────
PROMPTS_DIR="${RUN_ROOT}/_prompts"
mkdir -p "${PROMPTS_DIR}"
PROMPT_FILE="${PROMPTS_DIR}/backend_${BACKEND}_${SEARCH_REWARD}.txt"
if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "[mcts-best] sampling ${N_PROMPTS} prompts → ${PROMPT_FILE}"
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_prompts.py" \
        --n_prompts "${N_PROMPTS}" \
        --out_dir "${PROMPTS_DIR}" \
        --backends "${BACKEND}" \
        --tag "${SEARCH_REWARD}"
fi

# ── 3. Backend env ─────────────────────────────────────────────────────────
case "${BACKEND}" in
    sid)
        export SD35_BACKEND=sid; unset FLUX_BACKEND || true
        export STEPS=4; export BASELINE_CFG=1.0
        export CFG_SCALES="1.0 1.5 2.0 2.5"
        SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
        ;;
    senseflow_large)
        export SD35_BACKEND=senseflow_large; unset FLUX_BACKEND || true
        export STEPS=4; export BASELINE_CFG=1.0
        export CFG_SCALES="1.0 1.5 2.0 2.5"
        SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
        ;;
    sd35_base)
        export SD35_BACKEND=sd35_base; unset FLUX_BACKEND || true
        export STEPS=28; export BASELINE_CFG=4.5
        export CFG_SCALES="3.5 4.5 5.5 7.0"
        SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
        echo "[mcts-best] WARN sd35_base is 28-step; each cell ~5-15× longer than sid"
        ;;
    *) echo "[mcts-best] ERROR unknown BACKEND='${BACKEND}'" >&2; exit 1 ;;
esac

# ── 4. Shared run knobs ────────────────────────────────────────────────────
export METHODS="baseline bon_mcts"
export PROMPT_FILE
export START_INDEX=0
export END_INDEX="${N_PROMPTS}"
export SEEDS="${SEED}"
export N_VARIANTS=1
export USE_QWEN=0
export PRECOMPUTE_REWRITES=0
export REWARDS_OVERWRITE=0
export CORRECTION_STRENGTHS="0.0"
export SAVE_BEST_IMAGES=1
export SAVE_IMAGES=1                       # ← keep ALL images for inspection
export SAVE_VARIANTS=0
export EVAL_BACKENDS="imagereward hpsv3"
export REWARD_BACKEND="${SEARCH_REWARD}"
export REWARD_TYPE="${SEARCH_REWARD}"
export REWARD_BACKENDS="${SEARCH_REWARD}"
export EVAL_BEST_IMAGES=1
export EVAL_ALLOW_MISSING_BACKENDS=1
export EVAL_REWARD_DEVICE=cuda
export REWARD_SERVER_URL
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_SAMPLE}"
export BON_MCTS_SIM_ALLOC=split
export BON_MCTS_MIN_SIMS=8

SUMMARY="${RUN_ROOT}/summary.tsv"
printf "cell\tn_sims\ttopk\tn_seeds\tucb_c\trefine\tnoise_steps\tnoise_samples\tnoise_scale\tn_variants\tqwen_id\tinterp\tn_interp\tkey_steps\tlookahead_mode\tmean_search\teval_ir\teval_hpsv3\tnfe_est\n" > "${SUMMARY}"

# ── 5. Run cells ────────────────────────────────────────────────────────────
failed=()
for cell in ${CELLS}; do
    # Find the row in ALL_CELLS for this label.
    row=""
    for line in "${ALL_CELLS[@]}"; do
        first=$(awk '{print $1}' <<<"${line}")
        if [[ "${first}" == "${cell}" ]]; then row="${line}"; break; fi
    done
    if [[ -z "${row}" ]]; then
        echo "[mcts-best] WARN unknown cell '${cell}'; skipping" >&2
        continue
    fi

    read -r _ N_SIMS TOPK N_SEEDS UCB_C REFINE NOISE_STEPS NOISE_SAMPLES NOISE_SCALE NOISE_KEY CELL_N_VARIANTS CELL_USE_QWEN CELL_QWEN_ID INTERP_FAMILY N_INTERP KEY_STEP_COUNT LOOKAHEAD_MODE <<<"${row}"
    NOISE_STEPS=${NOISE_STEPS#\'}; NOISE_STEPS=${NOISE_STEPS%\'}
    # LOOKAHEAD_MODE: '-' means inherit suite default (rollout_tree_prior_adaptive_cfg).
    if [[ "${LOOKAHEAD_MODE}" == "-" || -z "${LOOKAHEAD_MODE}" ]]; then
        LOOKAHEAD_MODE=""
    fi

    cell_root="${RUN_ROOT}/${cell}"
    mkdir -p "${cell_root}"
    REWRITES_FILE="${PROMPTS_DIR}/${BACKEND}_${cell}_qwen_rewrites.json"

    echo
    echo "================================================================"
    echo "[mcts-best] cell=${cell}"
    echo "  N_SIMS=${N_SIMS} TOPK=${TOPK} N_SEEDS=${N_SEEDS} UCB_C=${UCB_C}"
    echo "  refine=${REFINE} noise_steps='${NOISE_STEPS}' samples=${NOISE_SAMPLES} scale=${NOISE_SCALE} key=${NOISE_KEY}"
    echo "  N_VARIANTS=${CELL_N_VARIANTS} USE_QWEN=${CELL_USE_QWEN} QWEN_ID=${CELL_QWEN_ID}"
    echo "  interp=${INTERP_FAMILY} n_interp=${N_INTERP} key_step_count=${KEY_STEP_COUNT}"
    echo "  → ${cell_root}"
    echo "================================================================"

    if N_SIMS="${N_SIMS}" \
       BON_MCTS_TOPK="${TOPK}" \
       BON_MCTS_N_SEEDS="${N_SEEDS}" \
       UCB_C="${UCB_C}" \
       BON_MCTS_REFINE_METHOD="${REFINE}" \
       MCTS_FRESH_NOISE_STEPS="${NOISE_STEPS}" \
       MCTS_FRESH_NOISE_SAMPLES="${NOISE_SAMPLES}" \
       MCTS_FRESH_NOISE_SCALE="${NOISE_SCALE}" \
       MCTS_FRESH_NOISE_KEY_STEPS="${NOISE_KEY}" \
       N_VARIANTS="${CELL_N_VARIANTS}" \
       USE_QWEN="${CELL_USE_QWEN}" \
       PRECOMPUTE_REWRITES="${CELL_USE_QWEN}" \
       QWEN_ID="${CELL_QWEN_ID}" \
       REWRITES_FILE="${REWRITES_FILE}" \
       MCTS_INTERP_FAMILY="${INTERP_FAMILY}" \
       MCTS_N_INTERP="${N_INTERP}" \
       MCTS_KEY_STEP_COUNT="${KEY_STEP_COUNT}" \
       ${LOOKAHEAD_MODE:+LOOKAHEAD_METHOD_MODE="${LOOKAHEAD_MODE}"} \
       OUT_ROOT="${cell_root}" \
       bash "${SUITE}"; then
        echo "[mcts-best] OK ${cell}"
        # Pull aggregate scores into the summary TSV.
        agg=$(find "${cell_root}" -name 'aggregate_ddp.json' -path '*/bon_mcts/*' 2>/dev/null | head -1)
        ir_eval=$(find "${cell_root}" -name 'best_images_multi_reward_aggregate.json' -path '*/bon_mcts/*' 2>/dev/null | head -1)
        ms="" eir="" eh=""
        if [[ -f "${agg}" ]]; then
            ms=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${agg}')).get('mean_search', ''))")
        fi
        if [[ -f "${ir_eval}" ]]; then
            eir=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${ir_eval}')); print(d.get('imagereward', {}).get('mean', ''))")
            eh=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${ir_eval}')); print(d.get('hpsv3', {}).get('mean', ''))")
        fi
        # NFE estimate: (N_SEEDS + topk*N_SIMS) * STEPS
        nfe=$(( (N_SEEDS + TOPK * N_SIMS) * STEPS ))
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${cell}" "${N_SIMS}" "${TOPK}" "${N_SEEDS}" "${UCB_C}" "${REFINE}" \
            "${NOISE_STEPS}" "${NOISE_SAMPLES}" "${NOISE_SCALE}" \
            "${CELL_N_VARIANTS}" "${CELL_QWEN_ID}" "${INTERP_FAMILY}" "${N_INTERP}" "${KEY_STEP_COUNT}" \
            "${LOOKAHEAD_MODE:-default}" \
            "${ms}" "${eir}" "${eh}" "${nfe}" \
            >> "${SUMMARY}"
    else
        rc=$?
        echo "[mcts-best] FAIL ${cell} rc=${rc}" >&2
        failed+=("${cell}")
    fi
done

# ── 6. Summary ─────────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "[mcts-best] DONE."
echo "  RUN_ROOT: ${RUN_ROOT}"
echo "  Summary:  ${SUMMARY}"
echo
column -t -s $'\t' "${SUMMARY}" | head -40
echo
if (( ${#failed[@]} > 0 )); then
    echo "[mcts-best] WARN failures: ${failed[*]}"
    exit 1
fi
