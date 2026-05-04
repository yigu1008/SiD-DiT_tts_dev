#!/usr/bin/env bash
# Local 6-cell prior ablation around the hybrid_ut_dt MCTS engine.
#
# Cells (matches your slide):
#   prior          : refine=hybrid_ut_dt, prior_mode=ut_dt, w_u=1, w_d=1   (full U_t/D_t)
#   no_prior       : refine=mcts (vanilla UCB1)                            ← no PUCT, no prior at all
#   only_d         : refine=hybrid_ut_dt, prior_mode=ut_dt, w_u=0, w_d=1
#   only_u         : refine=hybrid_ut_dt, prior_mode=ut_dt, w_u=1, w_d=0
#   uniform_prior  : refine=hybrid_ut_dt, prior_mode=uniform               (PUCT bonus, π = 1/|A|)
#   random_prior   : refine=hybrid_ut_dt, prior_mode=random                (PUCT bonus, π = Dirichlet(1) per call)
#
# Designed for local 48GB A100. Single GPU by default; override with
# CUDA_VISIBLE_DEVICES=4 (or 4,5,6,7 for DDP) before launching.
#
# Defaults (overridable via env):
#   SD35_BACKEND=sid    NUM_PROMPTS=10    SEED=42
#   N_SIMS=30           BON_MCTS_TOPK=2   BON_MCTS_N_SEEDS=8
#   REWARD_BACKEND=imagereward (loaded inline; no reward server)
#
# Per-cell wallclock at defaults: ~20-30 min on sid → ~3 h for all 6 cells.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Backend defaults (can be overridden) ─────────────────────────────────────
SD35_BACKEND="${SD35_BACKEND:-sid}"
case "${SD35_BACKEND}" in
    sid)             STEPS="${STEPS:-4}";  BASELINE_CFG="${BASELINE_CFG:-1.0}"; CFG_SCALES="${CFG_SCALES:-1.0 1.5 2.0 2.5}" ;;
    senseflow_large) STEPS="${STEPS:-4}";  BASELINE_CFG="${BASELINE_CFG:-1.0}"; CFG_SCALES="${CFG_SCALES:-1.0 1.5 2.0 2.5}" ;;
    sd35_base)       STEPS="${STEPS:-28}"; BASELINE_CFG="${BASELINE_CFG:-4.5}"; CFG_SCALES="${CFG_SCALES:-3.5 4.5 5.5 7.0}" ;;
    *) echo "Unknown SD35_BACKEND=${SD35_BACKEND}" >&2; exit 1 ;;
esac

# ── GPU defaults ─────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
NUM_GPUS_FROM_ENV="${NUM_GPUS:-}"
if [[ -z "${NUM_GPUS_FROM_ENV}" ]]; then
    # Count comma-separated devices.
    n=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')
    NUM_GPUS="${n}"
fi
export NUM_GPUS

unset REWARD_SERVER_URL || true   # inline rewards
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SID_FORCE_WANDB_STUB=1
export WANDB_DISABLED=true

# ── Run knobs ────────────────────────────────────────────────────────────────
NUM_PROMPTS="${NUM_PROMPTS:-10}"
SEED="${SEED:-42}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/hpsv2_subset.txt}"
OUT_ROOT_BASE="${OUT_ROOT_BASE:-/tmp/prior_ablation}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${OUT_ROOT_BASE}/${SD35_BACKEND}/run_${RUN_TS}"
mkdir -p "${RUN_ROOT}"

REWARD_BACKEND="${REWARD_BACKEND:-imagereward}"
EVAL_BACKENDS="${EVAL_BACKENDS:-imagereward}"

# ── MCTS / bon_mcts knobs (lighter tier for fast iteration) ─────────────────
N_SIMS="${N_SIMS:-30}"
UCB_C="${UCB_C:-1.0}"
BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS:-8}"
BON_MCTS_TOPK="${BON_MCTS_TOPK:-2}"
BON_MCTS_SIM_ALLOC="${BON_MCTS_SIM_ALLOC:-split}"
BON_MCTS_MIN_SIMS="${BON_MCTS_MIN_SIMS:-8}"

# ── Methods: only bon_mcts here (baseline added once at start) ──────────────
N_VARIANTS="${N_VARIANTS:-1}"
USE_QWEN="${USE_QWEN:-0}"
PRECOMPUTE_REWRITES="${PRECOMPUTE_REWRITES:-0}"
CORRECTION_STRENGTHS="${CORRECTION_STRENGTHS:-0.0}"
SAVE_BEST_IMAGES="${SAVE_BEST_IMAGES:-1}"
SAVE_IMAGES="${SAVE_IMAGES:-0}"
SAVE_VARIANTS="${SAVE_VARIANTS:-0}"
EVAL_BEST_IMAGES="${EVAL_BEST_IMAGES:-1}"
EVAL_REWARD_DEVICE="${EVAL_REWARD_DEVICE:-cuda}"
EVAL_ALLOW_MISSING_BACKENDS="${EVAL_ALLOW_MISSING_BACKENDS:-1}"

CELLS="${CELLS:-prior no_prior only_d only_u uniform_prior random_prior}"
FAIL_FAST="${FAIL_FAST:-0}"

# ── Per-cell knob assignment ────────────────────────────────────────────────
_apply_cell() {
    # Reset to neutral defaults each call.
    export BON_MCTS_REFINE_METHOD=hybrid_ut_dt
    export MCTS_HYBRID_PRIOR_MODE=ut_dt
    export MCTS_HYBRID_W_U=1.0
    export MCTS_HYBRID_W_D=1.0

    case "$1" in
        prior)
            : ;;  # full U_t/D_t prior; defaults above
        no_prior)
            export BON_MCTS_REFINE_METHOD=mcts        # vanilla UCB1, no PUCT
            ;;
        only_d)
            export MCTS_HYBRID_W_U=0.0
            export MCTS_HYBRID_W_D=1.0
            ;;
        only_u)
            export MCTS_HYBRID_W_U=1.0
            export MCTS_HYBRID_W_D=0.0
            ;;
        uniform_prior)
            export MCTS_HYBRID_PRIOR_MODE=uniform
            ;;
        random_prior)
            export MCTS_HYBRID_PRIOR_MODE=random
            ;;
        *)
            echo "[prior-ablation] ERROR unknown cell '$1'" >&2; return 1 ;;
    esac
}

# ── Banner ──────────────────────────────────────────────────────────────────
echo "================================================================"
echo "[prior-ablation]"
echo "  backend:      ${SD35_BACKEND}  STEPS=${STEPS} CFG_SCALES='${CFG_SCALES}'"
echo "  reward:       ${REWARD_BACKEND}  (eval=${EVAL_BACKENDS})"
echo "  prompts:      ${NUM_PROMPTS} from ${PROMPT_FILE}  seed=${SEED}"
echo "  GPUs:         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  NUM_GPUS=${NUM_GPUS}"
echo "  cells:        ${CELLS}"
echo "  bon_mcts:     n_seeds=${BON_MCTS_N_SEEDS} topk=${BON_MCTS_TOPK} alloc=${BON_MCTS_SIM_ALLOC}"
echo "                n_sims=${N_SIMS} ucb_c=${UCB_C}"
echo "  output:       ${RUN_ROOT}"
echo "================================================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>/dev/null || true

# ── Shared exports ──────────────────────────────────────────────────────────
export METHODS="bon_mcts"   # baseline run separately as control (below)
export SD35_BACKEND STEPS BASELINE_CFG CFG_SCALES
export PROMPT_FILE START_INDEX=0 END_INDEX="${NUM_PROMPTS}"
export SEED
export N_VARIANTS USE_QWEN PRECOMPUTE_REWRITES CORRECTION_STRENGTHS
export SAVE_BEST_IMAGES SAVE_IMAGES SAVE_VARIANTS
export EVAL_BACKENDS EVAL_BEST_IMAGES EVAL_REWARD_DEVICE EVAL_ALLOW_MISSING_BACKENDS
export REWARD_BACKEND REWARD_TYPE="${REWARD_BACKEND}" REWARD_BACKENDS="${REWARD_BACKEND}"
export N_SIMS UCB_C
export BON_MCTS_N_SEEDS BON_MCTS_TOPK BON_MCTS_SIM_ALLOC BON_MCTS_MIN_SIMS

# ── 1) Baseline reference (one-time, before per-cell loop) ─────────────────
echo
echo "================================================================"
echo "[prior-ablation] baseline (control, runs once)"
echo "================================================================"
export METHODS="baseline"
export OUT_ROOT="${RUN_ROOT}/_baseline"
mkdir -p "${OUT_ROOT}"
bash "${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh" || echo "[prior-ablation] WARN baseline failed"

# ── 2) Per-cell bon_mcts runs ───────────────────────────────────────────────
export METHODS="bon_mcts"
failed=()
for cell in ${CELLS}; do
    if ! _apply_cell "${cell}"; then
        failed+=("${cell}/bad-cell"); continue
    fi
    cell_root="${RUN_ROOT}/${cell}"
    mkdir -p "${cell_root}"
    export OUT_ROOT="${cell_root}"

    # Pass mcts_hybrid_* via env. They get translated to --mcts_hybrid_*
    # CLI args by the bon_mcts wrapper (sd35_ddp_experiment_bon_mcts.py
    # reads them from args after argparse). To force the env values into
    # argparse, we set them as suite EXTRA_ARGS. Simpler: env vars are
    # read directly by add_mcts_hybrid_args defaults — but argparse only
    # honors env at module import. To be safe, append CLI flags via
    # MCTS_PARAM_EXTRA (the suite forwards EXTRA env to torchrun).
    # Workaround: directly inject via _mcts_hybrid_extra_args.
    : "${MCTS_HYBRID_W_U:?}"; : "${MCTS_HYBRID_W_D:?}"; : "${MCTS_HYBRID_PRIOR_MODE:?}"

    echo
    echo "================================================================"
    echo "[prior-ablation] cell=${cell}"
    echo "  refine=${BON_MCTS_REFINE_METHOD}  prior_mode=${MCTS_HYBRID_PRIOR_MODE}"
    echo "  w_u=${MCTS_HYBRID_W_U}  w_d=${MCTS_HYBRID_W_D}"
    echo "  out=${cell_root}"
    echo "================================================================"

    if bash "${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"; then
        echo "[prior-ablation] OK ${cell}"
    else
        rc=$?
        echo "[prior-ablation] FAIL ${cell} rc=${rc}" >&2
        failed+=("${cell}")
        if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
    fi
done

echo
if (( ${#failed[@]} > 0 )); then
    echo "[prior-ablation] DONE with failures: ${failed[*]}"
    exit 1
fi
echo "[prior-ablation] DONE all 6 cells OK."
echo "  Inspect: find ${RUN_ROOT} -name aggregate_ddp.json"
echo "  Compare: cat ${RUN_ROOT}/*/run_*/bon_mcts/aggregate_ddp.json | jq '.mean_search_score'"
