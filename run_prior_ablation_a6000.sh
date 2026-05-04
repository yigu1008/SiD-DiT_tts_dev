#!/usr/bin/env bash
# A6000-tuned U_t/D_t prior ablation (sibling of run_prior_ablation_local.sh).
#
# 6 cells (matches the slide):
#   prior          : refine=hybrid_ut_dt, prior_mode=ut_dt, w_u=1, w_d=1
#   no_prior       : refine=mcts (vanilla UCB1)
#   only_d         : refine=hybrid_ut_dt, prior_mode=ut_dt, w_u=0, w_d=1
#   only_u         : refine=hybrid_ut_dt, prior_mode=ut_dt, w_u=1, w_d=0
#   uniform_prior  : refine=hybrid_ut_dt, prior_mode=uniform
#   random_prior   : refine=hybrid_ut_dt, prior_mode=random
#
# Differences from the A100 sibling (don't touch that one):
#   - Multi-GPU DDP support: NUM_GPUS auto-derived from CUDA_VISIBLE_DEVICES.
#     Each rank loads its own ImageReward (~3 GB) + SD3.5 (~17 GB), so on
#     4× A6000 (4 × 48 GB = 192 GB) you have plenty of room.
#   - Lighter defaults: 30 prompts × 2 seeds (vs 10 × 1).
#   - Phase-2 posthoc eval (HPSv3 + PickScore inline) on saved best_images,
#     skip with RUN_POSTHOC=0.
#   - No reward server. ImageReward loads inline.
#
# Memory per GPU on A6000 (48GB): pipeline ~17 + IR ~3 + activations ~5 +
# tree state ~1 ≈ 26 GB used. Fits comfortably.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "prior-ablation-a6000"

# ── Backend defaults ────────────────────────────────────────────────────────
SD35_BACKEND="${SD35_BACKEND:-sid}"
case "${SD35_BACKEND}" in
    sid)             STEPS="${STEPS:-4}";  BASELINE_CFG="${BASELINE_CFG:-1.0}"; CFG_SCALES="${CFG_SCALES:-1.0 1.5 2.0 2.5}" ;;
    senseflow_large) STEPS="${STEPS:-4}";  BASELINE_CFG="${BASELINE_CFG:-1.0}"; CFG_SCALES="${CFG_SCALES:-1.0 1.5 2.0 2.5}" ;;
    sd35_base)       STEPS="${STEPS:-28}"; BASELINE_CFG="${BASELINE_CFG:-4.5}"; CFG_SCALES="${CFG_SCALES:-3.5 4.5 5.5 7.0}" ;;
    *) echo "Unknown SD35_BACKEND=${SD35_BACKEND}" >&2; exit 1 ;;
esac

# ── DDP-friendly GPU defaults ───────────────────────────────────────────────
# Default: GPU 0 single-process. For DDP across 4 cards, set
# CUDA_VISIBLE_DEVICES=0,1,2,3 (or 4,5,6,7 etc.) — NUM_GPUS auto-derives.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -z "${NUM_GPUS:-}" ]]; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')
fi
export NUM_GPUS

unset REWARD_SERVER_URL || true                 # inline reward; no server
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SID_FORCE_WANDB_STUB=1
export WANDB_DISABLED=true

# ── Run knobs (A6000-friendly defaults) ─────────────────────────────────────
NUM_PROMPTS="${NUM_PROMPTS:-30}"
SEEDS="${SEEDS:-42 43}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/hpsv2_subset.txt}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT_BASE="${OUT_ROOT_BASE:-/tmp/prior_ablation_a6000}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${OUT_ROOT_BASE}/${SD35_BACKEND}/run_${RUN_TS}"
mkdir -p "${RUN_ROOT}"

REWARD_BACKEND="${REWARD_BACKEND:-imagereward}"
EVAL_BACKENDS="${EVAL_BACKENDS:-imagereward}"

# ── MCTS / bon_mcts knobs (lighter for A6000) ───────────────────────────────
N_SIMS="${N_SIMS:-30}"
UCB_C="${UCB_C:-1.0}"
BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS:-8}"
BON_MCTS_TOPK="${BON_MCTS_TOPK:-2}"
BON_MCTS_SIM_ALLOC="${BON_MCTS_SIM_ALLOC:-split}"
BON_MCTS_MIN_SIMS="${BON_MCTS_MIN_SIMS:-8}"

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
    export BON_MCTS_REFINE_METHOD=hybrid_ut_dt
    export MCTS_HYBRID_PRIOR_MODE=ut_dt
    export MCTS_HYBRID_W_U=1.0
    export MCTS_HYBRID_W_D=1.0

    case "$1" in
        prior)
            : ;;
        no_prior)
            export BON_MCTS_REFINE_METHOD=mcts        # vanilla UCB1
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
            echo "[prior-ablation-a6000] ERROR unknown cell '$1'" >&2; return 1 ;;
    esac
}

_sample_prompts() {
    local backend="$1"
    local prompt_file_default="${SCRIPT_DIR}/hpsv2_subset.txt"
    if [[ -f "${PROMPT_FILE}" && "${PROMPT_FILE}" == "${prompt_file_default}" ]]; then
        # Repo's pre-baked subset; just use it.
        return 0
    fi
    # If user wants a HF-sampled list, write into RUN_ROOT/_prompts/.
    local prompts_dir="${RUN_ROOT}/_prompts"
    mkdir -p "${prompts_dir}"
    local sampled="${prompts_dir}/backend_${backend}.txt"
    if [[ ! -f "${sampled}" ]]; then
        echo "[prior-ablation-a6000] sampling prompts → ${sampled}"
        env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
            "${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_prompts.py" \
            --n_prompts "${NUM_PROMPTS}" \
            --out_dir "${prompts_dir}" \
            --backends "${backend}" || \
            echo "[prior-ablation-a6000] WARN prompt sampling failed; falling back to ${prompt_file_default}"
    fi
    [[ -f "${sampled}" ]] && export PROMPT_FILE="${sampled}"
}

# ── Banner ──────────────────────────────────────────────────────────────────
echo "================================================================"
echo "[prior-ablation-a6000]   (DDP-aware, U_t/D_t prior cells)"
echo "  backend:      ${SD35_BACKEND}  STEPS=${STEPS} CFG_SCALES='${CFG_SCALES}'"
echo "  reward:       ${REWARD_BACKEND}  (eval=${EVAL_BACKENDS})"
echo "  prompts:      ${NUM_PROMPTS} from ${PROMPT_FILE}  seeds=${SEEDS}"
echo "  GPUs:         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  NUM_GPUS=${NUM_GPUS}"
echo "  cells:        ${CELLS}"
echo "  bon_mcts:     n_seeds=${BON_MCTS_N_SEEDS} topk=${BON_MCTS_TOPK} alloc=${BON_MCTS_SIM_ALLOC}"
echo "                n_sims=${N_SIMS} ucb_c=${UCB_C}"
echo "  output:       ${RUN_ROOT}"
echo "================================================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>/dev/null || true

# ── Shared exports ──────────────────────────────────────────────────────────
export METHODS="bon_mcts"
export SD35_BACKEND STEPS BASELINE_CFG CFG_SCALES
export PROMPT_FILE START_INDEX=0 END_INDEX="${NUM_PROMPTS}"
export N_VARIANTS USE_QWEN PRECOMPUTE_REWRITES CORRECTION_STRENGTHS
export SAVE_BEST_IMAGES SAVE_IMAGES SAVE_VARIANTS
export EVAL_BACKENDS EVAL_BEST_IMAGES EVAL_REWARD_DEVICE EVAL_ALLOW_MISSING_BACKENDS
export REWARD_BACKEND REWARD_TYPE="${REWARD_BACKEND}" REWARD_BACKENDS="${REWARD_BACKEND}"
export N_SIMS UCB_C
export BON_MCTS_N_SEEDS BON_MCTS_TOPK BON_MCTS_SIM_ALLOC BON_MCTS_MIN_SIMS

_sample_prompts "${SD35_BACKEND}"

# ── 1) Baseline reference (control, runs once at SEEDS[0]) ─────────────────
first_seed="${SEEDS%% *}"
echo
echo "================================================================"
echo "[prior-ablation-a6000] baseline (control, seed=${first_seed})"
echo "================================================================"
export METHODS="baseline"
export OUT_ROOT="${RUN_ROOT}/_baseline/seed${first_seed}"
mkdir -p "${OUT_ROOT}"
SEED="${first_seed}" bash "${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh" || \
    echo "[prior-ablation-a6000] WARN baseline failed"

# ── 2) Per-cell × per-seed bon_mcts runs ────────────────────────────────────
export METHODS="bon_mcts"
failed=()
for cell in ${CELLS}; do
    if ! _apply_cell "${cell}"; then
        failed+=("${cell}/bad-cell"); continue
    fi
    for seed in ${SEEDS}; do
        cell_root="${RUN_ROOT}/${cell}/seed${seed}"
        mkdir -p "${cell_root}"
        export OUT_ROOT="${cell_root}"

        echo
        echo "================================================================"
        echo "[prior-ablation-a6000] cell=${cell}  seed=${seed}"
        echo "  refine=${BON_MCTS_REFINE_METHOD}  prior_mode=${MCTS_HYBRID_PRIOR_MODE:-n/a}"
        echo "  w_u=${MCTS_HYBRID_W_U:-n/a}  w_d=${MCTS_HYBRID_W_D:-n/a}"
        echo "  out=${cell_root}"
        echo "================================================================"

        if SEED="${seed}" bash "${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"; then
            echo "[prior-ablation-a6000] OK ${cell}/seed${seed}"
        else
            rc=$?
            echo "[prior-ablation-a6000] FAIL ${cell}/seed${seed} rc=${rc}" >&2
            failed+=("${cell}/seed${seed}")
            if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
        fi
    done
done

if (( ${#failed[@]} > 0 )); then
    echo "[prior-ablation-a6000] phase-1 DONE with failures: ${failed[*]}"
fi
echo "[prior-ablation-a6000] phase-1 DONE."

# ── 3) Phase-2 posthoc: HPSv3 + PickScore inline on saved best_images ───────
if [[ "${RUN_POSTHOC:-1}" == "1" ]]; then
    echo
    echo "================================================================"
    echo "[prior-ablation-a6000] phase-2 posthoc: hpsv3 + pickscore on saved best_images"
    echo "================================================================"
    posthoc_failed=()
    for cell in ${CELLS}; do
        for seed in ${SEEDS}; do
            cell_root="${RUN_ROOT}/${cell}/seed${seed}"
            [[ -d "${cell_root}" ]] || continue
            for method_dir in $(find "${cell_root}" -maxdepth 3 -type d -name 'bon_mcts' 2>/dev/null); do
                echo "[posthoc] eval ${method_dir}"
                if "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_best_images_multi_reward.py" \
                    --layout sd35 \
                    --method_out "${method_dir}" \
                    --method bon_mcts \
                    --backends hpsv3 pickscore \
                    --reward_device cuda \
                    --out_json "${method_dir}/best_images_posthoc.json" \
                    --out_aggregate "${method_dir}/best_images_posthoc_aggregate.json" \
                    --allow_missing_backends; then
                    :
                else
                    echo "[posthoc] WARN failed: ${method_dir}" >&2
                    posthoc_failed+=("${cell}/seed${seed}")
                fi
            done
        done
    done
    if (( ${#posthoc_failed[@]} > 0 )); then
        echo "[prior-ablation-a6000] phase-2 posthoc WARN failures: ${posthoc_failed[*]}"
    else
        echo "[prior-ablation-a6000] phase-2 posthoc DONE."
    fi
fi

echo
echo "[prior-ablation-a6000] ALL DONE."
echo "  Output:   ${RUN_ROOT}"
echo "  Compare:  for c in ${CELLS}; do echo \"=== \$c ===\"; cat ${RUN_ROOT}/\$c/seed*/run_*/bon_mcts/aggregate_ddp.json 2>/dev/null | jq '.mean_search_score'; done"
if (( ${#failed[@]} > 0 )); then exit 1; fi
