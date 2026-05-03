#!/usr/bin/env bash
# Single-bash driver for the bon_mcts prescreen ablation.
#
# Loops sequentially over (backend × config_variant) and invokes the SD3.5
# suite once per cell. All cells share the same prompts / base seed / reward
# server so results are directly comparable.
#
# Caller env (typically set by the AMLT yaml or shell):
#   RUN_ROOT           - output dir parent
#   PROMPT_FILE        - prompt list (default: ./hpsv2_subset.txt)
#   REWARD_SERVER_URL  - shared reward server URL
#   REWARD_BACKEND     - search-time reward (hpsv3 or imagereward)
#   NUM_PROMPTS, SEED, NUM_GPUS, CUDA_VISIBLE_DEVICES, etc.
#
# Optional env:
#   BACKENDS    - subset of {sid senseflow_large}      (default: both)
#   VARIANTS    - subset of the 6 config_variants       (default: all 6)
#   FAIL_FAST   - "1" to abort on first variant failure (default: 0)
#
# Output layout (matches bon_mcts_compare.py expectations):
#   $RUN_ROOT/<backend>/<variant>/seed<S>/<run_dir>/{baseline,bon_mcts}/...

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Heartbeat to keep AMLT from suspending on inactivity.
if [[ -f "${SCRIPT_DIR}/_heartbeat.sh" ]]; then
  source "${SCRIPT_DIR}/_heartbeat.sh"
  start_heartbeat "bon-mcts-ablation"
fi

: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${REWARD_SERVER_URL:?REWARD_SERVER_URL must be set}"

PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/hpsv2_subset.txt}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
SEED="${SEED:-42}"
# N_SIMS=60 keeps each refine tree deep enough for ours_tree / hybrid priors
# to converge (topk=2 split → ~30 sims per tree). The 4-step suite auto-drops
# N_SIMS to 25 by default for cost; here we explicitly override it back up
# because shallow trees defeat the point of comparing prior strategies.
N_SIMS="${N_SIMS:-60}"
REWARD_BACKEND="${REWARD_BACKEND:-imagereward}"
EVAL_BACKENDS="${EVAL_BACKENDS:-imagereward hpsv3}"

BACKENDS="${BACKENDS:-sid senseflow_large}"
VARIANTS="${VARIANTS:-default wide_topk large_pool vanilla_refine improved_refine hybrid_refine}"
FAIL_FAST="${FAIL_FAST:-0}"

# ── Shared run knobs (identical across all cells) ───────────────────────────
export METHODS="baseline bon_mcts"
export PROMPT_FILE START_INDEX=0 END_INDEX="${NUM_PROMPTS}"
export SEED N_SIMS REWARD_BACKEND EVAL_BACKENDS
export REWARD_TYPE="${REWARD_BACKEND}"
export REWARD_BACKENDS="${REWARD_BACKEND}"
export N_VARIANTS=1 USE_QWEN=0 PRECOMPUTE_REWRITES=0 REWARDS_OVERWRITE=0
export CORRECTION_STRENGTHS="0.0"
export SAVE_BEST_IMAGES=1 SAVE_IMAGES=0 SAVE_VARIANTS=0
export EVAL_BEST_IMAGES=1 EVAL_REWARD_DEVICE=cuda EVAL_ALLOW_MISSING_BACKENDS=0

_apply_variant() {
    local variant="$1"
    case "${variant}" in
        default)
            export BON_MCTS_N_SEEDS=8
            export BON_MCTS_TOPK=2
            export BON_MCTS_SIM_ALLOC=split
            export BON_MCTS_MIN_SIMS=8
            export BON_MCTS_REFINE_METHOD=ours_tree
            unset BON_MCTS_PRESCREEN_CFG
            ;;
        wide_topk)
            export BON_MCTS_N_SEEDS=8
            export BON_MCTS_TOPK=4
            export BON_MCTS_SIM_ALLOC=split
            export BON_MCTS_MIN_SIMS=4
            export BON_MCTS_REFINE_METHOD=ours_tree
            unset BON_MCTS_PRESCREEN_CFG
            ;;
        large_pool)
            export BON_MCTS_N_SEEDS=16
            export BON_MCTS_TOPK=2
            export BON_MCTS_SIM_ALLOC=split
            export BON_MCTS_MIN_SIMS=8
            export BON_MCTS_REFINE_METHOD=ours_tree
            unset BON_MCTS_PRESCREEN_CFG
            ;;
        vanilla_refine)
            export BON_MCTS_N_SEEDS=8
            export BON_MCTS_TOPK=2
            export BON_MCTS_SIM_ALLOC=split
            export BON_MCTS_MIN_SIMS=8
            export BON_MCTS_REFINE_METHOD=mcts
            unset BON_MCTS_PRESCREEN_CFG
            ;;
        improved_refine)
            export BON_MCTS_N_SEEDS=8
            export BON_MCTS_TOPK=2
            export BON_MCTS_SIM_ALLOC=split
            export BON_MCTS_MIN_SIMS=8
            export BON_MCTS_REFINE_METHOD=mcts_improved
            unset BON_MCTS_PRESCREEN_CFG
            ;;
        hybrid_refine)
            export BON_MCTS_N_SEEDS=8
            export BON_MCTS_TOPK=2
            export BON_MCTS_SIM_ALLOC=split
            export BON_MCTS_MIN_SIMS=8
            export BON_MCTS_REFINE_METHOD=hybrid_ut_dt
            unset BON_MCTS_PRESCREEN_CFG
            ;;
        *)
            echo "[ablation] ERROR unknown variant '${variant}'" >&2
            return 1
            ;;
    esac
}

_apply_backend() {
    local backend="$1"
    export SD35_BACKEND="${backend}"
    case "${backend}" in
        sid)
            export STEPS=4
            export BASELINE_CFG=1.0
            export CFG_SCALES="1.0"
            ;;
        senseflow_large)
            export STEPS=4
            export BASELINE_CFG=1.0
            export CFG_SCALES="1.0"
            ;;
        sd35_base)
            export STEPS=28
            export BASELINE_CFG=4.5
            export CFG_SCALES="3.5 4.0 4.5 5.0 5.5 6.0 7.0"
            ;;
        *)
            echo "[ablation] ERROR unknown backend '${backend}'" >&2
            return 1
            ;;
    esac
}

failed=()
for backend in ${BACKENDS}; do
    if ! _apply_backend "${backend}"; then
        failed+=("${backend}/?")
        if [[ "${FAIL_FAST}" == "1" ]]; then exit 1; fi
        continue
    fi
    for variant in ${VARIANTS}; do
        if ! _apply_variant "${variant}"; then
            failed+=("${backend}/${variant}")
            if [[ "${FAIL_FAST}" == "1" ]]; then exit 1; fi
            continue
        fi
        cell_root="${RUN_ROOT}/${backend}/${variant}"
        mkdir -p "${cell_root}"
        export OUT_ROOT="${cell_root}"

        echo
        echo "================================================================"
        echo "[ablation] backend=${backend}  variant=${variant}"
        echo "  n_seeds=${BON_MCTS_N_SEEDS} topk=${BON_MCTS_TOPK} alloc=${BON_MCTS_SIM_ALLOC}"
        echo "  min_sims=${BON_MCTS_MIN_SIMS} refine=${BON_MCTS_REFINE_METHOD}"
        echo "  N_SIMS=${N_SIMS} BASELINE_CFG=${BASELINE_CFG} STEPS=${STEPS}"
        echo "  out=${cell_root}"
        echo "================================================================"

        if bash "${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"; then
            echo "[ablation] OK ${backend}/${variant}"
        else
            rc=$?
            echo "[ablation] FAIL ${backend}/${variant} rc=${rc}" >&2
            failed+=("${backend}/${variant}")
            if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
        fi
    done
done

echo
if (( ${#failed[@]} > 0 )); then
    echo "[ablation] DONE with failures: ${failed[*]}"
    exit 1
fi
echo "[ablation] DONE all (${BACKENDS} × ${VARIANTS}) cells OK."
echo "[ablation] Compare results: python3 bon_mcts_compare.py --root ${RUN_ROOT}"
