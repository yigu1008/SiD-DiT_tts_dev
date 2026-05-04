#!/usr/bin/env bash
# Single-bash: bon_mcts + baseline at composite_hpsv3_ir reward, across all
# 4 backends (sid, senseflow_large, sd35_base, flux_schnell).
#
# One reward server boot, one env build (paid by the AMLT yaml). The bash
# loops sequentially over backends and dispatches to the matching suite.
#
# Caller env (from AMLT yaml):
#   RUN_ROOT           - output dir parent
#   REWARD_SERVER_URL  - shared reward server URL (must host hpsv3 + imagereward)
#
# Optional:
#   BACKENDS           (default: all 4)
#   N_PROMPTS          (default 30 — keeps total wallclock reasonable)
#   SEEDS              (default "42 43")
#   N_SIMS             (default 30)
#   FAIL_FAST          (default 0)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "composite-all-backends"

: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${REWARD_SERVER_URL:?REWARD_SERVER_URL must be set}"

BACKENDS="${BACKENDS:-sid senseflow_large sd35_base flux_schnell}"
N_PROMPTS="${N_PROMPTS:-30}"
SEEDS="${SEEDS:-42 43}"
N_SIMS="${N_SIMS:-30}"
FAIL_FAST="${FAIL_FAST:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ── Shared run knobs ────────────────────────────────────────────────────────
export METHODS="baseline bon_mcts"
export START_INDEX=0
export END_INDEX="${N_PROMPTS}"
export N_VARIANTS=1
export USE_QWEN=0
export PRECOMPUTE_REWRITES=0
export REWARDS_OVERWRITE=0
export CORRECTION_STRENGTHS="0.0"
export SAVE_BEST_IMAGES=1
export SAVE_IMAGES=0
export SAVE_VARIANTS=0
export EVAL_BEST_IMAGES=1
export EVAL_REWARD_DEVICE=cuda
export EVAL_ALLOW_MISSING_BACKENDS=0

# ── Composite reward (the half-half normalized hpsv3+imagereward) ────────────
export REWARD_BACKEND="composite_hpsv3_ir"
export REWARD_TYPE="composite_hpsv3_ir"
export REWARD_BACKENDS="composite_hpsv3_ir"
# Eval covers both raw rewards so we can decompose the composite gain.
export EVAL_BACKENDS="imagereward hpsv3"

# ── Anchored bon_mcts knobs (matches default cell of mcts_param ablation) ───
export N_SIMS UCB_C="${UCB_C:-1.0}"
export BON_MCTS_N_SEEDS=8
export BON_MCTS_TOPK=2
export BON_MCTS_SIM_ALLOC=split
export BON_MCTS_MIN_SIMS=8
export BON_MCTS_REFINE_METHOD=mcts        # vanilla refine

# Reuse cherry_pick prompt sampler (HPSv2 + DrawBench).
PROMPTS_DIR="${RUN_ROOT}/_prompts"
mkdir -p "${PROMPTS_DIR}"

# ── Per-backend dispatch ────────────────────────────────────────────────────
_run_one_backend() {
    local backend="$1"
    local prompt_file="${PROMPTS_DIR}/backend_${backend}.txt"

    if [[ ! -f "${prompt_file}" ]]; then
        echo "[composite] sampling prompts → ${prompt_file}"
        env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
            "${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_prompts.py" \
            --n_prompts "${N_PROMPTS}" \
            --out_dir "${PROMPTS_DIR}" \
            --backends "${backend}"
    fi
    export PROMPT_FILE="${prompt_file}"

    local suite suite_kind
    case "${backend}" in
        sid)
            suite="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"; suite_kind="sd35"
            export SD35_BACKEND=sid; export STEPS=4
            export BASELINE_CFG=1.0; export CFG_SCALES="1.0"
            unset FLUX_BACKEND || true ;;
        senseflow_large)
            suite="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"; suite_kind="sd35"
            export SD35_BACKEND=senseflow_large; export STEPS=4
            export BASELINE_CFG=1.0; export CFG_SCALES="1.0"
            unset FLUX_BACKEND || true ;;
        sd35_base)
            suite="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"; suite_kind="sd35"
            export SD35_BACKEND=sd35_base; export STEPS=28
            export BASELINE_CFG=4.5; export CFG_SCALES="3.5 4.0 4.5 5.0 5.5 6.0 7.0"
            unset FLUX_BACKEND || true ;;
        flux_schnell)
            suite="${SCRIPT_DIR}/hpsv2_flux_schnell_ddp_suite.sh"; suite_kind="flux"
            export FLUX_BACKEND=flux; export STEPS=4
            export MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-schnell}"
            export BASELINE_GUIDANCE_SCALE=0.0; export BASELINE_CFG=0.0; export CFG_SCALES="0.0"
            unset SD35_BACKEND || true ;;
        *)
            echo "[composite] ERROR unknown backend '${backend}'" >&2; return 2 ;;
    esac

    local seed_failed=()
    for seed in ${SEEDS}; do
        local cell_root="${RUN_ROOT}/${backend}/seed${seed}"
        mkdir -p "${cell_root}"
        export OUT_ROOT="${cell_root}"
        echo
        echo "================================================================"
        echo "[composite] backend=${backend}  seed=${seed}  reward=composite_hpsv3_ir"
        echo "  N_SIMS=${N_SIMS} TOPK=${BON_MCTS_TOPK} N_SEEDS=${BON_MCTS_N_SEEDS}"
        echo "  out=${cell_root}"
        echo "================================================================"
        if SEED="${seed}" bash "${suite}"; then
            echo "[composite] OK ${backend}/seed${seed}"
        else
            local rc=$?
            echo "[composite] FAIL ${backend}/seed${seed} rc=${rc}" >&2
            seed_failed+=("${seed}")
            if [[ "${FAIL_FAST}" == "1" ]]; then return "${rc}"; fi
        fi
    done

    if (( ${#seed_failed[@]} > 0 )); then
        echo "[composite] WARN ${backend} seed failures: ${seed_failed[*]}"
        return 1
    fi
    return 0
}

# ── Loop ────────────────────────────────────────────────────────────────────
echo "[composite] backends=${BACKENDS}  reward=composite_hpsv3_ir"
echo "[composite] N_PROMPTS=${N_PROMPTS}  SEEDS=${SEEDS}  N_SIMS=${N_SIMS}"

backend_failed=()
for backend in ${BACKENDS}; do
    if _run_one_backend "${backend}"; then
        echo "[composite] DONE OK ${backend}"
    else
        rc=$?
        echo "[composite] FAIL backend=${backend} rc=${rc}" >&2
        backend_failed+=("${backend}")
        if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
    fi
done

if (( ${#backend_failed[@]} > 0 )); then
    echo "[composite] DONE with backend failures: ${backend_failed[*]}"
    exit 1
fi
echo "[composite] DONE all backends OK"
echo "  Compare aggregate_ddp.json files under ${RUN_ROOT}/<backend>/seed*/"
