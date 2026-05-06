#!/usr/bin/env bash
# Run baseline + greedy + sop + bon on the 3 distilled (4-step) backends,
# scaled so each search method consumes ~256 NFE per prompt.
#
# Backends: sid, senseflow_large, flux_schnell.
#
# NFE accounting (4 transformer steps per sample):
#   bon     : BON_N samples × steps                    = 64 × 4 = 256
#   greedy  : (n_var × |cfg_scales|) actions × steps   = action × 4
#             - sid / senseflow: N_VAR=4, |cfg|=16  → 64 × 4 = 256
#             - flux_schnell  : N_VAR=64, |cfg|=1   → 64 × 4 = 256
#   sop     : N_init + K·M·branch_steps + K            ≈ 256
#             with N=16, M=8, K=8, branch_every=1, end_frac=1.0
#
# Caller env (from AMLT yaml):
#   RUN_ROOT, REWARD_SERVER_URL, SEARCH_REWARD ∈ {imagereward, hpsv3}
#
# Optional:
#   BACKENDS  (default "sid senseflow_large flux_schnell")
#   SEEDS     (default "42")
#   N_PROMPTS (default 50)
#   FAIL_FAST (default 0)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "greedy-sop-bon-distilled"

: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${REWARD_SERVER_URL:?REWARD_SERVER_URL must be set}"

SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"
case "${SEARCH_REWARD}" in
    imagereward|hpsv3) : ;;
    *) echo "[gsbd] ERROR unknown SEARCH_REWARD='${SEARCH_REWARD}'" >&2; exit 1 ;;
esac

BACKENDS="${BACKENDS:-sid senseflow_large flux_schnell}"
SEEDS="${SEEDS:-42}"
N_PROMPTS="${N_PROMPTS:-50}"
FAIL_FAST="${FAIL_FAST:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ── Methods + shared knobs ──────────────────────────────────────────────────
export METHODS="baseline greedy sop bon"
export START_INDEX=0
export END_INDEX="${N_PROMPTS}"
export REWARDS_OVERWRITE=0
export CORRECTION_STRENGTHS="0.0"
export SAVE_BEST_IMAGES=1
export SAVE_IMAGES=0
export SAVE_VARIANTS=1
export EVAL_BACKENDS="imagereward hpsv3"
export REWARD_BACKEND="${SEARCH_REWARD}"
export REWARD_TYPE="${SEARCH_REWARD}"
export REWARD_BACKENDS="${SEARCH_REWARD}"
export EVAL_ALLOW_MISSING_BACKENDS=1
export EVAL_BEST_IMAGES=1
export EVAL_REWARD_DEVICE=cuda

# ── BoN: K=64 → 256 NFE on 4-step ──────────────────────────────────────────
export BON_N=64

# ── SoP: ~256 NFE on 4-step (N=16, M=8, K=8, branch_every=1) ───────────────
export SOP_INIT_PATHS=16
export SOP_BRANCH_FACTOR=8
export SOP_KEEP_TOP=8
export SOP_BRANCH_EVERY=1
export SOP_END_FRAC=1.0
export SOP_SCORE_DECODE="x0_pred"
export SOP_VARIANT_IDX=0

# Greedy needs Qwen rewrites for n_var > 1 → fixed across all 3 backends.
export USE_QWEN=1
export PRECOMPUTE_REWRITES=1
export QWEN_ID="${QWEN_ID:-Qwen/Qwen3-4B}"

PROMPTS_DIR="${RUN_ROOT}/_prompts"
mkdir -p "${PROMPTS_DIR}"

# Linspace helper (for cfg bank sizing).
_linspace() {
    local lo="$1" hi="$2" n="$3"
    "${PYTHON_BIN}" -c "
import numpy as np, sys
lo, hi, n = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
print(' '.join(f'{x:.4f}' for x in np.linspace(lo, hi, n)))
" "${lo}" "${hi}" "${n}"
}

_apply_backend() {
    case "$1" in
        sid)
            export SD35_BACKEND=sid; unset FLUX_BACKEND || true
            export STEPS=4; export BASELINE_CFG=1.0
            # 4 variants × 16 cfg = 64 → 256 NFE for greedy.
            export N_VARIANTS=4
            export CFG_SCALES="$(_linspace 1.0 2.5 16)"
            export _SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
            ;;
        senseflow_large)
            export SD35_BACKEND=senseflow_large; unset FLUX_BACKEND || true
            export STEPS=4; export BASELINE_CFG=1.0
            export N_VARIANTS=4
            export CFG_SCALES="$(_linspace 1.0 2.5 16)"
            export _SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
            ;;
        flux_schnell)
            # CFG-distilled (cfg=0). Hit 256 NFE for greedy via N_VAR=64 instead.
            export FLUX_BACKEND=flux; unset SD35_BACKEND || true
            export STEPS=4
            export MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-schnell}"
            export BASELINE_GUIDANCE_SCALE=0.0; export BASELINE_CFG=0.0
            export N_VARIANTS=64
            export CFG_SCALES="0.0"
            export _SUITE="${SCRIPT_DIR}/hpsv2_flux_schnell_ddp_suite.sh"
            ;;
        *) echo "[gsbd] ERROR unknown backend '$1'" >&2; return 1 ;;
    esac
}

_sample_prompts() {
    local backend="$1"
    local prompt_file="${PROMPTS_DIR}/backend_${backend}.txt"
    if [[ ! -f "${prompt_file}" ]]; then
        echo "[gsbd] sampling prompts → ${prompt_file}"
        env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
            "${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_prompts.py" \
            --n_prompts "${N_PROMPTS}" \
            --out_dir "${PROMPTS_DIR}" \
            --backends "${backend}"
    fi
    export PROMPT_FILE="${prompt_file}"
}

# ── Banner ──────────────────────────────────────────────────────────────────
echo "================================================================"
echo "[gsbd]   greedy + sop + bon @ ~256 NFE on distilled backends"
echo "  backends=${BACKENDS}  search_reward=${SEARCH_REWARD}"
echo "  N_PROMPTS=${N_PROMPTS}  SEEDS=${SEEDS}"
echo "  bon:    BON_N=${BON_N}"
echo "  sop:    INIT=${SOP_INIT_PATHS} M=${SOP_BRANCH_FACTOR} K=${SOP_KEEP_TOP} every=${SOP_BRANCH_EVERY}"
echo "  greedy: per-backend N_VARIANTS × |CFG|"
echo "================================================================"

failed=()
for backend in ${BACKENDS}; do
    _apply_backend "${backend}"
    _sample_prompts "${backend}"

    # Per-backend Qwen rewrites cache.
    export REWRITES_FILE="${PROMPTS_DIR}/${backend}_qwen_rewrites.json"

    for seed in ${SEEDS}; do
        cell_root="${RUN_ROOT}/${SEARCH_REWARD}/${backend}/seed${seed}"
        mkdir -p "${cell_root}"
        export OUT_ROOT="${cell_root}"

        echo
        echo "================================================================"
        echo "[gsbd] backend=${backend}  seed=${seed}  reward=${SEARCH_REWARD}"
        echo "  STEPS=${STEPS}  CFG_SCALES='${CFG_SCALES}'  N_VARIANTS=${N_VARIANTS}"
        echo "  out=${cell_root}"
        echo "================================================================"

        if SEED="${seed}" bash "${_SUITE}"; then
            echo "[gsbd] OK ${backend}/seed${seed}"
        else
            rc=$?
            echo "[gsbd] FAIL ${backend}/seed${seed} rc=${rc}" >&2
            failed+=("${backend}/seed${seed}")
            if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
        fi
    done
done

if (( ${#failed[@]} > 0 )); then
    echo "[gsbd] phase-1 DONE with failures: ${failed[*]}"
fi
echo "[gsbd] phase-1 DONE."

# ── Phase-2 posthoc: HPSv3 + PickScore on saved best_images ─────────────────
if [[ "${RUN_POSTHOC:-1}" == "1" ]]; then
    echo
    echo "================================================================"
    echo "[gsbd] phase-2 posthoc: hpsv3 + pickscore on saved best_images"
    echo "================================================================"
    posthoc_failed=()
    for backend in ${BACKENDS}; do
        case "${backend}" in
            flux*) layout=flux ;;
            *)     layout=sd35 ;;
        esac
        for seed in ${SEEDS}; do
            cell_root="${RUN_ROOT}/${SEARCH_REWARD}/${backend}/seed${seed}"
            [[ -d "${cell_root}" ]] || continue
            for method in greedy sop bon; do
                for method_dir in $(find "${cell_root}" -maxdepth 3 -type d -name "${method}" 2>/dev/null); do
                    echo "[posthoc] eval ${method_dir} (layout=${layout})"
                    if "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_best_images_multi_reward.py" \
                        --layout "${layout}" \
                        --method_out "${method_dir}" \
                        --method "${method}" \
                        --backends hpsv3 pickscore \
                        --reward_device cuda \
                        --out_json "${method_dir}/best_images_posthoc.json" \
                        --out_aggregate "${method_dir}/best_images_posthoc_aggregate.json" \
                        --allow_missing_backends; then
                        :
                    else
                        echo "[posthoc] WARN failed: ${method_dir}" >&2
                        posthoc_failed+=("${backend}/${method}/seed${seed}")
                    fi
                done
            done
        done
    done
    if (( ${#posthoc_failed[@]} > 0 )); then
        echo "[gsbd] phase-2 posthoc WARN failures: ${posthoc_failed[*]}"
    else
        echo "[gsbd] phase-2 posthoc DONE."
    fi
fi

echo "[gsbd] ALL DONE."
echo "  Output: ${RUN_ROOT}"
if (( ${#failed[@]} > 0 )); then exit 1; fi
