#!/usr/bin/env bash
# Deploy dynamic_cfg_x0 search method on senseflow_large + flux_schnell.
#
# Existing yamls already cover dynamic_cfg_x0 on sid + sd35_base
# (sd35_dynamic_cfg_x0_server.yaml) and flux + tdd_flux
# (flux_dynamic_cfg_x0_server.yaml). This bash fills the gap for the two
# remaining backends.
#
# Method = dynamic_cfg_x0: per-step adaptive CFG via decoded-x0 reward scoring.
# Each cell runs:
#   baseline             — single-pass at baseline_cfg
#   dynamic_cfg_x0       — adaptive per-step CFG selection
#
# Caller env (typically AMLT yaml):
#   RUN_ROOT, REWARD_SERVER_URL
#
# Optional:
#   BACKENDS  (default "senseflow_large flux_schnell")
#   SEEDS     (default "42")
#   N_PROMPTS (default 50)
#   FAIL_FAST (default 0)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "dynamic-cfg-deploy"

: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${REWARD_SERVER_URL:?REWARD_SERVER_URL must be set}"

BACKENDS="${BACKENDS:-senseflow_large flux_schnell}"
FAIL_FAST="${FAIL_FAST:-0}"

N_PROMPTS="${N_PROMPTS:-50}"
SEEDS="${SEEDS:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ── Shared knobs (search-time) ──────────────────────────────────────────────
export METHODS="baseline dynamic_cfg_x0"
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
export EVAL_BACKENDS="imagereward"
export REWARD_BACKEND="imagereward"
export REWARD_TYPE="imagereward"
export REWARD_BACKENDS="imagereward"
export EVAL_ALLOW_MISSING_BACKENDS=1
export EVAL_BEST_IMAGES=1
export EVAL_REWARD_DEVICE=cuda
export AUTO_BACKEND_STEPS=0

# ── Dynamic CFG x0 knobs (shared defaults) ─────────────────────────────────
export DYNAMIC_CFG_X0_SCORE_START_FRAC="${DYNAMIC_CFG_X0_SCORE_START_FRAC:-0.5}"
export DYNAMIC_CFG_X0_SCORE_END_FRAC="${DYNAMIC_CFG_X0_SCORE_END_FRAC:-1.0}"
export DYNAMIC_CFG_X0_SCORE_EVERY="${DYNAMIC_CFG_X0_SCORE_EVERY:-1}"
export DYNAMIC_CFG_X0_EVALUATORS="imagereward"
export DYNAMIC_CFG_X0_WEIGHT_SCHEDULE="${DYNAMIC_CFG_X0_WEIGHT_SCHEDULE:-piecewise}"
export DYNAMIC_CFG_X0_PROMPT_TYPE="${DYNAMIC_CFG_X0_PROMPT_TYPE:-general}"
export DYNAMIC_CFG_X0_CONFIDENCE_GATING="${DYNAMIC_CFG_X0_CONFIDENCE_GATING:-1}"
export DYNAMIC_CFG_X0_SMOOTH_WEIGHT="${DYNAMIC_CFG_X0_SMOOTH_WEIGHT:-0.0}"
export DYNAMIC_CFG_X0_HIGH_CFG_PENALTY="${DYNAMIC_CFG_X0_HIGH_CFG_PENALTY:-0.01}"
export DYNAMIC_CFG_X0_ADD_LOCAL_NEIGHBORHOOD="${DYNAMIC_CFG_X0_ADD_LOCAL_NEIGHBORHOOD:-0}"

PROMPTS_DIR="${RUN_ROOT}/_prompts"
mkdir -p "${PROMPTS_DIR}"

# ── Per-backend env ─────────────────────────────────────────────────────────
_apply_backend() {
    case "$1" in
        senseflow_large)
            export SD35_BACKEND=senseflow_large; unset FLUX_BACKEND || true
            export STEPS=4; export BASELINE_CFG=1.0
            export CFG_SCALES="1.0 1.25 1.5 1.75 2.0 2.25 2.5"
            export DYNAMIC_CFG_X0_GRID="1.0 1.25 1.5 1.75 2.0 2.25 2.5"
            export DYNAMIC_CFG_X0_CFG_MIN=0.5
            export DYNAMIC_CFG_X0_CFG_MAX=3.0
            export DYNAMIC_CFG_X0_CFG_SOFT_MAX=2.5
            export _SUITE="${SCRIPT_DIR}/hpsv2_sd35_flux_allalgos.sh"
            export RUN_SANA=0; export RUN_SD35=1; export RUN_FLUX=0
            ;;
        flux_schnell)
            export FLUX_BACKEND=flux_schnell; unset SD35_BACKEND || true
            export STEPS=4
            export MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-schnell}"
            # NOTE: flux_schnell is CFG-distilled at 0. Grid is narrow — the
            # method still picks per-step but headroom is small.
            export BASELINE_GUIDANCE_SCALE=0.0
            export BASELINE_CFG=0.0
            export CFG_SCALES="0.0 0.25 0.5 0.75 1.0"
            export DYNAMIC_CFG_X0_GRID="0.0 0.25 0.5 0.75 1.0"
            export DYNAMIC_CFG_X0_CFG_MIN=0.0
            export DYNAMIC_CFG_X0_CFG_MAX=1.5
            export DYNAMIC_CFG_X0_CFG_SOFT_MAX=1.0
            export _SUITE="${SCRIPT_DIR}/hpsv2_sd35_flux_allalgos.sh"
            export RUN_SANA=0; export RUN_SD35=0; export RUN_FLUX=1
            ;;
        *) echo "[dyncfg-deploy] ERROR unknown backend '$1'" >&2; return 1 ;;
    esac
}

_sample_prompts() {
    local backend="$1"
    local prompt_file="${PROMPTS_DIR}/backend_${backend}.txt"
    if [[ ! -f "${prompt_file}" ]]; then
        echo "[dyncfg-deploy] sampling prompts → ${prompt_file}"
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
echo "[dyncfg-deploy]   dynamic_cfg_x0 deploy"
echo "  backends=${BACKENDS}  N_PROMPTS=${N_PROMPTS}  SEEDS=${SEEDS}"
echo "  reward=imagereward (search), phase-2 posthoc=hpsv3+pickscore"
echo "================================================================"

failed=()
for backend in ${BACKENDS}; do
    _apply_backend "${backend}"
    _sample_prompts "${backend}"

    for seed in ${SEEDS}; do
        cell_root="${RUN_ROOT}/${backend}/seed${seed}"
        mkdir -p "${cell_root}"
        export OUT_ROOT="${cell_root}"

        echo
        echo "================================================================"
        echo "[dyncfg-deploy] backend=${backend}  seed=${seed}"
        echo "  STEPS=${STEPS} BASELINE_CFG=${BASELINE_CFG}"
        echo "  GRID='${DYNAMIC_CFG_X0_GRID}'  CFG_MIN=${DYNAMIC_CFG_X0_CFG_MIN}  CFG_MAX=${DYNAMIC_CFG_X0_CFG_MAX}"
        echo "  out=${cell_root}"
        echo "================================================================"

        if SEED="${seed}" bash "${_SUITE}"; then
            echo "[dyncfg-deploy] OK ${backend}/seed${seed}"
        else
            rc=$?
            echo "[dyncfg-deploy] FAIL ${backend}/seed${seed} rc=${rc}" >&2
            failed+=("${backend}/seed${seed}")
            if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
        fi
    done
done

if (( ${#failed[@]} > 0 )); then
    echo "[dyncfg-deploy] phase-1 DONE with failures: ${failed[*]}"
fi
echo "[dyncfg-deploy] phase-1 DONE."

# ── Phase-2 posthoc: HPSv3 + PickScore on saved best_images ─────────────────
if [[ "${RUN_POSTHOC:-1}" == "1" ]]; then
    echo
    echo "================================================================"
    echo "[dyncfg-deploy] phase-2 posthoc: hpsv3 + pickscore on saved best_images"
    echo "================================================================"
    posthoc_failed=()
    for backend in ${BACKENDS}; do
        for seed in ${SEEDS}; do
            cell_root="${RUN_ROOT}/${backend}/seed${seed}"
            [[ -d "${cell_root}" ]] || continue
            for method_dir in $(find "${cell_root}" -maxdepth 3 -type d -name 'dynamic_cfg_x0' 2>/dev/null); do
                echo "[posthoc] eval ${method_dir}"
                if "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_best_images_multi_reward.py" \
                    --method_out "${method_dir}" \
                    --method dynamic_cfg_x0 \
                    --backends hpsv3 pickscore \
                    --reward_device cuda \
                    --out_json "${method_dir}/best_images_posthoc.json" \
                    --out_aggregate "${method_dir}/best_images_posthoc_aggregate.json" \
                    --allow_missing_backends; then
                    :
                else
                    echo "[posthoc] WARN failed: ${method_dir}" >&2
                    posthoc_failed+=("${backend}/seed${seed}")
                fi
            done
        done
    done
    if (( ${#posthoc_failed[@]} > 0 )); then
        echo "[dyncfg-deploy] phase-2 posthoc WARN failures: ${posthoc_failed[*]}"
    else
        echo "[dyncfg-deploy] phase-2 posthoc DONE."
    fi
fi

echo "[dyncfg-deploy] ALL DONE."
echo "  Output: ${RUN_ROOT}"
if (( ${#failed[@]} > 0 )); then exit 1; fi
