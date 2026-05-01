#!/usr/bin/env bash
# Single-script driver for SoP across all 4 backends.
#
# Loops sequentially over: sd35_sid, senseflow_large, sd35_base, flux_schnell.
# Per-backend, sets the matching env (SD35_BACKEND / FLUX_BACKEND, STEPS,
# BASELINE_CFG, SOP_START_FRAC, OUT_ROOT) and invokes the appropriate suite
# (hpsv2_sd35_sid_ddp_suite.sh for SD3.5-family, hpsv2_flux_schnell_ddp_suite.sh
# for FLUX).
#
# Caller is expected to have set the shared knobs:
#   RUN_ROOT          - output dir parent (one subdir per backend will be created)
#   PROMPT_FILE       - prompt list
#   START_INDEX, END_INDEX
#   SEED
#   REWARD_SERVER_URL - shared reward server URL
#   REWARD_BACKEND    - search-time reward (hpsv3 or imagereward)
#   EVAL_BACKENDS     - eval-time reward backends (e.g. "imagereward hpsv3")
#   NUM_GPUS, CUDA_VISIBLE_DEVICES   - sampling GPUs (server already on GPU 0)
#   SOP_INIT_PATHS, SOP_BRANCH_FACTOR, SOP_KEEP_TOP, SOP_BRANCH_EVERY,
#   SOP_END_FRAC, SOP_SCORE_DECODE, SOP_VARIANT_IDX
#
# Optional:
#   BACKENDS          - space-separated subset of {sd35_sid senseflow sd35_base flux_schnell}
#                       (default: all four)
#   FAIL_FAST         - "1" to abort on first backend failure (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${PROMPT_FILE:?PROMPT_FILE must be set}"
: "${REWARD_BACKEND:?REWARD_BACKEND must be set}"

BACKENDS="${BACKENDS:-sd35_sid senseflow sd35_base flux_schnell}"
FAIL_FAST="${FAIL_FAST:-0}"

# ── Shared run knobs (sane defaults; caller can override) ───────────────────
export METHODS="${METHODS:-baseline sop}"
export N_VARIANTS="${N_VARIANTS:-1}"
export USE_QWEN="${USE_QWEN:-0}"
export PRECOMPUTE_REWRITES="${PRECOMPUTE_REWRITES:-0}"
export REWARDS_OVERWRITE="${REWARDS_OVERWRITE:-0}"
export CORRECTION_STRENGTHS="${CORRECTION_STRENGTHS:-0.0}"
export SAVE_BEST_IMAGES="${SAVE_BEST_IMAGES:-1}"
export SAVE_IMAGES="${SAVE_IMAGES:-0}"
export SAVE_VARIANTS="${SAVE_VARIANTS:-0}"
export EVAL_BEST_IMAGES="${EVAL_BEST_IMAGES:-1}"
export EVAL_REWARD_DEVICE="${EVAL_REWARD_DEVICE:-cuda}"
export EVAL_ALLOW_MISSING_BACKENDS="${EVAL_ALLOW_MISSING_BACKENDS:-0}"
export EVAL_BACKENDS="${EVAL_BACKENDS:-imagereward hpsv3}"

# SoP defaults if caller didn't set them.
export SOP_INIT_PATHS="${SOP_INIT_PATHS:-8}"
export SOP_BRANCH_FACTOR="${SOP_BRANCH_FACTOR:-4}"
export SOP_KEEP_TOP="${SOP_KEEP_TOP:-4}"
export SOP_BRANCH_EVERY="${SOP_BRANCH_EVERY:-1}"
export SOP_END_FRAC="${SOP_END_FRAC:-1.0}"
export SOP_SCORE_DECODE="${SOP_SCORE_DECODE:-x0_pred}"
export SOP_VARIANT_IDX="${SOP_VARIANT_IDX:-0}"

# ── Per-backend dispatch ────────────────────────────────────────────────────

_run_one_backend() {
    local backend="$1"
    local backend_out="${RUN_ROOT}/${backend}"
    mkdir -p "${backend_out}"

    case "${backend}" in
        sd35_sid)
            export RUN_SD35=1
            export RUN_FLUX=0
            export SD35_BACKEND=sid
            export STEPS=4
            export BASELINE_CFG=1.0
            export CFG_SCALES='1.0'
            export SOP_START_FRAC=0.5    # 4-step: skip steps 0,1
            export OUT_ROOT="${backend_out}"
            echo "[sop-all] === ${backend} (sd35 suite) ==="
            bash "${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
            ;;
        senseflow|senseflow_large)
            export RUN_SD35=1
            export RUN_FLUX=0
            export SD35_BACKEND=senseflow_large
            export STEPS=4
            export BASELINE_CFG=0.0
            export CFG_SCALES='0.0'
            export SOP_START_FRAC=0.5
            export OUT_ROOT="${backend_out}"
            echo "[sop-all] === ${backend} (sd35 suite) ==="
            bash "${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
            ;;
        sd35_base)
            export RUN_SD35=1
            export RUN_FLUX=0
            export SD35_BACKEND=sd35_base
            export STEPS=28
            export BASELINE_CFG=4.5
            export CFG_SCALES='4.5'
            export SOP_START_FRAC=0.25   # 28-step: original sane window
            export OUT_ROOT="${backend_out}"
            echo "[sop-all] === ${backend} (sd35 suite) ==="
            bash "${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
            ;;
        flux_schnell)
            export RUN_SD35=0
            export RUN_FLUX=1
            export FLUX_BACKEND=flux
            export STEPS=4
            export BASELINE_GUIDANCE_SCALE=0.0
            export BASELINE_CFG=0.0
            export CFG_SCALES='0.0'
            export MODEL_ID=black-forest-labs/FLUX.1-schnell
            export SOP_START_FRAC=0.5
            export OUT_ROOT="${backend_out}"
            echo "[sop-all] === ${backend} (flux suite) ==="
            bash "${SCRIPT_DIR}/hpsv2_flux_schnell_ddp_suite.sh"
            ;;
        *)
            echo "[sop-all] ERROR: unknown backend '${backend}'" >&2
            return 1
            ;;
    esac
}

echo "[sop-all] backends: ${BACKENDS}"
echo "[sop-all] RUN_ROOT=${RUN_ROOT}"
echo "[sop-all] REWARD_BACKEND=${REWARD_BACKEND} REWARD_SERVER_URL=${REWARD_SERVER_URL:-<unset>}"
echo "[sop-all] SoP knobs: N=${SOP_INIT_PATHS} M=${SOP_BRANCH_FACTOR} K=${SOP_KEEP_TOP} every=${SOP_BRANCH_EVERY} end=${SOP_END_FRAC} score=${SOP_SCORE_DECODE}"

failed=()
for backend in ${BACKENDS}; do
    if _run_one_backend "${backend}"; then
        echo "[sop-all] OK ${backend}"
    else
        rc=$?
        echo "[sop-all] FAIL ${backend} (rc=${rc})" >&2
        failed+=("${backend}")
        if [[ "${FAIL_FAST}" == "1" ]]; then
            echo "[sop-all] FAIL_FAST=1, aborting"
            exit "${rc}"
        fi
    fi
done

if (( ${#failed[@]} > 0 )); then
    echo "[sop-all] DONE with failures: ${failed[*]}"
    exit 1
fi
echo "[sop-all] DONE all backends OK"
