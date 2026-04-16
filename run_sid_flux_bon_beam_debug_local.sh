#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

# One-file merged debug runner for:
# - SD3.5 SiD backend (sid)
# - FLUX.1-schnell backend (flux)
# Search methods:
# - bon
# - beam
#
# Usage:
#   bash run_sid_flux_bon_beam_debug_local.sh
#   START_INDEX=0 END_INDEX=9 NUM_GPUS=1 bash run_sid_flux_bon_beam_debug_local.sh
#   METHODS="bon beam" BON_N=32 BEAM_WIDTH=8 bash run_sid_flux_bon_beam_debug_local.sh

DEFAULT_PROMPT_FILE="${SCRIPT_DIR}/hpsv2_subset.txt"
if [[ -f "${DEFAULT_PROMPT_FILE}" ]]; then
  PROMPT_FILE="${PROMPT_FILE:-${DEFAULT_PROMPT_FILE}}"
fi

OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/sid_flux_bon_beam_debug_out}"

# Merge both models in one run.
RUN_SANA="${RUN_SANA:-0}"
RUN_SD35="${RUN_SD35:-1}"
RUN_FLUX="${RUN_FLUX:-1}"
SD35_BACKEND="${SD35_BACKEND:-sid}"
FLUX_BACKEND="${FLUX_BACKEND:-flux}"

# bon/beam only by default for focused debugging.
METHODS="${METHODS:-bon beam}"
FLUX_METHODS="${FLUX_METHODS:-${METHODS}}"

# Debug-friendly defaults: one GPU + one prompt.
NUM_GPUS="${NUM_GPUS:-1}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-0}"
SEED="${SEED:-42}"

# Save artifacts for inspection.
SAVE_IMAGES="${SAVE_IMAGES:-1}"
SAVE_BEST_IMAGES="${SAVE_BEST_IMAGES:-1}"
SAVE_VARIANTS="${SAVE_VARIANTS:-1}"

# Keep rewards on CUDA for both search and post-eval.
REWARD_BACKEND="${REWARD_BACKEND:-imagereward}"
REWARD_BACKENDS="${REWARD_BACKENDS:-${REWARD_BACKEND}}"
REWARD_TYPE="${REWARD_TYPE:-${REWARD_BACKEND}}"
REWARD_DEVICE="${REWARD_DEVICE:-cuda}"
EVAL_BEST_IMAGES="${EVAL_BEST_IMAGES:-1}"
EVAL_BACKENDS="${EVAL_BACKENDS:-imagereward}"
EVAL_REWARD_DEVICE="${EVAL_REWARD_DEVICE:-cuda}"
EVAL_ALLOW_MISSING_BACKENDS="${EVAL_ALLOW_MISSING_BACKENDS:-1}"

# Search knobs.
BON_N="${BON_N:-16}"
BEAM_WIDTH="${BEAM_WIDTH:-4}"
N_VARIANTS="${N_VARIANTS:-4}"
STEPS="${STEPS:-4}"
CFG_SCALES="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5}"
BASELINE_CFG="${BASELINE_CFG:-1.0}"

# Reduce startup stalls from mounted storage.
LOCAL_REWARD_CACHE_ENABLE="${LOCAL_REWARD_CACHE_ENABLE:-1}"
LOCAL_REWARD_CACHE_ROOT="${LOCAL_REWARD_CACHE_ROOT:-/tmp/sid_reward_cache}"

export OUT_ROOT PROMPT_FILE
export RUN_SANA RUN_SD35 RUN_FLUX SD35_BACKEND FLUX_BACKEND
export METHODS FLUX_METHODS
export NUM_GPUS START_INDEX END_INDEX SEED
export SAVE_IMAGES SAVE_BEST_IMAGES SAVE_VARIANTS
export REWARD_BACKEND REWARD_BACKENDS REWARD_TYPE REWARD_DEVICE
export EVAL_BEST_IMAGES EVAL_BACKENDS EVAL_REWARD_DEVICE EVAL_ALLOW_MISSING_BACKENDS
export BON_N BEAM_WIDTH N_VARIANTS STEPS CFG_SCALES BASELINE_CFG
export LOCAL_REWARD_CACHE_ENABLE LOCAL_REWARD_CACHE_ROOT

echo "[sid+flux bon/beam debug]"
echo "  out_root=${OUT_ROOT}"
echo "  prompt_file=${PROMPT_FILE:-<default in suite>}"
echo "  range=[${START_INDEX}, ${END_INDEX}] num_gpus=${NUM_GPUS}"
echo "  methods=${METHODS} flux_methods=${FLUX_METHODS}"
echo "  sd35_backend=${SD35_BACKEND} flux_backend=${FLUX_BACKEND}"
echo "  reward=${REWARD_BACKEND} eval_backends=${EVAL_BACKENDS}"
echo "  bon_n=${BON_N} beam_width=${BEAM_WIDTH} n_variants=${N_VARIANTS}"
echo

bash "${SCRIPT_DIR}/hpsv2_sd35_flux_allalgos.sh" "$@"

