#!/usr/bin/env bash
# Run hpsv2_sd35_flux_allalgos on an SSH / TACC cluster.
# Baked from amlt/terminal.yaml (params section, single-trial defaults).
#
# Usage:
#   bash run_tacc.sh                        # use all defaults
#   NUM_GPUS=4 bash run_tacc.sh             # override GPU count
#   START_INDEX=0 END_INDEX=100 bash run_tacc.sh
#
# For SLURM (TACC), submit with:
#   sbatch --nodes=1 --ntasks=1 --cpus-per-task=8 \
#          --gres=gpu:8 --time=24:00:00 \
#          --partition=gpu run_tacc.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Environment setup (installs + env vars)
# ---------------------------------------------------------------------------
# SKIP_INSTALL=1 skips all pip installs in tacc_setup.sh (env already set up).
# Defaults to 1 — set SKIP_INSTALL=0 to force a full reinstall.
export SKIP_INSTALL="${SKIP_INSTALL:-1}"
source "${SCRIPT_DIR}/tacc_setup.sh"

# ---------------------------------------------------------------------------
# Run config — baked from terminal.yaml params (override via env)
# ---------------------------------------------------------------------------
unset REWARD_BACKEND REWARD_TYPE REWARD_BACKENDS 2>/dev/null || true

NUM_GPUS="${NUM_GPUS:-$(python3 - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)}"
export NUM_GPUS

# Prompt / output
# Outputs go to $SCRATCH (large, OK for temporary results).
# Model caches stay on $WORK (set via DATA_ROOT in tacc_setup.sh).
_OUT_BASE="${SCRATCH:-${DATA_ROOT}}"
HPSV2_PROMPT_DIR="${HPSV2_PROMPT_DIR:-${_OUT_BASE}/hpsv2_prompt_cache}"
export HPSV2_PROMPT_DIR
RUN_TAG="${RUN_TAG:-hpsv2_sd35_flux_allalgos}"
export OUT_ROOT="${OUT_ROOT:-${_OUT_BASE}/hpsv2_all_models_runs/${RUN_TAG}}"
export PROMPT_STYLE="${PROMPT_STYLE:-all}"
export START_INDEX="${START_INDEX:-0}"
export END_INDEX="${END_INDEX:--1}"
export USE_SUBSET="${USE_SUBSET:-1}"

# Methods / reward
export METHODS="${METHODS:-baseline greedy mcts ga smc}"
export REWARD_BACKEND=imagereward
export REWARD_TYPE=imagereward
export REWARD_BACKENDS=imagereward
export EVAL_BEST_IMAGES="${EVAL_BEST_IMAGES:-1}"
export EVAL_BACKENDS="${EVAL_BACKENDS:-imagereward hpsv2 pickscore}"
export REWARD_DEVICE="${REWARD_DEVICE:-cpu}"
export EVAL_REWARD_DEVICE="${EVAL_REWARD_DEVICE:-cpu}"
export EVAL_ALLOW_MISSING_BACKENDS="${EVAL_ALLOW_MISSING_BACKENDS:-1}"

# Sampling
export CFG_SCALES="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5}"
export BASELINE_CFG="${BASELINE_CFG:-1.0}"
export STEPS="${STEPS:-4}"
export N_VARIANTS="${N_VARIANTS:-3}"
export SEED="${SEED:-42}"

# MCTS
export N_SIMS="${N_SIMS:-50}"
export UCB_C="${UCB_C:-1.41}"

# SMC
export SMC_K="${SMC_K:-8}"
export SMC_GAMMA="${SMC_GAMMA:-0.10}"
export ESS_THRESHOLD="${ESS_THRESHOLD:-0.5}"
export RESAMPLE_START_FRAC="${RESAMPLE_START_FRAC:-0.3}"
export SMC_CFG_SCALE="${SMC_CFG_SCALE:-1.25}"
export SMC_VARIANT_IDX="${SMC_VARIANT_IDX:-0}"

# Reward correction (space-separated list of strengths, included as search actions)
export CORRECTION_STRENGTHS="${CORRECTION_STRENGTHS:-0.0}"

# GA
export GA_POPULATION="${GA_POPULATION:-24}"
export GA_GENERATIONS="${GA_GENERATIONS:-8}"
export GA_ELITES="${GA_ELITES:-4}"
export GA_MUTATION_PROB="${GA_MUTATION_PROB:-0.15}"
export GA_TOURNAMENT_K="${GA_TOURNAMENT_K:-4}"
export GA_SELECTION="${GA_SELECTION:-rank}"
export GA_RANK_PRESSURE="${GA_RANK_PRESSURE:-1.7}"
export GA_CROSSOVER="${GA_CROSSOVER:-uniform}"
export GA_LOG_TOPK="${GA_LOG_TOPK:-5}"
export GA_EVAL_BATCH="${GA_EVAL_BATCH:-2}"

# Models to run
export RUN_SANA="${RUN_SANA:-0}"
export RUN_SD35="${RUN_SD35:-1}"
export RUN_FLUX="${RUN_FLUX:-0}"

# Qwen rewrites
export USE_QWEN="${USE_QWEN:-1}"
export PRECOMPUTE_REWRITES="${PRECOMPUTE_REWRITES:-1}"
export REWRITES_OVERWRITE="${REWRITES_OVERWRITE:-0}"
export QWEN_PRECOMPUTE_DEVICE="${QWEN_PRECOMPUTE_DEVICE:-auto}"
export QWEN_PRECOMPUTE_BATCH_SIZE="${QWEN_PRECOMPUTE_BATCH_SIZE:-16}"
export QWEN_PRECOMPUTE_SAVE_EVERY="${QWEN_PRECOMPUTE_SAVE_EVERY:-1}"
export QWEN_PRECOMPUTE_CLEAR_CACHE="${QWEN_PRECOMPUTE_CLEAR_CACHE:-1}"

# Images
export SAVE_IMAGES="${SAVE_IMAGES:-0}"
export SAVE_BEST_IMAGES="${SAVE_BEST_IMAGES:-1}"
export SAVE_VARIANTS="${SAVE_VARIANTS:-0}"

# Rewrites cache
REWRITES_CACHE_DIR="${DATA_ROOT}/hpsv2_rewrite_cache"
mkdir -p "${REWRITES_CACHE_DIR}"
export REWRITES_FILE="${REWRITES_FILE:-${REWRITES_CACHE_DIR}/${RUN_TAG}_${PROMPT_STYLE}_${START_INDEX}_${END_INDEX}.json}"

# ---------------------------------------------------------------------------
# Prompt file
# ---------------------------------------------------------------------------
mkdir -p "${HPSV2_PROMPT_DIR}"
OUT_DIR="${HPSV2_PROMPT_DIR}" STYLE="${PROMPT_STYLE}" bash "${SCRIPT_DIR}/get_hpsv2_prompts.sh"

if [[ "${PROMPT_STYLE}" == "all" ]]; then
  export PROMPT_FILE="${HPSV2_PROMPT_DIR}/hpsv2_prompts.txt"
else
  export PROMPT_FILE="${HPSV2_PROMPT_DIR}/hpsv2_prompts_${PROMPT_STYLE}.txt"
fi

if [[ "${USE_SUBSET}" == "1" ]]; then
  export PROMPT_FILE="${SCRIPT_DIR}/hpsv2_subset.txt"
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "[run_tacc] RUN_TAG=${RUN_TAG} OUT_ROOT=${OUT_ROOT}"
echo "[run_tacc] PROMPT_FILE=${PROMPT_FILE} NUM_GPUS=${NUM_GPUS}"
echo "[run_tacc] METHODS=${METHODS} STEPS=${STEPS} N_VARIANTS=${N_VARIANTS}"

mkdir -p "${OUT_ROOT}"

bash "${SCRIPT_DIR}/hpsv2_sd35_flux_allalgos.sh"
