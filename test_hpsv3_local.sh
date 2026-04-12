#!/usr/bin/env bash
set -euo pipefail
#
# Local test script for HPSv3-based MCTS vs baseline/BoN/beam.
#
# Usage:
#   # --- Quick smoke test (2 prompts, SiD SD3.5) ---
#   bash test_hpsv3_local.sh
#
#   # --- SD3.5 base backend ---
#   BACKEND=sd35_base bash test_hpsv3_local.sh
#
#   # --- FLUX Schnell ---
#   MODEL=flux bash test_hpsv3_local.sh
#
#   # --- Run only specific methods ---
#   METHODS="baseline bon" bash test_hpsv3_local.sh
#
#   # --- Full 200-prompt subset ---
#   END_INDEX=200 bash test_hpsv3_local.sh
#
#   # --- Use imagereward for search instead (sanity check) ---
#   REWARD_BACKEND=imagereward bash test_hpsv3_local.sh
#
# Prerequisites:
#   pip install -r requirements.txt
#   pip install hpsv3 open_clip_torch omegaconf hydra-core
#   bash install_reward_deps.sh
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

# ── Configurable parameters ─────────────────────────────────────────────────
MODEL="${MODEL:-sd35}"               # sd35 | flux
BACKEND="${BACKEND:-sid}"            # sid | sd35_base | senseflow_large | senseflow_medium
METHODS="${METHODS:-baseline mcts_dynamiccfg_u_lookahead bon beam}"
REWARD_BACKEND="${REWARD_BACKEND:-hpsv3}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-2}"          # just 2 prompts by default for quick test
N_VARIANTS="${N_VARIANTS:-2}"        # fewer variants for speed
N_SIMS="${N_SIMS:-10}"               # fewer MCTS simulations for speed
BON_N="${BON_N:-4}"                  # fewer BoN candidates for speed
BEAM_WIDTH="${BEAM_WIDTH:-2}"        # fewer beams for speed
SEED="${SEED:-42}"
USE_QWEN="${USE_QWEN:-0}"           # off by default for local (needs Qwen model)
SAVE_BEST_IMAGES="${SAVE_BEST_IMAGES:-1}"
EVAL_BACKENDS="${EVAL_BACKENDS:-hpsv3 imagereward}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/test_hpsv3_local_out}"

# ── Model-specific defaults ─────────────────────────────────────────────────
if [[ "${MODEL}" == "flux" ]]; then
  # FLUX Schnell path
  STEPS="${STEPS:-4}"
  CFG_SCALES="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0}"
  BASELINE_CFG="${BASELINE_CFG:-1.0}"
  SMC_CFG_SCALE="${SMC_CFG_SCALE:-1.25}"
  # FLUX uses the flux suite script, not sd35
  FLUX_BACKEND="${FLUX_BACKEND:-flux}"
  SUITE_SCRIPT="hpsv2_flux_schnell_ddp_suite.sh"
  export MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-schnell}"

  # FLUX suite doesn't support mcts_dynamiccfg_u_lookahead, use plain mcts
  METHODS="${METHODS//mcts_dynamiccfg_u_lookahead/mcts}"
elif [[ "${BACKEND}" == "sd35_base" ]]; then
  STEPS="${STEPS:-28}"
  CFG_SCALES="${CFG_SCALES:-3.5 4.0 4.5 5.0 5.5 6.0 7.0}"
  BASELINE_CFG="${BASELINE_CFG:-4.5}"
  SMC_CFG_SCALE="${SMC_CFG_SCALE:-4.5}"
  SUITE_SCRIPT="hpsv2_sd35_sid_ddp_suite.sh"
elif [[ "${BACKEND}" == senseflow_* ]]; then
  STEPS="${STEPS:-4}"
  CFG_SCALES="${CFG_SCALES:-0.0}"
  BASELINE_CFG="${BASELINE_CFG:-0.0}"
  SMC_CFG_SCALE="${SMC_CFG_SCALE:-0.0}"
  SUITE_SCRIPT="hpsv2_sd35_sid_ddp_suite.sh"
else
  # SiD defaults
  STEPS="${STEPS:-4}"
  CFG_SCALES="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5}"
  BASELINE_CFG="${BASELINE_CFG:-1.0}"
  SMC_CFG_SCALE="${SMC_CFG_SCALE:-1.25}"
  SUITE_SCRIPT="hpsv2_sd35_sid_ddp_suite.sh"
fi

PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/hpsv2_subset.txt}"

# ── Validate prompt file exists ──────────────────────────────────────────────
if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found: ${PROMPT_FILE}" >&2
  exit 1
fi

# ── Print config ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "HPSv3 Local Test"
echo "============================================================"
echo "  model:           ${MODEL}"
echo "  backend:         ${BACKEND}"
echo "  methods:         ${METHODS}"
echo "  reward_backend:  ${REWARD_BACKEND}"
echo "  eval_backends:   ${EVAL_BACKENDS}"
echo "  steps:           ${STEPS}"
echo "  cfg_scales:      ${CFG_SCALES}"
echo "  baseline_cfg:    ${BASELINE_CFG}"
echo "  n_variants:      ${N_VARIANTS}"
echo "  n_sims:          ${N_SIMS}"
echo "  bon_n:           ${BON_N}"
echo "  beam_width:      ${BEAM_WIDTH}"
echo "  prompts:         ${START_INDEX}-${END_INDEX}"
echo "  use_qwen:        ${USE_QWEN}"
echo "  prompt_file:     ${PROMPT_FILE}"
echo "  out_root:        ${OUT_ROOT}"
echo "  suite_script:    ${SUITE_SCRIPT}"
echo "============================================================"

# ── Check CUDA ───────────────────────────────────────────────────────────────
"${PYTHON_BIN}" -c "
import torch
if not torch.cuda.is_available():
    print('WARNING: CUDA not available, this will be very slow or fail')
else:
    print(f'CUDA OK: {torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB)')
"

# ── Check HPSv3 is importable ───────────────────────────────────────────────
"${PYTHON_BIN}" -c "
try:
    import hpsv3
    cls = getattr(hpsv3, 'HPSv3RewardInferencer', None)
    print(f'HPSv3: OK (inferencer={cls is not None})')
except ImportError as e:
    print(f'HPSv3: NOT INSTALLED ({e})')
    print('Install with: pip install hpsv3 open_clip_torch omegaconf hydra-core')
    exit(1)
"

# ── Export env vars and run the suite script ─────────────────────────────────
export PROMPT_FILE
export OUT_ROOT
export METHODS
export START_INDEX
export END_INDEX
export REWARD_BACKEND
export REWARD_TYPE="${REWARD_BACKEND}"
export REWARD_BACKENDS="${REWARD_BACKEND}"
export EVAL_BEST_IMAGES="${SAVE_BEST_IMAGES}"
export EVAL_BACKENDS
export EVAL_REWARD_DEVICE="cpu"
export EVAL_ALLOW_MISSING_BACKENDS=1
export REWARD_DEVICE="${REWARD_DEVICE:-cuda}"
export CFG_SCALES
export BASELINE_CFG
export SMC_CFG_SCALE
export STEPS
export N_VARIANTS
export N_SIMS
export BON_N
export BEAM_WIDTH
export SEED
export USE_QWEN
export PRECOMPUTE_REWRITES=0
export SAVE_IMAGES=0
export SAVE_BEST_IMAGES
export SAVE_VARIANTS=0
export NUM_GPUS="${NUM_GPUS:-1}"
export SID_FORCE_WANDB_STUB=1
export WANDB_DISABLED=true

if [[ "${MODEL}" == "flux" ]]; then
  export FLUX_BACKEND
  export MODEL_ID
  bash "${SCRIPT_DIR}/${SUITE_SCRIPT}"
else
  export SD35_BACKEND="${BACKEND}"
  export LOOKAHEAD_METHOD_MODE="${LOOKAHEAD_METHOD_MODE:-rollout_tree_prior}"
  export LOOKAHEAD_U_T_DEF="${LOOKAHEAD_U_T_DEF:-latent_delta_rms}"
  export LOOKAHEAD_TAU="${LOOKAHEAD_TAU:-0.35}"
  export LOOKAHEAD_C_PUCT="${LOOKAHEAD_C_PUCT:-1.20}"
  export LOOKAHEAD_CFG_WIDTH_MIN="${LOOKAHEAD_CFG_WIDTH_MIN:-3}"
  export LOOKAHEAD_CFG_WIDTH_MAX="${LOOKAHEAD_CFG_WIDTH_MAX:-7}"
  bash "${SCRIPT_DIR}/${SUITE_SCRIPT}"
fi

echo
echo "============================================================"
echo "HPSv3 local test DONE"
echo "Outputs: ${OUT_ROOT}"
echo "============================================================"
