#!/usr/bin/env bash
# A6000-local port of amlt/actdiff_grid_sid.yaml.
#
# 11-method "drop ActDiff axes into existing search methods" grid on sid.
#
# Methods:
#   baseline           reference (no search)
#   bon                canonical BoN (fixed cfg/variant)
#   bon_actdiff_cfg    BoN + CFG-axis search
#   bon_actdiff_full   BoN + (CFG x prompt) axis search
#   sop                canonical SoP
#   sop_actdiff_cfg    SoP + CFG-axis at branching
#   sop_actdiff_full   SoP + (CFG x prompt) at branching
#   smc                canonical SMC
#   smc_actdiff_cfg    SMC + CFG-axis at resample
#   smc_actdiff_full   SMC + (CFG x prompt) at resample
#   bon_mcts           full ActDiff anchor (prescreen + MCTS refine)
#
# In-process reward (no server).  Defaults to imagereward (single, fast);
# override with SEARCH_REWARD=composite_hpsv3_ir for the cluster default.
#
# Just run:
#   bash run_actdiff_grid_sid_a6000.sh
# Override:
#   N_PROMPTS=20 (default; 100 on cluster but ~50h on A6000)
#   BON_N=76     (matched-NFE BoN budget for 4-step sid)
#   SEARCH_REWARD=imagereward|composite_hpsv3_ir
#   METHODS="baseline bon bon_actdiff_full bon_mcts"  (subset)
#
# ETA on A6000 (N_PROMPTS=20, 11 methods, imagereward):
#   - baseline / bon:               ~10-30 min each
#   - actdiff_cfg variants:         ~30-60 min each
#   - actdiff_full variants:        ~45-90 min each
#   - bon_mcts:                     ~90 min
#   - Total:                        ~10-14 hours

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_a6000_common.sh"
source "${SCRIPT_DIR}/_a6000_common_multigpu.sh"
source "${SCRIPT_DIR}/_heartbeat.sh" 2>/dev/null || true
type start_heartbeat >/dev/null 2>&1 && start_heartbeat "actdiff-grid-sid"

# ── Defaults (cluster has N_PROMPTS=100; A6000-friendly default is 20) ────
BACKEND="${BACKEND:-sid}"
N_PROMPTS="${N_PROMPTS:-20}"
SEED="${SEED:-42}"
SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"        # cluster uses composite_hpsv3_ir
METHODS="${METHODS:-baseline bon bon_actdiff_cfg bon_actdiff_full sop sop_actdiff_cfg sop_actdiff_full smc smc_actdiff_cfg smc_actdiff_full bon_mcts}"
BON_N="${BON_N:-76}"     # matched NFE: 16 prescreen + 60 refine = 76 (sid 4-step)
RUN_ROOT="${RUN_ROOT:-/data/ygu/runs/actdiff_grid_sid_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_ROOT}"

PROMPT_FILE="${PROMPT_FILE:-/data/ygu/dpg_bench_prompts.txt}"
export PROMPT_FILE

# Bank overrides (same as cluster yaml).
export BON_MCTS_NEG_BANK_NEG="${BON_MCTS_NEG_BANK_NEG:-||low quality, blurry, lowres, jpeg artifacts||bad anatomy, deformed, mutated, extra limbs||watermark, signature, text, frame, cropped}"
export BON_MCTS_SIGMA_BANK_SIGMA="${BON_MCTS_SIGMA_BANK_SIGMA:--0.05 0.0 0.05}"
export BON_MCTS_NEG_BANK_AXES="${BON_MCTS_NEG_BANK_AXES:-${BON_MCTS_NEG_BANK_NEG}}"
export BON_MCTS_SIGMA_BANK_AXES="${BON_MCTS_SIGMA_BANK_AXES:-${BON_MCTS_SIGMA_BANK_SIGMA}}"

# Multi-GPU: reward server on GPU 0, DDP sampling on GPUs 1..N-1.
TOTAL_GPUS="${TOTAL_GPUS:-8}"
a6000_setup_backend
# For composite_hpsv3_ir we need BOTH backends in the server.
case "${SEARCH_REWARD}" in
    composite_hpsv3_ir|composite_all4) _server_backends="hpsv3 imagereward" ;;
    *) _server_backends="${SEARCH_REWARD}" ;;
esac
mgpu_boot_reward_server "${RUN_ROOT}/reward_server.log" "${_server_backends}" || exit 1
trap 'mgpu_kill_reward_server' EXIT
mgpu_setup_sampling_gpus "${TOTAL_GPUS}"

# Memory + execution knobs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OFFLOAD_TEXT_ENCODER_AFTER_ENCODE="${OFFLOAD_TEXT_ENCODER_AFTER_ENCODE:-1}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"

# Hand off to the same inner script the cluster uses.
export BACKEND N_PROMPTS SEED SEARCH_REWARD METHODS BON_N RUN_ROOT

echo "================================================================"
echo "ACTDIFF GRID -- sid (A6000 local port)"
echo "  BACKEND        = ${BACKEND}"
echo "  N_PROMPTS      = ${N_PROMPTS}     (cluster default 100; A6000 default 20)"
echo "  SEED           = ${SEED}"
echo "  SEARCH_REWARD  = ${SEARCH_REWARD}  (override: composite_hpsv3_ir)"
echo "  BON_N          = ${BON_N}"
echo "  METHODS        = ${METHODS}"
echo "  PROMPT_FILE    = ${PROMPT_FILE}"
echo "  RUN_ROOT       = ${RUN_ROOT}"
echo "  CUDA_DEVICE    = ${CUDA_VISIBLE_DEVICES}"
echo "================================================================"

bash "${SCRIPT_DIR}/run_all_method_ablations.sh" 2>&1 | tee "${RUN_ROOT}/_run.log"

# Per-prompt decision trees for the bon_mcts method (if it ran).
if [[ -d "${RUN_ROOT}/bon_mcts" ]]; then
    mkdir -p "${RUN_ROOT}/trees"
    for pi in 0 5 10; do
        "${PYTHON_BIN:-python}" "${SCRIPT_DIR}/plot_actdiff_tree.py" \
            --mode real \
            --run_root "${RUN_ROOT}" \
            --method bon_mcts \
            --prompt_index "${pi}" \
            --out_dir "${RUN_ROOT}/trees" \
            --title "ActDiff trace -- actdiff_grid_sid / bon_mcts / prompt #${pi}" \
            2>/dev/null || true
    done
fi

# Pull together a method-level summary (does the suite already write summary.tsv?
# If yes, just preview it; if no, build a quick one).
if [[ -f "${RUN_ROOT}/summary.tsv" ]]; then
    echo
    echo "Summary preview (${RUN_ROOT}/summary.tsv):"
    head -20 "${RUN_ROOT}/summary.tsv"
fi

echo
echo "================================================================"
echo "DONE.  ${RUN_ROOT}/"
echo "  summary.tsv                       method-level IR / delta stats"
echo "  {method}/run_*/...                per-method rank files + images"
echo "  trees/*.png                       decision trees for prompts 0, 5, 10"
echo "================================================================"
