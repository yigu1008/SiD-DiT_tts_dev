#!/usr/bin/env bash
# A6000-local port of amlt/all_method_ablations.yaml.
#
# Runs 6 methods on sid backend over N prompts, sequentially, sharing the
# same pipeline load across methods (the inner script run_all_method_ablations.sh
# already amortizes load across methods).
#
# Methods:
#   baseline | greedy_prompt | bon_mcts | bon_mcts_neg | bon_mcts_sigma | bon_mcts_axes
#
# In-process ImageReward (no server, no port-reuse issues).
#
# Just run:
#   bash run_all_method_ablations_a6000.sh
# Override:
#   N_PROMPTS=20 (default 100 -- BIG; consider 20-30 for first pass)
#   BACKEND=sid|senseflow_large|sd35_base
#   SEED=42
#   METHODS="baseline bon_mcts"  (subset)
#   RUN_ROOT=<path>
#
# ETA on A6000 (N_PROMPTS=20, 6 methods):
#   - baseline:        ~10 min
#   - greedy_prompt:   ~30 min
#   - bon_mcts*:       ~90 min each → ~6h for 4 mcts variants
#   - Total:           ~7-9 hours
# For N_PROMPTS=100: scale by 5x (~35-45 hours).

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_a6000_common.sh"
source "${SCRIPT_DIR}/_heartbeat.sh" 2>/dev/null || true
type start_heartbeat >/dev/null 2>&1 && start_heartbeat "all-method-ablations"

# ── Defaults ─────────────────────────────────────────────────────────────
BACKEND="${BACKEND:-sid}"
N_PROMPTS="${N_PROMPTS:-20}"          # cluster default was 100 -- way too long for A6000
SEED="${SEED:-42}"
SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"
METHODS="${METHODS:-baseline greedy_prompt bon_mcts bon_mcts_neg bon_mcts_sigma bon_mcts_axes}"
RUN_ROOT="${RUN_ROOT:-/data/ygu/runs/all_method_ablations_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_ROOT}"

# Default DPG-Bench prompts (override via PROMPT_FILE=...).
PROMPT_FILE="${PROMPT_FILE:-/data/ygu/dpg_bench_prompts.txt}"
export PROMPT_FILE

# Bank overrides (same as cluster yaml).
export BON_MCTS_NEG_BANK_NEG="${BON_MCTS_NEG_BANK_NEG:-||low quality, blurry, lowres, jpeg artifacts||bad anatomy, deformed, mutated, extra limbs||watermark, signature, text, frame, cropped}"
export BON_MCTS_SIGMA_BANK_SIGMA="${BON_MCTS_SIGMA_BANK_SIGMA:--0.05 0.0 0.05}"
export BON_MCTS_NEG_BANK_AXES="${BON_MCTS_NEG_BANK_AXES:-${BON_MCTS_NEG_BANK_NEG}}"
export BON_MCTS_SIGMA_BANK_AXES="${BON_MCTS_SIGMA_BANK_AXES:-${BON_MCTS_SIGMA_BANK_SIGMA}}"

# In-process reward (no server -- A6000 single GPU).
a6000_use_inprocess_reward
a6000_setup_backend

# Pipeline + memory knobs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OFFLOAD_TEXT_ENCODER_AFTER_ENCODE="${OFFLOAD_TEXT_ENCODER_AFTER_ENCODE:-1}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
export NUM_GPUS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Pass everything the inner shell script wants.
export BACKEND N_PROMPTS SEED SEARCH_REWARD METHODS
export RUN_ROOT

echo "================================================================"
echo "ALL-METHOD ABLATIONS (A6000 local port)"
echo "  BACKEND        = ${BACKEND}"
echo "  N_PROMPTS      = ${N_PROMPTS}"
echo "  SEED           = ${SEED}"
echo "  SEARCH_REWARD  = ${SEARCH_REWARD}"
echo "  METHODS        = ${METHODS}"
echo "  PROMPT_FILE    = ${PROMPT_FILE}"
echo "  RUN_ROOT       = ${RUN_ROOT}"
echo "  CUDA_DEVICE    = ${CUDA_VISIBLE_DEVICES}"
echo "================================================================"

# Run the inner shared script (it loops over methods, sharing pipeline load).
bash "${SCRIPT_DIR}/run_all_method_ablations.sh" 2>&1 | tee "${RUN_ROOT}/_run.log"

# Synergy plot (will only have data for the methods that ran).
"${PYTHON_BIN:-python}" "${SCRIPT_DIR}/plot_synergy_2x2.py" \
    --summary "${RUN_ROOT}/summary.tsv" \
    --metric eval_ir \
    --out_png "${RUN_ROOT}/synergy_eval_ir.png" || true

# Per-prompt decision-tree visualizations for first 3 prompts (if bon_mcts ran).
if [[ -d "${RUN_ROOT}/bon_mcts" ]]; then
    mkdir -p "${RUN_ROOT}/trees"
    for pi in 0 5 10; do
        "${PYTHON_BIN:-python}" "${SCRIPT_DIR}/plot_actdiff_tree.py" \
            --mode real \
            --run_root "${RUN_ROOT}" \
            --method bon_mcts \
            --prompt_index "${pi}" \
            --out_dir "${RUN_ROOT}/trees" \
            --title "ActDiff trace -- all-method / bon_mcts / prompt #${pi}" \
            2>/dev/null || true
    done
fi

echo
echo "================================================================"
echo "DONE.  ${RUN_ROOT}/"
echo "  summary.tsv               method-level IR / delta stats"
echo "  {method}/run_*/...        per-method rank files + images"
echo "  synergy_eval_ir.png       method comparison plot"
echo "  trees/*.png               decision trees for prompts 0, 5, 10"
echo "================================================================"

if [[ -f "${RUN_ROOT}/summary.tsv" ]]; then
    echo
    echo "Summary preview:"
    head -20 "${RUN_ROOT}/summary.tsv"
fi
