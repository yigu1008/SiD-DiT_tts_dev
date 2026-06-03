#!/usr/bin/env bash
# Single-prompt visualization on A6000.
# Baked-in: raccoon prompt, Qwen rewrites (N_VARIANTS=3), full deliberation trace.
#
# Just run:
#   bash run_single_prompt_viz.sh
# Override:
#   PROMPT="..." | BACKEND=sid|senseflow_large|sd35_base | N_SIMS=64 | N_VARIANTS=3

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_a6000_common.sh"
source "${SCRIPT_DIR}/_heartbeat.sh" 2>/dev/null || true
type start_heartbeat >/dev/null 2>&1 && start_heartbeat "single-prompt-viz"

# Defaults
BACKEND="${BACKEND:-sid}"
N_SIMS="${N_SIMS:-64}"
SEED="${SEED:-42}"
N_VARIANTS="${N_VARIANTS:-3}"
USE_QWEN="${USE_QWEN:-1}"
RUN_ROOT="${RUN_ROOT:-/data/ygu/runs/raccoon_full_trace_$(date +%Y%m%d_%H%M%S)}"
export SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"
export REWRITE_STYLES_OVERRIDE="${REWRITE_STYLES_OVERRIDE:-Paraphrase this prompt for an image generator using completely different sentence structure and synonyms while preserving every visual detail.||Rewrite this prompt as if describing the same scene to a different illustrator, freely rearranging clauses and substituting words.||Restate the prompt with rich, vivid wording: invert the order of elements, replace verbs and adjectives, and add atmospheric detail without inventing new objects.}"
mkdir -p "${RUN_ROOT}"

echo "================================================================"
echo "SINGLE-PROMPT VIZ"
echo "  BACKEND=${BACKEND}  N_SIMS=${N_SIMS}  SEED=${SEED}  N_VARIANTS=${N_VARIANTS}"
echo "  USE_QWEN=${USE_QWEN}  RUN_ROOT=${RUN_ROOT}"
echo "================================================================"

# 1. prompt
a6000_bake_prompt "${RUN_ROOT}/_baked_prompt" "${1:-${PROMPT:-}}"
PROMPT_TEXT="$(head -n1 "${PROMPT_FILE}")"

# 2. rewrites
REWRITES_FILE="${RUN_ROOT}/rewrites.json"
if [[ "${USE_QWEN}" == "1" ]]; then
    a6000_run_qwen_rewrites "${PROMPT_FILE}" "${REWRITES_FILE}" "${N_VARIANTS}"
fi
a6000_verify_rewrites "${REWRITES_FILE}" "${PROMPT_TEXT}" "${N_VARIANTS}"
export REWRITES_FILE

# 3-5. reward, backend, bon_mcts
a6000_use_inprocess_reward
a6000_setup_backend
a6000_setup_bon_mcts_env "${RUN_ROOT}" 1
a6000_run_bon_mcts "${RUN_ROOT}"

# 6. viz
a6000_render_viz "${RUN_ROOT}" 1

# 7. verify
echo
a6000_verify_variant_usage "${RUN_ROOT}" "${N_VARIANTS}"

echo
echo "================================================================"
echo "DONE.  ${RUN_ROOT}/"
echo "  rewrites.json | run_*/bon_mcts/{images,logs} | step_images_inline/ | all_attempts/"
echo "  ${BACKEND}/ (tree) | ${BACKEND}_logs/ | trajectory_strips/"
echo "================================================================"
