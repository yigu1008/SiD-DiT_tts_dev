#!/usr/bin/env bash
set -euo pipefail

# One-GPU continuation after Flux OOD evaluation:
#   1. SenseFlow-SD3.5-Large, all requested methods
#   2. SD3.5-Base, excluding ActDiff (bon_mcts)
# For each method, reward backends are loaded and unloaded sequentially.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
RUN_ID="${RUN_ID:-genai200_v1}"
GPU="${GPU:-7}"
EXPECTED_PROMPTS="${EXPECTED_PROMPTS:-200}"
OOD_EVAL_BACKENDS="${OOD_EVAL_BACKENDS:-imagereward pickscore hpsv2}"
REPORT_BACKENDS="${REPORT_BACKENDS:-imagereward hpsv3 pickscore hpsv2}"
SENSE_METHODS="${SENSE_METHODS:-baseline fksteering bon beam sop ga dts dts_star dynamic_cfg_x0 bon_mcts}"
SD35_BASE_METHODS="${SD35_BASE_METHODS:-baseline fksteering bon beam sop ga dts dts_star dynamic_cfg_x0}"
SENSE_PORT="${SENSE_PORT:-5837}"
SD35_BASE_PORT="${SD35_BASE_PORT:-5857}"
PYTHON_BIN="${PYTHON_BIN:-/home/ygu/miniconda3/envs/sid_dit/bin/python}"
REWARD_ENV_CONDA_BASE="${REWARD_ENV_CONDA_BASE:-/home/ygu/miniconda3}"
STANDARD_REWARD_ENV_NAME="${STANDARD_REWARD_ENV_NAME:-reward}"
DRY_RUN="${DRY_RUN:-0}"

case "${DRY_RUN}" in 0|1) ;;
  *) echo "Error: DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac
if [[ " ${SD35_BASE_METHODS} " == *" bon_mcts "* ]]; then
  echo "Error: SD35_BASE_METHODS must exclude bon_mcts/ActDiff." >&2
  exit 2
fi
if [[ "${SENSE_PORT}" == "${SD35_BASE_PORT}" ]]; then
  echo "Error: use distinct SenseFlow and SD3.5-Base server ports." >&2
  exit 2
fi

echo "[sense+base-ood] GPU=${GPU}"
echo "[sense+base-ood] sequential backends=${OOD_EVAL_BACKENDS}"
echo "[sense+base-ood] SenseFlow methods=${SENSE_METHODS}"
echo "[sense+base-ood] SD3.5-Base methods=${SD35_BASE_METHODS}"
echo "[sense+base-ood] SD3.5-Base ActDiff is excluded"

run_model() {
  local model="$1"
  local methods="$2"
  local port="$3"
  echo "[sense+base-ood] starting model=${model} port=${port}"
  env \
    MODEL="${model}" GPU="${GPU}" REWARD_SERVER_PORT="${port}" \
    METHODS="${methods}" OOD_EVAL_BACKENDS="${OOD_EVAL_BACKENDS}" \
    REPORT_BACKENDS="${REPORT_BACKENDS}" \
    REPORT_STEM="vqa_ood_summary_partial" \
    ALLOW_INCOMPLETE_GENERATION=1 \
    HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT}" RUN_ID="${RUN_ID}" \
    EXPECTED_PROMPTS="${EXPECTED_PROMPTS}" PYTHON_BIN="${PYTHON_BIN}" \
    REWARD_ENV_CONDA_BASE="${REWARD_ENV_CONDA_BASE}" \
    STANDARD_REWARD_ENV_NAME="${STANDARD_REWARD_ENV_NAME}" \
    DRY_RUN="${DRY_RUN}" \
    bash "${SCRIPT_DIR}/run_vqa_ood_per_method.sh"
  echo "[sense+base-ood] finished model=${model}"
}

run_model senseflow_large "${SENSE_METHODS}" "${SENSE_PORT}"
run_model sd35_base "${SD35_BASE_METHODS}" "${SD35_BASE_PORT}"

echo "[sense+base-ood] complete"
