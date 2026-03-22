#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

PROMPT_FILE="${PROMPT_FILE:-/data/ygu/hpsv2_prompts.txt}"
OUT_ROOT="${OUT_ROOT:-/data/ygu/hpsv2_all_models_ddp}"
METHODS="${METHODS:-baseline greedy mcts ga}"
NUM_GPUS="${NUM_GPUS:-0}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:--1}"

RUN_SANA="${RUN_SANA:-1}"
RUN_SD35="${RUN_SD35:-1}"
RUN_FLUX="${RUN_FLUX:-1}"

mkdir -p "${OUT_ROOT}"

if [[ "${RUN_SANA}" == "1" ]]; then
  OUT_ROOT="${OUT_ROOT}/sana" \
  PROMPT_FILE="${PROMPT_FILE}" \
  METHODS="${METHODS}" \
  NUM_GPUS="${NUM_GPUS}" \
  START_INDEX="${START_INDEX}" \
  END_INDEX="${END_INDEX}" \
  "${SCRIPT_DIR}/hpsv2_sana_sid_ddp_suite.sh"
fi

if [[ "${RUN_SD35}" == "1" ]]; then
  OUT_ROOT="${OUT_ROOT}/sd35" \
  PROMPT_FILE="${PROMPT_FILE}" \
  METHODS="${METHODS}" \
  NUM_GPUS="${NUM_GPUS}" \
  START_INDEX="${START_INDEX}" \
  END_INDEX="${END_INDEX}" \
  "${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
fi

if [[ "${RUN_FLUX}" == "1" ]]; then
  OUT_ROOT="${OUT_ROOT}/flux" \
  PROMPT_FILE="${PROMPT_FILE}" \
  METHODS="${METHODS}" \
  NUM_GPUS="${NUM_GPUS}" \
  START_INDEX="${START_INDEX}" \
  END_INDEX="${END_INDEX}" \
  "${SCRIPT_DIR}/hpsv2_flux_schnell_ddp_suite.sh"
fi

echo "All requested model suites finished. Root: ${OUT_ROOT}"
