#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

DEFAULT_PROMPT_FILE="${SCRIPT_DIR}/hpsv2_subset.txt"
if [[ -f "${DEFAULT_PROMPT_FILE}" ]]; then
  PROMPT_FILE="${PROMPT_FILE:-${DEFAULT_PROMPT_FILE}}"
else
  PROMPT_FILE="${PROMPT_FILE:-/data/ygu/hpsv2_prompts.txt}"
fi
OUT_ROOT="${OUT_ROOT:-/data/ygu/hpsv2_all_models_ddp}"
METHODS="${METHODS:-baseline greedy mcts ga}"
NUM_GPUS="${NUM_GPUS:-0}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:--1}"
REWARD_BACKENDS="${REWARD_BACKENDS:-${REWARD_BACKEND:-${REWARD_TYPE:-pickscore hpsv2}}}"

RUN_SANA="${RUN_SANA:-1}"
RUN_SD35="${RUN_SD35:-1}"
RUN_FLUX="${RUN_FLUX:-1}"

mkdir -p "${OUT_ROOT}"

for reward_backend in ${REWARD_BACKENDS}; do
  echo "[all-models] reward backend: ${reward_backend}"

  if [[ "${RUN_SANA}" == "1" ]]; then
    OUT_ROOT="${OUT_ROOT}/sana/${reward_backend}" \
    PROMPT_FILE="${PROMPT_FILE}" \
    METHODS="${METHODS}" \
    NUM_GPUS="${NUM_GPUS}" \
    START_INDEX="${START_INDEX}" \
    END_INDEX="${END_INDEX}" \
    REWARD_TYPE="${reward_backend}" \
    bash "${SCRIPT_DIR}/hpsv2_sana_sid_ddp_suite.sh"
  fi

  if [[ "${RUN_SD35}" == "1" ]]; then
    OUT_ROOT="${OUT_ROOT}/sd35/${reward_backend}" \
    PROMPT_FILE="${PROMPT_FILE}" \
    METHODS="${METHODS}" \
    NUM_GPUS="${NUM_GPUS}" \
    START_INDEX="${START_INDEX}" \
    END_INDEX="${END_INDEX}" \
    REWARD_BACKEND="${reward_backend}" \
    bash "${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
  fi

  if [[ "${RUN_FLUX}" == "1" ]]; then
    OUT_ROOT="${OUT_ROOT}/flux/${reward_backend}" \
    PROMPT_FILE="${PROMPT_FILE}" \
    METHODS="${METHODS}" \
    NUM_GPUS="${NUM_GPUS}" \
    START_INDEX="${START_INDEX}" \
    END_INDEX="${END_INDEX}" \
    REWARD_BACKEND="${reward_backend}" \
    bash "${SCRIPT_DIR}/hpsv2_flux_schnell_ddp_suite.sh"
  fi
done

echo "All requested model suites finished. Root: ${OUT_ROOT}"
