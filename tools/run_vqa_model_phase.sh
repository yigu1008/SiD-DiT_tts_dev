#!/usr/bin/env bash
set -euo pipefail

# Run one model and one phase of the GenAI-200 VQAScore experiment.
#
# Examples:
#   MODEL=flux_schnell PHASE=generate GPUS=0,1,2,3 bash tools/run_vqa_model_phase.sh
#   MODEL=sd35_base PHASE=generate GPUS=4,5,6,7 bash tools/run_vqa_model_phase.sh
#   MODEL=flux_schnell PHASE=ood_eval GPUS=3 bash tools/run_vqa_model_phase.sh
#   MODEL=sd35_base PHASE=ood_eval GPUS=7 bash tools/run_vqa_model_phase.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${MODEL:?Set MODEL=flux_schnell or MODEL=sd35_base}"
PHASE="${PHASE:-generate}"
case "${MODEL}" in
  flux|flux_schnell)
    MODEL=flux_schnell
    DEFAULT_GENERATION_GPUS=0,1,2,3
    DEFAULT_OOD_GPUS=3
    REWARD_SERVER_BASE_PORT=5400
    POSTHOC_REWARD_SERVER_PORT=5490
    ;;
  base|sd35|sd35_base)
    MODEL=sd35_base
    DEFAULT_GENERATION_GPUS=4,5,6,7
    DEFAULT_OOD_GPUS=7
    REWARD_SERVER_BASE_PORT=5500
    POSTHOC_REWARD_SERVER_PORT=5590
    ;;
  *)
    echo "Error: MODEL must be flux_schnell or sd35_base." >&2
    exit 2
    ;;
esac
case "${PHASE}" in generate|ood_eval) ;;
  *) echo "Error: PHASE must be generate or ood_eval." >&2; exit 2 ;;
esac

if [[ -z "${GPUS:-}" ]]; then
  if [[ "${PHASE}" == "generate" ]]; then
    GPUS="${DEFAULT_GENERATION_GPUS}"
  else
    GPUS="${DEFAULT_OOD_GPUS}"
  fi
fi
IFS=',' read -r -a gpu_array <<< "${GPUS}"
if [[ "${PHASE}" == "generate" && ${#gpu_array[@]} -lt 3 ]]; then
  echo "Error: VQAScore generation requires at least three GPUs (2+ generation + 1 reward)." >&2
  exit 2
fi
if (( ${#gpu_array[@]} < 1 )); then
  echo "Error: GPUS must contain at least one GPU." >&2
  exit 2
fi
REWARD_GPU="${gpu_array[${#gpu_array[@]}-1]}"

HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
SOURCE_SID_RUN_ROOT="${SOURCE_SID_RUN_ROOT:-${HUMAN_EVAL_ROOT}/sid_vqascore_algorithm_sweep/genai200_v1}"
STUDY_ROOT="${STUDY_ROOT:-${HUMAN_EVAL_ROOT}/vqascore_remaining_models}"
RUN_ID="${RUN_ID:-genai200_v1}"
RUN_ROOT="${STUDY_ROOT}/${RUN_ID}"
METHODS="${METHODS:-baseline fksteering bon beam sop ga dts dts_star dynamic_cfg_x0 bon_mcts}"
OOD_EVAL_BACKENDS="${OOD_EVAL_BACKENDS:-imagereward hpsv3 pickscore}"

PYTHON_BIN="${PYTHON_BIN:-/home/ygu/miniconda3/envs/sid_dit/bin/python}"
REWARD_ENV_CONDA_BASE="${REWARD_ENV_CONDA_BASE:-/home/ygu/miniconda3}"
VQA_REWARD_ENV_NAME="${VQA_REWARD_ENV_NAME:-vqascore_reward}"
STANDARD_REWARD_ENV_NAME="${STANDARD_REWARD_ENV_NAME:-reward}"
DRY_RUN="${DRY_RUN:-0}"

if [[ " ${METHODS} " == *" das "* ]]; then
  echo "Error: this continuation matches the supplied ten-method table; DAS is excluded." >&2
  exit 2
fi
case "${DRY_RUN}" in 0|1) ;;
  *) echo "Error: DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac

echo "[model-phase] model=${MODEL} phase=${PHASE}"
echo "[model-phase] GPUs=${GPUS}; reward GPU=${REWARD_GPU}"
if [[ "${PHASE}" == "generate" ]]; then
  echo "[model-phase] generation GPU count=$((${#gpu_array[@]} - 1))"
fi
echo "[model-phase] run_root=${RUN_ROOT}"
echo "[model-phase] methods=${METHODS}"
if [[ "${PHASE}" == "ood_eval" ]]; then
  echo "[model-phase] OOD backends=${OOD_EVAL_BACKENDS}"
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

for executable in \
  "${PYTHON_BIN}" \
  "${REWARD_ENV_CONDA_BASE}/envs/${VQA_REWARD_ENV_NAME}/bin/python" \
  "${REWARD_ENV_CONDA_BASE}/envs/${STANDARD_REWARD_ENV_NAME}/bin/python"
do
  if [[ ! -x "${executable}" ]]; then
    echo "Error: required Python is not executable: ${executable}" >&2
    exit 1
  fi
done

if [[ "${PHASE}" == "ood_eval" ]]; then
  standard_reward_python="${REWARD_ENV_CONDA_BASE}/envs/${STANDARD_REWARD_ENV_NAME}/bin/python"
  PATH="$(dirname "${standard_reward_python}"):${PATH}" \
    "${standard_reward_python}" - <<'PY'
import click
import platformdirs
from google.protobuf import runtime_version
import ImageReward
import hpsv3
import transformers
print("[model-phase] standard OOD reward runtime imports OK")
PY
  POST_EVAL_ONLY=1
  SKIP_POST_EVAL=0
else
  POST_EVAL_ONLY=0
  SKIP_POST_EVAL=1
fi

mkdir -p "${RUN_ROOT}/${MODEL}"
exec env \
  CUDA_VISIBLE_DEVICES="${GPUS}" \
  REWARD_CUDA_VISIBLE_DEVICES="${REWARD_GPU}" \
  HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT}" \
  SOURCE_SID_RUN_ROOT="${SOURCE_SID_RUN_ROOT}" \
  STUDY_ROOT="${STUDY_ROOT}" RUN_ID="${RUN_ID}" \
  STUDY_MANIFEST="${RUN_ROOT}/study_manifest_${MODEL}.json" \
  AUDIT_OUT_CSV="${RUN_ROOT}/${MODEL}/vqascore_coverage.csv" \
  BACKENDS="${MODEL}" METHODS="${METHODS}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  REWARD_ENV_CONDA_BASE="${REWARD_ENV_CONDA_BASE}" \
  VQA_REWARD_ENV_NAME="${VQA_REWARD_ENV_NAME}" \
  STANDARD_REWARD_ENV_NAME="${STANDARD_REWARD_ENV_NAME}" \
  REWARD_SERVER_BASE_PORT="${REWARD_SERVER_BASE_PORT}" \
  POSTHOC_REWARD_SERVER_PORT="${POSTHOC_REWARD_SERVER_PORT}" \
  POST_EVAL_ONLY="${POST_EVAL_ONLY}" SKIP_POST_EVAL="${SKIP_POST_EVAL}" \
  GENERATION_END_INDEX="${GENERATION_END_INDEX:-}" \
  POSTHOC_EVAL_BACKENDS="${OOD_EVAL_BACKENDS}" \
  POSTHOC_SKIP_COMPLETE=1 POSTHOC_MERGE_SCOPE=model FAIL_FAST=1 \
  bash "${REPO}/tools/run_remaining_vqascore_algorithm_sweeps.sh"
