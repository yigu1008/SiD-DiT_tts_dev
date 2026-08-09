#!/usr/bin/env bash
set -euo pipefail

# Evaluate existing generated images in method-major order. For every method,
# reward backends are loaded one at a time and torn down before the next backend
# is loaded. Assign disjoint METHODS lists to separate GPUs for safe parallelism.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${MODEL:?Set MODEL=flux_schnell, sid, senseflow_large, senseflow_medium, or sd35_base}"
case "${MODEL}" in
  flux|flux_schnell)
    MODEL=flux_schnell; LAYOUT=flux; DEFAULT_GPU=6; DEFAULT_PORT=5816 ;;
  sid)
    MODEL=sid; LAYOUT=sd35; DEFAULT_GPU=6; DEFAULT_PORT=5826 ;;
  sense|senseflow|senseflow_large)
    MODEL=senseflow_large; LAYOUT=sd35; DEFAULT_GPU=6; DEFAULT_PORT=5836 ;;
  senseflow_medium)
    LAYOUT=sd35; DEFAULT_GPU=6; DEFAULT_PORT=5846 ;;
  base|sd35|sd35_base)
    MODEL=sd35_base; LAYOUT=sd35; DEFAULT_GPU=6; DEFAULT_PORT=5856 ;;
  *) echo "Error: unsupported MODEL=${MODEL}." >&2; exit 2 ;;
esac

HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
RUN_ID="${RUN_ID:-genai200_v1}"
EXPECTED_PROMPTS="${EXPECTED_PROMPTS:-200}"
METHODS="${METHODS:-baseline fksteering bon beam sop ga dts dts_star dynamic_cfg_x0 bon_mcts}"
OOD_EVAL_BACKENDS="${OOD_EVAL_BACKENDS:-imagereward pickscore hpsv2}"
REPORT_BACKENDS="${REPORT_BACKENDS:-imagereward hpsv3 pickscore hpsv2}"
REPORT_STEM="${REPORT_STEM:-vqa_ood_summary_partial}"
GPU="${GPU:-${DEFAULT_GPU}}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-${DEFAULT_PORT}}"
ALLOW_INCOMPLETE_GENERATION="${ALLOW_INCOMPLETE_GENERATION:-0}"
DRY_RUN="${DRY_RUN:-0}"

PYTHON_BIN="${PYTHON_BIN:-/home/ygu/miniconda3/envs/sid_dit/bin/python}"
REWARD_ENV_CONDA_BASE="${REWARD_ENV_CONDA_BASE:-/home/ygu/miniconda3}"
STANDARD_REWARD_ENV_NAME="${STANDARD_REWARD_ENV_NAME:-reward}"
STANDARD_REWARD_PY="${REWARD_ENV_CONDA_BASE}/envs/${STANDARD_REWARD_ENV_NAME}/bin/python"

if [[ "${MODEL}" == "sid" ]]; then
  STUDY_RUN_ROOT="${STUDY_RUN_ROOT:-${HUMAN_EVAL_ROOT}/sid_vqascore_algorithm_sweep/${RUN_ID}}"
else
  STUDY_RUN_ROOT="${STUDY_RUN_ROOT:-${HUMAN_EVAL_ROOT}/vqascore_remaining_models/${RUN_ID}}"
fi
MODEL_ROOT="${STUDY_RUN_ROOT}/${MODEL}"
RUN_ROOT="${MODEL_ROOT}/run_${RUN_ID}"
REPORT_DIR="${RUN_ROOT}/reports"

case "${ALLOW_INCOMPLETE_GENERATION}" in 0|1) ;;
  *) echo "Error: ALLOW_INCOMPLETE_GENERATION must be 0 or 1." >&2; exit 2 ;;
esac
case "${DRY_RUN}" in 0|1) ;;
  *) echo "Error: DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac
if [[ ! "${EXPECTED_PROMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: EXPECTED_PROMPTS must be positive." >&2
  exit 2
fi
if [[ "${REPORT_STEM}" == *"/"* || -z "${REPORT_STEM}" ]]; then
  echo "Error: REPORT_STEM must be a non-empty filename stem." >&2
  exit 2
fi
for method in ${METHODS}; do
  if [[ "${method}" == *"/"* || "${method}" == "." || "${method}" == ".." ]]; then
    echo "Error: invalid method name: ${method}" >&2
    exit 2
  fi
done

echo "[ood-per-method] model=${MODEL} layout=${LAYOUT} GPU=${GPU} port=${REWARD_SERVER_PORT}"
echo "[ood-per-method] methods=${METHODS}"
echo "[ood-per-method] sequential backends=${OOD_EVAL_BACKENDS}"
echo "[ood-per-method] run_root=${RUN_ROOT}"
if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

for executable in "${PYTHON_BIN}" "${STANDARD_REWARD_PY}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "Error: required Python is not executable: ${executable}" >&2
    exit 1
  fi
done
if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "Error: generated run directory is missing: ${RUN_ROOT}" >&2
  exit 1
fi
mkdir -p "${REPORT_DIR}"

PATH="$(dirname "${STANDARD_REWARD_PY}"):${PATH}" "${STANDARD_REWARD_PY}" - <<'PY'
import click
import platformdirs
from google.protobuf import runtime_version
import ImageReward
import hpsv2
import transformers
print("[ood-per-method] standard reward runtime imports OK")
PY

for method in ${METHODS}; do
  method_dir="${RUN_ROOT}/${method}"
  aggregate="${method_dir}/aggregate_ddp.json"
  if [[ ! -s "${aggregate}" ]]; then
    if [[ "${ALLOW_INCOMPLETE_GENERATION}" == "1" ]]; then
      echo "[ood-per-method] WARN: skipping ${method}; aggregate_ddp.json is missing"
      continue
    fi
    echo "Error: ${method}: missing ${aggregate}" >&2
    exit 1
  fi
  generated_count="$(${PYTHON_BIN} - "${aggregate}" <<'PY'
import json
import sys
print(int(json.load(open(sys.argv[1], encoding="utf-8")).get("num_samples", 0) or 0))
PY
)"
  if (( generated_count != EXPECTED_PROMPTS )) && [[ "${ALLOW_INCOMPLETE_GENERATION}" != "1" ]]; then
    echo "Error: ${method}: generated ${generated_count}/${EXPECTED_PROMPTS}" >&2
    exit 1
  fi
  if (( generated_count <= 0 )); then
    echo "[ood-per-method] WARN: skipping ${method}; aggregate contains no samples"
    continue
  fi

  echo "[ood-per-method] method=${method} generated=${generated_count}; loading rewards sequentially"
  OUT_ROOT="${MODEL_ROOT}" \
  REWARD_PY="${STANDARD_REWARD_PY}" STANDARD_REWARD_PY="${STANDARD_REWARD_PY}" \
  PYTHON_BIN="${PYTHON_BIN}" REWARD_SERVER_PORT="${REWARD_SERVER_PORT}" \
  REWARD_CUDA_VISIBLE_DEVICES="${GPU}" \
  POSTHOC_EVAL_BACKENDS="${OOD_EVAL_BACKENDS}" \
  POSTHOC_METHODS="${method}" POSTHOC_SKIP_COMPLETE=1 \
  POSTHOC_EXPECTED_COUNT="${generated_count}" POSTHOC_RUN_ID="${RUN_ID}" \
  POSTHOC_ALLOW_MISSING_BACKENDS=0 POSTHOC_LAYOUT="${LAYOUT}" \
  bash "${REPO}/post_eval_extra_rewards.sh"
done

# Refresh a combined, non-strict report without launching a reward model. It
# includes HPSv3 files already present and every newly completed backend.
"${PYTHON_BIN}" "${REPO}/tools/merge_posthoc_reward_evals.py" \
  --root "${STUDY_RUN_ROOT}" --include-models "${MODEL}" --run-id "${RUN_ID}" \
  --backends ${REPORT_BACKENDS} \
  --summary-csv "${REPORT_DIR}/${REPORT_STEM}.csv" \
  --expected-count "${EXPECTED_PROMPTS}" --no-strict

if [[ "${REPORT_STEM}" == "vqa_ood_summary_partial" ]]; then
  cp "${REPORT_DIR}/${REPORT_STEM}.csv" "${MODEL_ROOT}/vqa_ood_summary_partial.csv"
  cp "${REPORT_DIR}/${REPORT_STEM}.json" "${MODEL_ROOT}/vqa_ood_summary_partial.json"
fi
echo "[ood-per-method] complete: ${REPORT_DIR}/${REPORT_STEM}.csv"
