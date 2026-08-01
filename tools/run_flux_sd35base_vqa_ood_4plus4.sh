#!/usr/bin/env bash
set -euo pipefail

# Resume the 200-prompt VQAScore sweep with two isolated 4-GPU jobs:
#   GPUs 0-3: Flux-Schnell; GPUs 4-7: SD3.5-Base.
# Generation completes before resumable OOD evaluation starts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
SOURCE_SID_RUN_ROOT="${SOURCE_SID_RUN_ROOT:-${HUMAN_EVAL_ROOT}/sid_vqascore_algorithm_sweep/genai200_v1}"
STUDY_ROOT="${STUDY_ROOT:-${HUMAN_EVAL_ROOT}/vqascore_remaining_models}"
RUN_ID="${RUN_ID:-genai200_v1}"
RUN_ROOT="${STUDY_ROOT}/${RUN_ID}"

FLUX_GPUS="${FLUX_GPUS:-0,1,2,3}"
SD35_BASE_GPUS="${SD35_BASE_GPUS:-4,5,6,7}"
METHODS="${METHODS:-baseline fksteering bon beam sop ga dts dts_star dynamic_cfg_x0 bon_mcts}"
OOD_EVAL_BACKENDS="${OOD_EVAL_BACKENDS:-imagereward hpsv3 pickscore}"
PHASE="${PHASE:-all}"
DRY_RUN="${DRY_RUN:-0}"

PYTHON_BIN="${PYTHON_BIN:-/home/ygu/miniconda3/envs/sid_dit/bin/python}"
REWARD_ENV_CONDA_BASE="${REWARD_ENV_CONDA_BASE:-/home/ygu/miniconda3}"
VQA_REWARD_ENV_NAME="${VQA_REWARD_ENV_NAME:-vqascore_reward}"
STANDARD_REWARD_ENV_NAME="${STANDARD_REWARD_ENV_NAME:-reward}"

case "${PHASE}" in all|generate|ood_eval) ;; *)
  echo "Error: PHASE must be all, generate, or ood_eval." >&2
  exit 2
esac
case "${DRY_RUN}" in 0|1) ;; *)
  echo "Error: DRY_RUN must be 0 or 1." >&2
  exit 2
esac
if [[ " ${METHODS} " == *" das "* ]]; then
  echo "Error: this continuation matches the supplied ten-method table; DAS is excluded." >&2
  exit 2
fi

IFS=',' read -r -a flux_gpu_array <<< "${FLUX_GPUS}"
IFS=',' read -r -a sd35_gpu_array <<< "${SD35_BASE_GPUS}"
if (( ${#flux_gpu_array[@]} != 4 || ${#sd35_gpu_array[@]} != 4 )); then
  echo "Error: FLUX_GPUS and SD35_BASE_GPUS must each contain exactly four GPUs." >&2
  exit 2
fi
all_gpus=("${flux_gpu_array[@]}" "${sd35_gpu_array[@]}")
unique_gpu_count="$(printf '%s\n' "${all_gpus[@]}" | sort -u | wc -l | tr -d ' ')"
if (( unique_gpu_count != 8 )); then
  echo "Error: the Flux and SD3.5-Base GPU sets must not overlap." >&2
  exit 2
fi

echo "[4+4] run_root=${RUN_ROOT}"
echo "[4+4] Flux-Schnell GPUs=${FLUX_GPUS}; reward GPU=${flux_gpu_array[3]}"
echo "[4+4] SD3.5-Base GPUs=${SD35_BASE_GPUS}; reward GPU=${sd35_gpu_array[3]}"
echo "[4+4] methods=${METHODS}"
echo "[4+4] OOD eval backends=${OOD_EVAL_BACKENDS}"
echo "[4+4] phase=${PHASE}"
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

if [[ "${PHASE}" == "all" || "${PHASE}" == "ood_eval" ]]; then
  standard_reward_python="${REWARD_ENV_CONDA_BASE}/envs/${STANDARD_REWARD_ENV_NAME}/bin/python"
  PATH="$(dirname "${standard_reward_python}"):${PATH}" \
    "${standard_reward_python}" - <<'PY'
import click
import platformdirs
from google.protobuf import runtime_version
import ImageReward
import hpsv3
import transformers
print("[4+4] standard OOD reward runtime imports OK")
PY
fi

mkdir -p "${RUN_ROOT}/launcher_logs"
RUNNER="${REPO}/tools/run_remaining_vqascore_algorithm_sweeps.sh"

launch_pair() {
  local phase="$1"
  local flux_log="${RUN_ROOT}/launcher_logs/flux_schnell_${phase}.log"
  local sd35_log="${RUN_ROOT}/launcher_logs/sd35_base_${phase}.log"
  local post_eval_only=0
  local skip_post_eval=1
  if [[ "${phase}" == "ood_eval" ]]; then
    post_eval_only=1
    skip_post_eval=0
  fi

  echo "[4+4] launching ${phase}; logs:"
  echo "  ${flux_log}"
  echo "  ${sd35_log}"

  env \
    CUDA_VISIBLE_DEVICES="${FLUX_GPUS}" \
    REWARD_CUDA_VISIBLE_DEVICES="${flux_gpu_array[3]}" \
    HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT}" \
    SOURCE_SID_RUN_ROOT="${SOURCE_SID_RUN_ROOT}" \
    STUDY_ROOT="${STUDY_ROOT}" RUN_ID="${RUN_ID}" \
    STUDY_MANIFEST="${RUN_ROOT}/study_manifest_flux_schnell.json" \
    AUDIT_OUT_CSV="${RUN_ROOT}/flux_schnell/vqascore_coverage.csv" \
    BACKENDS=flux_schnell METHODS="${METHODS}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    REWARD_ENV_CONDA_BASE="${REWARD_ENV_CONDA_BASE}" \
    VQA_REWARD_ENV_NAME="${VQA_REWARD_ENV_NAME}" \
    STANDARD_REWARD_ENV_NAME="${STANDARD_REWARD_ENV_NAME}" \
    REWARD_SERVER_BASE_PORT=5400 POSTHOC_REWARD_SERVER_PORT=5490 \
    POST_EVAL_ONLY="${post_eval_only}" SKIP_POST_EVAL="${skip_post_eval}" \
    POSTHOC_EVAL_BACKENDS="${OOD_EVAL_BACKENDS}" \
    POSTHOC_SKIP_COMPLETE=1 POSTHOC_MERGE_SCOPE=model FAIL_FAST=1 \
    bash "${RUNNER}" >"${flux_log}" 2>&1 &
  local flux_pid=$!

  env \
    CUDA_VISIBLE_DEVICES="${SD35_BASE_GPUS}" \
    REWARD_CUDA_VISIBLE_DEVICES="${sd35_gpu_array[3]}" \
    HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT}" \
    SOURCE_SID_RUN_ROOT="${SOURCE_SID_RUN_ROOT}" \
    STUDY_ROOT="${STUDY_ROOT}" RUN_ID="${RUN_ID}" \
    STUDY_MANIFEST="${RUN_ROOT}/study_manifest_sd35_base.json" \
    AUDIT_OUT_CSV="${RUN_ROOT}/sd35_base/vqascore_coverage.csv" \
    BACKENDS=sd35_base METHODS="${METHODS}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    REWARD_ENV_CONDA_BASE="${REWARD_ENV_CONDA_BASE}" \
    VQA_REWARD_ENV_NAME="${VQA_REWARD_ENV_NAME}" \
    STANDARD_REWARD_ENV_NAME="${STANDARD_REWARD_ENV_NAME}" \
    REWARD_SERVER_BASE_PORT=5500 POSTHOC_REWARD_SERVER_PORT=5590 \
    POST_EVAL_ONLY="${post_eval_only}" SKIP_POST_EVAL="${skip_post_eval}" \
    POSTHOC_EVAL_BACKENDS="${OOD_EVAL_BACKENDS}" \
    POSTHOC_SKIP_COMPLETE=1 POSTHOC_MERGE_SCOPE=model FAIL_FAST=1 \
    bash "${RUNNER}" >"${sd35_log}" 2>&1 &
  local sd35_pid=$!

  local failed=0
  if ! wait "${flux_pid}"; then
    echo "Error: Flux-Schnell ${phase} failed; tail follows." >&2
    tail -n 100 "${flux_log}" >&2 || true
    failed=1
  fi
  if ! wait "${sd35_pid}"; then
    echo "Error: SD3.5-Base ${phase} failed; tail follows." >&2
    tail -n 100 "${sd35_log}" >&2 || true
    failed=1
  fi
  if (( failed )); then
    return 1
  fi
  echo "[4+4] ${phase} complete"
}

if [[ "${PHASE}" == "all" || "${PHASE}" == "generate" ]]; then
  launch_pair generate
fi
if [[ "${PHASE}" == "all" || "${PHASE}" == "ood_eval" ]]; then
  launch_pair ood_eval
fi

echo "[4+4] done: ${RUN_ROOT}"
