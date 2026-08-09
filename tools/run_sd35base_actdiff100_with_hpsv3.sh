#!/usr/bin/env bash
set -euo pipefail

# Four-GPU concurrent layout:
#   first two GPUs : SD3.5-Base ActDiff generation ranks
#   third GPU      : online VQAScore reward server for ActDiff
#   fourth GPU     : post-hoc HPSv3 evaluation for an already generated model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
RUN_ID="${RUN_ID:-genai200_v1}"
GPUS="${GPUS:-4,5,6,7}"
TARGET_PROMPTS="${TARGET_PROMPTS:-100}"
EXPECTED_PROMPTS="${EXPECTED_PROMPTS:-200}"
OOD_MODEL="${OOD_MODEL:-flux_schnell}"
OOD_METHODS="${OOD_METHODS:-baseline fksteering bon beam sop ga dts dts_star dynamic_cfg_x0 bon_mcts}"
OOD_EXPECTED_PROMPTS="${OOD_EXPECTED_PROMPTS:-200}"
OOD_PORT="${OOD_PORT:-5795}"
PYTHON_BIN="${PYTHON_BIN:-/home/ygu/miniconda3/envs/sid_dit/bin/python}"
REWARD_ENV_CONDA_BASE="${REWARD_ENV_CONDA_BASE:-/home/ygu/miniconda3}"
VQA_REWARD_ENV_NAME="${VQA_REWARD_ENV_NAME:-vqascore_reward}"
STANDARD_REWARD_ENV_NAME="${STANDARD_REWARD_ENV_NAME:-reward}"
DRY_RUN="${DRY_RUN:-0}"

case "${DRY_RUN}" in 0|1) ;;
  *) echo "Error: DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac
case "${OOD_MODEL}" in
  flux|flux_schnell) OOD_MODEL=flux_schnell ;;
  sense|senseflow|senseflow_large) OOD_MODEL=senseflow_large ;;
  senseflow_medium) ;;
  base|sd35|sd35_base) OOD_MODEL=sd35_base ;;
  *) echo "Error: OOD_MODEL must be flux_schnell, senseflow_large, senseflow_medium, or sd35_base." >&2; exit 2 ;;
esac

IFS=',' read -r -a gpu_array <<< "${GPUS}"
if (( ${#gpu_array[@]} != 4 )); then
  echo "Error: GPUS must contain exactly four physical GPU IDs." >&2
  exit 2
fi
unique_count="$(printf '%s\n' "${gpu_array[@]}" | sort -u | wc -l | tr -d ' ')"
if (( unique_count != 4 )); then
  echo "Error: GPUS contains duplicate IDs: ${GPUS}" >&2
  exit 2
fi

ACTDIFF_GPUS="${gpu_array[0]},${gpu_array[1]},${gpu_array[2]}"
VQA_GPU="${gpu_array[2]}"
OOD_GPU="${gpu_array[3]}"
LOG_ROOT="${HUMAN_EVAL_ROOT}/launcher_logs/${RUN_ID}"
ACTDIFF_LOG="${LOG_ROOT}/sd35_base_actdiff_first${TARGET_PROMPTS}.log"
OOD_LOG="${LOG_ROOT}/${OOD_MODEL}_hpsv3_eval.log"

echo "[actdiff+hpsv3] SD3.5-Base generation GPUs=${gpu_array[0]},${gpu_array[1]}"
echo "[actdiff+hpsv3] VQAScore GPU=${VQA_GPU}"
echo "[actdiff+hpsv3] HPSv3 GPU=${OOD_GPU} model=${OOD_MODEL}"
echo "[actdiff+hpsv3] ActDiff target=${TARGET_PROMPTS}/${EXPECTED_PROMPTS}"
echo "[actdiff+hpsv3] logs=${LOG_ROOT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  GPUS="${ACTDIFF_GPUS}" TARGET_PROMPTS="${TARGET_PROMPTS}" \
    EXPECTED_PROMPTS="${EXPECTED_PROMPTS}" DRY_RUN=1 \
    bash "${SCRIPT_DIR}/run_sd35base_actdiff_after_ga.sh"
  MODEL="${OOD_MODEL}" GPU="${OOD_GPU}" OOD_EVAL_BACKENDS=hpsv3 \
    METHODS="${OOD_METHODS}" EXPECTED_PROMPTS="${OOD_EXPECTED_PROMPTS}" \
    REWARD_SERVER_PORT="${OOD_PORT}" DRY_RUN=1 \
    bash "${SCRIPT_DIR}/run_vqa_ood_eval_model.sh"
  exit 0
fi

mkdir -p "${LOG_ROOT}"
children=()
cleanup() {
  local pid
  for pid in "${children[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
}
trap cleanup INT TERM EXIT

env \
  HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT}" RUN_ID="${RUN_ID}" \
  GPUS="${ACTDIFF_GPUS}" TARGET_PROMPTS="${TARGET_PROMPTS}" \
  EXPECTED_PROMPTS="${EXPECTED_PROMPTS}" REQUIRE_GA_COMPLETE=0 \
  PYTHON_BIN="${PYTHON_BIN}" REWARD_ENV_CONDA_BASE="${REWARD_ENV_CONDA_BASE}" \
  VQA_REWARD_ENV_NAME="${VQA_REWARD_ENV_NAME}" \
  bash "${SCRIPT_DIR}/run_sd35base_actdiff_after_ga.sh" \
  >"${ACTDIFF_LOG}" 2>&1 &
actdiff_pid=$!
children+=("${actdiff_pid}")

env \
  HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT}" RUN_ID="${RUN_ID}" \
  MODEL="${OOD_MODEL}" GPU="${OOD_GPU}" METHODS="${OOD_METHODS}" \
  EXPECTED_PROMPTS="${OOD_EXPECTED_PROMPTS}" OOD_EVAL_BACKENDS=hpsv3 \
  REWARD_SERVER_PORT="${OOD_PORT}" FINALIZE_REPORT=0 \
  SNAPSHOT_STEM="vqa_ood_summary_hpsv3_only_partial" \
  PYTHON_BIN="${PYTHON_BIN}" REWARD_ENV_CONDA_BASE="${REWARD_ENV_CONDA_BASE}" \
  STANDARD_REWARD_ENV_NAME="${STANDARD_REWARD_ENV_NAME}" \
  bash "${SCRIPT_DIR}/run_vqa_ood_eval_model.sh" \
  >"${OOD_LOG}" 2>&1 &
ood_pid=$!
children+=("${ood_pid}")

failed=0
if ! wait "${actdiff_pid}"; then
  echo "Error: SD3.5-Base ActDiff failed; tail follows." >&2
  tail -n 100 "${ACTDIFF_LOG}" >&2 || true
  failed=1
fi
if ! wait "${ood_pid}"; then
  echo "Error: ${OOD_MODEL} HPSv3 evaluation failed; tail follows." >&2
  tail -n 100 "${OOD_LOG}" >&2 || true
  failed=1
fi
children=()
trap - INT TERM EXIT
if (( failed )); then
  exit 1
fi

# Rebuild the canonical partial report from every backend file currently on
# disk. This is metadata-only and does not launch another reward model.
MODELS="${OOD_MODEL}" METHODS="${OOD_METHODS}" \
OOD_EVAL_BACKENDS="imagereward hpsv3 pickscore" \
HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT}" RUN_ID="${RUN_ID}" \
EXPECTED_PROMPTS="${OOD_EXPECTED_PROMPTS}" PYTHON_BIN="${PYTHON_BIN}" \
bash "${SCRIPT_DIR}/rebuild_vqa_ood_reports.sh"

echo "[actdiff+hpsv3] complete"
echo "  ActDiff: ${ACTDIFF_LOG}"
echo "  HPSv3:   ${OOD_LOG}"
