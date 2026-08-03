#!/usr/bin/env bash
set -euo pipefail

# Finish post-hoc ImageReward/HPSv3/PickScore evaluation for the existing
# GenAI-200 SiD-SD3.5 and SenseFlow-SD3.5-Large VQAScore sweeps. The two
# models run concurrently on separate reward GPUs. Existing complete backend
# files are validated and reused; image generation is never launched here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${REPO}/tools/run_vqa_ood_eval_model.sh"

HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
RUN_ID="${RUN_ID:-genai200_v1}"
EXPECTED_PROMPTS="${EXPECTED_PROMPTS:-200}"
METHODS="${METHODS:-baseline fksteering bon beam sop ga dts dts_star dynamic_cfg_x0 bon_mcts}"
OOD_EVAL_BACKENDS="${OOD_EVAL_BACKENDS:-imagereward hpsv3 pickscore}"

SID_GPU="${SID_GPU:-0}"
SENSE_GPU="${SENSE_GPU:-4}"
SID_PORT="${SID_PORT:-5690}"
SENSE_PORT="${SENSE_PORT:-5790}"

SID_STUDY_RUN_ROOT="${SID_STUDY_RUN_ROOT:-${HUMAN_EVAL_ROOT}/sid_vqascore_algorithm_sweep/${RUN_ID}}"
SENSE_STUDY_RUN_ROOT="${SENSE_STUDY_RUN_ROOT:-${HUMAN_EVAL_ROOT}/vqascore_remaining_models/${RUN_ID}}"
SUMMARY_CSV="${SUMMARY_CSV:-${HUMAN_EVAL_ROOT}/sid_sense_vqa_ood_summary_${RUN_ID}.csv}"
PYTHON_BIN="${PYTHON_BIN:-/home/ygu/miniconda3/envs/sid_dit/bin/python}"
DRY_RUN="${DRY_RUN:-0}"

case "${DRY_RUN}" in 0|1) ;;
  *) echo "Error: DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac
if [[ "${SID_GPU}" == "${SENSE_GPU}" ]]; then
  echo "Error: SID_GPU and SENSE_GPU must be different." >&2
  exit 2
fi
if [[ "${SID_PORT}" == "${SENSE_PORT}" ]]; then
  echo "Error: SID_PORT and SENSE_PORT must be different." >&2
  exit 2
fi

LOG_ROOT="${HUMAN_EVAL_ROOT}/launcher_logs/${RUN_ID}"
mkdir -p "${LOG_ROOT}"
SID_LOG="${LOG_ROOT}/sid_ood_eval.log"
SENSE_LOG="${LOG_ROOT}/senseflow_large_ood_eval.log"

echo "[sid+sense] run_id=${RUN_ID} expected_prompts=${EXPECTED_PROMPTS}"
echo "[sid+sense] SiD-SD3.5 GPU=${SID_GPU} port=${SID_PORT}"
echo "[sid+sense] SenseFlow-SD3.5-Large GPU=${SENSE_GPU} port=${SENSE_PORT}"
echo "[sid+sense] backends=${OOD_EVAL_BACKENDS}"
echo "[sid+sense] methods=${METHODS}"
echo "[sid+sense] logs=${LOG_ROOT}"

common=(
  "HUMAN_EVAL_ROOT=${HUMAN_EVAL_ROOT}"
  "RUN_ID=${RUN_ID}"
  "EXPECTED_PROMPTS=${EXPECTED_PROMPTS}"
  "METHODS=${METHODS}"
  "OOD_EVAL_BACKENDS=${OOD_EVAL_BACKENDS}"
  "PYTHON_BIN=${PYTHON_BIN}"
  "DRY_RUN=${DRY_RUN}"
)

env "${common[@]}" MODEL=sid GPU="${SID_GPU}" \
  REWARD_SERVER_PORT="${SID_PORT}" STUDY_RUN_ROOT="${SID_STUDY_RUN_ROOT}" \
  bash "${RUNNER}" >"${SID_LOG}" 2>&1 &
sid_pid=$!

env "${common[@]}" MODEL=senseflow_large GPU="${SENSE_GPU}" \
  REWARD_SERVER_PORT="${SENSE_PORT}" STUDY_RUN_ROOT="${SENSE_STUDY_RUN_ROOT}" \
  bash "${RUNNER}" >"${SENSE_LOG}" 2>&1 &
sense_pid=$!

failed=0
if ! wait "${sid_pid}"; then
  echo "Error: SiD-SD3.5 OOD evaluation failed; tail follows." >&2
  tail -n 100 "${SID_LOG}" >&2 || true
  failed=1
fi
if ! wait "${sense_pid}"; then
  echo "Error: SenseFlow-SD3.5-Large OOD evaluation failed; tail follows." >&2
  tail -n 100 "${SENSE_LOG}" >&2 || true
  failed=1
fi
if (( failed )); then
  exit 1
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[sid+sense] dry run complete"
  exit 0
fi

SID_SUMMARY="${SID_STUDY_RUN_ROOT}/sid/vqa_ood_summary.csv"
SENSE_SUMMARY="${SENSE_STUDY_RUN_ROOT}/senseflow_large/vqa_ood_summary.csv"
SID_SUMMARY="${SID_SUMMARY}" SENSE_SUMMARY="${SENSE_SUMMARY}" \
SUMMARY_CSV="${SUMMARY_CSV}" "${PYTHON_BIN}" - <<'PY'
import csv
import os
from pathlib import Path

sources = [Path(os.environ["SID_SUMMARY"]), Path(os.environ["SENSE_SUMMARY"])]
rows = []
fieldnames = None
for source in sources:
    if not source.is_file():
        raise SystemExit(f"missing model summary: {source}")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        elif reader.fieldnames != fieldnames:
            raise SystemExit(f"summary columns differ: {source}")
        rows.extend(reader)

destination = Path(os.environ["SUMMARY_CSV"])
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_name(destination.name + ".tmp")
with temporary.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
temporary.replace(destination)
print(f"[sid+sense] wrote {len(rows)} rows: {destination}")
PY

echo "[sid+sense] complete: ${SUMMARY_CSV}"
