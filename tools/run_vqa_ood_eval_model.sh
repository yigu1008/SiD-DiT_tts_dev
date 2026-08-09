#!/usr/bin/env bash
set -euo pipefail

# Post-hoc OOD evaluation for an already generated GenAI-200 VQAScore sweep.
# This script never launches generation or a VQAScore reward server.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${MODEL:?Set MODEL=flux_schnell, sid, senseflow_large, senseflow_medium, or sd35_base}"
case "${MODEL}" in
  flux|flux_schnell)
    MODEL=flux_schnell; LAYOUT=flux; DEFAULT_GPU=3; DEFAULT_PORT=5490 ;;
  sid)
    MODEL=sid; LAYOUT=sd35; DEFAULT_GPU=4; DEFAULT_PORT=5690 ;;
  sense|senseflow|senseflow_large)
    MODEL=senseflow_large; LAYOUT=sd35; DEFAULT_GPU=5; DEFAULT_PORT=5790 ;;
  senseflow_medium)
    MODEL=senseflow_medium; LAYOUT=sd35; DEFAULT_GPU=6; DEFAULT_PORT=5890 ;;
  base|sd35|sd35_base)
    MODEL=sd35_base; LAYOUT=sd35; DEFAULT_GPU=7; DEFAULT_PORT=5590 ;;
  *)
    echo "Error: unsupported MODEL=${MODEL}." >&2
    exit 2
    ;;
esac

HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
RUN_ID="${RUN_ID:-genai200_v1}"
EXPECTED_PROMPTS="${EXPECTED_PROMPTS:-200}"
METHODS="${METHODS:-baseline fksteering bon beam sop ga dts dts_star dynamic_cfg_x0 bon_mcts}"
OOD_EVAL_BACKENDS="${OOD_EVAL_BACKENDS:-imagereward hpsv3 pickscore}"
GPU="${GPU:-${DEFAULT_GPU}}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-${DEFAULT_PORT}}"

if [[ "${MODEL}" == "sid" ]]; then
  STUDY_RUN_ROOT="${STUDY_RUN_ROOT:-${HUMAN_EVAL_ROOT}/sid_vqascore_algorithm_sweep/${RUN_ID}}"
else
  STUDY_RUN_ROOT="${STUDY_RUN_ROOT:-${HUMAN_EVAL_ROOT}/vqascore_remaining_models/${RUN_ID}}"
fi
MODEL_ROOT="${STUDY_RUN_ROOT}/${MODEL}"
REPORT_DIR="${MODEL_ROOT}/run_${RUN_ID}/reports"

PYTHON_BIN="${PYTHON_BIN:-/home/ygu/miniconda3/envs/sid_dit/bin/python}"
REWARD_ENV_CONDA_BASE="${REWARD_ENV_CONDA_BASE:-/home/ygu/miniconda3}"
STANDARD_REWARD_ENV_NAME="${STANDARD_REWARD_ENV_NAME:-reward}"
STANDARD_REWARD_PY="${REWARD_ENV_CONDA_BASE}/envs/${STANDARD_REWARD_ENV_NAME}/bin/python"
DRY_RUN="${DRY_RUN:-0}"

case "${DRY_RUN}" in 0|1) ;;
  *) echo "Error: DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac
if [[ ! "${EXPECTED_PROMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: EXPECTED_PROMPTS must be positive." >&2
  exit 2
fi

echo "[ood-only] model=${MODEL} layout=${LAYOUT}"
echo "[ood-only] model_root=${MODEL_ROOT}"
echo "[ood-only] GPU=${GPU} port=${REWARD_SERVER_PORT}"
echo "[ood-only] backends=${OOD_EVAL_BACKENDS}"
echo "[ood-only] methods=${METHODS}"
if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

for executable in "${PYTHON_BIN}" "${STANDARD_REWARD_PY}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "Error: required Python is not executable: ${executable}" >&2
    exit 1
  fi
done
if [[ ! -d "${MODEL_ROOT}/run_${RUN_ID}" ]]; then
  echo "Error: generated run directory is missing: ${MODEL_ROOT}/run_${RUN_ID}" >&2
  exit 1
fi
mkdir -p "${REPORT_DIR}"

MODEL_ROOT="${MODEL_ROOT}" RUN_ID="${RUN_ID}" METHODS="${METHODS}" \
EXPECTED_PROMPTS="${EXPECTED_PROMPTS}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["MODEL_ROOT"]) / f"run_{os.environ['RUN_ID']}"
expected = int(os.environ["EXPECTED_PROMPTS"])
failures = []
for method in os.environ["METHODS"].split():
    path = root / method / "aggregate_ddp.json"
    if not path.is_file():
        failures.append(f"{method}: missing aggregate_ddp.json")
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    count = int(payload.get("num_samples", 0) or 0)
    if count != expected:
        failures.append(f"{method}: generated {count}/{expected}")
if failures:
    raise SystemExit("generation coverage is incomplete:\n  " + "\n  ".join(failures))
print(f"[ood-only] generation preflight OK: {len(os.environ['METHODS'].split())} methods")
PY

PATH="$(dirname "${STANDARD_REWARD_PY}"):${PATH}" \
  "${STANDARD_REWARD_PY}" - <<'PY'
import click
import platformdirs
from google.protobuf import runtime_version
import ImageReward
import hpsv3
import transformers
print("[ood-only] standard reward runtime imports OK")
PY

OUT_ROOT="${MODEL_ROOT}" \
REWARD_PY="${STANDARD_REWARD_PY}" \
STANDARD_REWARD_PY="${STANDARD_REWARD_PY}" \
PYTHON_BIN="${PYTHON_BIN}" \
REWARD_SERVER_PORT="${REWARD_SERVER_PORT}" \
REWARD_CUDA_VISIBLE_DEVICES="${GPU}" \
POSTHOC_EVAL_BACKENDS="${OOD_EVAL_BACKENDS}" \
POSTHOC_SKIP_COMPLETE=1 \
POSTHOC_EXPECTED_COUNT="${EXPECTED_PROMPTS}" \
POSTHOC_RUN_ID="${RUN_ID}" \
POSTHOC_ALLOW_MISSING_BACKENDS=0 \
POSTHOC_LAYOUT="${LAYOUT}" \
POSTHOC_SNAPSHOT_ROOT="${STUDY_RUN_ROOT}" \
POSTHOC_SNAPSHOT_MODEL="${MODEL}" \
POSTHOC_SNAPSHOT_SUMMARY="${REPORT_DIR}/vqa_ood_summary_partial.csv" \
bash "${REPO}/post_eval_extra_rewards.sh"

"${PYTHON_BIN}" "${REPO}/tools/merge_posthoc_reward_evals.py" \
  --root "${STUDY_RUN_ROOT}" \
  --include-models "${MODEL}" \
  --run-id "${RUN_ID}" \
  --backends ${OOD_EVAL_BACKENDS} \
  --summary-csv "${REPORT_DIR}/vqa_ood_summary.csv" \
  --expected-count "${EXPECTED_PROMPTS}" \
  --strict

"${PYTHON_BIN}" "${REPO}/tools/audit_vqascore_sweep_coverage.py" \
  --root "${STUDY_RUN_ROOT}" \
  --models "${MODEL}" \
  --methods ${METHODS} \
  --expected-prompts "${EXPECTED_PROMPTS}" \
  --eval-backends ${OOD_EVAL_BACKENDS} \
  --run-id "${RUN_ID}" \
  --out-csv "${REPORT_DIR}/vqascore_coverage.csv"

# Compatibility mirrors for older reporting commands. Canonical run-scoped
# reports remain under run_<id>/reports so multiple runs cannot overwrite one
# another.
cp "${REPORT_DIR}/vqa_ood_summary.csv" "${MODEL_ROOT}/vqa_ood_summary.csv"
cp "${REPORT_DIR}/vqa_ood_summary.json" "${MODEL_ROOT}/vqa_ood_summary.json"
cp "${REPORT_DIR}/vqascore_coverage.csv" "${MODEL_ROOT}/vqascore_coverage.csv"

echo "[ood-only] complete: ${REPORT_DIR}/vqa_ood_summary.csv"
