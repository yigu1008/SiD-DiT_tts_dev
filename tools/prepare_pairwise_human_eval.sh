#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-${REPO}/configs/pairwise_human_eval.yaml}"
SUMMARY_DIR="${SUMMARY_DIR:-${HUMAN_EVAL_ROOT}/legacy_summary}"

# This is postprocessing only. It never launches generation or reward models.
"${PYTHON_BIN}" "${REPO}/tools/summarize_human_eval_legacy.py" \
  --root "${HUMAN_EVAL_ROOT}" \
  --output-dir "${SUMMARY_DIR}" \
  --materialize none

"${PYTHON_BIN}" "${REPO}/tools/pairwise_human_eval.py" \
  --config "${CONFIG}" \
  --root "${HUMAN_EVAL_ROOT}" \
  import-legacy \
  --manifest "${SUMMARY_DIR}/legacy_manifest.csv" \
  "$@"

"${PYTHON_BIN}" "${REPO}/tools/pairwise_human_eval.py" \
  --config "${CONFIG}" \
  --root "${HUMAN_EVAL_ROOT}" \
  validate

"${PYTHON_BIN}" "${REPO}/tools/pairwise_human_eval.py" \
  --config "${CONFIG}" \
  --root "${HUMAN_EVAL_ROOT}" \
  build-tasks \
  --overwrite
