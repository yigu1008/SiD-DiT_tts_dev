#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SUMMARY_DIR="${SUMMARY_DIR:-${HUMAN_EVAL_ROOT}/legacy_summary}"
PAIRWISE_OUTPUT="${PAIRWISE_OUTPUT:-${HUMAN_EVAL_ROOT}/pairwise_human_eval}"
BLIND_SEED="${BLIND_SEED:-20260729}"
ANCHOR="${ANCHOR:-bon_mcts}"
OPPONENTS="${OPPONENTS:-baseline bon das}"
MATERIALIZE="${MATERIALIZE:-copy}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"

if [[ "${SKIP_SUMMARY}" != "1" ]]; then
  "${PYTHON_BIN}" "${REPO}/tools/summarize_human_eval_legacy.py" \
    --root "${HUMAN_EVAL_ROOT}" \
    --output-dir "${SUMMARY_DIR}" \
    --materialize none
fi

read -r -a opponent_args <<< "${OPPONENTS}"

exec "${PYTHON_BIN}" "${REPO}/tools/package_pairwise_human_eval.py" \
  --manifest "${SUMMARY_DIR}/legacy_manifest.csv" \
  --output-dir "${PAIRWISE_OUTPUT}" \
  --seed "${BLIND_SEED}" \
  --anchor "${ANCHOR}" \
  --opponents "${opponent_args[@]}" \
  --materialize "${MATERIALIZE}" \
  "$@"
