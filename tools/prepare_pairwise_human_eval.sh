#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-${REPO}/configs/pairwise_human_eval.yaml}"
SUMMARY_DIR="${SUMMARY_DIR:-${HUMAN_EVAL_ROOT}/legacy_summary}"
BLIND_SEED="${BLIND_SEED:-20260729}"
ALLOW_INCOMPLETE=0
OVERWRITE_DERIVED=0

usage() {
  cat <<'EOF'
Usage: tools/prepare_pairwise_human_eval.sh [options]

Options:
  --allow-incomplete   Build all usable ActDiff pairs and log missing pairs.
  --overwrite          Replace derived images/tasks only; never legacy_runs.
  --overwrite-derived  Alias for --overwrite.
  -h, --help           Show this help.
EOF
}

while (($#)); do
  case "$1" in
    --allow-incomplete)
      ALLOW_INCOMPLETE=1
      ;;
    --overwrite|--overwrite-derived)
      OVERWRITE_DERIVED=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

common_optional=()
if [[ "${ALLOW_INCOMPLETE}" == "1" ]]; then
  common_optional+=(--allow-incomplete)
fi
import_optional=("${common_optional[@]}")
build_optional=("${common_optional[@]}")
if [[ "${OVERWRITE_DERIVED}" == "1" ]]; then
  import_optional+=(--overwrite)
  build_optional+=(--overwrite)
fi

# This is postprocessing only. It never launches generation or reward models.
# legacy_runs is read-only. All writes go to legacy_summary, images, and tasks.
"${PYTHON_BIN}" "${REPO}/tools/summarize_human_eval_legacy.py" \
  --root "${HUMAN_EVAL_ROOT}" \
  --output-dir "${SUMMARY_DIR}" \
  --materialize none

"${PYTHON_BIN}" "${REPO}/tools/pairwise_human_eval.py" \
  --config "${CONFIG}" \
  --root "${HUMAN_EVAL_ROOT}" \
  import-legacy \
  --manifest "${SUMMARY_DIR}/legacy_manifest.csv" \
  "${import_optional[@]}"

"${PYTHON_BIN}" "${REPO}/tools/pairwise_human_eval.py" \
  --config "${CONFIG}" \
  --root "${HUMAN_EVAL_ROOT}" \
  validate \
  "${common_optional[@]}"

"${PYTHON_BIN}" "${REPO}/tools/pairwise_human_eval.py" \
  --config "${CONFIG}" \
  --root "${HUMAN_EVAL_ROOT}" \
  build-tasks \
  --seed "${BLIND_SEED}" \
  "${build_optional[@]}"
