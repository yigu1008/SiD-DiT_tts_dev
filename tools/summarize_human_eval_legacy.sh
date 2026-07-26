#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MATERIALIZE="${MATERIALIZE:-symlink}"

exec "${PYTHON_BIN}" "${REPO}/tools/summarize_human_eval_legacy.py" \
  --root "${HUMAN_EVAL_ROOT}" \
  --materialize "${MATERIALIZE}" \
  "$@"
