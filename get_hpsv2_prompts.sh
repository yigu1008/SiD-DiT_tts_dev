#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

OUT_DIR="${OUT_DIR:-/data/ygu}"
STYLE="${STYLE:-all}"

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import hpsv2
PY
then
  echo "[prompts] installing hpsv2 into current env ..."
  "${PYTHON_BIN}" -m pip install --no-cache-dir hpsv2
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/export_hpsv2_prompts.py" \
  --style "${STYLE}" \
  --out_dir "${OUT_DIR}" \
  "$@"
