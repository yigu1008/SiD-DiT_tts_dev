#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

OUT_DIR="${OUT_DIR:-/data/ygu}"
STYLE="${STYLE:-all}"
MINI_PER_STYLE="${MINI_PER_STYLE:-0}"
MINI_SEED="${MINI_SEED:-42}"
MINI_OUT_FILE="${MINI_OUT_FILE:-}"

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import hpsv2
PY
then
  echo "[prompts] installing hpsv2 into current env ..."
  if ! "${PYTHON_BIN}" -m pip install --no-cache-dir --no-deps hpsv2; then
    "${PYTHON_BIN}" -m pip install --no-cache-dir hpsv2
  fi
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/export_hpsv2_prompts.py" \
  --style "${STYLE}" \
  --out_dir "${OUT_DIR}" \
  "$@"

if [[ "${STYLE}" == "all" && "${MINI_PER_STYLE}" -gt 0 ]]; then
  echo "[prompts] creating deterministic subset: per_style=${MINI_PER_STYLE} seed=${MINI_SEED}"
  subset_args=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/make_hpsv2_subset.py"
    --in_dir "${OUT_DIR}"
    --per_style "${MINI_PER_STYLE}"
    --seed "${MINI_SEED}"
  )
  if [[ -n "${MINI_OUT_FILE}" ]]; then
    subset_args+=(--out_file "${MINI_OUT_FILE}")
  fi
  "${subset_args[@]}"
fi
