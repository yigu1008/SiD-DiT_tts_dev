#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

# Some local setups have numpy only on `python` (not `python3`).
if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import numpy  # noqa: F401
PY
then
  if command -v python >/dev/null 2>&1; then
    export PYTHON_BIN="python"
  fi
fi

RUN_DIR="${RUN_DIR:-}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Error: set RUN_DIR to your axis-pipeline output directory." >&2
  exit 1
fi

BACKENDS="${BACKENDS:-imagereward hpsv2}"
REWARD_DEVICE="${REWARD_DEVICE:-cpu}"
INCLUDE_MODES="${INCLUDE_MODES:-fixed stepaware}"
ALLOW_MISSING_BACKENDS="${ALLOW_MISSING_BACKENDS:-0}"

OUT_JSON="${OUT_JSON:-${RUN_DIR}/reward_validation.json}"
OUT_TSV="${OUT_TSV:-${RUN_DIR}/reward_validation.tsv}"
OUT_FINAL_REWARDS_TXT="${OUT_FINAL_REWARDS_TXT:-${RUN_DIR}/reward_final_output.txt}"
OUT_SIMPLE_SUMMARY_TXT="${OUT_SIMPLE_SUMMARY_TXT:-${RUN_DIR}/reward_summary_simple.txt}"

extra=()
if [[ "${ALLOW_MISSING_BACKENDS}" == "1" ]]; then
  extra+=(--allow_missing_backends)
fi

echo "[axis-eval] run_dir=${RUN_DIR}"
echo "[axis-eval] backends=[${BACKENDS}] reward_device=${REWARD_DEVICE} modes=[${INCLUDE_MODES}]"

"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_axis_prompt_bank_rewards.py" \
  --run_dir "${RUN_DIR}" \
  --backends ${BACKENDS} \
  --reward_device "${REWARD_DEVICE}" \
  --include_modes ${INCLUDE_MODES} \
  --out_json "${OUT_JSON}" \
  --out_tsv "${OUT_TSV}" \
  --out_final_rewards_txt "${OUT_FINAL_REWARDS_TXT}" \
  --out_simple_summary_txt "${OUT_SIMPLE_SUMMARY_TXT}" \
  "${extra[@]}"
