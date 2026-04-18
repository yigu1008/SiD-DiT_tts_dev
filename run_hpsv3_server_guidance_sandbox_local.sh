#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
REWARD_SERVER_URL="${REWARD_SERVER_URL:-http://localhost:5100}"
RUN_SAMPLING="${RUN_SAMPLING:-0}"
OUT_JSON="${OUT_JSON:-${SCRIPT_DIR}/hpsv3_server_guidance_sandbox_report.json}"
SAMPLING_OUT_DIR="${SAMPLING_OUT_DIR:-${SCRIPT_DIR}/sandbox_hpsv3_server_guidance_out}"

echo "[hpsv3-sandbox] python=${PYTHON_BIN}"
echo "[hpsv3-sandbox] reward_server_url=${REWARD_SERVER_URL}"
echo "[hpsv3-sandbox] run_sampling=${RUN_SAMPLING}"
echo "[hpsv3-sandbox] out_json=${OUT_JSON}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/sandbox_hpsv3_server_guidance.py"
  --reward_server_url "${REWARD_SERVER_URL}"
  --out_json "${OUT_JSON}"
  --sampling_out_dir "${SAMPLING_OUT_DIR}"
)

if [[ "${RUN_SAMPLING}" == "1" ]]; then
  cmd+=(--run_sampling)
else
  cmd+=(--no-run_sampling)
fi

"${cmd[@]}" "$@"

echo "[hpsv3-sandbox] done"
