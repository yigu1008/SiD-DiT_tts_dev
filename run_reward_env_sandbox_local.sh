#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/reward_sandbox_out}"
OUT_JSON="${OUT_JSON:-}"

CASES="${CASES:-imagereward,hpsv3_imscore,hpsv3_official}"
ALLOW_FAIL_CASES="${ALLOW_FAIL_CASES:-hpsv3_official}"
DEVICE="${DEVICE:-cuda}"
PROMPT="${PROMPT:-a cinematic portrait of a woman in soft rim light, 85mm, ultra detailed}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"

INSTALL_DEPS="${INSTALL_DEPS:-0}"
FORCE_OFFLINE="${FORCE_OFFLINE:-0}"
STRICT_DEVICE="${STRICT_DEVICE:-0}"

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/install_reward_deps.sh"
fi

if [[ "${FORCE_OFFLINE}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
else
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
fi

mkdir -p "${OUT_DIR}"
if [[ -z "${OUT_JSON}" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  OUT_JSON="${OUT_DIR}/reward_env_sandbox_${ts}.json"
fi

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/reward_env_sandbox_check.py"
  --python-bin "${PYTHON_BIN}"
  --cases "${CASES}"
  --allow-fail-cases "${ALLOW_FAIL_CASES}"
  --device "${DEVICE}"
  --prompt "${PROMPT}"
  --image-size "${IMAGE_SIZE}"
  --timeout-sec "${TIMEOUT_SEC}"
  --out-json "${OUT_JSON}"
  --force-local-reward
)

if [[ "${STRICT_DEVICE}" == "1" ]]; then
  cmd+=(--strict-device)
fi

echo "[sandbox-run] python=${PYTHON_BIN}"
echo "[sandbox-run] cases=${CASES}"
echo "[sandbox-run] allow_fail_cases=${ALLOW_FAIL_CASES}"
echo "[sandbox-run] device=${DEVICE} offline=${FORCE_OFFLINE} strict_device=${STRICT_DEVICE}"
echo "[sandbox-run] out_json=${OUT_JSON}"

"${cmd[@]}"

echo "[sandbox-run] done"
echo "[sandbox-run] report: ${OUT_JSON}"
