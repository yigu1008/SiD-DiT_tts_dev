#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
IMPL="${IMPL:-imscore}"                    # imscore | official
MODEL_ID="${MODEL_ID:-RE-N-Y/hpsv3}"
PROMPT="${PROMPT:-A photo of a cat sitting in a sink.}"
RESOLUTIONS="${RESOLUTIONS:-224,512,1024}" # comma-separated
MODES="${MODES:-judge,grad}"               # judge,grad
BATCH_SIZE="${BATCH_SIZE:-1}"
DTYPE="${DTYPE:-float32}"                  # float16|float32|bfloat16
REPEATS="${REPEATS:-1}"
RESERVE_GB="${RESERVE_GB:-0}"              # pre-reserve VRAM to mimic SD memory residency
OUT_DIR="${OUT_DIR:-./hpsv3_grad_probe_out}"

mkdir -p "${OUT_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_JSON="${OUT_DIR}/hpsv3_grad_probe_${TS}.json"

echo "[run] python=${PYTHON_BIN} device=${DEVICE} impl=${IMPL}"
echo "[run] resolutions=${RESOLUTIONS} modes=${MODES} reserve_gb=${RESERVE_GB}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/hpsv3_grad_correction_probe.py" \
  --impl "${IMPL}" \
  --model_id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --prompt "${PROMPT}" \
  --resolutions "${RESOLUTIONS}" \
  --modes "${MODES}" \
  --batch_size "${BATCH_SIZE}" \
  --dtype "${DTYPE}" \
  --reserve_gb "${RESERVE_GB}" \
  --repeats "${REPEATS}" \
  --out_json "${OUT_JSON}"

echo "[run] wrote ${OUT_JSON}"

