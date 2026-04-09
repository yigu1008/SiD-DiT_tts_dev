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

PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/hpsv2_subset.txt}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/sd35_axis_prompt_bank_out}"

START_INDEX="${START_INDEX:-0}"
if [[ -n "${NUM_PROMPTS:-}" && "${NUM_PROMPTS}" -gt 0 ]]; then
  END_INDEX=$(( START_INDEX + NUM_PROMPTS ))
else
  END_INDEX="${END_INDEX:--1}"
fi

BACKEND="${SD35_BACKEND:-sid}"
STEPS="${STEPS:-4}"
BASELINE_CFG="${BASELINE_CFG:-1.0}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"

SEED_BASE="${SEED_BASE:-42}"
NUM_SEEDS="${NUM_SEEDS:-3}"
SELECTION_K="${SELECTION_K:-6}"

USE_QWEN="${USE_QWEN:-1}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen3-4B}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
QWEN_DEVICE="${QWEN_DEVICE:-auto}"
REWRITES_CACHE_FILE="${REWRITES_CACHE_FILE:-}"
REWRITES_OVERWRITE="${REWRITES_OVERWRITE:-0}"

TEXT_EMBED_BACKEND="${TEXT_EMBED_BACKEND:-sd35}" # sd35|clip
CLIP_MODEL_ID="${CLIP_MODEL_ID:-openai/clip-vit-large-patch14}"
TEXT_DEVICE="${TEXT_DEVICE:-auto}"
COMPUTE_IMAGE_DIVERSITY="${COMPUTE_IMAGE_DIVERSITY:-1}"
RUN_STEPAWARE="${RUN_STEPAWARE:-1}"
STEPAWARE_POLICY="${STEPAWARE_POLICY:-cycle}" # cycle|random
SAVE_IMAGES="${SAVE_IMAGES:-1}"

QWEN_ARGS=()
if [[ "${USE_QWEN}" != "1" ]]; then
  QWEN_ARGS+=(--no_qwen)
fi
if [[ "${REWRITES_OVERWRITE}" == "1" ]]; then
  QWEN_ARGS+=(--rewrites_overwrite)
else
  QWEN_ARGS+=(--no-rewrites_overwrite)
fi
if [[ -n "${REWRITES_CACHE_FILE}" ]]; then
  QWEN_ARGS+=(--rewrites_cache_file "${REWRITES_CACHE_FILE}")
fi

BOOL_ARGS=()
if [[ "${COMPUTE_IMAGE_DIVERSITY}" == "1" ]]; then
  BOOL_ARGS+=(--compute_image_diversity)
else
  BOOL_ARGS+=(--no-compute_image_diversity)
fi
if [[ "${RUN_STEPAWARE}" == "1" ]]; then
  BOOL_ARGS+=(--run_stepaware)
else
  BOOL_ARGS+=(--no-run_stepaware)
fi
if [[ "${SAVE_IMAGES}" == "1" ]]; then
  BOOL_ARGS+=(--save_images)
else
  BOOL_ARGS+=(--no-save_images)
fi

mkdir -p "${OUT_DIR}"

echo "[axis-pipeline] prompt_file=${PROMPT_FILE} range=[${START_INDEX},${END_INDEX})"
echo "[axis-pipeline] backend=${BACKEND} steps=${STEPS} cfg=${BASELINE_CFG} seeds=${NUM_SEEDS}"
echo "[axis-pipeline] text_embed_backend=${TEXT_EMBED_BACKEND} image_div=${COMPUTE_IMAGE_DIVERSITY}"
echo "[axis-pipeline] out_dir=${OUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/sd35_axis_prompt_bank_pipeline.py" \
  --prompt_file "${PROMPT_FILE}" \
  --start_index "${START_INDEX}" \
  --end_index "${END_INDEX}" \
  --out_dir "${OUT_DIR}" \
  --backend "${BACKEND}" \
  --steps "${STEPS}" \
  --baseline_cfg "${BASELINE_CFG}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --seed_base "${SEED_BASE}" \
  --num_seeds "${NUM_SEEDS}" \
  --selection_k "${SELECTION_K}" \
  --qwen_id "${QWEN_ID}" \
  --qwen_dtype "${QWEN_DTYPE}" \
  --qwen_device "${QWEN_DEVICE}" \
  --text_embed_backend "${TEXT_EMBED_BACKEND}" \
  --clip_model_id "${CLIP_MODEL_ID}" \
  --text_device "${TEXT_DEVICE}" \
  --stepaware_policy "${STEPAWARE_POLICY}" \
  "${QWEN_ARGS[@]}" \
  "${BOOL_ARGS[@]}"
