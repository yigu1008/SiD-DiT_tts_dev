#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

MODEL="${MODEL:-Tongyi-MAI/Z-Image-Turbo}"
PROMPT="${PROMPT:-a studio portrait of an elderly woman smiling, soft window light, 85mm lens, photorealistic}"
OUTDIR="${OUTDIR:-./zimage_scheduler_sigma_out}"

DTYPE="${DTYPE:-bf16}"
WIDTH="${WIDTH:-768}"
HEIGHT="${HEIGHT:-768}"
STEPS="${STEPS:-9}"
SEED="${SEED:-42}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-0.0}"

REWARD_BACKEND="${REWARD_BACKEND:-auto}"
REWARD_MODEL="${REWARD_MODEL:-ImageReward-v1.0}"

SIGMA_FAMILIES_STR="${SIGMA_FAMILIES_STR:-default karras exponential beta}"
read -r -a SIGMA_FAMILIES <<< "${SIGMA_FAMILIES_STR}"

EXTRA_ARGS=()
if [[ -n "${NEGATIVE_PROMPT:-}" ]]; then
  EXTRA_ARGS+=(--negative_prompt "${NEGATIVE_PROMPT}")
fi
if [[ -n "${ATTENTION:-}" ]]; then
  EXTRA_ARGS+=(--attention "${ATTENTION}")
fi
if [[ "${COMPILE_TRANSFORMER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--compile_transformer)
fi
if [[ "${LOG_INTERMEDIATES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--log_intermediates)
fi
if [[ "${SAVE_INTERMEDIATE_IMAGES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save_intermediate_images)
fi
if [[ "${SCORE_INTERMEDIATES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--score_intermediates)
fi
if [[ "${NO_IMAGE_REWARD:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no_image_reward)
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/Z_image_test.py" \
  --model "${MODEL}" \
  --prompt "${PROMPT}" \
  --outdir "${OUTDIR}" \
  --dtype "${DTYPE}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --steps "${STEPS}" \
  --seed "${SEED}" \
  --guidance_scale "${GUIDANCE_SCALE}" \
  --scheduler_test_theme sigma \
  --scheduler_sigma_families "${SIGMA_FAMILIES[@]}" \
  --reward_backend "${REWARD_BACKEND}" \
  --reward_model "${REWARD_MODEL}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
