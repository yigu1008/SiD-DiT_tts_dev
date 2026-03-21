#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

MODEL="${MODEL:-Tongyi-MAI/Z-Image-Turbo}"
PROMPT="${PROMPT:-a studio portrait of an elderly woman smiling, soft window light, 85mm lens, photorealistic}"
OUTDIR="${OUTDIR:-./zimage_scheduler_stochastic_out}"

DTYPE="${DTYPE:-bf16}"
WIDTH="${WIDTH:-768}"
HEIGHT="${HEIGHT:-768}"
STEPS="${STEPS:-9}"
SEED="${SEED:-42}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-0.0}"

REWARD_BACKEND="${REWARD_BACKEND:-auto}"
REWARD_MODEL="${REWARD_MODEL:-ImageReward-v1.0}"

STOCHASTIC_VALUES_STR="${STOCHASTIC_VALUES_STR:-false true}"
read -r -a STOCHASTIC_VALUES <<< "${STOCHASTIC_VALUES_STR}"

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
if [[ -n "${S_CHURN:-}" || -n "${S_TMIN:-}" || -n "${S_TMAX:-}" || -n "${S_NOISE:-}" ]]; then
  EXTRA_ARGS+=(--enable_scheduler_step_kwargs)
  if [[ -n "${S_CHURN:-}" ]]; then
    EXTRA_ARGS+=(--s_churn "${S_CHURN}")
  fi
  if [[ -n "${S_TMIN:-}" ]]; then
    EXTRA_ARGS+=(--s_tmin "${S_TMIN}")
  fi
  if [[ -n "${S_TMAX:-}" ]]; then
    EXTRA_ARGS+=(--s_tmax "${S_TMAX}")
  fi
  if [[ -n "${S_NOISE:-}" ]]; then
    EXTRA_ARGS+=(--s_noise "${S_NOISE}")
  fi
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
  --scheduler_test_theme stochastic \
  --scheduler_stochastic_values "${STOCHASTIC_VALUES[@]}" \
  --reward_backend "${REWARD_BACKEND}" \
  --reward_model "${REWARD_MODEL}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
