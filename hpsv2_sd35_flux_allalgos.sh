#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

# Two-model entrypoint: SD3.5 + FLUX, all algorithms.
export RUN_SANA="${RUN_SANA:-0}"
if [[ -z "${RUN_SD35:-}" ]]; then
  # TDD-FLUX jobs are typically FLUX-only; avoid accidental SD3.5 CKPT checks.
  if [[ "${FLUX_BACKEND:-flux}" == "tdd_flux" ]]; then
    export RUN_SD35="0"
  else
    export RUN_SD35="1"
  fi
else
  export RUN_SD35
fi
export RUN_FLUX="${RUN_FLUX:-1}"
export METHODS="${METHODS:-baseline greedy mcts ga smc}"
export USE_QWEN="${USE_QWEN:-1}"
export SD35_BACKEND="${SD35_BACKEND:-sid}"
export FLUX_BACKEND="${FLUX_BACKEND:-flux}"

if [[ "${RUN_SD35}" == "0" ]]; then
  # Prevent inherited env from unrelated SD3.5 jobs from leaking into flux-only runs.
  unset SD35_CKPT SD35_LORA_PATH SD35_LORA_SCALE || true
fi

bash "${SCRIPT_DIR}/hpsv2_all_models_ddp_suite.sh" "$@"
