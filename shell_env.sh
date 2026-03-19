#!/usr/bin/env bash

# Shared shell environment bootstrap for local + AMLT runs.
# Usage:
#   source "$(dirname "$0")/shell_env.sh"

# Keep existing HF_HOME if user already configured it.
export HF_HOME="${HF_HOME:-${HUGGINGFACE_HUB_CACHE:-${TRANSFORMERS_CACHE:-$HOME/.cache/huggingface}}}"
mkdir -p "${HF_HOME}"

# Follow G-OPD style: always include user local bin.
export PATH="$HOME/.local/bin:$PATH"

# Prevent ~/.local site-packages from shadowing conda env packages.
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# Optional explicit override.
if [[ -n "${SID_ENV_PATH:-}" ]]; then
  export PATH="${SID_ENV_PATH}:$PATH"
fi

# Legacy local env path compatibility.
if [[ -d "/home/ygu/miniconda3/envs/sid_dit/bin" ]]; then
  export PATH="/home/ygu/miniconda3/envs/sid_dit/bin:$PATH"
fi

# AMLT ptca env compatibility.
if [[ -d "/opt/conda/envs/ptca/bin" ]]; then
  export PATH="/opt/conda/envs/ptca/bin:$PATH"
fi

# Stable python executable selection.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    export PYTHON_BIN="python3"
  else
    export PYTHON_BIN="python"
  fi
fi

# Prefer the CUDA/cuDNN libraries bundled with the active torch install.
# This avoids pulling incompatible system cuDNN from stale LD_LIBRARY_PATH entries.
if [[ "${SID_PREFER_TORCH_LIBS:-1}" == "1" ]]; then
  TORCH_LIB_DIR="$("${PYTHON_BIN}" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null || true)"
  if [[ -n "${TORCH_LIB_DIR}" && -d "${TORCH_LIB_DIR}" ]]; then
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
      export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH}"
    else
      export LD_LIBRARY_PATH="${TORCH_LIB_DIR}"
    fi
  fi
fi
