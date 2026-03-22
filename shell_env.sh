#!/usr/bin/env bash

# Shared shell environment bootstrap for local + AMLT runs.
# Usage:
#   source "$(dirname "$0")/shell_env.sh"

# Keep existing HF_HOME if user already configured it.
export HF_HOME="${HF_HOME:-${HUGGINGFACE_HUB_CACHE:-${TRANSFORMERS_CACHE:-$HOME/.cache/huggingface}}}"
mkdir -p "${HF_HOME}"

# Follow G-OPD style: always include user local bin.
export PATH="$HOME/.local/bin:$PATH"

# Prevent ~/.local site-packages from shadowing conda env packages by default.
# Set SID_ALLOW_USER_SITE=1 to allow `pip install --user` packages at runtime.
if [[ "${SID_ALLOW_USER_SITE:-0}" == "1" ]]; then
  unset PYTHONNOUSERSITE || true
else
  export PYTHONNOUSERSITE=1
fi
unset PYTHONPATH || true

# Optional overlay path for cluster/user-writable Python deps.
# Example:
#   export SID_EXTRA_PYTHONPATH="$HOME/.sid_pydeps/py3.10"
# This keeps base env clean while allowing patched packages (e.g., wandb/xxhash).
if [[ -n "${SID_EXTRA_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${SID_EXTRA_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
fi

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
