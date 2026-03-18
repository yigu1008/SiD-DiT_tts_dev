#!/usr/bin/env bash

# Shared shell environment bootstrap for local + AMLT runs.
# Usage:
#   source "$(dirname "$0")/shell_env.sh"

# Keep existing HF_HOME if user already configured it.
export HF_HOME="${HF_HOME:-${HUGGINGFACE_HUB_CACHE:-${TRANSFORMERS_CACHE:-$HOME/.cache/huggingface}}}"
mkdir -p "${HF_HOME}"

# Follow G-OPD style: always include user local bin.
export PATH="$HOME/.local/bin:$PATH"

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
