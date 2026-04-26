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
# Prefer a Python that can import torch (needed by local multi-GPU launchers).
if [[ -z "${PYTHON_BIN:-}" ]]; then
  _sid_pick_torch_python() {
    local cand
    local -a cands=()
    # Current shell python first.
    if command -v python3 >/dev/null 2>&1; then cands+=("python3"); fi
    if command -v python >/dev/null 2>&1; then cands+=("python"); fi
    # Common local/cluster envs used in this repo.
    if [[ -x "/Users/guyi/miniconda3/bin/python3" ]]; then cands+=("/Users/guyi/miniconda3/bin/python3"); fi
    if [[ -x "/opt/conda/envs/ptca/bin/python" ]]; then cands+=("/opt/conda/envs/ptca/bin/python"); fi
    if [[ -x "/home/ygu/miniconda3/envs/sid_dit/bin/python" ]]; then cands+=("/home/ygu/miniconda3/envs/sid_dit/bin/python"); fi

    for cand in "${cands[@]}"; do
      if "${cand}" - <<'PY' >/dev/null 2>&1
import torch
print(torch.__version__)
PY
      then
        echo "${cand}"
        return 0
      fi
    done

    # Fallback (may not have torch, but keeps behavior predictable).
    if command -v python3 >/dev/null 2>&1; then
      echo "python3"
    else
      echo "python"
    fi
    return 0
  }

  export PYTHON_BIN="$(_sid_pick_torch_python)"
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
