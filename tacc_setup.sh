#!/usr/bin/env bash
# Environment setup for SSH / TACC cluster.
# Mirrors the environment.setup + early command steps from amlt/terminal.yaml.
#
# Usage:
#   source tacc_setup.sh          # to set env vars in current shell
#   bash tacc_setup.sh            # to just run installs
#
# Key overrides (set before sourcing):
#   PYTHON_BIN   - path to python (default: auto-detect ptca conda env or python3)
#   CONDA_SH     - path to conda.sh (default: /opt/conda/etc/profile.d/conda.sh)
#   DATA_ROOT    - root for model caches and outputs (default: $SCRATCH or $HOME/data)

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Conda
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
if [[ -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
fi

# Python binary — prefer ptca env, fall back to system python3
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/opt/conda/envs/ptca/bin/python" ]]; then
    PYTHON_BIN="/opt/conda/envs/ptca/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi
export PYTHON_BIN
PIP="${PYTHON_BIN} -m pip"

echo "[setup] PYTHON_BIN=${PYTHON_BIN}"
"${PYTHON_BIN}" -V

# Data root: prefer $SCRATCH (TACC), fall back to $HOME/data
if [[ -z "${DATA_ROOT:-}" ]]; then
  DATA_ROOT="${SCRATCH:-${HOME}/data}"
fi
export DATA_ROOT

# ---------------------------------------------------------------------------
# PATH
# ---------------------------------------------------------------------------
export PATH="/home/aiscuser/.local/bin:${PATH}"
export PATH="${PATH}:/home/aiscuser/.local/bin:/root/.local/bin:/opt/conda/envs/ptca/bin"

# ---------------------------------------------------------------------------
# HuggingFace / cache dirs
# ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-${DATA_ROOT}/model_cache/hf_cache}"
export IMAGEREWARD_CACHE="${IMAGEREWARD_CACHE:-${DATA_ROOT}/model_cache/ImageReward}"
export HPS_ROOT="${HPS_ROOT:-${DATA_ROOT}/model_cache/hpsv2}"

mkdir -p \
  "${HF_HOME}" \
  "${IMAGEREWARD_CACHE}" \
  "${HPS_ROOT}" \
  "${DATA_ROOT}/model_cache/clip"

# Symlink CLIP cache to expected location
if [[ -d "${HOME}/.cache" ]] || mkdir -p "${HOME}/.cache"; then
  ln -sfn "${DATA_ROOT}/model_cache/clip" "${HOME}/.cache/clip" || true
fi

# ---------------------------------------------------------------------------
# NCCL / distributed env
# ---------------------------------------------------------------------------
export AZFUSE_USE_FUSE="${AZFUSE_USE_FUSE:-0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"

# ---------------------------------------------------------------------------
# Pip installs
# ---------------------------------------------------------------------------
echo "[setup] upgrading build tooling ..."
${PIP} install --upgrade "setuptools>=70,<76" wheel

echo "[setup] installing requirements.txt ..."
${PIP} install -r "${SCRIPT_DIR}/requirements.txt"

echo "[setup] verifying torch + CUDA ..."
"${PYTHON_BIN}" -c "
import os, torch
print('[torch-check]', torch.__version__,
      'cuda=', torch.version.cuda,
      'is_available=', torch.cuda.is_available(),
      'count=', torch.cuda.device_count(),
      'cvd=', os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'))
assert torch.cuda.is_available() and (torch.version.cuda is not None), \
    'CPU-only torch detected after requirements install'
"

echo "[setup] installing xformers ..."
${PIP} install --no-cache-dir --force-reinstall --no-deps "xformers==0.0.31.post1" || true
"${PYTHON_BIN}" -c "import xformers, xformers.ops; print('xformers ok', xformers.__version__)" \
  || (${PIP} uninstall -y xformers && echo "[setup] xformers disabled (fallback path)")

echo "[setup] installing reward deps ..."
PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/install_reward_deps.sh"

echo "[setup] verifying reward dep core imports ..."
"${PYTHON_BIN}" -c "
import xxhash, clip
from timm.data import ImageNetInfo
print('reward deps core ok', xxhash.__version__, getattr(clip, '__file__', 'ok'), ImageNetInfo.__name__)
"

echo "[setup] preloading reward model checkpoints ..."
"${PYTHON_BIN}" "${SCRIPT_DIR}/preload_reward_models.py"

echo "[setup] done"
