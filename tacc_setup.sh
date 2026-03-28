#!/usr/bin/env bash
# Environment setup for SSH / TACC cluster.
# Mirrors the environment.setup + early command steps from amlt/terminal.yaml.
#
# Usage:
#   source tacc_setup.sh               # set env vars + run installs
#   SKIP_INSTALL=1 source tacc_setup.sh  # env vars only, skip all pip installs
#
# Key overrides (set before sourcing):
#   SKIP_INSTALL  - set to 1 to skip all pip installs (env is already set up)
#   PYTHON_BIN    - path to python (default: auto-detect sid_dit/ptca conda env or python3)
#   CONDA_SH      - path to conda.sh
#   DATA_ROOT     - root for model caches and outputs (default: $SCRATCH or $HOME/data)

set -euo pipefail

SKIP_INSTALL="${SKIP_INSTALL:-0}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Conda
CONDA_SH="${CONDA_SH:-}"
for _candidate in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
  if [[ -z "${CONDA_SH}" && -f "${_candidate}" ]]; then
    CONDA_SH="${_candidate}"
  fi
done
if [[ -n "${CONDA_SH}" && -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
fi

# Python binary — prefer sid_dit env (TACC), then ptca (Azure), then python3
if [[ -z "${PYTHON_BIN:-}" ]]; then
  for _candidate in \
      "${HOME}/miniconda3/envs/sid_dit/bin/python" \
      "${HOME}/anaconda3/envs/sid_dit/bin/python" \
      "/opt/conda/envs/ptca/bin/python" \
      "$(command -v python3 2>/dev/null || true)"; do
    if [[ -x "${_candidate}" ]]; then
      PYTHON_BIN="${_candidate}"
      break
    fi
  done
fi
export PYTHON_BIN
PIP="${PYTHON_BIN} -m pip"

echo "[setup] PYTHON_BIN=${PYTHON_BIN}"
"${PYTHON_BIN}" -V

# Data root: prefer $WORK (TACC, large quota, not auto-purged),
# then $SCRATCH (large but purged after ~10 days), then $HOME/data.
if [[ -z "${DATA_ROOT:-}" ]]; then
  DATA_ROOT="${WORK:-${SCRATCH:-${HOME}/data}}"
fi
export DATA_ROOT

# ---------------------------------------------------------------------------
# PATH
# ---------------------------------------------------------------------------
export PATH="${HOME}/.local/bin:${PATH}"
export PATH="${PATH}:/opt/conda/envs/ptca/bin"

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

mkdir -p "${HOME}/.cache"
ln -sfn "${DATA_ROOT}/model_cache/clip" "${HOME}/.cache/clip" || true

# ---------------------------------------------------------------------------
# NCCL / distributed env
# ---------------------------------------------------------------------------
export AZFUSE_USE_FUSE="${AZFUSE_USE_FUSE:-0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"

# ---------------------------------------------------------------------------
# Pip installs (skipped when SKIP_INSTALL=1)
# ---------------------------------------------------------------------------
if [[ "${SKIP_INSTALL}" == "1" ]]; then
  echo "[setup] SKIP_INSTALL=1 — skipping all pip installs"
else
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
print('reward deps core ok', getattr(xxhash, '__version__', 'ok'), getattr(clip, '__file__', 'ok'), ImageNetInfo.__name__)
"

  echo "[setup] preloading reward model checkpoints ..."
  "${PYTHON_BIN}" "${SCRIPT_DIR}/preload_reward_models.py"
fi

echo "[setup] done"
