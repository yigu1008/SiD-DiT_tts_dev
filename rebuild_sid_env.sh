#!/usr/bin/env bash
set -euo pipefail

# Full environment reset for sid_dit.
# This script removes the existing conda env and recreates it from scratch.
#
# Usage:
#   ./rebuild_sid_env.sh
#   ENV_NAME=sid_dit CONDA_ROOT=/home/ygu/miniconda3 ./rebuild_sid_env.sh
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 ./rebuild_sid_env.sh

ENV_NAME="${ENV_NAME:-sid_dit}"
CONDA_ROOT="${CONDA_ROOT:-/home/ygu/miniconda3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "[error] conda.sh not found at ${CONDA_ROOT}/etc/profile.d/conda.sh"
  exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

echo "[env] removing existing conda env: ${ENV_NAME}"
conda env remove -n "${ENV_NAME}" -y || true

echo "[env] creating clean conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
conda create -n "${ENV_NAME}" -y "python=${PYTHON_VERSION}" pip

echo "[env] upgrading pip tooling"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel

echo "[env] installing torch stack from ${TORCH_INDEX_URL}"
conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url "${TORCH_INDEX_URL}"

echo "[env] pinning cuDNN for CUDA 12.6"
conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir \
  --index-url "${PYPI_INDEX_URL}" \
  nvidia-cudnn-cu12==9.5.1.17

echo "[env] installing SiD-DiT/SANA dependency pins"
conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir \
  --index-url "${PYPI_INDEX_URL}" \
  accelerate==1.8.1 \
  blobfile==3.0.0 \
  click==8.2.1 \
  datasets==2.19.0 \
  diffusers==0.33.1 \
  ftfy==6.3.1 \
  huggingface-hub==0.33.0 \
  numpy==1.26.4 \
  open-clip-torch==2.32.0 \
  pillow==10.3.0 \
  requests==2.31.0 \
  safetensors==0.5.3 \
  scipy==1.13.0 \
  timm==0.9.16 \
  tokenizers==0.21.1 \
  tqdm==4.66.4 \
  transformers==4.52.4 \
  wcwidth==0.2.13 \
  protobuf==6.31.1 \
  sentencepiece==0.2.0 \
  fsspec==2024.3.1 \
  imageio==2.34.2 \
  python-dotenv==1.0.1 \
  PyWavelets==1.6.0

echo "[env] installing reward packages"
if ! conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir \
  --index-url "${PYPI_INDEX_URL}" ImageReward; then
  echo "[env] PyPI ImageReward unavailable, falling back to GitHub"
  conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir \
    "git+https://github.com/THUDM/ImageReward.git"
fi

echo "[env] pip check (informational)"
conda run -n "${ENV_NAME}" python -m pip check || true

echo "[env] version sanity check"
conda run -n "${ENV_NAME}" python - <<'PY'
import sys
import torch
import diffusers
import transformers
import accelerate
import huggingface_hub

print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu0", torch.cuda.get_device_name(0))
PY

echo "[done] rebuilt env '${ENV_NAME}'"
echo "[next] source ${CONDA_ROOT}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
