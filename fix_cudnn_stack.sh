#!/usr/bin/env bash
set -euo pipefail

# Rebuild a coherent CUDA/cuDNN/PyTorch stack for sid_dit-style environments.
# Usage:
#   ./fix_cudnn_stack.sh
#   PYTHON_BIN=/home/ygu/miniconda3/envs/sid_dit/bin/python ./fix_cudnn_stack.sh
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 ./fix_cudnn_stack.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

PY="${PYTHON_BIN:-python3}"
echo "[env] python: ${PY}"
"${PY}" -V

echo "[env] force no-user install mode"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export PIP_USER=0
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"

echo "[env] purge conflicting user-site packages"
USER_SITE="$("${PY}" -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
if [[ -n "${USER_SITE}" && -d "${USER_SITE}" ]]; then
  rm -rf "${USER_SITE}/torch" "${USER_SITE}/torch-"*".dist-info" "${USER_SITE}/torchgen" || true
  rm -rf "${USER_SITE}/torchvision" "${USER_SITE}/torchvision-"*".dist-info" || true
  rm -rf "${USER_SITE}/torchaudio" "${USER_SITE}/torchaudio-"*".dist-info" || true
  rm -rf "${USER_SITE}/accelerate" "${USER_SITE}/accelerate-"*".dist-info" || true
  rm -rf "${USER_SITE}/diffusers" "${USER_SITE}/diffusers-"*".dist-info" || true
  rm -rf "${USER_SITE}/transformers" "${USER_SITE}/transformers-"*".dist-info" || true
  rm -rf "${USER_SITE}/huggingface_hub" "${USER_SITE}/huggingface_hub-"*".dist-info" || true
fi

echo "[env] remove conflicting wheels first"
"${PY}" -m pip uninstall -y \
  torch torchvision torchaudio xformers \
  accelerate diffusers transformers tokenizers huggingface-hub datasets fsspec \
  safetensors numpy pillow imageio python-dotenv PyWavelets \
  nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 || true

echo "[env] purge broken torch leftovers in env site-packages"
ENV_SITE="$("${PY}" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [[ -n "${ENV_SITE}" && -d "${ENV_SITE}" ]]; then
  rm -rf "${ENV_SITE}/torch" "${ENV_SITE}/torch-"*".dist-info" "${ENV_SITE}/torchgen" || true
  rm -rf "${ENV_SITE}/torchvision" "${ENV_SITE}/torchvision-"*".dist-info" || true
  rm -rf "${ENV_SITE}/torchaudio" "${ENV_SITE}/torchaudio-"*".dist-info" || true
  rm -rf "${ENV_SITE}/accelerate" "${ENV_SITE}/accelerate-"*".dist-info" || true
  rm -rf "${ENV_SITE}/diffusers" "${ENV_SITE}/diffusers-"*".dist-info" || true
  rm -rf "${ENV_SITE}/transformers" "${ENV_SITE}/transformers-"*".dist-info" || true
  rm -rf "${ENV_SITE}/tokenizers" "${ENV_SITE}/tokenizers-"*".dist-info" || true
  rm -rf "${ENV_SITE}/huggingface_hub" "${ENV_SITE}/huggingface_hub-"*".dist-info" || true
fi

echo "[env] install torch bundle from ${TORCH_INDEX_URL} (includes matching cuDNN)"
"${PY}" -m pip install --no-cache-dir --force-reinstall \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url "${TORCH_INDEX_URL}"

echo "[env] pin cuDNN for CUDA 12.6"
"${PY}" -m pip install --no-cache-dir --force-reinstall \
  --index-url "${PYPI_INDEX_URL}" \
  nvidia-cudnn-cu12==9.5.1.17

echo "[env] install model stack pins"
"${PY}" -m pip install --no-cache-dir --force-reinstall \
  --index-url "${PYPI_INDEX_URL}" \
  accelerate==1.8.1 \
  diffusers==0.33.1 \
  transformers==4.52.4 \
  tokenizers==0.21.1 \
  huggingface-hub==0.33.0 \
  datasets==2.19.0 \
  fsspec==2024.3.1 \
  safetensors==0.5.3 \
  numpy==1.26.4 \
  pillow==10.3.0 \
  imageio==2.34.2 \
  python-dotenv==1.0.1 \
  PyWavelets==1.6.0

echo "[env] install reward package (with fallback)"
if ! "${PY}" -m pip install --no-cache-dir --force-reinstall \
  --index-url "${PYPI_INDEX_URL}" "image-reward==1.5"; then
  echo "[env] PyPI image-reward unavailable, falling back to GitHub"
  "${PY}" -m pip install --no-cache-dir --force-reinstall \
    --index-url "${PYPI_INDEX_URL}" "setuptools==75.8.0"
  "${PY}" -m pip install --no-cache-dir --force-reinstall \
    --no-build-isolation \
    "git+https://github.com/THUDM/ImageReward.git"
fi

echo "[env] install clip module required by ImageReward"
if ! "${PY}" -m pip install --no-cache-dir --force-reinstall \
  "git+https://github.com/openai/CLIP.git"; then
  echo "[env] openai/CLIP install failed, trying clip-anytorch fallback"
  "${PY}" -m pip install --no-cache-dir --force-reinstall \
    --index-url "${PYPI_INDEX_URL}" clip-anytorch
fi

echo "[env] pip check"
"${PY}" -m pip check || true

echo "[env] verify runtime versions"
"${PY}" - <<'PY'
import sys
import torch
import accelerate
import diffusers
import transformers
import huggingface_hub
import datasets
import fsspec

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch.cuda", torch.version.cuda)
print("torch.cudnn", torch.backends.cudnn.version())
print("accelerate", accelerate.__version__)
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("datasets", datasets.__version__)
print("fsspec", fsspec.__version__)
print("cuda_available", torch.cuda.is_available())
print("torch_file", getattr(torch, "__file__", "n/a"))
print("diffusers_file", getattr(diffusers, "__file__", "n/a"))
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
try:
    import ImageReward as RM
    print("ImageReward", getattr(RM, "__file__", "ok"))
except Exception as exc:
    print("ImageReward import failed:", exc)
PY

echo "[done] stack rebuilt"
