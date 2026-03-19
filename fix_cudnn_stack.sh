#!/usr/bin/env bash
set -euo pipefail

# Rebuild a coherent CUDA/cuDNN/PyTorch stack for sid_dit-style environments.
# Usage:
#   ./fix_cudnn_stack.sh
#   PYTHON_BIN=/home/ygu/miniconda3/envs/sid_dit/bin/python ./fix_cudnn_stack.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

PY="${PYTHON_BIN:-python3}"
echo "[env] python: ${PY}"
"${PY}" -V

echo "[env] force no-user install mode"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export PIP_USER=0

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
  nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 || true

echo "[env] purge broken torch leftovers in env site-packages"
ENV_SITE="$("${PY}" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [[ -n "${ENV_SITE}" && -d "${ENV_SITE}" ]]; then
  rm -rf "${ENV_SITE}/torch" "${ENV_SITE}/torch-"*".dist-info" "${ENV_SITE}/torchgen" || true
  rm -rf "${ENV_SITE}/torchvision" "${ENV_SITE}/torchvision-"*".dist-info" || true
  rm -rf "${ENV_SITE}/torchaudio" "${ENV_SITE}/torchaudio-"*".dist-info" || true
fi

echo "[env] install torch/cu126 bundle (includes matching cuDNN)"
"${PY}" -m pip install --no-cache-dir \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126

echo "[env] install model stack pins"
"${PY}" -m pip install --no-cache-dir \
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
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

echo "[done] stack rebuilt"
