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

echo "[env] remove conflicting wheels first"
"${PY}" -m pip uninstall -y \
  torch torchvision torchaudio xformers \
  nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 || true

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
  pillow==10.3.0

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
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

echo "[done] stack rebuilt"
