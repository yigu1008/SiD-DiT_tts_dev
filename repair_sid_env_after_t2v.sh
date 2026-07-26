#!/usr/bin/env bash
set -euo pipefail

# Targeted repair for a sid_dit environment mutated by:
#   pip install t2v-metrics==3.0
#
# This preserves the Conda environment and unrelated packages. It removes the
# legacy metric package and restores only the generation/runtime pins that its
# resolver is known to replace.
#
# Usage:
#   CONDA_ROOT=/home/ygu/miniconda3 ENV_NAME=sid_dit \
#     bash repair_sid_env_after_t2v.sh

CONDA_ROOT="${CONDA_ROOT:-/home/ygu/miniconda3}"
ENV_NAME="${ENV_NAME:-sid_dit}"
PREFIX="${SID_ENV_PREFIX:-${CONDA_ROOT}/envs/${ENV_NAME}}"
PY="${PYTHON_BIN:-${PREFIX}/bin/python}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
AUDIT_DIR="${SID_REPAIR_AUDIT_DIR:-${HOME}/.cache/sid_env_repair}"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [[ ! -x "${PY}" ]]; then
  echo "Error: target Python not found: ${PY}" >&2
  exit 1
fi

mkdir -p "${AUDIT_DIR}"
echo "[repair] target: ${PY}"
"${PY}" -m pip freeze > "${AUDIT_DIR}/before_${ENV_NAME}_${STAMP}.txt"

echo "[repair] removing legacy VQAScore package from the generation env"
"${PY}" -m pip uninstall -y t2v-metrics || true

echo "[repair] restoring PyTorch/CUDA 12.6 stack"
"${PY}" -m pip install --no-cache-dir --force-reinstall \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url "${TORCH_INDEX_URL}"
"${PY}" -m pip install --no-cache-dir --force-reinstall --no-deps \
  xformers==0.0.31.post1 \
  --index-url "${TORCH_INDEX_URL}"
"${PY}" -m pip install --no-cache-dir --force-reinstall --no-deps \
  nvidia-cudnn-cu12==9.5.1.17 \
  --index-url "${PYPI_INDEX_URL}"

echo "[repair] restoring SiD/SenseFlow model stack"
"${PY}" -m pip install --no-cache-dir --force-reinstall \
  --index-url "${PYPI_INDEX_URL}" \
  accelerate==1.8.1 \
  diffusers==0.33.1 \
  huggingface-hub==0.33.0 \
  peft==0.18.1 \
  pillow==10.3.0 \
  protobuf==6.31.1 \
  qwen-vl-utils==0.0.14 \
  safetensors==0.5.3 \
  timm==1.0.15 \
  tokenizers==0.21.1 \
  transformers==4.52.4 \
  wandb==0.20.1

# Resolver passes for optional packages can revisit protobuf/tokenizers.
# Reassert the critical ABI/API pins one final time without dependencies.
"${PY}" -m pip install --no-cache-dir --force-reinstall --no-deps \
  --index-url "${PYPI_INDEX_URL}" \
  protobuf==6.31.1 \
  tokenizers==0.21.1 \
  transformers==4.52.4 \
  timm==1.0.15

echo "[repair] verifying exact versions and imports"
"${PY}" - <<'PY'
import importlib.metadata as md

expected = {
    "torch": "2.7.1",
    "torchvision": "0.22.1",
    "torchaudio": "2.7.1",
    "xformers": "0.0.31.post1",
    "accelerate": "1.8.1",
    "diffusers": "0.33.1",
    "huggingface-hub": "0.33.0",
    "peft": "0.18.1",
    "Pillow": "10.3.0",
    "protobuf": "6.31.1",
    "safetensors": "0.5.3",
    "timm": "1.0.15",
    "tokenizers": "0.21.1",
    "transformers": "4.52.4",
    "wandb": "0.20.1",
}
errors = []
for package, wanted in expected.items():
    try:
        found = md.version(package)
    except md.PackageNotFoundError:
        found = "<missing>"
    comparable = found.split("+", 1)[0]
    if comparable != wanted:
        errors.append(f"{package}={found}, expected {wanted}")
if errors:
    raise SystemExit("Version verification failed:\n  " + "\n  ".join(errors))

try:
    md.version("t2v-metrics")
except md.PackageNotFoundError:
    pass
else:
    raise SystemExit("t2v-metrics is still installed in the generation environment")

import torch
import torchvision
import xformers
import xformers.ops
import accelerate
import diffusers
import peft
import timm
import transformers
from google.protobuf import runtime_version

print(
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"cudnn={torch.backends.cudnn.version()} cuda_available={torch.cuda.is_available()}"
)
print(
    f"transformers={transformers.__version__} diffusers={diffusers.__version__} "
    f"accelerate={accelerate.__version__} peft={peft.__version__}"
)
print(f"torchvision={torchvision.__version__} xformers={xformers.__version__} timm={timm.__version__}")
print("protobuf runtime_version import OK")
PY

"${PY}" -m pip freeze > "${AUDIT_DIR}/after_${ENV_NAME}_${STAMP}.txt"
echo "[repair] before/after snapshots: ${AUDIT_DIR}"
echo "[repair] complete: ${PREFIX}"
