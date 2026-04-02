#!/usr/bin/env bash
set -euo pipefail

# Install fragile runtime deps into a user-writable overlay directory.
# This avoids permission issues with shared/system conda envs on cluster nodes.
#
# Usage:
#   PYTHON_BIN=/opt/conda/envs/ptca/bin/python bash prepare_cluster_overlay_deps.sh
#   SID_OVERLAY_DIR=/mnt/data/$USER/sid_pydeps bash prepare_cluster_overlay_deps.sh

PY="${PYTHON_BIN:-python3}"
OVERLAY_BASE="${SID_OVERLAY_DIR:-$HOME/.sid_pydeps}"
PY_TAG="$("${PY}" - <<'PY'
import sys
print(f"py{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
TARGET_DIR="${OVERLAY_BASE}/${PY_TAG}"

mkdir -p "${TARGET_DIR}"
echo "[overlay] python: ${PY}"
"${PY}" -V
echo "[overlay] target: ${TARGET_DIR}"

echo "[overlay] installing runtime deps to overlay"
"${PY}" -m pip install --no-cache-dir --target "${TARGET_DIR}" --upgrade --no-deps \
  "xxhash>=3.4.1" \
  "ftfy>=6.2.3" \
  "regex>=2024.11.6" \
  "pandas>=2.1.4" \
  "pyarrow>=14.0.2" \
  "datasets>=2.19.0" \
  "timm==1.0.15" \
  "wandb" \
  "protobuf>=4.25,<6" \
  "pyyaml>=6.0.1" \
  "click>=8.1.7" \
  "typing-extensions>=4.11.0" \
  "sentry-sdk>=2.0.0" \
  "gitpython>=3.1.43"

# Safety: never allow core torch stack in overlay (can shadow CUDA-enabled env torch).
for pat in "torch*" "torchvision*" "torchaudio*" "triton*" "nvidia*"; do
  find "${TARGET_DIR}" -maxdepth 1 -name "${pat}" -print -exec rm -rf {} + 2>/dev/null || true
done

echo "[overlay] verify imports using overlay"
PYTHONPATH="${TARGET_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" - <<'PY'
import xxhash
import ftfy
import regex
import pandas
import pyarrow
import datasets
import importlib.metadata as md
import wandb
from timm.data import ImageNetInfo
import torch
print("xxhash", xxhash.__version__)
print("ftfy", ftfy.__version__, md.version("ftfy"))
print("regex", regex.__version__, md.version("regex"))
print("pandas", pandas.__version__, md.version("pandas"))
print("pyarrow", pyarrow.__version__, md.version("pyarrow"))
print("datasets", datasets.__version__, md.version("datasets"))
print("wandb", wandb.__version__)
print("timm ImageNetInfo", ImageNetInfo.__name__)
print("torch", torch.__version__, "from", torch.__file__)
PY

echo
echo "[overlay] done"
echo "[overlay] export before running experiments:"
echo "  export SID_EXTRA_PYTHONPATH=\"${TARGET_DIR}\""
