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

echo "[overlay] installing standalone deps to overlay (--no-deps)"
"${PY}" -m pip install --no-cache-dir --target "${TARGET_DIR}" --upgrade --no-deps \
  "xxhash>=3.4.1" \
  "ftfy>=6.2.3" \
  "regex>=2024.11.6" \
  "timm==1.0.15" \
  "protobuf>=4.25" \
  "pyyaml>=6.0.1" \
  "click>=8.1.7" \
  "typing-extensions>=4.11.0" \
  "sentry-sdk>=2.0.0" \
  "gitpython>=3.1.43"

# datasets, pandas, pyarrow, httpx, wandb have real transitive dep trees;
# install with deps so httpx->httpcore->h11, anyio, certifi etc. are all present.
# None of these pull in torch, so the overlay stays lightweight.
echo "[overlay] installing deps with transitive resolution (datasets, httpx, wandb)"
"${PY}" -m pip install --no-cache-dir --target "${TARGET_DIR}" --upgrade \
  "pandas>=2.1.4" \
  "pyarrow>=14.0.2" \
  "datasets>=2.19.0" \
  "httpx>=0.23.0" \
  "wandb"

# Safety: never allow core torch stack in overlay (can shadow CUDA-enabled env torch).
for pat in "torch*" "torchvision*" "torchaudio*" "triton*" "nvidia*"; do
  find "${TARGET_DIR}" -maxdepth 1 -name "${pat}" -print -exec rm -rf {} + 2>/dev/null || true
done

echo "[overlay] verify imports using overlay"
PYTHONPATH="${TARGET_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" - <<'PY'
import importlib.metadata as md
import xxhash, ftfy, regex
print("xxhash", xxhash.__version__, md.version("xxhash"))
print("ftfy", ftfy.__version__, md.version("ftfy"))
print("regex", regex.__version__, md.version("regex"))
try:
    import pandas, pyarrow, datasets
    print("pandas", pandas.__version__, md.version("pandas"))
    print("pyarrow", pyarrow.__version__, md.version("pyarrow"))
    print("datasets", datasets.__version__, md.version("datasets"))
except Exception as e:
    print("WARNING: datasets/pandas/pyarrow import failed:", e)
try:
    import httpx
    print("httpx", httpx.__version__)
except Exception as e:
    print("WARNING: httpx import failed:", e)
try:
    import wandb
    print("wandb", wandb.__version__)
except Exception as e:
    print("WARNING: wandb import failed:", e)
from timm.data import ImageNetInfo
print("timm ImageNetInfo", ImageNetInfo.__name__)
import torch
print("torch", torch.__version__, "from", torch.__file__)
PY

echo
echo "[overlay] done"
echo "[overlay] export before running experiments:"
echo "  export SID_EXTRA_PYTHONPATH=\"${TARGET_DIR}\""
