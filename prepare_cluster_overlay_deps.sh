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
"${PY}" -m pip install --no-cache-dir --target "${TARGET_DIR}" --upgrade \
  "xxhash>=3.4.1" \
  "wandb" \
  "protobuf<7" \
  "pyyaml>=6.0.1" \
  "click>=8.1.7" \
  "typing-extensions>=4.11.0" \
  "sentry-sdk>=2.0.0" \
  "gitpython>=3.1.43"

echo "[overlay] verify imports using overlay"
PYTHONPATH="${TARGET_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" - <<'PY'
import xxhash
import wandb
print("xxhash", xxhash.__version__)
print("wandb", wandb.__version__)
PY

echo
echo "[overlay] done"
echo "[overlay] export before running experiments:"
echo "  export SID_EXTRA_PYTHONPATH=\"${TARGET_DIR}\""
