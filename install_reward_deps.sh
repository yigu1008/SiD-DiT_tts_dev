#!/usr/bin/env bash
set -euo pipefail

# Minimal reward dependency installer:
# - ImageReward
# - CLIP
# - timm (pinned)
#
# Usage:
#   ./install_reward_deps.sh
#   PYTHON_BIN=/path/to/python ./install_reward_deps.sh

PY="${PYTHON_BIN:-python3}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"

echo "[install] python: ${PY}"
"${PY}" -V

echo "[install] build tooling"
"${PY}" -m pip install --no-cache-dir --upgrade "setuptools>=70,<76" wheel

echo "[install] timm==0.9.16"
"${PY}" -m pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" "timm==0.9.16"

echo "[install] image-reward (PyPI), fallback to THUDM/ImageReward"
if ! "${PY}" -m pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" "image-reward==1.5"; then
  "${PY}" -m pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" "setuptools==75.8.0"
  "${PY}" -m pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/THUDM/ImageReward.git"
fi

echo "[install] CLIP (OpenAI), fallback to clip-anytorch"
if ! "${PY}" -m pip install --no-cache-dir "git+https://github.com/openai/CLIP.git"; then
  "${PY}" -m pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" "clip-anytorch"
fi

echo "[verify] imports"
"${PY}" - <<'PY'
import timm
print("timm", timm.__version__)
import clip
print("clip", getattr(clip, "__file__", "ok"))
import ImageReward as RM
print("ImageReward", getattr(RM, "__file__", "ok"))
PY

echo "[done] reward dependencies installed"
