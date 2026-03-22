#!/usr/bin/env bash
set -euo pipefail

# Reward dependency installer (without touching torch/diffusers stack):
# - ImageReward
# - CLIP
# - UnifiedReward runtime deps
# - timm (pinned for ImageReward compatibility)
#
# Usage:
#   ./install_reward_deps.sh
#   PYTHON_BIN=/path/to/python ./install_reward_deps.sh

PY="${PYTHON_BIN:-python3}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

echo "[install] UnifiedReward runtime deps (qwen-vl-utils, openai client)"
"${PY}" -m pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" \
  "qwen-vl-utils>=0.0.14" \
  "openai>=1.40.0"

echo "[verify] imports"
"${PY}" - <<'PY' "${SCRIPT_DIR}"
import sys
from pathlib import Path
repo_root = Path(sys.argv[1]).resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
import timm
print("timm", timm.__version__)
import clip
print("clip", getattr(clip, "__file__", "ok"))
import ImageReward as RM
print("ImageReward", getattr(RM, "__file__", "ok"))
import qwen_vl_utils
print("qwen_vl_utils", getattr(qwen_vl_utils, "__file__", "ok"))
import openai
print("openai", getattr(openai, "__version__", "ok"))
from reward_unified import UnifiedRewardScorer
print("UnifiedRewardScorer", getattr(UnifiedRewardScorer, "__name__", "ok"))
PY

echo "[done] reward dependencies installed"
