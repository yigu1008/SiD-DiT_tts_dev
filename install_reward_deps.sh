#!/usr/bin/env bash
set -euo pipefail

# Reward dependency installer (without touching torch/diffusers stack):
# - ImageReward
# - CLIP
# - UnifiedReward runtime deps
# - timm (PickScore-compatible)
#
# Usage:
#   ./install_reward_deps.sh
#   PYTHON_BIN=/path/to/python ./install_reward_deps.sh

PY="${PYTHON_BIN:-python3}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRAINTS="${SCRIPT_DIR}/constraints.txt"

# --constraint enforces lower bounds on ftfy/regex/xxhash throughout all installs.
# No individual package can silently downgrade them.
_pip() { "${PY}" -m pip install --no-cache-dir --constraint "${CONSTRAINTS}" "$@"; }

echo "[install] python: ${PY}"
"${PY}" -V

echo "[install] build tooling"
"${PY}" -m pip install --no-cache-dir --upgrade "setuptools>=70,<76" wheel

echo "[install] core runtime deps (ImageReward transitive deps)"
_pip --index-url "${PYPI_INDEX_URL}" \
  "xxhash>=3.4.1" \
  "ftfy>=6.2.3" \
  "regex>=2024.11.6" \
  "tqdm>=4.66.4"

echo "[install] timm==1.0.15 (PickScore-compatible)"
_pip --index-url "${PYPI_INDEX_URL}" "timm==1.0.15"

echo "[install] image-reward (PyPI), fallback to THUDM/ImageReward"
if ! _pip --index-url "${PYPI_INDEX_URL}" "image-reward==1.5"; then
  _pip --index-url "${PYPI_INDEX_URL}" "setuptools==75.8.0"
  _pip --no-build-isolation "git+https://github.com/THUDM/ImageReward.git"
fi

echo "[install] CLIP (OpenAI), fallback to clip-anytorch"
if ! _pip "git+https://github.com/openai/CLIP.git"; then
  _pip --index-url "${PYPI_INDEX_URL}" "clip-anytorch"
fi

echo "[install] open-clip-torch (needed by old hpsv2 API)"
if ! _pip --index-url "${PYPI_INDEX_URL}" "open-clip-torch"; then
  echo "[install] warning: open-clip-torch install failed; old-API hpsv2 path will be skipped."
fi

echo "[install] wandb (required by ImageReward import path)"
if ! _pip --force-reinstall "wandb"; then
  echo "[install] warning: wandb reinstall failed (likely permissions)."
  echo "[install] warning: trying user-writable overlay install for cluster ..."
  if PYTHON_BIN="${PY}" SID_OVERLAY_DIR="${SID_OVERLAY_DIR:-$HOME/.sid_pydeps}" bash "${SCRIPT_DIR}/prepare_cluster_overlay_deps.sh"; then
    echo "[install] overlay prepared. Set SID_EXTRA_PYTHONPATH to the printed path before launch."
  else
    echo "[install] warning: overlay install also failed."
  fi
  echo "[install] warning: continuing; reward_unified runtime can still use a wandb stub for ImageReward inference."
fi

echo "[install] UnifiedReward runtime deps (qwen-vl-utils, openai client)"
_pip --index-url "${PYPI_INDEX_URL}" \
  "qwen-vl-utils>=0.0.14" \
  "openai>=1.40.0"

echo "[install] optional HPS backends (hpsv3/hpsv2)"
# Install hpsv3 with --no-deps to prevent it from downgrading transformers to 4.45.2.
# hpsv3 hard-pins transformers==4.45.2 in its metadata but works fine with newer versions.
if ! _pip --no-deps --index-url "${PYPI_INDEX_URL}" "hpsv3"; then
  echo "[install] warning: hpsv3 install failed; continuing."
fi
if ! _pip --index-url "${PYPI_INDEX_URL}" \
  "omegaconf>=2.3.0" "hydra-core>=1.3.2"; then
  echo "[install] warning: omegaconf/hydra-core install failed; continuing."
fi
# hpsv2x is a drop-in replacement for hpsv2 that includes the missing BPE vocab file
# (bpe_simple_vocab_16e6.txt.gz was omitted from the official hpsv2 PyPI release).
# It still imports as `import hpsv2`. See: https://pypi.org/project/hpsv2x/
if ! _pip --index-url "${PYPI_INDEX_URL}" "hpsv2x"; then
  echo "[install] warning: hpsv2x install failed; falling back to hpsv2 (may have missing BPE file)."
  if ! _pip --index-url "${PYPI_INDEX_URL}" "hpsv2"; then
    echo "[install] warning: hpsv2 install also failed; continuing."
  fi
fi

echo "[install] restoring protobuf/wandb compatibility (hpsv2 may downgrade protobuf)"
_pip --index-url "${PYPI_INDEX_URL}" "protobuf>=4.25,<6"
if ! _pip --index-url "${PYPI_INDEX_URL}" --force-reinstall "wandb>=0.19,<0.21"; then
  echo "[install] warning: wandb reinstall failed."
fi

# Final restore — belt-and-suspenders after all installs complete.
echo "[install] final restore: transformers, timm, ftfy, regex, xxhash"
_pip --index-url "${PYPI_INDEX_URL}" \
  "transformers>=4.51.0" "tokenizers>=0.19" "timm==1.0.15" \
  "ftfy>=6.2.3" "regex>=2024.11.6" "xxhash>=3.4.1"

echo "[verify] imports"
"${PY}" - <<'PY' "${SCRIPT_DIR}"
import sys
from pathlib import Path
repo_root = Path(sys.argv[1]).resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
import timm
print("timm", timm.__version__)
from timm.data import ImageNetInfo
print("timm ImageNetInfo", ImageNetInfo.__name__)
import clip
print("clip", getattr(clip, "__file__", "ok"))
try:
    import ImageReward as RM
    print("ImageReward", getattr(RM, "__file__", "ok"))
except Exception as exc:
    print("ImageReward import warning:", exc)
import qwen_vl_utils
print("qwen_vl_utils", getattr(qwen_vl_utils, "__file__", "ok"))
import openai
print("openai", getattr(openai, "__version__", "ok"))
try:
    import hpsv3
    print("hpsv3", getattr(hpsv3, "__file__", "ok"))
except Exception as exc:
    print("hpsv3 import warning:", exc)
try:
    import omegaconf
    print("omegaconf", getattr(omegaconf, "__version__", "ok"))
except Exception as exc:
    print("omegaconf import warning:", exc)
try:
    import hydra
    print("hydra", getattr(hydra, "__version__", "ok"))
except Exception as exc:
    print("hydra import warning:", exc)
try:
    import hpsv2
    print("hpsv2", getattr(hpsv2, "__file__", "ok"))
except Exception as exc:
    print("hpsv2 import warning:", exc)
try:
    import wandb
    print("wandb", getattr(wandb, "__version__", "ok"))
except Exception as exc:
    print("wandb import warning:", exc)
import xxhash
print("xxhash", getattr(xxhash, "__version__", "ok"))
from reward_unified import UnifiedRewardScorer
print("UnifiedRewardScorer", getattr(UnifiedRewardScorer, "__name__", "ok"))
PY

echo "[done] reward dependencies installed"

# Write stamp so ensure_*_runtime functions skip the check on future runs.
# Delete ~/.cache/sid_deps/reward_deps_ok to force a re-check (e.g. after env rebuild).
_stamp="${HOME}/.cache/sid_deps/reward_deps_ok"
mkdir -p "$(dirname "${_stamp}")" && touch "${_stamp}"
echo "[done] stamp written: ${_stamp}"
