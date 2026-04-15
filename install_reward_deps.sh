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
STAMP_VERSION="${STAMP_VERSION:-v2}"
_stamp="${HOME}/.cache/sid_deps/reward_deps_ok_${STAMP_VERSION}"

_quick_verify() {
  "${PY}" - <<'PY' >/dev/null 2>&1
import importlib.metadata as md
import xxhash
import ftfy
import regex
import pandas
import pyarrow
import datasets
import timm
import clip
from timm.data import ImageNetInfo
for pkg in ("xxhash", "ftfy", "regex", "pandas", "pyarrow", "datasets", "timm"):
    md.version(pkg)
print(xxhash.__version__, ftfy.__version__, regex.__version__, ImageNetInfo.__name__)
PY
}

# --constraint enforces lower bounds on ftfy/regex/xxhash throughout all installs.
# No individual package can silently downgrade them.
_pip() { "${PY}" -m pip install --no-cache-dir --constraint "${CONSTRAINTS}" "$@"; }

echo "[install] python: ${PY}"
"${PY}" -V

if [[ "${FORCE_INSTALL_DEPS:-0}" != "1" ]] && [[ -f "${_stamp}" ]]; then
  if _quick_verify; then
    echo "[install] stamp + quick verify passed; skipping reinstall (${_stamp})"
    exit 0
  fi
  echo "[install] stale deps stamp detected; reinstalling runtime deps."
fi

echo "[install] build tooling"
"${PY}" -m pip install --no-cache-dir --upgrade "setuptools>=70,<76" wheel

echo "[install] core runtime deps (ImageReward transitive deps)"
_pip --index-url "${PYPI_INDEX_URL}" \
  "xxhash>=3.4.1" \
  "ftfy>=6.2.3" \
  "regex>=2024.11.6" \
  "tqdm>=4.66.4"

echo "[install] core data runtime deps (datasets stack)"
_pip --index-url "${PYPI_INDEX_URL}" \
  "pandas>=2.1.4" \
  "pyarrow>=14.0.2" \
  "datasets>=2.19.0"

echo "[install] timm==1.0.15 (PickScore-compatible)"
_pip --index-url "${PYPI_INDEX_URL}" "timm==1.0.15"

echo "[install] image-reward (PyPI), fallback to THUDM/ImageReward"
# --no-deps: image-reward pins old transformers which would break torchvision.
# Runtime deps (CLIP, torch, transformers, etc.) are already installed.
if ! _pip --no-deps --index-url "${PYPI_INDEX_URL}" "image-reward==1.5"; then
  if ! _pip --no-deps --no-build-isolation "git+https://github.com/THUDM/ImageReward.git"; then
    echo "[install] warning: image-reward install failed (PyPI and git); ImageReward backend unavailable."
  fi
fi

echo "[install] CLIP (OpenAI), fallback to clip-anytorch"
if ! _pip "git+https://github.com/openai/CLIP.git"; then
  if ! _pip --index-url "${PYPI_INDEX_URL}" "clip-anytorch"; then
    echo "[install] warning: CLIP install failed (git and clip-anytorch); CLIP-dependent backends unavailable."
  fi
fi

echo "[install] open-clip-torch (needed by old hpsv2 API)"
if ! _pip --index-url "${PYPI_INDEX_URL}" "open-clip-torch"; then
  echo "[install] warning: open-clip-torch install failed; old-API hpsv2 path will be skipped."
fi

echo "[install] wandb (required by ImageReward import path)"
if ! _pip "wandb"; then
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
if ! _pip --index-url "${PYPI_INDEX_URL}" \
  "qwen-vl-utils>=0.0.14" \
  "openai>=1.40.0"; then
  echo "[install] warning: qwen-vl-utils/openai install failed; UnifiedReward backend unavailable."
fi

echo "[install] optional HPS backends (hpsv3/hpsv2)"
# Install hpsv3 with --no-deps to prevent it from downgrading transformers to 4.45.2.
# hpsv3 hard-pins transformers==4.45.2 in its metadata but works fine with newer versions.
if ! _pip --no-deps --index-url "${PYPI_INDEX_URL}" "hpsv3"; then
  echo "[install] warning: hpsv3 install failed; continuing."
fi
# Optional alternative HPSv3 path via imscore.
# Keep --no-deps to avoid unexpected resolver changes in the main env.
if ! _pip --no-deps --index-url "${PYPI_INDEX_URL}" "imscore"; then
  echo "[install] warning: imscore install failed; hpsv3-imscore fallback unavailable."
fi
# hpsv3 runtime deps (installed separately since hpsv3 uses --no-deps).
# Declared deps missing due to --no-deps:
#   fire (arg parser), omegaconf/hydra-core (config), peft/trl (model loading),
#   einops (tensor ops), opencv-python (image I/O), deepspeed (module-level import),
#   qwen-vl-utils (VL model), safetensors (weight loading), accelerate (device map)
# Undeclared imports found in hpsv3 source:
#   matplotlib (visualization), prettytable (formatting), pandas (data handling),
#   pydantic (validation), requests (HTTP), tensorboard (torch.utils.tensorboard),
#   packaging (version checks)
if ! _pip --index-url "${PYPI_INDEX_URL}" \
  "fire" "omegaconf>=2.3.0" "hydra-core>=1.3.2" \
  "peft>=0.8.0" "trl>=0.7.0" "einops>=0.6.0" \
  "opencv-python>=4.5.0" "deepspeed>=0.12.0" \
  "safetensors>=0.3.0" "accelerate>=0.20.0" \
  "qwen-vl-utils>=0.0.8" \
  "matplotlib" "prettytable" "pandas" "pydantic" "requests" \
  "tensorboard" "packaging"; then
  echo "[install] warning: hpsv3 runtime deps partially failed; continuing."
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
# hpsv2 depends on clint (for clint.textui.progress) but doesn't always pull it in
_pip --index-url "${PYPI_INDEX_URL}" "clint" || echo "[install] warning: clint install failed; hpsv2 may be broken."

echo "[install] restoring protobuf/wandb compatibility (hpsv2 may downgrade protobuf)"
# Use >=4.25 only — no upper bound. requirements.txt pins protobuf==6.31.1, so
# capping at <6 would cause pip to conflict with any base-image package that
# requires protobuf>=6, killing the script via set -euo pipefail before the
# final restore runs (and thus before the stamp is written).
if ! _pip --index-url "${PYPI_INDEX_URL}" "protobuf>=4.25"; then
  echo "[install] warning: protobuf restore failed; continuing."
fi
if ! _pip --index-url "${PYPI_INDEX_URL}" "wandb>=0.19,<0.21"; then
  echo "[install] warning: wandb reinstall failed."
fi

# Final restore — belt-and-suspenders after all installs complete.
echo "[install] final restore: transformers/timm + text/hash + data stack"
_pip --index-url "${PYPI_INDEX_URL}" \
  "transformers==4.52.4" "tokenizers==0.21.1" "qwen-vl-utils==0.0.14" "timm==1.0.15" \
  "ftfy>=6.2.3" "regex>=2024.11.6" "xxhash>=3.4.1" \
  "pandas>=2.1.4" "pyarrow>=14.0.2" "datasets>=2.19.0"
_pip --index-url "${PYPI_INDEX_URL}" --no-deps \
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
try:
    import clip
    print("clip", getattr(clip, "__file__", "ok"))
except Exception as exc:
    print("clip import warning:", exc)
try:
    import ImageReward as RM
    print("ImageReward", getattr(RM, "__file__", "ok"))
except Exception as exc:
    print("ImageReward import warning:", exc)
try:
    import qwen_vl_utils
    print("qwen_vl_utils", getattr(qwen_vl_utils, "__file__", "ok"))
except Exception as exc:
    print("qwen_vl_utils import warning:", exc)
try:
    import openai
    print("openai", getattr(openai, "__version__", "ok"))
except Exception as exc:
    print("openai import warning:", exc)
try:
    import hpsv3
    print("hpsv3", getattr(hpsv3, "__file__", "ok"))
except Exception as exc:
    print("hpsv3 import warning:", exc)
try:
    import imscore
    print("imscore", getattr(imscore, "__file__", "ok"))
except Exception as exc:
    print("imscore import warning:", exc)
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
import pandas
import pyarrow
import datasets
print("pandas", getattr(pandas, "__version__", "ok"))
print("pyarrow", getattr(pyarrow, "__version__", "ok"))
print("datasets", getattr(datasets, "__version__", "ok"))
import importlib.metadata as md
print("metadata ftfy/regex/xxhash", md.version("ftfy"), md.version("regex"), md.version("xxhash"))
print("metadata pandas/pyarrow/datasets", md.version("pandas"), md.version("pyarrow"), md.version("datasets"))
from reward_unified import UnifiedRewardScorer
print("UnifiedRewardScorer", getattr(UnifiedRewardScorer, "__name__", "ok"))
PY

echo "[verify] pip check (non-fatal)"
"${PY}" -m pip check || true

echo "[done] reward dependencies installed"

# Write stamp so ensure_*_runtime functions skip the check on future runs.
# Delete this file to force a re-check (e.g. after env rebuild):
#   ~/.cache/sid_deps/reward_deps_ok_<STAMP_VERSION>
mkdir -p "$(dirname "${_stamp}")" && touch "${_stamp}"
echo "[done] stamp written: ${_stamp}"
