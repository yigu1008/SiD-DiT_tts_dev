#!/usr/bin/env bash
set -euo pipefail

# Build an isolated reward-server environment for legacy CLIP-FlanT5
# VQAScore plus ImageReward.  Never install t2v-metrics==3.0 into sid_dit:
# its pinned torch/transformers stack conflicts with the generation runtime.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-${REWARD_ENV_CONDA_BASE:-/home/ygu/miniconda3}}"
ENV_NAME="${VQA_REWARD_ENV_NAME:-vqascore_reward}"
PREFIX="${VQA_REWARD_ENV_PREFIX:-${CONDA_BASE}/envs/${ENV_NAME}}"
CONDA="${CONDA_BASE}/bin/conda"
PY="${PREFIX}/bin/python"
PIP="${PREFIX}/bin/pip"

if [[ ! -x "${CONDA}" ]]; then
  echo "Error: conda not found: ${CONDA}" >&2
  exit 1
fi

if [[ ! -x "${PY}" ]]; then
  echo "[vqa-env] creating ${PREFIX}"
  "${CONDA}" create -y -p "${PREFIX}" -c conda-forge \
    python=3.10 pip ffmpeg=6.1.2
else
  echo "[vqa-env] reusing ${PREFIX}"
  if [[ ! -x "${PREFIX}/bin/ffmpeg" ]]; then
    "${CONDA}" install -y -p "${PREFIX}" -c conda-forge ffmpeg=6.1.2
  fi
fi

"${PIP}" install --no-cache-dir --upgrade "pip<26" "setuptools<80" wheel

echo "[vqa-env] installing the legacy GenAI-Bench VQAScore runtime"
"${PIP}" install --no-cache-dir "t2v-metrics==3.0"

# The legacy PyPI wheel eagerly imports all VQAScore model modules but does
# not declare its Git-based LLaVA/PyTorchVideo packages.  They are required
# even when the selected model is CLIP-FlanT5.  These are the upstream v3.0
# installation commands.
echo "[vqa-env] installing undeclared legacy VQAScore backends"
"${PIP}" install --no-cache-dir \
  "git+https://github.com/LLaVA-VL/LLaVA-NeXT.git"
"${PIP}" install --no-cache-dir \
  "git+https://github.com/linzhiqiu/pytorchvideo.git"

echo "[vqa-env] installing ImageReward without changing VQAScore's pins"
"${PIP}" install --no-cache-dir --no-deps "image-reward==1.5" || \
  "${PIP}" install --no-cache-dir --no-deps --no-build-isolation \
    "git+https://github.com/THUDM/ImageReward.git"
"${PIP}" install --no-cache-dir "fairscale==0.4.4" ftfy regex tqdm
"${PIP}" install --no-cache-dir "git+https://github.com/openai/CLIP.git" || \
  "${PIP}" install --no-cache-dir clip-anytorch

# ImageReward/__init__.py imports its training-only ReFL module by default.
# The reward server only needs inference, so avoid pulling a second diffusers
# stack into this deliberately isolated environment.
IR_INIT="$("${PY}" -c 'import importlib.util; s=importlib.util.find_spec("ImageReward"); print(s.origin if s else "")' 2>/dev/null || true)"
if [[ -n "${IR_INIT}" && -f "${IR_INIT}" ]] && ! grep -q '^# (patched-no-ReFL)' "${IR_INIT}"; then
  sed -i 's|^from \.ReFL import|# (patched-no-ReFL) from .ReFL import|' "${IR_INIT}"
fi

echo "[vqa-env] verifying imports and ffmpeg"
PATH="${PREFIX}/bin:${PATH}" "${PY}" - "${SCRIPT_DIR}" <<'PY'
import importlib.metadata as md
import shutil
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
import reward_server

reward_server._inject_wandb_stub()
import ImageReward  # noqa: F401,E402
import t2v_metrics  # noqa: F401,E402

assert shutil.which("ffmpeg"), "ffmpeg is not visible"
version = md.version("t2v-metrics")
assert version.split(".", 1)[0] == "3", version
print(f"t2v-metrics={version}")
print(f"ffmpeg={shutil.which('ffmpeg')}")
print("ImageReward import OK")
PY

echo "[vqa-env] ready: ${PY}"
