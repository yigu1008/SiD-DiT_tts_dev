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

# The 3.0 wheel eagerly imports every optional VQA/CLIP/ITM backend.
# Restrict this reward-only environment to the requested CLIP-FlanT5 family,
# avoiding irrelevant LLaVA/InternVideo/FlashAttention dependencies.
echo "[vqa-env] applying CLIP-FlanT5-only import registry"
"${PY}" "${SCRIPT_DIR}/tools/patch_t2v_metrics_clip_flant5_only.py"
"${PY}" "${SCRIPT_DIR}/tools/patch_t2v_metrics_clip_flant5_only.py" --check

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
models = set(t2v_metrics.list_all_models())
expected = {
    "clip-flant5-xxl",
    "clip-flant5-xl",
    "clip-flant5-xxl-no-system",
    "clip-flant5-xxl-no-system-no-user",
}
assert models == expected, f"unexpected VQAScore registry: {sorted(models)}"
allowed_vqa_prefixes = (
    "t2v_metrics.models.vqascore_models.clip_t5",
    "t2v_metrics.models.vqascore_models.mm_utils",
    "t2v_metrics.models.vqascore_models.vqa_model",
)
unexpected_t2v_modules = sorted(
    name
    for name in sys.modules
    if (
        name.startswith("t2v_metrics.clipscore")
        or name.startswith("t2v_metrics.itmscore")
        or name.startswith("t2v_metrics.models.clipscore_models")
        or name.startswith("t2v_metrics.models.itmscore_models")
        or (
            name.startswith("t2v_metrics.models.vqascore_models.")
            and not name.startswith(allowed_vqa_prefixes)
        )
    )
)
assert not unexpected_t2v_modules, (
    "unrelated t2v-metrics backends imported during VQAScore preflight: "
    f"{unexpected_t2v_modules}"
)
print(f"t2v-metrics={version}")
print(f"ffmpeg={shutil.which('ffmpeg')}")
print(f"VQAScore registry OK: {sorted(models)}")
print("ImageReward import OK")
PY

if [[ "${VQA_LOAD_SMOKE_TEST:-0}" == "1" ]]; then
  echo "[vqa-env] loading ${VQASCORE_MODEL:-clip-flant5-xxl} and scoring one image"
  PATH="${PREFIX}/bin:${PATH}" "${PY}" \
    "${SCRIPT_DIR}/tools/smoke_test_vqascore_reward.py" \
    --model "${VQASCORE_MODEL:-clip-flant5-xxl}" \
    --device "${VQA_SMOKE_DEVICE:-cuda:0}"
else
  echo "[vqa-env] model-load smoke test skipped (set VQA_LOAD_SMOKE_TEST=1 to run it)"
fi

echo "[vqa-env] ready: ${PY}"
