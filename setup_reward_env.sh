#!/usr/bin/env bash
set -euo pipefail

# Create a separate conda env for reward scoring (hpsv3 + ImageReward).
# This env uses transformers==4.45.2 which is incompatible with the main
# generation env (needs transformers>=4.51 for Qwen3/diffusers).
#
# Usage:
#   bash setup_reward_env.sh
#   # or with a custom base:
#   CONDA_BASE=/opt/conda bash setup_reward_env.sh

CONDA_BASE="${CONDA_BASE:-/opt/conda}"
ENV_NAME="${REWARD_ENV_NAME:-reward}"
PYTHON_VERSION="${REWARD_PYTHON_VERSION:-3.10}"
CUDA_VERSION="${REWARD_CUDA_VERSION:-cu126}"

CONDA="${CONDA_BASE}/bin/conda"

# Resolve env prefix. Prefer ${CONDA_BASE}/envs/${ENV_NAME} when writable,
# otherwise fall back to a writable location under $HOME. This lets the
# script run as a non-root user on shared images where /opt/conda/envs is
# owned by root (conda would otherwise silently drop the env into
# $HOME/.conda/envs, leaving the hardcoded paths below broken).
if [ -n "${REWARD_ENV_PREFIX:-}" ]; then
    PREFIX="${REWARD_ENV_PREFIX}"
elif [ -w "${CONDA_BASE}/envs" ]; then
    PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"
else
    PREFIX="${HOME}/.conda/envs/${ENV_NAME}"
fi

PY="${PREFIX}/bin/python"
PIP="${PREFIX}/bin/pip"

echo "[setup_reward_env] env prefix: ${PREFIX}"

# Export for downstream steps: the caller can `source` a sentinel file
# to pick up the resolved prefix without hardcoding the path.
SENTINEL_DIR="${REWARD_ENV_SENTINEL_DIR:-/tmp}"
mkdir -p "${SENTINEL_DIR}" || true
echo "${PREFIX}" > "${SENTINEL_DIR}/reward_env_prefix_${ENV_NAME}"

echo "[setup_reward_env] Creating conda env '${ENV_NAME}' (python=${PYTHON_VERSION}) ..."

# Create env at explicit prefix if missing
if [ ! -x "${PY}" ]; then
    "${CONDA}" create -y -p "${PREFIX}" python="${PYTHON_VERSION}" pip
else
    echo "[setup_reward_env] Env already exists at ${PREFIX}, updating ..."
fi

# Install setuptools + wheel before anything else: ImageReward's setup.py
# does `from pkg_resources import ...`. setuptools>=80 made pkg_resources
# optional / moved it out of the default install, breaking ImageReward's
# build, so pin to <80. We install with --no-build-isolation so the env's
# own setuptools must be the one that's used.
echo "[setup_reward_env] Installing build tooling (setuptools<80, wheel, pip) ..."
"${PIP}" install --no-cache-dir --upgrade "setuptools<80" "wheel" "pip"

echo "[setup_reward_env] Installing PyTorch ..."
"${PIP}" install --no-cache-dir \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"

echo "[setup_reward_env] Installing transformers stack (4.45.2) ..."
"${PIP}" install --no-cache-dir \
    "transformers==4.45.2" \
    "tokenizers==0.20.3" \
    "trl==0.12.2" \
    "huggingface-hub==0.36.2" \
    "safetensors>=0.3.0" \
    "accelerate>=0.20.0"

echo "[setup_reward_env] Installing hpsv3 (no-deps to avoid transformers pin) ..."
"${PIP}" install --no-cache-dir --no-deps hpsv3

echo "[setup_reward_env] Installing hpsv3 runtime deps ..."
"${PIP}" install --no-cache-dir \
    "fire" "omegaconf>=2.3.0" "hydra-core>=1.3.2" \
    "peft>=0.8.0" "einops>=0.6.0" \
    "opencv-python>=4.5.0" "deepspeed>=0.12.0" \
    "qwen-vl-utils>=0.0.8" \
    "matplotlib" "prettytable" "pandas" "pydantic" "requests" \
    "tensorboard" "packaging"

echo "[setup_reward_env] Installing ImageReward ..."
# Use --no-deps to avoid clobbering the transformers==4.45.2 / trl==0.12.2 pin
# above. Use --no-build-isolation so the git install reuses the existing
# setuptools/wheel rather than fetching a fresh build env (which has been
# observed to fail behind locked-down package mirrors).
install_imagereward_from_clone() {
    local sp_dir
    sp_dir="$(${PY} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    local tmp_dir
    tmp_dir="$(mktemp -d)"
    git clone --depth=1 https://github.com/THUDM/ImageReward.git "${tmp_dir}/ImageReward" || return 1
    if [ -d "${tmp_dir}/ImageReward/ImageReward" ]; then
        cp -r "${tmp_dir}/ImageReward/ImageReward" "${sp_dir}/" || return 1
        echo "[setup_reward_env] ImageReward copied into ${sp_dir}/ImageReward"
        rm -rf "${tmp_dir}"
        return 0
    fi
    rm -rf "${tmp_dir}"
    return 1
}

"${PIP}" install --no-cache-dir --no-deps --no-build-isolation "git+https://github.com/THUDM/ImageReward.git" || \
    "${PIP}" install --no-cache-dir --no-deps "git+https://github.com/THUDM/ImageReward.git" || \
    "${PIP}" install --no-cache-dir --no-deps "image-reward==1.5" || \
    install_imagereward_from_clone || \
    echo "[setup_reward_env] WARNING: ImageReward install failed"

# ImageReward's BLIP backbone imports `fairscale.nn.checkpoint.checkpoint_activations`
# at module load. Without it, `import ImageReward` raises ModuleNotFoundError
# even though the package is technically installed. ImageReward upstream pins
# fairscale==0.4.4 so we follow that.
echo "[setup_reward_env] Installing fairscale (ImageReward runtime dep) ..."
"${PIP}" install --no-cache-dir "fairscale==0.4.4" || \
    "${PIP}" install --no-cache-dir "fairscale" || \
    echo "[setup_reward_env] WARNING: fairscale install failed; ImageReward will not import"

echo "[setup_reward_env] Installing CLIP ..."
"${PIP}" install --no-cache-dir "git+https://github.com/openai/CLIP.git" || \
    "${PIP}" install --no-cache-dir "clip-anytorch" || \
    echo "[setup_reward_env] WARNING: CLIP install failed"

echo "[setup_reward_env] Installing remaining deps ..."
"${PIP}" install --no-cache-dir \
    "xxhash>=3.4.1" "ftfy>=6.2.3" "regex>=2024.11.6" "tqdm>=4.66.4" \
    "open-clip-torch" "hpsv2" "clint" \
    "pillow>=10.0.0" "numpy>=1.24.0" "scipy>=1.10.0"

echo "[setup_reward_env] Verifying ..."
"${PY}" -c "
import torch
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')
import transformers
print(f'transformers={transformers.__version__}')
import trl
print(f'trl={trl.__version__}')
import huggingface_hub
print(f'huggingface_hub={huggingface_hub.__version__}')
try:
    import hpsv3; print('hpsv3 OK')
except Exception as e: print(f'hpsv3 FAILED: {e}')
try:
    import ImageReward; print('ImageReward OK')
except Exception as e: print(f'ImageReward FAILED: {e}')
try:
    import hpsv2; print('hpsv2 OK')
except Exception as e: print(f'hpsv2 FAILED: {e}')
try:
    import torchvision; print(f'torchvision={torchvision.__version__}')
except Exception as e: print(f'torchvision FAILED: {e}')
"

echo "[setup_reward_env] Done. Reward env python: ${PY}"
echo "[setup_reward_env] To start the reward server:"
echo "  ${PY} reward_server.py --port 5100 --device cuda:0 --backends hpsv3 imagereward"
