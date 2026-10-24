#!/usr/bin/env bash
# Idempotent: build $SCRATCH/envs/sid_dit (sampler + Qwen2.5-VL server env; pins = amlt/jev_decider_sd35base.yaml).
# Run on a COMPUTE node (login nodes cap vmem at 8 GB). Caches/tmp kept off $HOME (home quota is full).
set -euo pipefail; S=/scratch/09398/yigu; E=$S/envs/sid_dit
export TMPDIR=/tmp PIP_CACHE_DIR=$S/pip_cache XDG_CACHE_HOME=$S/xdg_cache; mkdir -p $PIP_CACHE_DIR $XDG_CACHE_HOME
[ -x $E/bin/python ] || /opt/apps/python/3.12.11/bin/python3 -m venv --without-pip $E
P=$S/jev_probe/venv_small/bin/python     # has pip 23; venv's own ensurepip hung when $HOME was full
$P -m pip -q --python $E/bin/python install "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu128
$P -m pip -q --python $E/bin/python install "transformers==4.52.4" "tokenizers==0.21.1" "qwen-vl-utils==0.0.14" accelerate pillow numpy requests diffusers sentencepiece protobuf scipy
$E/bin/python -c "import torch,transformers,qwen_vl_utils,diffusers;print('ENV_OK',torch.__version__,torch.cuda.is_available(),transformers.__version__,diffusers.__version__)"
