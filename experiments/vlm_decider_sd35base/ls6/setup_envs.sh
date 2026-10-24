#!/usr/bin/env bash
# Build the four python envs on $SCRATCH (pins = amlt/jev_decider_sd35base.yaml, which is the authoritative list).
# Idempotent; run on a COMPUTE node (login nodes cap vmem at 8 GB).   bash setup_envs.sh [conda reward jev vqascore sid_dit preload]
set -uo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
export CONDA_PKGS_DIRS=$S/conda_pkgs; mkdir -p $CONDA_PKGS_DIRS $TMPDIR $RUN_DIR/setup/logs
STAGES=${*:-conda reward jev vqascore sid_dit preload}
say(){ echo "[$(date '+%F %T')] $*"; }
has(){ case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
CONDA=$CONDA_BASE/bin/conda

if has conda; then say "== conda"
  if [ ! -x $CONDA ]; then curl -fsSL -o $TMPDIR/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash $TMPDIR/miniforge.sh -b -p $CONDA_BASE || { say "FAILED conda"; exit 1; }; fi; $CONDA --version; fi

if has reward; then say "== reward env ($REWARD_PY_PREFIX): HPSv3 + ImageReward + PickScore, transformers 4.45.2"
  CONDA_BASE=$CONDA_BASE REWARD_ENV_PREFIX=$REWARD_PY_PREFIX REWARD_ENV_SENTINEL_DIR=$RUN_DIR/state bash $REPO_DIR/setup_reward_env.sh || say "WARN setup_reward_env.sh rc=$?"
  RP=$REWARD_PY_PREFIX/bin/python
  $RP -m pip install --no-cache-dir --upgrade "setuptools<80" wheel || true
  $RP -m pip install --no-cache-dir "fairscale==0.4.4" "timm>=1.0" || true
  $RP -m pip install --no-cache-dir --no-deps "diffusers>=0.27,<0.31" "datasets" "wandb" || true
  IR_INIT=$($RP -c 'import importlib.util; s=importlib.util.find_spec("ImageReward"); print(s.origin if s else "")' 2>/dev/null || echo "")
  [ -f "$IR_INIT" ] && { grep -q "^# (patched-no-ReFL)" "$IR_INIT" || sed -i "s|^from \.ReFL import|# (patched-no-ReFL) from .ReFL import|" "$IR_INIT"; }
  $RP -c "import torch,transformers,hpsv3,ImageReward,hpsv2; print('REWARD_ENV_OK', torch.__version__, torch.cuda.is_available(), transformers.__version__)" || say "FAILED reward env verify"; fi

if has jev; then say "== jev env ($JEV_PY_PREFIX): py3.11, torch 2.11 cu128, transformers 5.17 (Gemma-4 / Jev-Omni)"
  [ -x $JEV_PY_PREFIX/bin/python ] || $CONDA create -y -p $JEV_PY_PREFIX python=3.11 pip
  JP=$JEV_PY_PREFIX/bin/python
  $JP -m pip install --no-cache-dir "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128 \
    || $JP -m pip install --no-cache-dir "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu126
  $JP -m pip install --no-cache-dir "transformers==5.17.0" "accelerate==1.15.0" "safetensors>=0.5" pillow numpy "huggingface_hub>=1.0"
  $JP -c "import torch,transformers; print('JEV_ENV_OK', torch.__version__, torch.cuda.is_available(), transformers.__version__)" || say "FAILED jev env verify"; fi

if has vqascore; then say "== vqascore env ($VQA_PY_PREFIX): py3.10, torch 2.5.1 cu124, t2v_metrics 3.0 (held-out judge only)"
  # t2v_metrics==3.0 declares a bogus PyPI 'apex' dep (-> cryptacular, C build fails) and imports every video/LLaVA backend at
  # package import. Recipe: install it --no-deps, add the deps VQAScore(clip-flant5) needs under a constraints file that pins
  # torch/torchvision/torchaudio 2.5.1 cu124 + transformers 4.49 + numpy<2, ffmpeg from conda (import-time check), then
  # patch_t2v_metrics.py makes every non-CLIP-FlanT5 backend import optional.
  [ -x $VQA_PY_PREFIX/bin/python ] || $CONDA create -y -p $VQA_PY_PREFIX python=3.10 pip
  VP=$VQA_PY_PREFIX/bin/python; VC=$RUN_DIR/setup/vqascore_constraints.txt
  printf 'torch==2.5.1\ntorchvision==0.20.1\ntorchaudio==2.5.1\ntransformers==4.49.0\nnumpy<2\npillow<11\naccelerate==1.14.0\nopencv-python<5\nopencv-python-headless<5\n' > $VC
  $VP -m pip install --no-cache-dir "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" --index-url https://download.pytorch.org/whl/cu124
  $VP -m pip install --no-cache-dir -c $VC "transformers==4.49.0" "accelerate==1.14.0" "pillow<11" "numpy<2"
  $VP -m pip install --no-cache-dir --no-deps "t2v_metrics==3.0"
  $VP -m pip install --no-cache-dir -c $VC ftfy tqdm "huggingface-hub<1.0" open-clip-torch pandas scipy sentencepiece datasets tokenizers omegaconf iopath fairscale "fire==0.4.0" tiktoken "peft==0.5.0" einops einops-exts protobuf requests PyYAML "timm>=1.0.15" pycocoevalcap matplotlib "opencv-python-headless<5" gdown easydict shortuuid pydantic icecream markdown2 termcolor func_timeout openai decord av imageio librosa
  $CONDA install -y -q -p $VQA_PY_PREFIX -c conda-forge ffmpeg
  $VP -I $RUN_DIR/setup/patch_t2v_metrics.py
  PATH=$VQA_PY_PREFIX/bin:$PATH $VP -c "import torch,t2v_metrics; from t2v_metrics import VQAScore; print('VQA_ENV_OK', torch.__version__, torch.cuda.is_available())" || say "FAILED vqascore env verify"; fi

if has sid_dit; then say "== sid_dit completion ($SIDDIT_PY_PREFIX): requirements.txt + reward deps, as the yaml does for ptca"
  SP=$SIDDIT_PY_PREFIX/bin/python; cd $REPO_DIR
  $SP -m pip install --upgrade "setuptools>=70,<76" wheel
  grep -v '^xformers' requirements.txt | sed 's/ --index-url.*//' > $TMPDIR/req_sid_dit.txt   # torch 2.7.1+cu128 already present satisfies ==2.7.1
  $SP -m pip install -r $TMPDIR/req_sid_dit.txt || say "WARN requirements.txt rc=$?"
  $SP -m pip install --no-deps "xformers==0.0.31.post1" || say "WARN xformers skipped"
  PYTHON_BIN=$SP bash install_reward_deps.sh || say "WARN install_reward_deps.sh rc=$?"
  $SP -m pip install "transformers==4.52.4" "tokenizers==0.21.1" "qwen-vl-utils==0.0.14" "diffusers==0.33.1" "pyarrow>=14.0.2,<21"   # pyarrow>=21 needs numpy 2; project pins numpy 1.26.4
  $SP -c "import torch,transformers,diffusers,qwen_vl_utils,timm; import sampling_unified_sd35_lookahead_reweighting; print('SID_DIT_ENV_OK', torch.__version__, transformers.__version__, diffusers.__version__)" || say "FAILED sid_dit verify"; fi

if has preload; then say "== preload reward weights (IR / HPSv3 / PickScore) into the caches"
  REWARD_BACKENDS='imagereward hpsv3 pickscore' $REWARD_PY_PREFIX/bin/python $REPO_DIR/preload_reward_models.py || say "WARN preload rc=$?"; fi
say "setup_envs done: $STAGES"
