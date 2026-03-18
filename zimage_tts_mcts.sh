#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/data/ygu/.cache}"
export PATH="${SID_ENV_PATH:-/home/ygu/miniconda3/envs/sid_dit/bin}:$PATH"

python zimage_tts.py \
  --search_method mcts \
  --model Tongyi-MAI/Z-Image-Turbo \
  --prompt_file prompts.txt \
  --steps 9 \
  --cfg_scales 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \
  --n_variants 3 \
  --n_sims 30 \
  --ucb_c 1.41 \
  --seed 42 \
  --outdir ./zimage_tts_mcts_out \
  "$@"

