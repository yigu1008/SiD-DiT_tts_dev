#!/usr/bin/env bash
set -euo pipefail

# Optional local env defaults (override by exporting before running this script)
export HF_HOME="${HF_HOME:-/data/ygu/.cache}"
export PATH="${SID_ENV_PATH:-/home/ygu/miniconda3/envs/sid_dit/bin}:$PATH"

python sampling_unified_sd35.py \
  --search_method mcts \
  --prompt_file prompts.txt \
  --n_variants 3 \
  --cfg_scales 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25 3.5 \
  --steps 4 \
  --n_sims 50 \
  --ucb_c 1.41 \
  --seed 42 \
  --width 1024 \
  --height 1024 \
  --out_dir ./imagereward_sd35_mcts_out \
  "$@"

