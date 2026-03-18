#!/usr/bin/env bash
set -euo pipefail

# Optional local env defaults (override by exporting before running this script)
export HF_HOME="${HF_HOME:-/data/ygu/.cache}"
export PATH="${SID_ENV_PATH:-/home/ygu/miniconda3/envs/sid_dit/bin}:$PATH"

python sampling_unified.py \
  --search_method greedy \
  --reward_type imagereward \
  --prompt_file prompts.txt \
  --n_variants 3 \
  --cfg_scales 1.0 1.25 1.5 1.75 2.0 2.25 2.5 \
  --steps 4 \
  --seed 42 \
  --out_dir ./imagereward_greedy_out \
  "$@"

