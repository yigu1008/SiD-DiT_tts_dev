#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

"${PYTHON_BIN}" "${SCRIPT_DIR}/geneval_greedy.py" \
  --geneval_prompts "${GENEVAL_PROMPTS:-/home/ygu/geneval/prompts/evaluation_metadata.jsonl}" \
  --geneval_python "${GENEVAL_PYTHON:-/home/ygu/miniconda3/envs/geneval/bin/python}" \
  --geneval_repo "${GENEVAL_REPO:-/home/ygu/geneval}" \
  --detector_path "${DETECTOR_PATH:-/home/ygu/geneval/dectect/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth}" \
  --cfg_scales ${CFG_SCALES:-1.0 1.5 2.0 2.5 3.0 4.0} \
  --steps "${STEPS:-4}" \
  --n_samples "${N_SAMPLES:-4}" \
  --start_index "${START_INDEX:-0}" \
  --end_index "${END_INDEX:-10}" \
  "$@"
