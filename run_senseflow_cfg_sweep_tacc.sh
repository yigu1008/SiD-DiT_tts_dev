#!/usr/bin/env bash
# Run SenseFlow CFG sweep on TACC cluster.
#
# Usage:
#   bash run_senseflow_cfg_sweep_tacc.sh
#
#   # SenseFlow-Medium:
#   BACKEND=senseflow_medium bash run_senseflow_cfg_sweep_tacc.sh
#
#   # Compare against SD3.5 base:
#   BACKEND=sd35_base CFG_SCALES="0.0 1.0 3.5 4.5 7.0" bash run_senseflow_cfg_sweep_tacc.sh
#
#   # More prompts:
#   END_INDEX=10 bash run_senseflow_cfg_sweep_tacc.sh
#
#   # Custom prompt:
#   PROMPT="a cat wearing a hat" bash run_senseflow_cfg_sweep_tacc.sh
#
# For SLURM:
#   sbatch --nodes=1 --ntasks=1 --cpus-per-task=4 \
#          --gres=gpu:1 --time=02:00:00 \
#          --partition=gpu run_senseflow_cfg_sweep_tacc.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Environment setup ────────────────────────────────────────────────────────
export SKIP_INSTALL="${SKIP_INSTALL:-1}"
source "${SCRIPT_DIR}/tacc_setup.sh"

# ── Clean up stale HF cache in home1 (quota-limited) ────────────────────────
# Models should live under $DATA_ROOT (WORK/SCRATCH), not $HOME.
if [[ -d "${HOME}/.cache/huggingface/hub" ]]; then
  _home_hf="${HOME}/.cache/huggingface"
  if [[ "${HF_HOME}" != "${_home_hf}"* ]]; then
    echo "[senseflow-sweep] WARNING: stale HF cache found at ${_home_hf}"
    echo "[senseflow-sweep] HF_HOME is set to ${HF_HOME}"
    echo "[senseflow-sweep] removing ${_home_hf} to free home1 quota ..."
    rm -rf "${_home_hf}"
    echo "[senseflow-sweep] cleaned up home1 HF cache"
  fi
fi

# ── Preload SenseFlow model weights ──────────────────────────────────────────
BACKEND="${BACKEND:-senseflow_large}"

echo "[senseflow-sweep] HF_HOME=${HF_HOME}"
echo "[senseflow-sweep] preloading model weights ..."
"${PYTHON_BIN}" -c "
from huggingface_hub import snapshot_download
import os
token = os.environ.get('HF_TOKEN', None)
snapshot_download('stabilityai/stable-diffusion-3.5-large', token=token, resume_download=True, max_workers=1)
print('SD3.5-large cache OK')
snapshot_download('domiso/SenseFlow', token=token, resume_download=True, max_workers=1)
print('SenseFlow cache OK')
"

# ── Config ───────────────────────────────────────────────────────────────────
CFG_SCALES="${CFG_SCALES:-0.0 0.5 1.0 1.5 2.0 3.0 4.0 5.0}"
PROMPT="${PROMPT:-}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/hpsv2_subset.txt}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-3}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-${SCRATCH:-${DATA_ROOT}}/senseflow_cfg_sweep/${BACKEND}}"

mkdir -p "${OUT_DIR}"

echo "[senseflow-sweep] backend:    ${BACKEND}"
echo "[senseflow-sweep] cfg_scales: ${CFG_SCALES}"
echo "[senseflow-sweep] out_dir:    ${OUT_DIR}"

# ── Run ──────────────────────────────────────────────────────────────────────
CMD=(
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/test_senseflow_cfg_sweep.py"
  --backend "${BACKEND}"
  --cfg_scales ${CFG_SCALES}
  --seed "${SEED}"
  --out_dir "${OUT_DIR}"
  --make_grid
)

if [[ -n "${PROMPT}" ]]; then
  CMD+=(--prompt "${PROMPT}")
else
  CMD+=(
    --prompt_file "${PROMPT_FILE}"
    --start_index "${START_INDEX}"
    --end_index "${END_INDEX}"
  )
fi

echo "[senseflow-sweep] command: ${CMD[*]}"
"${CMD[@]}"

echo "[senseflow-sweep] done. Outputs: ${OUT_DIR}"
