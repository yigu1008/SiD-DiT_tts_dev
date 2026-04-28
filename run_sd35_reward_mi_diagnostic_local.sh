#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/hpsv2_subset.txt}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/mi_diag_sd35_out}"
mkdir -p "${OUT_DIR}"

MODE="${MODE:-full}"  # generate | train | full

# Data grid
N_PROMPTS="${N_PROMPTS:-50}"
N_SEEDS="${N_SEEDS:-16}"        # set 8 for pilot
N_REWRITES="${N_REWRITES:-3}"   # original + 3 rewrites = 4 variants
CFG_SCALES="${CFG_SCALES:-1.0 3.0 5.0 7.0 9.0}"
DEFAULT_CFG="${DEFAULT_CFG:-5.0}"
REWARD_NOISE_STD="${REWARD_NOISE_STD:-0.01}"
SEED_BASE="${SEED_BASE:-42}"

# Sampler/reward
BACKEND="${BACKEND:-sid}"       # sid | sd35_base | senseflow_large | senseflow_medium
STEPS="${STEPS:-4}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
TIME_SCALE="${TIME_SCALE:-1000.0}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-256}"
REWARD_BACKEND="${REWARD_BACKEND:-imagereward}"
USE_QWEN="${USE_QWEN:-1}"

# MINE
MINE_HIDDEN_DIM="${MINE_HIDDEN_DIM:-256}"
MINE_EMBEDDING_DIM="${MINE_EMBEDDING_DIM:-32}"
MINE_BATCH_SIZE="${MINE_BATCH_SIZE:-1024}"
MINE_LR="${MINE_LR:-1e-4}"
MINE_STEPS="${MINE_STEPS:-10000}"
MINE_EVAL_EVERY="${MINE_EVAL_EVERY:-250}"
MINE_RESTARTS="${MINE_RESTARTS:-3}"
MINE_DEVICE="${MINE_DEVICE:-cuda}"
MINE_VAL_FRAC="${MINE_VAL_FRAC:-0.2}"

DATASET_CSV="${DATASET_CSV:-${OUT_DIR}/mi_diag_sd35_dataset.csv}"
REPORT_JSON="${REPORT_JSON:-${OUT_DIR}/mi_diag_sd35_report.json}"
TABLE_CSV="${TABLE_CSV:-${OUT_DIR}/mi_diag_sd35_table.csv}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found: ${PROMPT_FILE}" >&2
  exit 1
fi

read -r -a CFG_ARR <<< "${CFG_SCALES}"
if [[ "${#CFG_ARR[@]}" -eq 0 ]]; then
  echo "Error: CFG_SCALES is empty." >&2
  exit 1
fi

QWEN_ARGS=()
if [[ "${USE_QWEN}" != "1" ]]; then
  QWEN_ARGS+=(--no_qwen)
fi

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/sd35_reward_mi_diagnostic.py"
  --mode "${MODE}"
  --prompt_file "${PROMPT_FILE}"
  --n_prompts "${N_PROMPTS}"
  --seed_base "${SEED_BASE}"
  --n_seeds "${N_SEEDS}"
  --n_rewrites "${N_REWRITES}"
  --cfg_scales "${CFG_ARR[@]}"
  --default_cfg "${DEFAULT_CFG}"
  --reward_noise_std "${REWARD_NOISE_STD}"
  --dataset_csv "${DATASET_CSV}"
  --backend "${BACKEND}"
  --steps "${STEPS}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --time_scale "${TIME_SCALE}"
  --max_sequence_length "${MAX_SEQUENCE_LENGTH}"
  --reward_backend "${REWARD_BACKEND}"
  --mine_hidden_dim "${MINE_HIDDEN_DIM}"
  --mine_embedding_dim "${MINE_EMBEDDING_DIM}"
  --mine_batch_size "${MINE_BATCH_SIZE}"
  --mine_lr "${MINE_LR}"
  --mine_steps "${MINE_STEPS}"
  --mine_eval_every "${MINE_EVAL_EVERY}"
  --mine_restarts "${MINE_RESTARTS}"
  --mine_device "${MINE_DEVICE}"
  --mine_val_frac "${MINE_VAL_FRAC}"
  --mine_report_json "${REPORT_JSON}"
  --mine_table_csv "${TABLE_CSV}"
  --resume
)

if [[ ${#QWEN_ARGS[@]} -gt 0 ]]; then
  CMD+=("${QWEN_ARGS[@]}")
fi

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[mi-diag] mode=${MODE} backend=${BACKEND} reward_backend=${REWARD_BACKEND}"
echo "[mi-diag] prompts=${N_PROMPTS} seeds=${N_SEEDS} variants=$((N_REWRITES+1)) cfgs=${#CFG_ARR[@]}"
echo "[mi-diag] dataset=${DATASET_CSV}"
echo "[mi-diag] report=${REPORT_JSON}"
echo "[mi-diag] table=${TABLE_CSV}"

"${CMD[@]}"

