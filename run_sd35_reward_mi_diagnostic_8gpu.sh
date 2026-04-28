#!/usr/bin/env bash
# 8-GPU sharded driver for sd35_reward_mi_diagnostic.py.
#
# Splits the prompt file into NUM_GPUS contiguous shards, launches one
# background `--mode generate` process per shard pinned to CUDA_VISIBLE_DEVICES=k,
# waits for all, then merges shard CSVs with globally-unique prompt_id, then
# runs `--mode train` once on the merged CSV (single GPU).
#
# Inherits the same env knobs as run_sd35_reward_mi_diagnostic_local.sh
# (PROMPT_FILE, N_PROMPTS, N_SEEDS, N_REWRITES, CFG_SCALES, BACKEND, STEPS,
# REWARD_BACKEND, USE_QWEN, MINE_*, etc.). Adds:
#   NUM_GPUS    : how many shards to spawn (default 8)
#   SHARD_DIR   : per-shard scratch dir (default ${OUT_DIR}/shards)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/hpsv2_subset.txt}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/mi_diag_sd35_out}"
mkdir -p "${OUT_DIR}"

NUM_GPUS="${NUM_GPUS:-8}"
SHARD_DIR="${SHARD_DIR:-${OUT_DIR}/shards}"
mkdir -p "${SHARD_DIR}"

MODE="${MODE:-full}"
N_PROMPTS="${N_PROMPTS:-50}"
N_SEEDS="${N_SEEDS:-16}"
N_REWRITES="${N_REWRITES:-3}"
CFG_SCALES="${CFG_SCALES:-1.0 3.0 5.0 7.0 9.0}"
DEFAULT_CFG="${DEFAULT_CFG:-5.0}"
SEED_BASE="${SEED_BASE:-42}"
REWARD_NOISE_STD="${REWARD_NOISE_STD:-0.01}"

BACKEND="${BACKEND:-sid}"
STEPS="${STEPS:-4}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
TIME_SCALE="${TIME_SCALE:-1000.0}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-256}"
REWARD_BACKEND="${REWARD_BACKEND:-imagereward}"
USE_QWEN="${USE_QWEN:-1}"

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

# ── Split prompts ──────────────────────────────────────────────────────────
# Pick first N_PROMPTS lines, then split into NUM_GPUS contiguous shards.
# Emit shard sizes so we can renumber prompt_id during merge.
"${PYTHON_BIN}" - <<PY
import os, math
prompt_path = os.environ["PROMPT_FILE"]
n_prompts = int(os.environ["N_PROMPTS"])
n_shards  = int(os.environ["NUM_GPUS"])
shard_dir = os.environ["SHARD_DIR"]
with open(prompt_path, "r", encoding="utf-8") as f:
    prompts = [ln.rstrip("\n") for ln in f if ln.strip()]
if n_prompts > 0:
    prompts = prompts[:n_prompts]
total = len(prompts)
base = total // n_shards
rem  = total - base * n_shards
sizes = [base + (1 if k < rem else 0) for k in range(n_shards)]
offsets = [sum(sizes[:k]) for k in range(n_shards)]
os.makedirs(shard_dir, exist_ok=True)
for k in range(n_shards):
    start = offsets[k]
    end   = start + sizes[k]
    with open(os.path.join(shard_dir, f"prompts_{k:02d}.txt"), "w", encoding="utf-8") as f:
        for p in prompts[start:end]:
            f.write(p + "\n")
with open(os.path.join(shard_dir, "shard_sizes.txt"), "w", encoding="utf-8") as f:
    for s in sizes:
        f.write(f"{s}\n")
print(f"[8gpu] split {total} prompts → {n_shards} shards: sizes={sizes}")
PY

mapfile -t SHARD_SIZES < "${SHARD_DIR}/shard_sizes.txt"

# ── Mode dispatch ──────────────────────────────────────────────────────────
if [[ "${MODE}" == "generate" || "${MODE}" == "full" ]]; then
  echo "[8gpu] launching ${NUM_GPUS} shards for generation."
  pids=()
  for ((k=0; k<NUM_GPUS; k++)); do
    shard_size="${SHARD_SIZES[$k]}"
    if [[ "${shard_size}" == "0" ]]; then
      continue
    fi
    shard_prompts="${SHARD_DIR}/prompts_$(printf '%02d' "$k").txt"
    shard_csv="${SHARD_DIR}/dataset_$(printf '%02d' "$k").csv"
    shard_log="${SHARD_DIR}/gen_$(printf '%02d' "$k").log"

    SHARD_CMD=(
      "${PYTHON_BIN}" "${SCRIPT_DIR}/sd35_reward_mi_diagnostic.py"
      --mode generate
      --prompt_file "${shard_prompts}"
      --n_prompts "${shard_size}"
      --seed_base "${SEED_BASE}"
      --n_seeds "${N_SEEDS}"
      --n_rewrites "${N_REWRITES}"
      --cfg_scales "${CFG_ARR[@]}"
      --default_cfg "${DEFAULT_CFG}"
      --reward_noise_std "${REWARD_NOISE_STD}"
      --dataset_csv "${shard_csv}"
      --backend "${BACKEND}"
      --steps "${STEPS}"
      --width "${WIDTH}"
      --height "${HEIGHT}"
      --time_scale "${TIME_SCALE}"
      --max_sequence_length "${MAX_SEQUENCE_LENGTH}"
      --reward_backend "${REWARD_BACKEND}"
      --resume
    )
    if [[ ${#QWEN_ARGS[@]} -gt 0 ]]; then
      SHARD_CMD+=("${QWEN_ARGS[@]}")
    fi

    echo "[8gpu] shard ${k}: GPU=${k} prompts=${shard_size} csv=${shard_csv}"
    CUDA_VISIBLE_DEVICES="${k}" "${SHARD_CMD[@]}" > "${shard_log}" 2>&1 &
    pids+=("$!")
  done

  rc=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      rc=1
    fi
  done
  if [[ "${rc}" != "0" ]]; then
    echo "[8gpu] one or more shards failed; check ${SHARD_DIR}/gen_*.log" >&2
    exit "${rc}"
  fi

  # Merge shard CSVs with globally-unique prompt_id (offset by cumulative sizes).
  "${PYTHON_BIN}" - <<PY
import csv, os
shard_dir   = os.environ["SHARD_DIR"]
out_csv     = os.environ["DATASET_CSV"]
n_shards    = int(os.environ["NUM_GPUS"])
sizes = [int(x.strip()) for x in open(os.path.join(shard_dir, "shard_sizes.txt"))]
offsets = [sum(sizes[:k]) for k in range(n_shards)]
header = None
written = 0
os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
with open(out_csv, "w", encoding="utf-8", newline="") as fout:
    writer = None
    for k in range(n_shards):
        if sizes[k] == 0:
            continue
        path = os.path.join(shard_dir, f"dataset_{k:02d}.csv")
        if not os.path.exists(path):
            print(f"[8gpu] WARN: missing shard csv {path}")
            continue
        with open(path, "r", encoding="utf-8", newline="") as fin:
            reader = csv.DictReader(fin)
            if writer is None:
                header = list(reader.fieldnames or [])
                writer = csv.DictWriter(fout, fieldnames=header)
                writer.writeheader()
            for row in reader:
                row["prompt_id"] = str(int(row["prompt_id"]) + offsets[k])
                writer.writerow(row)
                written += 1
print(f"[8gpu] merged {written} rows → {out_csv}")
PY
fi

# ── MINE training (single GPU) ─────────────────────────────────────────────
if [[ "${MODE}" == "train" || "${MODE}" == "full" ]]; then
  echo "[8gpu] running MINE training on merged dataset."
  CUDA_VISIBLE_DEVICES="${MINE_GPU:-0}" "${PYTHON_BIN}" "${SCRIPT_DIR}/sd35_reward_mi_diagnostic.py" \
    --mode train \
    --prompt_file "${PROMPT_FILE}" \
    --n_prompts "${N_PROMPTS}" \
    --seed_base "${SEED_BASE}" \
    --n_seeds "${N_SEEDS}" \
    --n_rewrites "${N_REWRITES}" \
    --cfg_scales "${CFG_ARR[@]}" \
    --default_cfg "${DEFAULT_CFG}" \
    --reward_noise_std "${REWARD_NOISE_STD}" \
    --dataset_csv "${DATASET_CSV}" \
    --backend "${BACKEND}" \
    --steps "${STEPS}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --time_scale "${TIME_SCALE}" \
    --max_sequence_length "${MAX_SEQUENCE_LENGTH}" \
    --reward_backend "${REWARD_BACKEND}" \
    --mine_hidden_dim "${MINE_HIDDEN_DIM}" \
    --mine_embedding_dim "${MINE_EMBEDDING_DIM}" \
    --mine_batch_size "${MINE_BATCH_SIZE}" \
    --mine_lr "${MINE_LR}" \
    --mine_steps "${MINE_STEPS}" \
    --mine_eval_every "${MINE_EVAL_EVERY}" \
    --mine_restarts "${MINE_RESTARTS}" \
    --mine_device "${MINE_DEVICE}" \
    --mine_val_frac "${MINE_VAL_FRAC}" \
    --mine_report_json "${REPORT_JSON}" \
    --mine_table_csv "${TABLE_CSV}"
fi

echo "[8gpu] done."
