#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/data/ygu/.cache}"
export PATH="${SID_ENV_PATH:-/home/ygu/miniconda3/envs/sid_dit/bin}:$PATH"

NUM_GPUS="${NUM_GPUS:-$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)}"

PROMPT_FILE="${PROMPT_FILE:-parti_prompts.txt}"
OUT_DIR="${OUT_DIR:-./sd35_ddp_out}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE does not exist: ${PROMPT_FILE}" >&2
  exit 1
fi
mkdir -p "${OUT_DIR}"

PROMPT_FILE_ABS="$(python - <<'PY' "${PROMPT_FILE}"
import pathlib,sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PY
)"
OUT_DIR_ABS="$(python - <<'PY' "${OUT_DIR}"
import pathlib,sys
p = pathlib.Path(sys.argv[1]).expanduser().resolve()
p.mkdir(parents=True, exist_ok=True)
print(p)
PY
)"

EXTRA_ARGS=()
if [[ "${NO_QWEN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no_qwen)
fi
if [[ "${SEED_PER_PROMPT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--seed_per_prompt)
fi
if [[ "${SAVE_IMAGES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save_images)
fi
if [[ "${SAVE_VARIANTS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save_variants)
fi
if [[ -n "${REWRITES_FILE:-}" ]]; then
  EXTRA_ARGS+=(--rewrites_file "${REWRITES_FILE}")
fi

torchrun --standalone --nproc_per_node "${NUM_GPUS}" sd35_ddp_experiment.py \
  --prompt_file "${PROMPT_FILE_ABS}" \
  --start_index "${START_INDEX:-0}" \
  --end_index "${END_INDEX:--1}" \
  --modes ${MODES:-base greedy mcts} \
  --cfg_scales ${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5} \
  --baseline_cfg "${BASELINE_CFG:-1.0}" \
  --steps "${STEPS:-4}" \
  --n_variants "${N_VARIANTS:-3}" \
  --n_sims "${N_SIMS:-50}" \
  --ucb_c "${UCB_C:-1.41}" \
  --seed "${SEED:-42}" \
  --out_dir "${OUT_DIR_ABS}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

python summarize_sd35_ddp.py --log_dir "${OUT_DIR_ABS}/logs" --out_dir "${OUT_DIR_ABS}"
