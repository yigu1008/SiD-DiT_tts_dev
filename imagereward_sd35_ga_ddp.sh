#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

NUM_GPUS="${NUM_GPUS:-$("${PYTHON_BIN}" - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)}"

PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/prompts.txt}"
OUT_DIR="${OUT_DIR:-./imagereward_sd35_ga_ddp_out}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE does not exist: ${PROMPT_FILE}" >&2
  exit 1
fi
mkdir -p "${OUT_DIR}"

ensure_imagereward_runtime() {
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import xxhash
import clip
import ImageReward as RM
print(xxhash.__version__, getattr(RM, "__file__", "ok"))
PY
  then
    return 0
  fi
  echo "[deps] ImageReward runtime deps missing. Installing with install_reward_deps.sh ..."
  PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/install_reward_deps.sh"
}

ensure_imagereward_runtime

PROMPT_FILE_ABS="$("${PYTHON_BIN}" - <<'PY' "${PROMPT_FILE}"
import pathlib,sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PY
)"
OUT_DIR_ABS="$("${PYTHON_BIN}" - <<'PY' "${OUT_DIR}"
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

torchrun --standalone --nproc_per_node "${NUM_GPUS}" "${SCRIPT_DIR}/sd35_ddp_experiment.py" \
  --prompt_file "${PROMPT_FILE_ABS}" \
  --start_index "${START_INDEX:-0}" \
  --end_index "${END_INDEX:--1}" \
  --modes base ga \
  --cfg_scales ${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5} \
  --baseline_cfg "${BASELINE_CFG:-1.0}" \
  --steps "${STEPS:-4}" \
  --n_variants "${N_VARIANTS:-3}" \
  --reward_backend imagereward \
  --image_reward_model "${IMAGE_REWARD_MODEL:-ImageReward-v1.0}" \
  --seed "${SEED:-42}" \
  --ga_population "${GA_POPULATION:-24}" \
  --ga_generations "${GA_GENERATIONS:-8}" \
  --ga_elites "${GA_ELITES:-3}" \
  --ga_mutation_prob "${GA_MUTATION_PROB:-0.10}" \
  --ga_tournament_k "${GA_TOURNAMENT_K:-3}" \
  --ga_selection "${GA_SELECTION:-rank}" \
  --ga_rank_pressure "${GA_RANK_PRESSURE:-1.7}" \
  --ga_crossover "${GA_CROSSOVER:-uniform}" \
  --ga_log_topk "${GA_LOG_TOPK:-3}" \
  --ga_phase_constraints \
  --out_dir "${OUT_DIR_ABS}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_sd35_ddp.py" --log_dir "${OUT_DIR_ABS}/logs" --out_dir "${OUT_DIR_ABS}"
