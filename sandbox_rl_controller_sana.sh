#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

PROMPT_STYLE="${PROMPT_STYLE:-all}"
PROMPT_DIR="${PROMPT_DIR:-/data/ygu}"
PROMPT_FILE="${PROMPT_FILE:-${PROMPT_DIR}/hpsv2_prompts.txt}"
OUT_DIR="${OUT_DIR:-/data/ygu/sandbox_rl_controller_sana}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "[sandbox] prompt file not found, exporting HPSv2 prompts first ..."
  OUT_DIR="${PROMPT_DIR}" STYLE="${PROMPT_STYLE}" bash "${SCRIPT_DIR}/get_hpsv2_prompts.sh"
fi

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found after HPSv2 export: ${PROMPT_FILE}" >&2
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/sandbox_rl_controller_sana.py" \
  --prompt_file "${PROMPT_FILE}" \
  --max_prompts "${MAX_PROMPTS:-1}" \
  --out_dir "${OUT_DIR}" \
  --model_id "${MODEL_ID:-YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow}" \
  --dtype "${DTYPE:-bf16}" \
  --steps "${STEPS:-4}" \
  --width "${WIDTH:-512}" \
  --height "${HEIGHT:-512}" \
  --seed "${SEED:-42}" \
  --guidance_scale "${GUIDANCE_SCALE:-1.0}" \
  --preview_every "${PREVIEW_EVERY:-1}" \
  --reward_type "${REWARD_TYPE:-imagereward}" \
  --reward_device "${REWARD_DEVICE:-cpu}" \
  --openloop_population "${OPENLOOP_POP:-16}" \
  --openloop_generations "${OPENLOOP_GENS:-8}" \
  --openloop_elites "${OPENLOOP_ELITES:-2}" \
  --controller_population "${CTRL_POP:-16}" \
  --controller_generations "${CTRL_GENS:-8}" \
  --controller_elites "${CTRL_ELITES:-2}" \
  --trace_topk_per_gen "${TRACE_TOPK_PER_GEN:-2}" \
  --save_images \
  "$@"

