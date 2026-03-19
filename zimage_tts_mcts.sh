#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

MODEL="${MODEL:-Tongyi-MAI/Z-Image-Turbo}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/prompts.txt}"
OUTDIR="${OUTDIR:-./zimage_tts_mcts_out}"

REWARD_BACKEND="${REWARD_BACKEND:-unifiedreward}"
REWARD_MODEL="${REWARD_MODEL:-CodeGoat24/UnifiedReward-2.0-qwen3vl-4b}"
REWARD_PROMPT_MODE="${REWARD_PROMPT_MODE:-standard}"
REWARD_MAX_NEW_TOKENS="${REWARD_MAX_NEW_TOKENS:-64}"

DTYPE="${DTYPE:-fp16}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-640}"
STEPS="${STEPS:-5}"
N_VARIANTS="${N_VARIANTS:-1}"
N_SIMS="${N_SIMS:-12}"
UCB_C="${UCB_C:-1.41}"
SEED="${SEED:-42}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-256}"
KICK_EPS="${KICK_EPS:-0.015}"
NUM_KICK_DIRS="${NUM_KICK_DIRS:-2}"

EXTRA_ARGS=()
if [[ -n "${ATTENTION:-}" ]]; then
  EXTRA_ARGS+=(--attention "${ATTENTION}")
fi
if [[ "${COMPILE_TRANSFORMER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--compile_transformer)
fi
if [[ "${ENABLE_CFG_ABLATION:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enable_cfg_ablation)
fi
if [[ "${ALLOW_SKIP:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--allow_skip)
fi
if [[ "${ALLOW_REPEAT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--allow_repeat)
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/zimage_tts.py" \
  --search_method mcts \
  --model "${MODEL}" \
  --prompt_file "${PROMPT_FILE}" \
  --reward_backend "${REWARD_BACKEND}" \
  --reward_model "${REWARD_MODEL}" \
  --reward_prompt_mode "${REWARD_PROMPT_MODE}" \
  --reward_max_new_tokens "${REWARD_MAX_NEW_TOKENS}" \
  --dtype "${DTYPE}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --steps "${STEPS}" \
  --n_variants "${N_VARIANTS}" \
  --n_sims "${N_SIMS}" \
  --ucb_c "${UCB_C}" \
  --max_sequence_length "${MAX_SEQUENCE_LENGTH}" \
  --kick_eps "${KICK_EPS}" \
  --num_kick_dirs "${NUM_KICK_DIRS}" \
  --seed "${SEED}" \
  --outdir "${OUTDIR}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
