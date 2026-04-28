#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FLOW_GRPO_CKPT="${FLOW_GRPO_CKPT:-${SD35_CKPT:-}}"
FLOW_GRPO_LORA_PATH="${FLOW_GRPO_LORA_PATH:-${SD35_LORA_PATH:-}}"
FLOW_GRPO_LORA_SCALE="${FLOW_GRPO_LORA_SCALE:-${SD35_LORA_SCALE:-1.0}}"

if [[ -z "${FLOW_GRPO_CKPT}" && -z "${FLOW_GRPO_LORA_PATH}" ]]; then
  echo "Error: set FLOW_GRPO_CKPT/SD35_CKPT (full checkpoint) or FLOW_GRPO_LORA_PATH/SD35_LORA_PATH (LoRA)." >&2
  echo "Example (ckpt): FLOW_GRPO_CKPT=/abs/path/flow_grpo.pt bash run_sd35_flow_grpo_bon_mcts_local.sh" >&2
  echo "Example (LoRA): FLOW_GRPO_LORA_PATH=/abs/path/flow_grpo_lora.safetensors bash run_sd35_flow_grpo_bon_mcts_local.sh" >&2
  exit 1
fi

if [[ -n "${FLOW_GRPO_CKPT}" && ! -f "${FLOW_GRPO_CKPT}" ]]; then
  if [[ -n "${FLOW_GRPO_LORA_PATH}" ]]; then
    echo "[flow-grpo] warning: FLOW_GRPO_CKPT not found, ignoring because LoRA path is set: ${FLOW_GRPO_CKPT}" >&2
    FLOW_GRPO_CKPT=""
  else
    echo "Error: checkpoint file not found: ${FLOW_GRPO_CKPT}" >&2
    exit 1
  fi
fi

if [[ -n "${FLOW_GRPO_LORA_PATH}" && ! -e "${FLOW_GRPO_LORA_PATH}" ]]; then
  echo "Error: LoRA path not found: ${FLOW_GRPO_LORA_PATH}" >&2
  exit 1
fi

export SD35_BACKEND="${SD35_BACKEND:-sd35_base}"
export BON_MCTS_REFINE_METHOD="${BON_MCTS_REFINE_METHOD:-ours_tree}"
export LOOKAHEAD_METHOD_MODE="${LOOKAHEAD_METHOD_MODE:-rollout_tree_prior_adaptive_cfg}"

if [[ -n "${FLOW_GRPO_CKPT}" ]]; then
  echo "[flow-grpo] ckpt=${FLOW_GRPO_CKPT}"
fi
if [[ -n "${FLOW_GRPO_LORA_PATH}" ]]; then
  echo "[flow-grpo] lora_path=${FLOW_GRPO_LORA_PATH} lora_scale=${FLOW_GRPO_LORA_SCALE}"
fi
echo "[flow-grpo] backend=${SD35_BACKEND} refine=${BON_MCTS_REFINE_METHOD} lookahead_mode=${LOOKAHEAD_METHOD_MODE}"

extra_args=()
if [[ -n "${FLOW_GRPO_CKPT}" ]]; then
  extra_args+=(--ckpt "${FLOW_GRPO_CKPT}")
fi
if [[ -n "${FLOW_GRPO_LORA_PATH}" ]]; then
  extra_args+=(--lora_path "${FLOW_GRPO_LORA_PATH}" --lora_scale "${FLOW_GRPO_LORA_SCALE}")
fi

bash "${SCRIPT_DIR}/run_sd35_base_bon_mcts_local.sh" "${extra_args[@]}" "$@"
