#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

FLOW_GRPO_CKPT="${FLOW_GRPO_CKPT:-${SD35_CKPT:-}}"
FLOW_GRPO_LORA_PATH="${FLOW_GRPO_LORA_PATH:-${SD35_LORA_PATH:-}}"
FLOW_GRPO_LORA_SCALE="${FLOW_GRPO_LORA_SCALE:-${SD35_LORA_SCALE:-1.0}}"
FLOW_GRPO_MODEL_ID="${FLOW_GRPO_MODEL_ID:-${SD35_MODEL_ID:-stabilityai/stable-diffusion-3.5-medium}}"

_resolve_hf_repo_from_input() {
  local v="${1:-}"
  if [[ "${v}" =~ ^https?://huggingface\.co/([^/]+/[^/]+) ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "${v}" =~ ^[^/]+/[^/]+$ ]]; then
    echo "${v}"
    return 0
  fi
  return 1
}

if [[ -n "${FLOW_GRPO_LORA_PATH}" && ! -e "${FLOW_GRPO_LORA_PATH}" ]]; then
  if repo_id="$(_resolve_hf_repo_from_input "${FLOW_GRPO_LORA_PATH}")"; then
    echo "[flow-grpo] resolving LoRA repo from Hugging Face: ${repo_id}"
    FLOW_GRPO_LORA_PATH="$("${PYTHON_BIN}" - <<'PY' "${repo_id}"
import os
import sys
from huggingface_hub import snapshot_download
repo_id = sys.argv[1]
path = snapshot_download(
    repo_id=repo_id,
    cache_dir=os.environ.get("HF_HOME"),
    token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    resume_download=True,
    max_workers=1,
)
print(path)
PY
)"
    echo "[flow-grpo] resolved LoRA local snapshot: ${FLOW_GRPO_LORA_PATH}"
  fi
fi

if [[ -z "${FLOW_GRPO_CKPT}" && -z "${FLOW_GRPO_LORA_PATH}" ]]; then
  echo "Error: set FLOW_GRPO_CKPT/SD35_CKPT (full checkpoint) or FLOW_GRPO_LORA_PATH/SD35_LORA_PATH (LoRA)." >&2
  echo "Example (ckpt): FLOW_GRPO_CKPT=/abs/path/flow_grpo.pt bash run_sd35_flow_grpo_bon_mcts_local.sh" >&2
  echo "Example (LoRA file): FLOW_GRPO_LORA_PATH=/abs/path/flow_grpo_lora.safetensors bash run_sd35_flow_grpo_bon_mcts_local.sh" >&2
  echo "Example (HF repo): FLOW_GRPO_LORA_PATH=jieliu/SD3.5M-FlowGRPO-PickScore bash run_sd35_flow_grpo_bon_mcts_local.sh" >&2
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

echo "[flow-grpo] model_id=${FLOW_GRPO_MODEL_ID}"
if [[ -n "${FLOW_GRPO_CKPT}" ]]; then
  echo "[flow-grpo] ckpt=${FLOW_GRPO_CKPT}"
fi
if [[ -n "${FLOW_GRPO_LORA_PATH}" ]]; then
  echo "[flow-grpo] lora_path=${FLOW_GRPO_LORA_PATH} lora_scale=${FLOW_GRPO_LORA_SCALE}"
fi
echo "[flow-grpo] backend=${SD35_BACKEND} refine=${BON_MCTS_REFINE_METHOD} lookahead_mode=${LOOKAHEAD_METHOD_MODE}"

extra_args=()
extra_args+=(--model_id "${FLOW_GRPO_MODEL_ID}")
if [[ -n "${FLOW_GRPO_CKPT}" ]]; then
  extra_args+=(--ckpt "${FLOW_GRPO_CKPT}")
fi
if [[ -n "${FLOW_GRPO_LORA_PATH}" ]]; then
  extra_args+=(--lora_path "${FLOW_GRPO_LORA_PATH}" --lora_scale "${FLOW_GRPO_LORA_SCALE}")
fi

bash "${SCRIPT_DIR}/run_sd35_base_bon_mcts_local.sh" "${extra_args[@]}" "$@"
