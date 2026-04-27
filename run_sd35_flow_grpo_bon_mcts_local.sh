#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FLOW_GRPO_CKPT="${FLOW_GRPO_CKPT:-${SD35_CKPT:-}}"
if [[ -z "${FLOW_GRPO_CKPT}" ]]; then
  echo "Error: set FLOW_GRPO_CKPT (or SD35_CKPT) to your post-trained Flow-GRPO checkpoint path." >&2
  echo "Example: FLOW_GRPO_CKPT=/abs/path/flow_grpo.pt bash run_sd35_flow_grpo_bon_mcts_local.sh" >&2
  exit 1
fi

if [[ ! -f "${FLOW_GRPO_CKPT}" ]]; then
  echo "Error: checkpoint file not found: ${FLOW_GRPO_CKPT}" >&2
  exit 1
fi

export SD35_BACKEND="${SD35_BACKEND:-sd35_base}"
export BON_MCTS_REFINE_METHOD="${BON_MCTS_REFINE_METHOD:-ours_tree}"
export LOOKAHEAD_METHOD_MODE="${LOOKAHEAD_METHOD_MODE:-rollout_tree_prior_adaptive_cfg}"

echo "[flow-grpo] ckpt=${FLOW_GRPO_CKPT}"
echo "[flow-grpo] backend=${SD35_BACKEND} refine=${BON_MCTS_REFINE_METHOD} lookahead_mode=${LOOKAHEAD_METHOD_MODE}"

bash "${SCRIPT_DIR}/run_sd35_base_bon_mcts_local.sh" --ckpt "${FLOW_GRPO_CKPT}" "$@"

