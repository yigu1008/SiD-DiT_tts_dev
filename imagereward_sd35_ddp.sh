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
OUT_DIR="${OUT_DIR:-./imagereward_sd35_ddp_out}"

MODES="${MODES:-base greedy mcts}"
CFG_SCALES="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5}"
BASELINE_CFG="${BASELINE_CFG:-1.0}"
STEPS="${STEPS:-4}"
SEED="${SEED:-42}"

REWARD_BACKEND="${REWARD_BACKEND:-imagereward}"
IMAGE_REWARD_MODEL="${IMAGE_REWARD_MODEL:-ImageReward-v1.0}"
UNIFIEDREWARD_MODEL="${UNIFIEDREWARD_MODEL:-CodeGoat24/UnifiedReward-qwen-7b}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-1.0 1.0}"
REWARD_API_BASE="${REWARD_API_BASE:-}"
REWARD_API_KEY="${REWARD_API_KEY:-unifiedreward}"
REWARD_API_MODEL="${REWARD_API_MODEL:-UnifiedReward-7b-v1.5}"
REWARD_MAX_NEW_TOKENS="${REWARD_MAX_NEW_TOKENS:-512}"
REWARD_PROMPT_MODE="${REWARD_PROMPT_MODE:-standard}"

# Keep ImageReward inference independent from cluster wandb/protobuf drift.
SID_FORCE_WANDB_STUB="${SID_FORCE_WANDB_STUB:-1}"
WANDB_DISABLED="${WANDB_DISABLED:-true}"
export SID_FORCE_WANDB_STUB WANDB_DISABLED

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE does not exist: ${PROMPT_FILE}" >&2
  exit 1
fi
mkdir -p "${OUT_DIR}"

_DEPS_STAMP="${HOME}/.cache/sid_deps/reward_deps_ok_v2"
_stamp_deps() { mkdir -p "$(dirname "${_DEPS_STAMP}")" && touch "${_DEPS_STAMP}"; }

ensure_imagereward_runtime() {
  local backend_lc
  backend_lc="$(echo "${REWARD_BACKEND}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${backend_lc}" != "imagereward" && "${backend_lc}" != "auto" && "${backend_lc}" != "blend" ]]; then
    return 0
  fi
  if [[ "${FORCE_INSTALL_DEPS:-0}" != "1" ]] && [[ -f "${_DEPS_STAMP}" ]]; then return 0; fi
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import xxhash
import clip
try:
    import transformers.modeling_utils as _tmu
    if not hasattr(_tmu, "apply_chunking_to_forward"):
        def _acf(fn, cs, cd, *ts):
            if cs > 0:
                import torch; n = ts[0].shape[cd]
                return torch.cat([fn(*[t.narrow(cd,c*cs,cs) for t in ts]) for c in range(n//cs)],dim=cd)
            return fn(*ts)
        _tmu.apply_chunking_to_forward = _acf
    if not hasattr(_tmu, "find_pruneable_heads_and_indices"):
        import torch as _t
        def _fphai(heads, n, hs, pruned):
            mask=_t.ones(n,hs); heads=set(heads)-pruned
            for h in sorted(heads):
                h2=h-sum(1 for p in sorted(pruned) if p<h); mask[h2]=0
            mask=mask.view(-1).eq(1); return heads,_t.arange(len(mask))[mask].long()
        _tmu.find_pruneable_heads_and_indices=_fphai
    if not hasattr(_tmu, "prune_linear_layer"):
        import torch.nn as _nn
        def _pll(layer, index, dim=0):
            index=index.to(layer.weight.device)
            W=layer.weight.index_select(dim,index).clone().detach()
            b=layer.bias[index].clone().detach() if layer.bias is not None and dim==0 else (layer.bias.clone().detach() if layer.bias is not None else None)
            ns=list(layer.weight.size()); ns[dim]=len(index)
            nl=_nn.Linear(ns[1],ns[0],bias=layer.bias is not None).to(layer.weight.device)
            nl.weight.requires_grad=False; nl.weight.copy_(W.contiguous()); nl.weight.requires_grad=True
            if b is not None: nl.bias.requires_grad=False; nl.bias.copy_(b.contiguous()); nl.bias.requires_grad=True
            return nl
        _tmu.prune_linear_layer=_pll
except Exception:
    pass
import importlib.util as _iu
if _iu.find_spec("ImageReward") is None:
    raise RuntimeError("ImageReward module not found")
print(getattr(xxhash, '__version__', 'ok'), "ImageReward module found")
PY
  then
    _stamp_deps; return 0
  fi
  echo "[deps] ImageReward runtime deps missing. Installing with install_reward_deps.sh ..."
  PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/install_reward_deps.sh"
  _stamp_deps
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
if [[ -n "${REWARD_API_BASE}" ]]; then
  EXTRA_ARGS+=(--reward_api_base "${REWARD_API_BASE}")
fi

torchrun --standalone --nproc_per_node "${NUM_GPUS}" "${SCRIPT_DIR}/sd35_ddp_experiment.py" \
  --backend sid \
  --prompt_file "${PROMPT_FILE_ABS}" \
  --start_index "${START_INDEX:-0}" \
  --end_index "${END_INDEX:--1}" \
  --modes ${MODES} \
  --cfg_scales ${CFG_SCALES} \
  --baseline_cfg "${BASELINE_CFG}" \
  --steps "${STEPS}" \
  --n_variants "${N_VARIANTS:-3}" \
  --n_sims "${N_SIMS:-50}" \
  --ucb_c "${UCB_C:-1.41}" \
  --reward_backend "${REWARD_BACKEND}" \
  --image_reward_model "${IMAGE_REWARD_MODEL}" \
  --unifiedreward_model "${UNIFIEDREWARD_MODEL}" \
  --reward_model "${UNIFIEDREWARD_MODEL}" \
  --reward_weights ${REWARD_WEIGHTS} \
  --reward_api_key "${REWARD_API_KEY}" \
  --reward_api_model "${REWARD_API_MODEL}" \
  --reward_max_new_tokens "${REWARD_MAX_NEW_TOKENS}" \
  --reward_prompt_mode "${REWARD_PROMPT_MODE}" \
  --seed "${SEED}" \
  --out_dir "${OUT_DIR_ABS}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_sd35_ddp.py" --log_dir "${OUT_DIR_ABS}/logs" --out_dir "${OUT_DIR_ABS}"
