#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

NUM_GPUS="${NUM_GPUS:-$("${PYTHON_BIN}" - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)}"

PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/parti_prompts.txt}"
OUT_DIR="${OUT_DIR:-./sd35_cfg_ablation_out}"
REWARD_BACKEND="${REWARD_BACKEND:-unifiedreward}"
REWARD_MODEL="${REWARD_MODEL:-CodeGoat24/UnifiedReward-qwen-7b}"
UNIFIEDREWARD_MODEL="${UNIFIEDREWARD_MODEL:-${REWARD_MODEL}}"
IMAGE_REWARD_MODEL="${IMAGE_REWARD_MODEL:-ImageReward-v1.0}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-1.0 1.0}"
REWARD_API_BASE="${REWARD_API_BASE:-}"
REWARD_API_KEY="${REWARD_API_KEY:-unifiedreward}"
REWARD_API_MODEL="${REWARD_API_MODEL:-UnifiedReward-7b-v1.5}"
REWARD_MAX_NEW_TOKENS="${REWARD_MAX_NEW_TOKENS:-512}"
REWARD_PROMPT_MODE="${REWARD_PROMPT_MODE:-standard}"

MODES="${MODES:-base greedy mcts ga smc}"
CFG_SCALES="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5}"
BASELINE_CFG="${BASELINE_CFG:-1.0}"
CFG_ONLY="${CFG_ONLY:-0}"
RUN_CFG_ABLATION="${RUN_CFG_ABLATION:-0}"
CFG_ABLATION_VALUES="${CFG_ABLATION_VALUES:-${CFG_SCALES}}"
CFG_ABLATION_MODES="${CFG_ABLATION_MODES:-base}"
N_VARIANTS_DEFAULT="${N_VARIANTS:-3}"

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
if [[ "${SEED_PER_PROMPT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--seed_per_prompt)
fi
if [[ "${SAVE_IMAGES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save_images)
fi
if [[ "${SAVE_VARIANTS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save_variants)
fi
if [[ -n "${REWRITES_FILE:-}" && "${CFG_ONLY}" != "1" && "${RUN_CFG_ABLATION}" != "1" ]]; then
  EXTRA_ARGS+=(--rewrites_file "${REWRITES_FILE}")
fi
if [[ -n "${REWARD_API_BASE}" ]]; then
  EXTRA_ARGS+=(--reward_api_base "${REWARD_API_BASE}")
fi
if [[ "${NO_QWEN:-0}" == "1" || "${CFG_ONLY}" == "1" || "${RUN_CFG_ABLATION}" == "1" ]]; then
  EXTRA_ARGS+=(--no_qwen)
fi

if [[ "${CFG_ONLY}" == "1" || "${RUN_CFG_ABLATION}" == "1" ]]; then
  N_VARIANTS_EFFECTIVE="${CFG_ONLY_N_VARIANTS:-0}"
  EXTRA_ARGS+=(--correction_strengths 0.0)
  echo "[cfg-only] forcing single-prompt variant action space (n_variants=${N_VARIANTS_EFFECTIVE}, correction_strengths=[0.0])"
else
  N_VARIANTS_EFFECTIVE="${N_VARIANTS_DEFAULT}"
  if [[ -n "${CORRECTION_STRENGTHS:-}" ]]; then
    read -r -a corr_strengths_arr <<< "${CORRECTION_STRENGTHS}"
    EXTRA_ARGS+=(--correction_strengths "${corr_strengths_arr[@]}")
  fi
fi

CLI_EXTRA_ARGS=("$@")
COMMON_ARGS=(
  --backend sid
  --prompt_file "${PROMPT_FILE_ABS}"
  --start_index "${START_INDEX:-0}"
  --end_index "${END_INDEX:--1}"
  --steps "${STEPS:-4}"
  --n_variants "${N_VARIANTS_EFFECTIVE}"
  --n_sims "${N_SIMS:-50}"
  --ucb_c "${UCB_C:-1.41}"
  --smc_k "${SMC_K:-8}"
  --smc_gamma "${SMC_GAMMA:-0.10}"
  --ess_threshold "${ESS_THRESHOLD:-0.5}"
  --resample_start_frac "${RESAMPLE_START_FRAC:-0.3}"
  --smc_cfg_scale "${SMC_CFG_SCALE:-1.25}"
  --smc_variant_idx "${SMC_VARIANT_IDX:-0}"
  --reward_backend "${REWARD_BACKEND}"
  --reward_model "${REWARD_MODEL}"
  --unifiedreward_model "${UNIFIEDREWARD_MODEL}"
  --image_reward_model "${IMAGE_REWARD_MODEL}"
  --reward_weights ${REWARD_WEIGHTS}
  --reward_api_key "${REWARD_API_KEY}"
  --reward_api_model "${REWARD_API_MODEL}"
  --reward_max_new_tokens "${REWARD_MAX_NEW_TOKENS}"
  --reward_prompt_mode "${REWARD_PROMPT_MODE}"
  --seed "${SEED:-42}"
)

run_sd35_job() {
  local out_dir="$1"
  local baseline_cfg="$2"
  local cfg_values_str="$3"
  local modes_str="$4"
  local -a cfg_values_arr
  local -a modes_arr
  read -r -a cfg_values_arr <<< "${cfg_values_str}"
  read -r -a modes_arr <<< "${modes_str}"

  torchrun --standalone --nproc_per_node "${NUM_GPUS}" "${SCRIPT_DIR}/sd35_ddp_experiment.py" \
    "${COMMON_ARGS[@]}" \
    --modes "${modes_arr[@]}" \
    --cfg_scales "${cfg_values_arr[@]}" \
    --baseline_cfg "${baseline_cfg}" \
    --out_dir "${out_dir}" \
    "${EXTRA_ARGS[@]}" \
    "${CLI_EXTRA_ARGS[@]}"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_sd35_ddp.py" --log_dir "${out_dir}/logs" --out_dir "${out_dir}"
}

if [[ "${RUN_CFG_ABLATION}" == "1" ]]; then
  ABL_ROOT="${OUT_DIR_ABS}/cfg_ablation"
  mkdir -p "${ABL_ROOT}"
  read -r -a cfg_ablation_arr <<< "${CFG_ABLATION_VALUES}"
  if [[ "${#cfg_ablation_arr[@]}" -eq 0 ]]; then
    echo "Error: CFG_ABLATION_VALUES is empty." >&2
    exit 1
  fi
  echo "[cfg-ablation] cfg_values=[${CFG_ABLATION_VALUES}] modes=[${CFG_ABLATION_MODES}] out=${ABL_ROOT}"
  for cfg in "${cfg_ablation_arr[@]}"; do
    cfg_slug="$(echo "${cfg}" | sed 's/-/m/g; s/\./p/g')"
    run_dir="${ABL_ROOT}/cfg_${cfg_slug}"
    mkdir -p "${run_dir}"
    echo "[cfg-ablation] running cfg=${cfg} -> ${run_dir}"
    run_sd35_job "${run_dir}" "${cfg}" "${cfg}" "${CFG_ABLATION_MODES}"
  done
  "${PYTHON_BIN}" - <<'PY' "${ABL_ROOT}" "${CFG_ABLATION_VALUES}"
import json, os, sys
root = sys.argv[1]
cfg_vals = [x for x in sys.argv[2].split() if x]
rows = []
for cfg in cfg_vals:
    slug = cfg.replace("-", "m").replace(".", "p")
    agg = os.path.join(root, f"cfg_{slug}", "aggregate_summary.json")
    if not os.path.exists(agg):
        continue
    with open(agg, "r", encoding="utf-8") as f:
        obj = json.load(f)
    ms = obj.get("mode_stats", {})
    base = ms.get("base", {})
    rows.append((cfg, int(base.get("count", 0)), float(base.get("mean_score", 0.0))))
rows.sort(key=lambda x: float(x[0]))
out_tsv = os.path.join(root, "cfg_ablation_summary.tsv")
with open(out_tsv, "w", encoding="utf-8") as f:
    f.write("cfg\tcount\tmean_base_score\n")
    for cfg, count, score in rows:
        f.write(f"{cfg}\t{count}\t{score:.6f}\n")
print(f"[cfg-ablation] wrote {out_tsv}")
PY
else
  run_sd35_job "${OUT_DIR_ABS}" "${BASELINE_CFG}" "${CFG_SCALES}" "${MODES}"
fi
