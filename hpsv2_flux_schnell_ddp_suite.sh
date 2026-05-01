#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

DEFAULT_PROMPT_FILE="${SCRIPT_DIR}/hpsv2_subset.txt"
if [[ -f "${DEFAULT_PROMPT_FILE}" ]]; then
  PROMPT_FILE="${PROMPT_FILE:-${DEFAULT_PROMPT_FILE}}"
else
  PROMPT_FILE="${PROMPT_FILE:-/data/ygu/hpsv2_prompts.txt}"
fi
OUT_ROOT="${OUT_ROOT:-/data/ygu/hpsv2_flux_schnell_ddp}"
METHODS="${METHODS:-baseline greedy mcts ga}"
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-schnell}"
FLUX_BACKEND="${FLUX_BACKEND:-flux}"
FLUX_SIGMAS="${FLUX_SIGMAS:-}"

START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:--1}"
SAVE_FIRST_K="${SAVE_FIRST_K:-10}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"
SAVE_BEST_IMAGES="${SAVE_BEST_IMAGES:-1}"

STEPS="${STEPS:-4}"
SEED="${SEED:-42}"
N_SAMPLES="${N_SAMPLES:-1}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
BASELINE_GUIDANCE_SCALE="${BASELINE_GUIDANCE_SCALE:-1.0}"

REWARD_BACKEND="${REWARD_BACKEND:-imagereward}"
REWARD_DEVICE="${REWARD_DEVICE:-cpu}"
REWARD_MODEL="${REWARD_MODEL:-CodeGoat24/UnifiedReward-qwen-7b}"
UNIFIEDREWARD_MODEL="${UNIFIEDREWARD_MODEL:-${REWARD_MODEL}}"
IMAGE_REWARD_MODEL="${IMAGE_REWARD_MODEL:-ImageReward-v1.0}"
PICKSCORE_MODEL="${PICKSCORE_MODEL:-yuvalkirstain/PickScore_v1}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-1.0 1.0}"
REWARD_API_BASE="${REWARD_API_BASE:-}"
REWARD_API_KEY="${REWARD_API_KEY:-unifiedreward}"
REWARD_API_MODEL="${REWARD_API_MODEL:-UnifiedReward-7b-v1.5}"
REWARD_MAX_NEW_TOKENS="${REWARD_MAX_NEW_TOKENS:-512}"
REWARD_PROMPT_MODE="${REWARD_PROMPT_MODE:-standard}"
EVAL_BEST_IMAGES="${EVAL_BEST_IMAGES:-1}"
EVAL_BACKENDS="${EVAL_BACKENDS:-imagereward hpsv2 pickscore}"
EVAL_REWARD_DEVICE="${EVAL_REWARD_DEVICE:-cpu}"
EVAL_ALLOW_MISSING_BACKENDS="${EVAL_ALLOW_MISSING_BACKENDS:-0}"

# Keep ImageReward inference independent from cluster wandb/protobuf drift.
SID_FORCE_WANDB_STUB="${SID_FORCE_WANDB_STUB:-1}"
WANDB_DISABLED="${WANDB_DISABLED:-true}"
export SID_FORCE_WANDB_STUB WANDB_DISABLED

# ── Reward server (two-env architecture) ──────────────────────────────────────
# GPU allocation: the LAST visible GPU is reserved for the reward server.
# GPU_IDS array is trimmed so generation excludes that GPU.
USE_REWARD_SERVER="${USE_REWARD_SERVER:-0}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5100}"
REWARD_SERVER_BACKENDS="${REWARD_SERVER_BACKENDS:-hpsv3 imagereward}"
REWARD_ENV_NAME="${REWARD_ENV_NAME:-reward}"
REWARD_ENV_CONDA_BASE="${REWARD_ENV_CONDA_BASE:-/opt/conda}"
REWARD_SERVER_PID=""
REWARD_SERVER_GPU=""

start_reward_server() {
  if [[ "${USE_REWARD_SERVER}" != "1" ]]; then return 0; fi

  # GPU reservation happens later, after GPU_IDS is built (see _reserve_reward_gpu)
  local reward_py="${REWARD_ENV_CONDA_BASE}/envs/${REWARD_ENV_NAME}/bin/python"
  local fallback_py="${REWARD_ENV_CONDA_BASE}/envs/ptca/bin/python"
  if [[ ! -x "${reward_py}" ]]; then
    echo "[reward-server] Reward env not found at ${reward_py}. Setting up ..."
    if ! CONDA_BASE="${REWARD_ENV_CONDA_BASE}" REWARD_ENV_NAME="${REWARD_ENV_NAME}" \
      bash "${SCRIPT_DIR}/setup_reward_env.sh"; then
      echo "[reward-server] WARNING: setup_reward_env.sh failed; trying fallback python." >&2
    fi
    if [[ ! -x "${reward_py}" ]]; then
      if [[ -x "${fallback_py}" ]]; then
        reward_py="${fallback_py}"
        echo "[reward-server] Fallback to main env python: ${reward_py}"
      else
        echo "[reward-server] ERROR: no usable reward python found." >&2
        echo "[reward-server] Checked: ${REWARD_ENV_CONDA_BASE}/envs/${REWARD_ENV_NAME}/bin/python and ${fallback_py}" >&2
        ls -la "${REWARD_ENV_CONDA_BASE}/envs" >&2 || true
        exit 1
      fi
    fi
  fi

  echo "[reward-server] Starting on GPU ${REWARD_SERVER_GPU}, port ${REWARD_SERVER_PORT} ..."
  echo "[reward-server] Backends: ${REWARD_SERVER_BACKENDS}"
  CUDA_VISIBLE_DEVICES="${REWARD_SERVER_GPU}" \
    "${reward_py}" -u "${SCRIPT_DIR}/reward_server.py" \
    --port "${REWARD_SERVER_PORT}" \
    --device "cuda:0" \
    --backends ${REWARD_SERVER_BACKENDS} \
    --image_reward_model "${IMAGE_REWARD_MODEL}" \
    --pickscore_model "${PICKSCORE_MODEL}" \
    &>"${RUN_DIR}/reward_server.log" &
  REWARD_SERVER_PID="$!"
  echo "[reward-server] PID=${REWARD_SERVER_PID} log=${RUN_DIR}/reward_server.log"

  local max_wait=300
  local waited=0
  while (( waited < max_wait )); do
    if ! kill -0 "${REWARD_SERVER_PID}" 2>/dev/null; then
      echo "[reward-server] ERROR: server process died during startup." >&2
      echo "[reward-server] Last 30 lines of log:" >&2
      tail -30 "${RUN_DIR}/reward_server.log" >&2
      exit 1
    fi
    if curl -s "http://localhost:${REWARD_SERVER_PORT}/health" >/dev/null 2>&1; then
      local health
      health="$(curl -s "http://localhost:${REWARD_SERVER_PORT}/health")"
      echo "[reward-server] Healthy after ${waited}s — ${health}"
      break
    fi
    sleep 3
    waited=$(( waited + 3 ))
  done
  if (( waited >= max_wait )); then
    echo "[reward-server] ERROR: server not healthy after ${max_wait}s." >&2
    echo "[reward-server] Last 30 lines of log:" >&2
    tail -30 "${RUN_DIR}/reward_server.log" >&2
    kill "${REWARD_SERVER_PID}" 2>/dev/null || true
    exit 1
  fi

  export REWARD_SERVER_URL="http://localhost:${REWARD_SERVER_PORT}"
  echo "[reward-server] REWARD_SERVER_URL=${REWARD_SERVER_URL}"
}

stop_reward_server() {
  if [[ -n "${REWARD_SERVER_PID}" ]] && kill -0 "${REWARD_SERVER_PID}" 2>/dev/null; then
    echo "[reward-server] Stopping server PID=${REWARD_SERVER_PID} ..."
    kill "${REWARD_SERVER_PID}" 2>/dev/null || true
    wait "${REWARD_SERVER_PID}" 2>/dev/null || true
    REWARD_SERVER_PID=""
  fi
}

trap stop_reward_server EXIT

GA_POPULATION="${GA_POPULATION:-24}"
GA_GENERATIONS="${GA_GENERATIONS:-8}"
GA_ELITES="${GA_ELITES:-3}"
GA_MUTATION_PROB="${GA_MUTATION_PROB:-0.10}"
GA_TOURNAMENT_K="${GA_TOURNAMENT_K:-3}"
GA_SELECTION="${GA_SELECTION:-rank}"
GA_RANK_PRESSURE="${GA_RANK_PRESSURE:-1.7}"
GA_CROSSOVER="${GA_CROSSOVER:-uniform}"
GA_INIT_MODE="${GA_INIT_MODE:-random}"
GA_LOG_TOPK="${GA_LOG_TOPK:-3}"
GA_GUIDANCE_SCALES="${GA_GUIDANCE_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5}"
N_VARIANTS="${N_VARIANTS:-5}"
CFG_SCALES="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5}"
N_SIMS="${N_SIMS:-50}"
UCB_C="${UCB_C:-1.41}"
MCTS_FRESH_NOISE_STEPS="${MCTS_FRESH_NOISE_STEPS:-}"
MCTS_FRESH_NOISE_SAMPLES="${MCTS_FRESH_NOISE_SAMPLES:-1}"
MCTS_FRESH_NOISE_SCALE="${MCTS_FRESH_NOISE_SCALE:-1.0}"
NOISE_INJECT_MODE="${NOISE_INJECT_MODE:-combined}"
NOISE_INJECT_SEED_BUDGET="${NOISE_INJECT_SEED_BUDGET:-8}"
NOISE_INJECT_CANDIDATE_STEPS="${NOISE_INJECT_CANDIDATE_STEPS:-}"
NOISE_INJECT_GAMMA_BANK="${NOISE_INJECT_GAMMA_BANK:-0.0 0.25 0.5}"
NOISE_INJECT_EPS_SAMPLES="${NOISE_INJECT_EPS_SAMPLES:-4}"
NOISE_INJECT_STEPS_PER_ROLLOUT="${NOISE_INJECT_STEPS_PER_ROLLOUT:-1}"
NOISE_INJECT_INCLUDE_NO_INJECT="${NOISE_INJECT_INCLUDE_NO_INJECT:-1}"
NOISE_INJECT_MAX_POLICIES="${NOISE_INJECT_MAX_POLICIES:-0}"
NOISE_INJECT_VARIANT_IDX="${NOISE_INJECT_VARIANT_IDX:-0}"
NOISE_INJECT_GUIDANCE="${NOISE_INJECT_GUIDANCE:-}"

SMC_K="${SMC_K:-12}"
SMC_GAMMA="${SMC_GAMMA:-0.10}"
SMC_POTENTIAL="${SMC_POTENTIAL:-tempering}"
SMC_LAMBDA="${SMC_LAMBDA:-10.0}"
SMC_GUIDANCE_SCALE="${SMC_GUIDANCE_SCALE:-1.25}"
SMC_CHUNK="${SMC_CHUNK:-4}"

BON_N="${BON_N:-16}"
BEAM_WIDTH="${BEAM_WIDTH:-4}"
BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS:-8}"
BON_MCTS_TOPK="${BON_MCTS_TOPK:-2}"
BON_MCTS_SEED_STRIDE="${BON_MCTS_SEED_STRIDE:-1}"
BON_MCTS_SEED_OFFSET="${BON_MCTS_SEED_OFFSET:-0}"
BON_MCTS_SIM_ALLOC="${BON_MCTS_SIM_ALLOC:-split}"
BON_MCTS_MIN_SIMS="${BON_MCTS_MIN_SIMS:-8}"
BON_MCTS_PRESCREEN_GUIDANCE="${BON_MCTS_PRESCREEN_GUIDANCE:-${BON_MCTS_PRESCREEN_CFG:-}}"

DTS_M_ITER="${DTS_M_ITER:-64}"
DTS_LAMBDA="${DTS_LAMBDA:-1.0}"
DTS_PW_C="${DTS_PW_C:-1.0}"
DTS_PW_ALPHA="${DTS_PW_ALPHA:-0.5}"
DTS_C_UCT="${DTS_C_UCT:-1.0}"
DTS_SDE_NOISE_SCALE="${DTS_SDE_NOISE_SCALE:-0.0}"
DTS_CFG_BANK="${DTS_CFG_BANK:-}"

_DEFAULT_GUIDANCE_SCALES_STR="1.0 1.25 1.5 1.75 2.0 2.25 2.5"
if [[ "${FLUX_BACKEND}" == "senseflow_flux" ]]; then
  if [[ "${CFG_SCALES}" == "${_DEFAULT_GUIDANCE_SCALES_STR}" ]]; then
    CFG_SCALES="0.0"
  fi
  if [[ "${GA_GUIDANCE_SCALES}" == "${_DEFAULT_GUIDANCE_SCALES_STR}" ]]; then
    GA_GUIDANCE_SCALES="0.0"
  fi
  if [[ "${BASELINE_GUIDANCE_SCALE}" == "1.0" ]]; then
    BASELINE_GUIDANCE_SCALE="0.0"
  fi
  if [[ "${SMC_GUIDANCE_SCALE}" == "1.25" ]]; then
    SMC_GUIDANCE_SCALE="0.0"
  fi
  if [[ -z "${FLUX_SIGMAS}" ]]; then
    FLUX_SIGMAS="1.0 0.75 0.5 0.25"
  fi
fi

NUM_GPUS="${NUM_GPUS:-0}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found: ${PROMPT_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"
chmod -R u+rwX "${OUT_ROOT}" 2>/dev/null || true
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_ROOT}/run_${RUN_TS}"
mkdir -p "${RUN_DIR}"
SUITE_TSV="${RUN_DIR}/suite_summary.tsv"

GPU_IDS_STR="$("${PYTHON_BIN}" - <<'PY'
import os
import torch
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if cvd:
    ids = [x.strip() for x in cvd.split(",") if x.strip()]
else:
    ids = [str(i) for i in range(torch.cuda.device_count())]
print(",".join(ids))
PY
)"
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "Error: no visible GPUs. Set CUDA_VISIBLE_DEVICES or check CUDA runtime." >&2
  exit 1
fi
if (( NUM_GPUS <= 0 || NUM_GPUS > ${#GPU_IDS[@]} )); then
  NUM_GPUS="${#GPU_IDS[@]}"
fi
GPU_IDS=("${GPU_IDS[@]:0:${NUM_GPUS}}")

# ── Reserve last GPU for reward server ──────────────────────────────────────
if [[ "${USE_REWARD_SERVER}" == "1" ]]; then
  if (( ${#GPU_IDS[@]} >= 2 )); then
    REWARD_SERVER_GPU="${GPU_IDS[$((${#GPU_IDS[@]}-1))]}"
    GPU_IDS=("${GPU_IDS[@]:0:$((${#GPU_IDS[@]}-1))}")
    NUM_GPUS="${#GPU_IDS[@]}"
    echo "[reward-server] Reserved GPU ${REWARD_SERVER_GPU} for reward server"
    echo "[reward-server] Generation GPUs (${NUM_GPUS}): ${GPU_IDS[*]}"
  else
    REWARD_SERVER_GPU="${GPU_IDS[0]}"
    echo "[reward-server] Only 1 GPU available — sharing it between reward server and generation."
  fi
fi

read -r TOTAL_PROMPTS EFFECTIVE_END <<EOF
$("${PYTHON_BIN}" - <<'PY' "${PROMPT_FILE}" "${START_INDEX}" "${END_INDEX}"
import sys
path = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])
with open(path, encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]
total = len(prompts)
if end < 0:
    end = total
end = min(end, total)
start = max(0, min(start, end))
print(total, end)
PY
)
EOF
RANGE_TOTAL=$(( EFFECTIVE_END - START_INDEX ))
if (( RANGE_TOTAL <= 0 )); then
  echo "Error: empty prompt range start=${START_INDEX} end=${EFFECTIVE_END}" >&2
  exit 1
fi

echo "FLUX schnell DDP suite"
echo "  prompt_file: ${PROMPT_FILE}"
echo "  prompts_total: ${TOTAL_PROMPTS} selected: ${RANGE_TOTAL} range=[${START_INDEX},${EFFECTIVE_END})"
echo "  gpus(${NUM_GPUS}): ${GPU_IDS[*]}"
echo "  flux_backend: ${FLUX_BACKEND} flux_sigmas: ${FLUX_SIGMAS:-<none>}"
if [[ "${MCTS_FRESH_NOISE_SAMPLES}" != "1" || -n "${MCTS_FRESH_NOISE_STEPS}" ]]; then
  echo "  fresh_noise: steps='${MCTS_FRESH_NOISE_STEPS}' samples=${MCTS_FRESH_NOISE_SAMPLES} scale=${MCTS_FRESH_NOISE_SCALE}"
fi
if [[ "${METHODS}" == *"noise"* || "${METHODS}" == *"noiseinj"* || "${METHODS}" == *"noise_inject"* ]]; then
  echo "  noise_inject: mode=${NOISE_INJECT_MODE} seed_budget=${NOISE_INJECT_SEED_BUDGET} steps='${NOISE_INJECT_CANDIDATE_STEPS}'"
  echo "               gammas=[${NOISE_INJECT_GAMMA_BANK}] eps_samples=${NOISE_INJECT_EPS_SAMPLES} steps_per_rollout=${NOISE_INJECT_STEPS_PER_ROLLOUT} cfg_scales='${CFG_SCALES}'"
fi
if [[ "${METHODS}" == *"bon_mcts"* ]]; then
  echo "  bon_mcts: prescreen_n=${BON_MCTS_N_SEEDS} topk=${BON_MCTS_TOPK} sim_alloc=${BON_MCTS_SIM_ALLOC} min_sims=${BON_MCTS_MIN_SIMS}"
fi
if [[ "${METHODS}" == *"dts"* ]]; then
  echo "  dts: M=${DTS_M_ITER} lam=${DTS_LAMBDA} pw=(C=${DTS_PW_C},a=${DTS_PW_ALPHA}) c_uct=${DTS_C_UCT} sde=${DTS_SDE_NOISE_SCALE} cfg_bank='${DTS_CFG_BANK}'"
fi
echo "  reward_backend: ${REWARD_BACKEND} reward_device: ${REWARD_DEVICE}"
echo "  eval_best_images: ${EVAL_BEST_IMAGES} eval_backends: ${EVAL_BACKENDS} eval_device: ${EVAL_REWARD_DEVICE}"
echo "  out: ${RUN_DIR}"

eval_backend_requested() {
  local target="${1,,}"
  local b
  for b in ${EVAL_BACKENDS}; do
    if [[ "${b,,}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

_DEPS_STAMP="${HOME}/.cache/sid_deps/reward_deps_ok_v2"
_stamp_deps() { mkdir -p "$(dirname "${_DEPS_STAMP}")" && touch "${_DEPS_STAMP}"; }

ensure_imagereward_runtime() {
  # External reward-server mode: reward backends run in a separate process/env.
  if [[ "${USE_REWARD_SERVER}" != "1" ]] && [[ -n "${REWARD_SERVER_URL:-}" ]]; then
    return 0
  fi
  local backend_lc
  backend_lc="$(echo "${REWARD_BACKEND}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${backend_lc}" != "imagereward" && "${backend_lc}" != "auto" && "${backend_lc}" != "blend" && "${backend_lc}" != "all" ]] && ! eval_backend_requested "imagereward"; then
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

ensure_pickscore_runtime() {
  # External reward-server mode: reward backends run in a separate process/env.
  if [[ "${USE_REWARD_SERVER}" != "1" ]] && [[ -n "${REWARD_SERVER_URL:-}" ]]; then
    return 0
  fi
  local backend_lc
  backend_lc="$(echo "${REWARD_BACKEND}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${backend_lc}" != "pickscore" && "${backend_lc}" != "auto" && "${backend_lc}" != "all" ]] && ! eval_backend_requested "pickscore"; then
    return 0
  fi
  if [[ "${FORCE_INSTALL_DEPS:-0}" != "1" ]] && [[ -f "${_DEPS_STAMP}" ]]; then return 0; fi
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import timm
from timm.data import ImageNetInfo
print(timm.__version__, ImageNetInfo.__name__)
PY
  then
    _stamp_deps; return 0
  fi
  echo "[deps] PickScore runtime deps missing/incompatible (timm ImageNetInfo). Installing with install_reward_deps.sh ..."
  PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/install_reward_deps.sh"
  _stamp_deps
}

ensure_pickscore_runtime

ensure_hpsv2_runtime() {
  # External reward-server mode: reward backends run in a separate process/env.
  if [[ "${USE_REWARD_SERVER}" != "1" ]] && [[ -n "${REWARD_SERVER_URL:-}" ]]; then
    return 0
  fi
  local backend_lc
  backend_lc="$(echo "${REWARD_BACKEND}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${backend_lc}" != "hpsv2" && "${backend_lc}" != "auto" && "${backend_lc}" != "blend" && "${backend_lc}" != "all" ]] && ! eval_backend_requested "hpsv2"; then
    return 0
  fi
  if [[ "${FORCE_INSTALL_DEPS:-0}" != "1" ]] && [[ -f "${_DEPS_STAMP}" ]]; then return 0; fi
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import hpsv2
print(getattr(hpsv2, "__file__", "ok"))
PY
  then
    return 0
  fi
  echo "[deps] HPSv2 missing. Installing hpsv2 ..."
  "${PYTHON_BIN}" -m pip install --no-cache-dir "hpsv2" || true
}

ensure_hpsv2_runtime

ensure_hpsv3_runtime() {
  # External reward-server mode: reward backends run in a separate process/env.
  if [[ "${USE_REWARD_SERVER}" != "1" ]] && [[ -n "${REWARD_SERVER_URL:-}" ]]; then
    return 0
  fi
  local backend_lc
  backend_lc="$(echo "${REWARD_BACKEND}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${backend_lc}" != "hpsv3" && "${backend_lc}" != "auto" && "${backend_lc}" != "all" ]] && ! eval_backend_requested "hpsv3"; then
    return 0
  fi
  if [[ "${FORCE_INSTALL_DEPS:-0}" != "1" ]] && [[ -f "${_DEPS_STAMP}" ]]; then return 0; fi
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import hpsv3
import omegaconf
import hydra
print(getattr(hpsv3, "__file__", "ok"), getattr(omegaconf, "__version__", "ok"), getattr(hydra, "__version__", "ok"))
PY
  then
    _stamp_deps; return 0
  fi
  echo "[deps] HPSv3 runtime deps missing. Installing with install_reward_deps.sh ..."
  PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/install_reward_deps.sh"
  _stamp_deps
}

ensure_hpsv3_runtime

ensure_xformers_runtime() {
  local xf_ver
  xf_ver="${XFORMERS_VERSION:-0.0.31.post1}"
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import xformers
import xformers.ops
print(xformers.__version__)
PY
  then
    return 0
  fi

  echo "[deps] xformers import failed (ABI mismatch likely). Reinstalling xformers==${xf_ver} ..."
  if "${PYTHON_BIN}" -m pip install --no-cache-dir --force-reinstall --no-deps "xformers==${xf_ver}"; then
    if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import xformers
import xformers.ops
print(xformers.__version__)
PY
    then
      return 0
    fi
  fi

  echo "[deps] xformers still broken; uninstalling xformers to force non-xformers diffusers path."
  "${PYTHON_BIN}" -m pip uninstall -y xformers || true
}

ensure_xformers_runtime

post_eval_best_images() {
  local method_out="$1"
  local method_name="$2"
  if [[ "${EVAL_BEST_IMAGES}" != "1" ]]; then
    return 0
  fi
  if [[ "${SAVE_FIRST_K}" == "0" ]]; then
    echo "[post-eval] skip method=${method_name}: SAVE_FIRST_K=0 (no images expected)"
    return 0
  fi
  if [[ "${SAVE_IMAGES}" != "1" && "${SAVE_BEST_IMAGES}" != "1" ]]; then
    return 0
  fi
  local -a cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_best_images_multi_reward.py"
    --layout flux
    --method_out "${method_out}"
    --method "${method_name}"
    --backends ${EVAL_BACKENDS}
    --reward_device "${EVAL_REWARD_DEVICE}"
    --image_reward_model "${IMAGE_REWARD_MODEL}"
    --pickscore_model "${PICKSCORE_MODEL}"
    --unifiedreward_model "${UNIFIEDREWARD_MODEL}"
    --reward_api_base "${REWARD_API_BASE}"
    --reward_api_key "${REWARD_API_KEY}"
    --reward_api_model "${REWARD_API_MODEL}"
    --reward_max_new_tokens "${REWARD_MAX_NEW_TOKENS}"
    --reward_prompt_mode "${REWARD_PROMPT_MODE}"
    --out_json "${method_out}/best_images_multi_reward.json"
    --out_aggregate "${method_out}/best_images_multi_reward_aggregate.json"
  )
  if [[ "${EVAL_ALLOW_MISSING_BACKENDS}" == "1" ]]; then
    cmd+=(--allow_missing_backends)
  fi
  "${cmd[@]}"
}

append_method_summary() {
  local method_out="$1"
  local method_name="$2"
  local elapsed_sec="$3"
  "${PYTHON_BIN}" - <<'PY' "${method_out}" "${method_name}" "${elapsed_sec}" "${SUITE_TSV}" "${EVAL_BACKENDS}"
import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

method_out = sys.argv[1]
method = sys.argv[2]
elapsed = int(sys.argv[3])
suite_tsv = sys.argv[4]
eval_backends = [x for x in str(sys.argv[5]).split() if x]

baseline = []
search = []
deltas = []
nfe_by_gen = defaultdict(list)

for summary_path in glob.glob(os.path.join(method_out, "rank_*", "summary.json")):
    with open(summary_path, encoding="utf-8") as f:
        rows = json.load(f)
    for row in rows:
        for sample in row.get("samples", []):
            b = float(sample.get("baseline_score", 0.0))
            baseline.append(b)
            if method == "baseline":
                search.append(b)
                deltas.append(0.0)
            else:
                s = float(sample.get("search_score", b))
                d = float(sample.get("delta_score", s - b))
                search.append(s)
                deltas.append(d)

            diag = sample.get("diagnostics", {})
            for gen_row in diag.get("history", []):
                gen = int(gen_row.get("generation", 0)) + 1
                nfe = float(gen_row.get("nfe_per_generation", 0))
                nfe_by_gen[gen].append(nfe)

if not baseline:
    raise RuntimeError(f"No samples found under {method_out}")

mean_baseline = float(statistics.fmean(baseline))
mean_search = float(statistics.fmean(search))
mean_delta = float(statistics.fmean(deltas))

aggregate = {
    "method": method,
    "elapsed_sec": elapsed,
    "num_samples": len(baseline),
    "mean_baseline_score": mean_baseline,
    "mean_search_score": mean_search,
    "mean_delta_score": mean_delta,
}
with open(os.path.join(method_out, "aggregate_ddp.json"), "w", encoding="utf-8") as f:
    json.dump(aggregate, f, indent=2)

eval_means = {b: "" for b in eval_backends}
eval_agg_path = os.path.join(method_out, "best_images_multi_reward_aggregate.json")
if os.path.exists(eval_agg_path):
    with open(eval_agg_path, encoding="utf-8") as f:
        eval_agg = json.load(f)
    stats = eval_agg.get("backend_stats", {})
    for b in eval_backends:
        mean_val = stats.get(b, {}).get("mean")
        if mean_val is not None:
            eval_means[b] = f"{float(mean_val):.6f}"

if nfe_by_gen:
    with open(os.path.join(method_out, "ga_nfe_per_generation.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "mean_nfe_per_generation", "num_samples"])
        for gen in sorted(nfe_by_gen):
            vals = nfe_by_gen[gen]
            writer.writerow([gen, f"{statistics.fmean(vals):.6f}", len(vals)])

need_header = (not os.path.exists(suite_tsv)) or os.path.getsize(suite_tsv) == 0
with open(suite_tsv, "a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    if need_header:
        writer.writerow(
            ["method", "elapsed_sec", "num_samples", "mean_baseline", "mean_search", "mean_delta", "note"]
            + [f"eval_{b}" for b in eval_backends]
        )
    writer.writerow(
        [method, elapsed, len(baseline), f"{mean_baseline:.6f}", f"{mean_search:.6f}", f"{mean_delta:+.6f}", ""]
        + [eval_means[b] for b in eval_backends]
    )
PY
}

run_flux_sharded() {
  local method_name="$1"
  local flux_search_method="$2"
  local runner_script="${SCRIPT_DIR}/sampling_flux_unified.py"
  shift 2
  if (( $# >= 2 )) && [[ "$1" == "--runner_script" ]]; then
    runner_script="$2"
    shift 2
  fi
  local -a method_args=("$@")
  local -a runner_extra=()
  if [[ "${runner_script}" == "${SCRIPT_DIR}/sampling_flux_bon_mcts.py" ]]; then
    runner_extra+=(
      --bon_mcts_n_seeds "${BON_MCTS_N_SEEDS}"
      --bon_mcts_topk "${BON_MCTS_TOPK}"
      --bon_mcts_seed_stride "${BON_MCTS_SEED_STRIDE}"
      --bon_mcts_seed_offset "${BON_MCTS_SEED_OFFSET}"
      --bon_mcts_sim_alloc "${BON_MCTS_SIM_ALLOC}"
      --bon_mcts_min_sims "${BON_MCTS_MIN_SIMS}"
    )
    if [[ -n "${BON_MCTS_PRESCREEN_GUIDANCE}" ]]; then
      runner_extra+=(--bon_mcts_prescreen_guidance "${BON_MCTS_PRESCREEN_GUIDANCE}")
    fi
  fi
  local method_out="${RUN_DIR}/${method_name}"
  local method_logs="${method_out}/logs"
  mkdir -p "${method_out}" "${method_logs}"
  local -a reward_extra=()
  local -a backend_extra=(--backend "${FLUX_BACKEND}")
  if [[ -n "${FLUX_SIGMAS}" ]]; then
    backend_extra+=(--sigmas ${FLUX_SIGMAS})
  fi
  # Do NOT pass SENSEFLOW_LOCAL_DIR as --transformer_id.  The backend config
  # already sets transformer_id="domiso/SenseFlow" with the correct subfolder.
  # from_pretrained resolves the HF repo ID from cache in offline mode.
  if [[ -n "${REWARD_API_BASE}" ]]; then
    reward_extra+=(--reward_api_base "${REWARD_API_BASE}")
  fi

  local begin_ts
  begin_ts="$(date +%s)"
  echo "[$(date '+%F %T')] method=${method_name} (flux=${flux_search_method}) start"

  local chunk_size=$(( (RANGE_TOTAL + NUM_GPUS - 1) / NUM_GPUS ))
  local -a pids=()
  local launched=0

  for rank in "${!GPU_IDS[@]}"; do
    local shard_start=$(( START_INDEX + rank * chunk_size ))
    local shard_end=$(( shard_start + chunk_size ))
    if (( shard_start >= EFFECTIVE_END )); then
      continue
    fi
    if (( shard_end > EFFECTIVE_END )); then
      shard_end="${EFFECTIVE_END}"
    fi

    local rank_out="${method_out}/rank_${rank}"
    local rank_prompt="${method_out}/prompts_rank_${rank}.txt"
    local log_file="${method_logs}/rank_${rank}.log"
    local gpu="${GPU_IDS[$rank]}"
    mkdir -p "${rank_out}"

    "${PYTHON_BIN}" - <<'PY' "${PROMPT_FILE}" "${rank_prompt}" "${shard_start}" "${shard_end}"
import sys
src, dst, start, end = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(src, encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]
subset = prompts[start:end]
with open(dst, "w", encoding="utf-8") as f:
    for line in subset:
        f.write(line + "\n")
PY

    launched=$((launched + 1))
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${runner_script}" \
      --search_method "${flux_search_method}" \
      "${backend_extra[@]}" \
      --model_id "${MODEL_ID}" \
      --prompt_file "${rank_prompt}" \
      --n_prompts -1 \
      --n_samples "${N_SAMPLES}" \
      --steps "${STEPS}" \
      --width "${WIDTH}" \
      --height "${HEIGHT}" \
      --seed "${SEED}" \
      --dtype bf16 \
      --device cuda \
      --auto_select_gpu \
      --reward_backend "${REWARD_BACKEND}" \
      --reward_device "${REWARD_DEVICE}" \
      --reward_model "${REWARD_MODEL}" \
      --unifiedreward_model "${UNIFIEDREWARD_MODEL}" \
      --image_reward_model "${IMAGE_REWARD_MODEL}" \
      --pickscore_model "${PICKSCORE_MODEL}" \
      --reward_weights ${REWARD_WEIGHTS} \
      --reward_api_key "${REWARD_API_KEY}" \
      --reward_api_model "${REWARD_API_MODEL}" \
      --reward_max_new_tokens "${REWARD_MAX_NEW_TOKENS}" \
      --reward_prompt_mode "${REWARD_PROMPT_MODE}" \
      "${reward_extra[@]}" \
      --offload_text_encoder_after_encode \
      --decode_device auto \
      --decode_cpu_if_free_below_gb 16 \
      --empty_cache_after_decode \
      --baseline_guidance_scale "${BASELINE_GUIDANCE_SCALE}" \
      --save_first_k "${SAVE_FIRST_K}" \
      --out_dir "${rank_out}" \
      "${runner_extra[@]}" \
      "${method_args[@]}" \
      >"${log_file}" 2>&1 &
    pids+=("$!")
    echo "  rank=${rank} gpu=${gpu} range=[${shard_start},${shard_end}) log=${log_file}"
  done

  if (( launched == 0 )); then
    echo "Error: no shards launched for method=${method_name}." >&2
    exit 1
  fi

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed != 0 )); then
    echo "Error: method=${method_name} failed on at least one shard." >&2
    exit 1
  fi

  local end_ts
  end_ts="$(date +%s)"
  local elapsed=$(( end_ts - begin_ts ))
  echo "[$(date '+%F %T')] method=${method_name} done elapsed=${elapsed}s"

  post_eval_best_images "${method_out}" "${method_name}"
  append_method_summary "${method_out}" "${method_name}" "${elapsed}"
}

# ── Start reward server (if two-env mode) ──────────────────────────────────────
start_reward_server

# ── Downgrade transformers for reward model compatibility ────────────────────
# ImageReward and hpsv3 need transformers<=4.46 (removed APIs in newer versions).
# Skip the downgrade when using the reward server — the reward env has its own
# transformers==4.45.2, and the main env can keep transformers>=4.51.
if [[ "${USE_REWARD_SERVER}" == "1" ]] || [[ -n "${REWARD_SERVER_URL:-}" ]]; then
  echo "[deps] Skipping transformers downgrade — reward server handles reward model deps."
elif [[ "${DOWNGRADE_TRANSFORMERS:-1}" == "1" ]]; then
  echo "[deps] downgrading transformers to 4.45.2 for reward model compat ..."
  # Use --no-deps to avoid pip cascading downgrades that break torch/torchvision
  "${PYTHON_BIN}" -m pip install --no-cache-dir --no-deps \
    "transformers==4.45.2" "tokenizers==0.20.3" "trl==0.12.2" \
    "huggingface-hub==0.36.2" 2>&1 | tail -5
  "${PYTHON_BIN}" -c "import transformers, trl, huggingface_hub; print(f'[deps] transformers={transformers.__version__} trl={trl.__version__} huggingface_hub={huggingface_hub.__version__}')"
  # Verify torch/torchvision still work
  "${PYTHON_BIN}" -c "import torch, torchvision; print(f'[deps] torch={torch.__version__} torchvision={torchvision.__version__}')"
fi

for method in ${METHODS}; do
  case "${method}" in
    baseline)
      # Baseline is always computed in each FLUX run; use a cheap SMC pass to collect baseline stats only.
      run_flux_sharded "baseline" "smc" \
        --smc_k 2 \
        --smc_gamma 0.0 \
        --smc_guidance_scale "${BASELINE_GUIDANCE_SCALE}" \
        --smc_chunk 2
      ;;
    ga)
      run_flux_sharded "ga" "ga" \
        --ga_population "${GA_POPULATION}" \
        --ga_generations "${GA_GENERATIONS}" \
        --ga_elites "${GA_ELITES}" \
        --ga_mutation_prob "${GA_MUTATION_PROB}" \
        --ga_tournament_k "${GA_TOURNAMENT_K}" \
        --ga_selection "${GA_SELECTION}" \
        --ga_rank_pressure "${GA_RANK_PRESSURE}" \
        --ga_crossover "${GA_CROSSOVER}" \
        --ga_init_mode "${GA_INIT_MODE}" \
        --ga_log_topk "${GA_LOG_TOPK}" \
        --ga_phase_constraints \
        --ga_guidance_scales ${GA_GUIDANCE_SCALES}
      ;;
    greedy)
      # Greedy scores every (variant × cfg) combination at every step, so the
      # search space must be small.  Default to 3 representative CFG values;
      # override via GREEDY_CFG_SCALES env var.
      _GREEDY_CFG="${GREEDY_CFG_SCALES:-1.0 1.5 2.0}"
      _GREEDY_VARIANTS="${GREEDY_N_VARIANTS:-${N_VARIANTS}}"
      run_flux_sharded "greedy" "greedy" \
        --n_variants "${_GREEDY_VARIANTS}" \
        --cfg_scales ${_GREEDY_CFG}
      ;;
    mcts)
      run_flux_sharded "mcts" "mcts" \
        --n_variants "${N_VARIANTS}" \
        --cfg_scales ${CFG_SCALES} \
        --n_sims "${N_SIMS}" \
        --ucb_c "${UCB_C}" \
        --mcts_fresh_noise_steps "${MCTS_FRESH_NOISE_STEPS}" \
        --mcts_fresh_noise_samples "${MCTS_FRESH_NOISE_SAMPLES}" \
        --mcts_fresh_noise_scale "${MCTS_FRESH_NOISE_SCALE}"
      ;;
    noise|noiseinj|noise_inject)
      noise_args=(
        --cfg_scales ${CFG_SCALES}
        --noise_inject_mode "${NOISE_INJECT_MODE}"
        --noise_inject_seed_budget "${NOISE_INJECT_SEED_BUDGET}"
        --noise_inject_gamma_bank ${NOISE_INJECT_GAMMA_BANK}
        --noise_inject_eps_samples "${NOISE_INJECT_EPS_SAMPLES}"
        --noise_inject_steps_per_rollout "${NOISE_INJECT_STEPS_PER_ROLLOUT}"
        --noise_inject_max_policies "${NOISE_INJECT_MAX_POLICIES}"
        --noise_inject_variant_idx "${NOISE_INJECT_VARIANT_IDX}"
      )
      if [[ "${NOISE_INJECT_INCLUDE_NO_INJECT}" != "1" ]]; then
        noise_args+=(--no-noise_inject_include_no_inject)
      else
        noise_args+=(--noise_inject_include_no_inject)
      fi
      if [[ -n "${NOISE_INJECT_CANDIDATE_STEPS}" ]]; then
        noise_args+=(--noise_inject_candidate_steps "${NOISE_INJECT_CANDIDATE_STEPS}")
      fi
      if [[ -n "${NOISE_INJECT_GUIDANCE}" ]]; then
        noise_args+=(--noise_inject_guidance "${NOISE_INJECT_GUIDANCE}")
      fi
      run_flux_sharded "noise" "noise_inject" "${noise_args[@]}"
      ;;
    smc)
      smc_args=(
        --smc_k "${SMC_K}"
        --smc_gamma "${SMC_GAMMA}"
        --smc_potential "${SMC_POTENTIAL}"
        --smc_lambda "${SMC_LAMBDA}"
        --smc_guidance_scale "${SMC_GUIDANCE_SCALE}"
        --smc_chunk "${SMC_CHUNK}"
      )
      if [[ "${SMC_VARIANT_EXPANSION:-0}" == "1" ]]; then
        smc_args+=(--smc_variant_expansion)
        if [[ -n "${SMC_EXPANSION_VARIANTS:-}" ]]; then
          smc_args+=(--smc_expansion_variants ${SMC_EXPANSION_VARIANTS})
        fi
        if [[ -n "${SMC_EXPANSION_GUIDANCES:-}" ]]; then
          smc_args+=(--smc_expansion_guidances ${SMC_EXPANSION_GUIDANCES})
        fi
        if [[ -n "${SMC_EXPANSION_FACTOR:-}" ]]; then
          smc_args+=(--smc_expansion_factor "${SMC_EXPANSION_FACTOR}")
        fi
        if [[ -n "${SMC_EXPANSION_PROPOSAL:-}" ]]; then
          smc_args+=(--smc_expansion_proposal "${SMC_EXPANSION_PROPOSAL}")
        fi
        if [[ -n "${SMC_EXPANSION_TAU:-}" ]]; then
          smc_args+=(--smc_expansion_tau "${SMC_EXPANSION_TAU}")
        fi
        if [[ "${SMC_EXPANSION_LOOKAHEAD:-0}" == "1" ]]; then
          smc_args+=(--smc_expansion_lookahead)
        fi
      fi
      run_flux_sharded "smc" "smc" "${smc_args[@]}"
      ;;
    bon)
      run_flux_sharded "bon" "bon" \
        --bon_n "${BON_N}"
      ;;
    bon_mcts)
      run_flux_sharded "bon_mcts" "mcts" \
        --runner_script "${SCRIPT_DIR}/sampling_flux_bon_mcts.py" \
        --n_variants "${N_VARIANTS}" \
        --cfg_scales ${CFG_SCALES} \
        --n_sims "${N_SIMS}" \
        --ucb_c "${UCB_C}" \
        --mcts_fresh_noise_steps "${MCTS_FRESH_NOISE_STEPS}" \
        --mcts_fresh_noise_samples "${MCTS_FRESH_NOISE_SAMPLES}" \
        --mcts_fresh_noise_scale "${MCTS_FRESH_NOISE_SCALE}"
      ;;
    dts|dts_star)
      run_flux_sharded "${method}" "${method}" \
        --runner_script "${SCRIPT_DIR}/sampling_flux_unified_dts.py" \
        --n_variants 1 \
        --cfg_scales ${CFG_SCALES} \
        --dts_m_iter "${DTS_M_ITER}" \
        --dts_lambda "${DTS_LAMBDA}" \
        --dts_pw_c "${DTS_PW_C}" \
        --dts_pw_alpha "${DTS_PW_ALPHA}" \
        --dts_c_uct "${DTS_C_UCT}" \
        --dts_sde_noise_scale "${DTS_SDE_NOISE_SCALE}" \
        --dts_cfg_bank "${DTS_CFG_BANK}"
      ;;
    beam)
      run_flux_sharded "beam" "beam" \
        --beam_width "${BEAM_WIDTH}" \
        --n_variants "${N_VARIANTS}" \
        --cfg_scales ${CFG_SCALES}
      ;;
    dynamic_cfg_x0)
      dyncfg_args=(
        --runner_script "${SCRIPT_DIR}/sampling_flux_unified_dynamic_cfg_x0.py"
        --n_variants 1
        --cfg_scales ${CFG_SCALES}
        --dynamic_cfg_x0
        --dynamic_cfg_x0_cfg_grid ${DYNAMIC_CFG_X0_GRID:-1.5 2.0 2.5 3.0 3.5}
        --dynamic_cfg_x0_score_start_frac "${DYNAMIC_CFG_X0_SCORE_START_FRAC:-0.25}"
        --dynamic_cfg_x0_score_end_frac "${DYNAMIC_CFG_X0_SCORE_END_FRAC:-1.0}"
        --dynamic_cfg_x0_score_every "${DYNAMIC_CFG_X0_SCORE_EVERY:-2}"
        --dynamic_cfg_x0_evaluators ${DYNAMIC_CFG_X0_EVALUATORS:-imagereward hpsv3}
        --dynamic_cfg_x0_weight_schedule "${DYNAMIC_CFG_X0_WEIGHT_SCHEDULE:-piecewise}"
        --dynamic_cfg_x0_prompt_type "${DYNAMIC_CFG_X0_PROMPT_TYPE:-general}"
        --dynamic_cfg_x0_cfg_smooth_weight "${DYNAMIC_CFG_X0_SMOOTH_WEIGHT:-0.05}"
        --dynamic_cfg_x0_high_cfg_penalty "${DYNAMIC_CFG_X0_HIGH_CFG_PENALTY:-0.02}"
        --dynamic_cfg_x0_cfg_soft_max "${DYNAMIC_CFG_X0_CFG_SOFT_MAX:-3.5}"
        --dynamic_cfg_x0_cfg_min "${DYNAMIC_CFG_X0_CFG_MIN:-0.0}"
        --dynamic_cfg_x0_cfg_max "${DYNAMIC_CFG_X0_CFG_MAX:-5.0}"
      )
      if [[ "${DYNAMIC_CFG_X0_CONFIDENCE_GATING:-1}" == "0" ]]; then
        dyncfg_args+=(--dynamic_cfg_x0_no_confidence_gating)
      fi
      if [[ "${DYNAMIC_CFG_X0_ADD_LOCAL_NEIGHBORHOOD:-0}" == "1" ]]; then
        dyncfg_args+=(--dynamic_cfg_x0_add_local_neighborhood)
      fi
      run_flux_sharded "dynamic_cfg_x0" "ga" "${dyncfg_args[@]}"
      ;;
    sop)
      sop_args=(
        --runner_script "${SCRIPT_DIR}/sampling_flux_unified_sop.py"
        --n_variants 1
        --cfg_scales ${CFG_SCALES}
        --sop_init_paths "${SOP_INIT_PATHS:-8}"
        --sop_branch_factor "${SOP_BRANCH_FACTOR:-4}"
        --sop_keep_top "${SOP_KEEP_TOP:-4}"
        --sop_branch_every "${SOP_BRANCH_EVERY:-1}"
        --sop_start_frac "${SOP_START_FRAC:-0.25}"
        --sop_end_frac "${SOP_END_FRAC:-1.0}"
        --sop_score_decode "${SOP_SCORE_DECODE:-x0_pred}"
        --sop_variant_idx "${SOP_VARIANT_IDX:-0}"
      )
      run_flux_sharded "sop" "ga" "${sop_args[@]}"
      ;;
    *)
      echo "Error: unsupported method '${method}' for FLUX suite." >&2
      exit 1
      ;;
  esac
done

echo
echo "Suite summary: ${SUITE_TSV}"
cat "${SUITE_TSV}"
echo
echo "Outputs: ${RUN_DIR}"
