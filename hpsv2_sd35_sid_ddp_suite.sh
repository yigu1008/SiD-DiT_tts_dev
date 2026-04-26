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
OUT_ROOT="${OUT_ROOT:-/data/ygu/hpsv2_sd35_sid_ddp}"
METHODS="${METHODS:-baseline greedy mcts ga}"
SD35_BACKEND="${SD35_BACKEND:-sid}"
SD35_SIGMAS="${SD35_SIGMAS:-}"

START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:--1}"
NUM_GPUS="${NUM_GPUS:-$("${PYTHON_BIN}" - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)}"

STEPS="${STEPS:-4}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
SEED="${SEED:-42}"
N_VARIANTS="${N_VARIANTS:-3}"
CFG_SCALES="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5}"
BASELINE_CFG="${BASELINE_CFG:-1.0}"
N_SIMS="${N_SIMS:-60}"
UCB_C="${UCB_C:-1.41}"
SMC_K="${SMC_K:-8}"
SMC_GAMMA="${SMC_GAMMA:-0.10}"
ESS_THRESHOLD="${ESS_THRESHOLD:-0.5}"
RESAMPLE_START_FRAC="${RESAMPLE_START_FRAC:-0.3}"
SMC_CFG_SCALE="${SMC_CFG_SCALE:-1.25}"
SMC_VARIANT_IDX="${SMC_VARIANT_IDX:-0}"
CORRECTION_STRENGTHS="${CORRECTION_STRENGTHS:-0.0}"
USE_QWEN="${USE_QWEN:-0}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen3-4B}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
QWEN_TIMEOUT_SEC="${QWEN_TIMEOUT_SEC:-240}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1}"
SAVE_IMAGES="${SAVE_IMAGES:-0}"
SAVE_BEST_IMAGES="${SAVE_BEST_IMAGES:-0}"
SAVE_VARIANTS="${SAVE_VARIANTS:-0}"
PRECOMPUTE_REWRITES="${PRECOMPUTE_REWRITES:-1}"
REWRITES_FILE="${REWRITES_FILE:-}"
REWRITES_OVERWRITE="${REWRITES_OVERWRITE:-0}"
QWEN_PRECOMPUTE_DEVICE="${QWEN_PRECOMPUTE_DEVICE:-auto}"
QWEN_PRECOMPUTE_BATCH_SIZE="${QWEN_PRECOMPUTE_BATCH_SIZE:-4}"
QWEN_PRECOMPUTE_SAVE_EVERY="${QWEN_PRECOMPUTE_SAVE_EVERY:-1}"
QWEN_PRECOMPUTE_CLEAR_CACHE="${QWEN_PRECOMPUTE_CLEAR_CACHE:-1}"
QWEN_PRECOMPUTE_MAX_NEW_TOKENS="${QWEN_PRECOMPUTE_MAX_NEW_TOKENS:-120}"
QWEN_PRECOMPUTE_TEMPERATURE="${QWEN_PRECOMPUTE_TEMPERATURE:-0.6}"
QWEN_PRECOMPUTE_TOP_P="${QWEN_PRECOMPUTE_TOP_P:-0.9}"

REWARD_BACKEND="${REWARD_BACKEND:-imagereward}"
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
# HPSv3 score shaping:
# - raw: use native HPSv3 outputs
# - 13ish: affine transform with default offset +13.0 for easier comparison to 13.x conventions
export HPSV3_SCORE_STYLE="${HPSV3_SCORE_STYLE:-raw}"
export HPSV3_SCORE_SCALE="${HPSV3_SCORE_SCALE:-1.0}"
if [[ -z "${HPSV3_SCORE_OFFSET:-}" ]]; then
  _hpsv3_style_lc="$(echo "${HPSV3_SCORE_STYLE}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${_hpsv3_style_lc}" == "13ish" || "${_hpsv3_style_lc}" == "offset13" || "${_hpsv3_style_lc}" == "plus13" ]]; then
    export HPSV3_SCORE_OFFSET="13.0"
  else
    export HPSV3_SCORE_OFFSET="0.0"
  fi
else
  export HPSV3_SCORE_OFFSET
fi
EVAL_BEST_IMAGES="${EVAL_BEST_IMAGES:-1}"
EVAL_BACKENDS="${EVAL_BACKENDS:-imagereward hpsv2 pickscore}"
EVAL_REWARD_DEVICE="${EVAL_REWARD_DEVICE:-cuda}"
EVAL_ALLOW_MISSING_BACKENDS="${EVAL_ALLOW_MISSING_BACKENDS:-1}"
LOCAL_REWARD_CACHE_ENABLE="${LOCAL_REWARD_CACHE_ENABLE:-1}"
LOCAL_REWARD_CACHE_ROOT="${LOCAL_REWARD_CACHE_ROOT:-/tmp/sid_reward_cache}"

# Keep ImageReward inference independent from cluster wandb/protobuf drift.
SID_FORCE_WANDB_STUB="${SID_FORCE_WANDB_STUB:-1}"
WANDB_DISABLED="${WANDB_DISABLED:-true}"
export SID_FORCE_WANDB_STUB WANDB_DISABLED

# ── Reward server (two-env architecture) ──────────────────────────────────────
# When USE_REWARD_SERVER=1, a separate conda env (with transformers==4.45.2)
# runs reward_server.py on a dedicated GPU.  The main env talks to it via HTTP,
# avoiding the transformers version conflict entirely.
#
# GPU allocation: the LAST visible GPU is reserved for the reward server.
# NUM_GPUS is decremented by 1 so torchrun only uses the remaining GPUs.
USE_REWARD_SERVER="${USE_REWARD_SERVER:-0}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5100}"
REWARD_SERVER_BACKENDS="${REWARD_SERVER_BACKENDS:-hpsv3 imagereward}"
REWARD_ENV_NAME="${REWARD_ENV_NAME:-reward}"
REWARD_ENV_CONDA_BASE="${REWARD_ENV_CONDA_BASE:-/opt/conda}"
REWARD_SERVER_PID=""
REWARD_SERVER_GPU=""

_resolve_reward_server_gpu() {
  # Determine which physical GPU the reward server gets (last visible GPU).
  # Updates NUM_GPUS and CUDA_VISIBLE_DEVICES so generation excludes that GPU.
  local all_gpus
  all_gpus="$("${PYTHON_BIN}" - <<'PY'
import os, torch
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if cvd:
    ids = [x.strip() for x in cvd.split(",") if x.strip()]
else:
    ids = [str(i) for i in range(torch.cuda.device_count())]
print(",".join(ids))
PY
)"
  IFS=',' read -r -a _all <<< "${all_gpus}"
  local n="${#_all[@]}"
  if (( n < 2 )); then
    # Single GPU: share it (reward server + generation both on GPU 0)
    echo "[reward-server] Only 1 GPU available — sharing it between reward server and generation."
    REWARD_SERVER_GPU="${_all[0]}"
    # NUM_GPUS stays as-is (1)
    return
  fi
  # Reserve the last GPU for the reward server
  REWARD_SERVER_GPU="${_all[$((n-1))]}"
  # Generation gets all GPUs except the last
  local gen_gpus=("${_all[@]:0:$((n-1))}")
  export CUDA_VISIBLE_DEVICES="$(IFS=','; echo "${gen_gpus[*]}")"
  NUM_GPUS="${#gen_gpus[@]}"
  echo "[reward-server] Reserved GPU ${REWARD_SERVER_GPU} for reward server"
  echo "[reward-server] Generation GPUs (${NUM_GPUS}): ${CUDA_VISIBLE_DEVICES}"
}

start_reward_server() {
  if [[ "${USE_REWARD_SERVER}" != "1" ]]; then return 0; fi

  _resolve_reward_server_gpu

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
  # Give the server its own CUDA_VISIBLE_DEVICES so it sees exactly 1 GPU as cuda:0
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

  # Wait for server to become healthy (models need time to load)
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
GA_LOG_TOPK="${GA_LOG_TOPK:-3}"
GA_EVAL_BATCH="${GA_EVAL_BATCH:-2}"
GA_PHASE_CONSTRAINTS="${GA_PHASE_CONSTRAINTS:-1}"
BON_N="${BON_N:-16}"
BEAM_WIDTH="${BEAM_WIDTH:-4}"
BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS:-8}"
BON_MCTS_TOPK="${BON_MCTS_TOPK:-2}"
BON_MCTS_SEED_STRIDE="${BON_MCTS_SEED_STRIDE:-1}"
BON_MCTS_SEED_OFFSET="${BON_MCTS_SEED_OFFSET:-0}"
BON_MCTS_SIM_ALLOC="${BON_MCTS_SIM_ALLOC:-split}"
BON_MCTS_MIN_SIMS="${BON_MCTS_MIN_SIMS:-8}"
BON_MCTS_PRESCREEN_CFG="${BON_MCTS_PRESCREEN_CFG:-}"
# Diffusion Tree Sampling (DTS) / Diffusion Tree Search (DTS*)
DTS_M_ITER="${DTS_M_ITER:-64}"
DTS_LAMBDA="${DTS_LAMBDA:-1.0}"
DTS_PW_C="${DTS_PW_C:-1.0}"
DTS_PW_ALPHA="${DTS_PW_ALPHA:-0.5}"
DTS_C_UCT="${DTS_C_UCT:-1.0}"
DTS_SDE_NOISE_SCALE="${DTS_SDE_NOISE_SCALE:-0.0}"
NOISE_INJECT_MODE="${NOISE_INJECT_MODE:-combined}"
NOISE_INJECT_SEED_BUDGET="${NOISE_INJECT_SEED_BUDGET:-8}"
NOISE_INJECT_CANDIDATE_STEPS="${NOISE_INJECT_CANDIDATE_STEPS:-}"
NOISE_INJECT_GAMMA_BANK="${NOISE_INJECT_GAMMA_BANK:-0.0 0.25 0.5}"
NOISE_INJECT_EPS_SAMPLES="${NOISE_INJECT_EPS_SAMPLES:-4}"
NOISE_INJECT_STEPS_PER_ROLLOUT="${NOISE_INJECT_STEPS_PER_ROLLOUT:-1}"
NOISE_INJECT_INCLUDE_NO_INJECT="${NOISE_INJECT_INCLUDE_NO_INJECT:-1}"
NOISE_INJECT_MAX_POLICIES="${NOISE_INJECT_MAX_POLICIES:-0}"
NOISE_INJECT_VARIANT_IDX="${NOISE_INJECT_VARIANT_IDX:-0}"
NOISE_INJECT_CFG="${NOISE_INJECT_CFG:-}"

# Lookahead U-based MCTS + adaptive per-step CFG (new SD3.5 option)
LOOKAHEAD_METHOD_MODE="${LOOKAHEAD_METHOD_MODE:-rollout_tree_prior_adaptive_cfg}"
LOOKAHEAD_U_T_DEF="${LOOKAHEAD_U_T_DEF:-latent_delta_rms}"
LOOKAHEAD_TAU="${LOOKAHEAD_TAU:-0.35}"
LOOKAHEAD_C_PUCT="${LOOKAHEAD_C_PUCT:-1.20}"
LOOKAHEAD_U_REF="${LOOKAHEAD_U_REF:-0.0}"
LOOKAHEAD_W_CFG="${LOOKAHEAD_W_CFG:-1.0}"
LOOKAHEAD_W_VARIANT="${LOOKAHEAD_W_VARIANT:-0.25}"
LOOKAHEAD_W_CS="${LOOKAHEAD_W_CS:-0.10}"
LOOKAHEAD_W_Q="${LOOKAHEAD_W_Q:-0.20}"
LOOKAHEAD_W_EXPLORE="${LOOKAHEAD_W_EXPLORE:-0.05}"
LOOKAHEAD_CFG_WIDTH_MIN="${LOOKAHEAD_CFG_WIDTH_MIN:-3}"
LOOKAHEAD_CFG_WIDTH_MAX="${LOOKAHEAD_CFG_WIDTH_MAX:-7}"
LOOKAHEAD_CFG_ANCHOR_COUNT="${LOOKAHEAD_CFG_ANCHOR_COUNT:-2}"
LOOKAHEAD_MIN_VISITS_FOR_CENTER="${LOOKAHEAD_MIN_VISITS_FOR_CENTER:-3}"
LOOKAHEAD_LOG_ACTION_TOPK="${LOOKAHEAD_LOG_ACTION_TOPK:-12}"

# Dynamic-CFG-only MCTS (no lookahead prior)
MCTS_CFG_MODE="${MCTS_CFG_MODE:-adaptive}"
MCTS_CFG_ROOT_BANK="${MCTS_CFG_ROOT_BANK:-1.0 1.5 2.0 2.5}"
MCTS_CFG_ANCHORS="${MCTS_CFG_ANCHORS:-1.0 2.0}"
MCTS_CFG_STEP_ANCHOR_COUNT="${MCTS_CFG_STEP_ANCHOR_COUNT:-2}"
MCTS_CFG_MIN_PARENT_VISITS="${MCTS_CFG_MIN_PARENT_VISITS:-3}"
MCTS_CFG_ROUND_NDIGITS="${MCTS_CFG_ROUND_NDIGITS:-6}"
MCTS_CFG_LOG_ACTION_TOPK="${MCTS_CFG_LOG_ACTION_TOPK:-12}"
MCTS_KEY_MODE="${MCTS_KEY_MODE:-count}"
MCTS_KEY_STEPS="${MCTS_KEY_STEPS:-}"
MCTS_KEY_STEP_COUNT="${MCTS_KEY_STEP_COUNT:-2}"
MCTS_KEY_STEP_STRIDE="${MCTS_KEY_STEP_STRIDE:-0}"
MCTS_KEY_DEFAULT_COUNT="${MCTS_KEY_DEFAULT_COUNT:-2}"
MCTS_FRESH_NOISE_STEPS="${MCTS_FRESH_NOISE_STEPS:-}"
MCTS_FRESH_NOISE_SAMPLES="${MCTS_FRESH_NOISE_SAMPLES:-1}"
MCTS_FRESH_NOISE_SCALE="${MCTS_FRESH_NOISE_SCALE:-1.0}"
MCTS_FRESH_NOISE_KEY_STEPS="${MCTS_FRESH_NOISE_KEY_STEPS:-0}"

_DEFAULT_CFG_SCALES_STR="1.0 1.25 1.5 1.75 2.0 2.25 2.5"
if [[ "${SD35_BACKEND}" == "senseflow_large" || "${SD35_BACKEND}" == "senseflow_medium" ]]; then
  if [[ "${CFG_SCALES}" == "${_DEFAULT_CFG_SCALES_STR}" ]]; then
    CFG_SCALES="0.0"
  fi
  if [[ "${BASELINE_CFG}" == "1.0" ]]; then
    BASELINE_CFG="0.0"
  fi
fi

# SD3.5-base guidance defaults (keep SID defaults unchanged).
if [[ "${SD35_BACKEND}" == "sd35_base" ]]; then
  if [[ "${CFG_SCALES}" == "${_DEFAULT_CFG_SCALES_STR}" ]]; then
    CFG_SCALES="3.5 4.0 4.5 5.0 5.5 6.0 7.0"
  fi
  if [[ "${BASELINE_CFG}" == "1.0" ]]; then
    BASELINE_CFG="4.5"
  fi
  if [[ "${MCTS_CFG_ROOT_BANK}" == "1.0 1.5 2.0 2.5" ]]; then
    MCTS_CFG_ROOT_BANK="4.0 4.5 5.0 5.5"
  fi
  if [[ "${MCTS_CFG_ANCHORS}" == "1.0 2.0" ]]; then
    MCTS_CFG_ANCHORS="3.5 7.0"
  fi
  if [[ "${SMC_CFG_SCALE}" == "1.25" ]]; then
    SMC_CFG_SCALE="4.5"
  fi
fi

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
if [[ -z "${REWRITES_FILE}" ]]; then
  REWRITES_FILE="${RUN_DIR}/rewrites_cache.json"
fi

stage_local_reward_cache() {
  if [[ "${LOCAL_REWARD_CACHE_ENABLE}" != "1" ]]; then
    return 0
  fi
  local local_root="${LOCAL_REWARD_CACHE_ROOT%/}"
  local local_ir="${local_root}/ImageReward"
  local local_clip="${local_root}/clip"
  local local_hps="${local_root}/hpsv2"
  local src_ir="${IMAGEREWARD_CACHE:-}"
  local src_clip="${CLIP_CACHE_DIR:-}"
  local src_hps="${HPS_ROOT:-}"

  mkdir -p "${local_ir}" "${local_clip}" "${local_hps}"

  _copy_cache_dir() {
    local src="$1"
    local dst="$2"
    if [[ -z "${src}" || "${src}" == "${dst}" || ! -d "${src}" ]]; then
      return 0
    fi
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "${src}/" "${dst}/" >/dev/null 2>&1 || true
    else
      cp -a "${src}/." "${dst}/" 2>/dev/null || true
    fi
  }

  _copy_cache_dir "${src_ir}" "${local_ir}"
  _copy_cache_dir "${src_clip}" "${local_clip}"
  _copy_cache_dir "${src_hps}" "${local_hps}"

  export IMAGEREWARD_CACHE="${local_ir}"
  export CLIP_CACHE_DIR="${local_clip}"
  export HPS_ROOT="${local_hps}"

  mkdir -p "${HOME}/.cache"
  ln -sfn "${CLIP_CACHE_DIR}" "${HOME}/.cache/clip" || true
  ln -sfn "${IMAGEREWARD_CACHE}" "${HOME}/.cache/ImageReward" || true
  echo "[cache] local reward cache enabled:"
  echo "  IMAGEREWARD_CACHE=${IMAGEREWARD_CACHE}"
  echo "  CLIP_CACHE_DIR=${CLIP_CACHE_DIR}"
  echo "  HPS_ROOT=${HPS_ROOT}"
}

stage_local_reward_cache

echo "SD3.5L SiD DDP suite"
echo "  prompt_file: ${PROMPT_FILE}"
echo "  modes: ${METHODS}"
if [[ "${METHODS}" == *"mcts_lookahead_dynamiccfg"* || "${METHODS}" == *"mcts_lookahead"* || "${METHODS}" == *"mcts_u_lookahead_only"* || "${METHODS}" == *"mcts_dynamiccfg_u_lookahead"* ]]; then
  echo "  lookahead_mode: ${LOOKAHEAD_METHOD_MODE} (u_t_def=${LOOKAHEAD_U_T_DEF})"
fi
if [[ "${METHODS}" == *"mcts_dynamiccfg_only"* ]]; then
  echo "  dynamic_cfg_mode: ${MCTS_CFG_MODE} root_bank=[${MCTS_CFG_ROOT_BANK}] anchors=[${MCTS_CFG_ANCHORS}]"
  echo "  dynamic_cfg_key_steps: mode=${MCTS_KEY_MODE} steps='${MCTS_KEY_STEPS}' count=${MCTS_KEY_STEP_COUNT} stride=${MCTS_KEY_STEP_STRIDE}"
fi
if [[ "${METHODS}" == *"noise"* || "${METHODS}" == *"noiseinj"* || "${METHODS}" == *"noise_inject"* ]]; then
  echo "  noise_inject: mode=${NOISE_INJECT_MODE} seed_budget=${NOISE_INJECT_SEED_BUDGET} steps='${NOISE_INJECT_CANDIDATE_STEPS}'"
  echo "               gammas=[${NOISE_INJECT_GAMMA_BANK}] eps_samples=${NOISE_INJECT_EPS_SAMPLES} steps_per_rollout=${NOISE_INJECT_STEPS_PER_ROLLOUT}"
fi
if [[ "${MCTS_FRESH_NOISE_SAMPLES}" != "1" || -n "${MCTS_FRESH_NOISE_STEPS}" || "${MCTS_FRESH_NOISE_KEY_STEPS}" == "1" ]]; then
  echo "  fresh_noise: steps='${MCTS_FRESH_NOISE_STEPS}' samples=${MCTS_FRESH_NOISE_SAMPLES} scale=${MCTS_FRESH_NOISE_SCALE} key_steps=${MCTS_FRESH_NOISE_KEY_STEPS}"
fi
echo "  sd35_backend: ${SD35_BACKEND} sd35_sigmas: ${SD35_SIGMAS:-<none>}"
echo "  nproc_per_node: ${NUM_GPUS}"
echo "  reward_backend: ${REWARD_BACKEND}"
echo "  eval_best_images: ${EVAL_BEST_IMAGES} eval_backends: ${EVAL_BACKENDS} eval_device: ${EVAL_REWARD_DEVICE}"
echo "  ga: pop=${GA_POPULATION} gens=${GA_GENERATIONS} eval_batch=${GA_EVAL_BATCH}"
if [[ "${METHODS}" == *"bon_mcts"* ]]; then
  echo "  bon_mcts: prescreen_n=${BON_MCTS_N_SEEDS} topk=${BON_MCTS_TOPK} sim_alloc=${BON_MCTS_SIM_ALLOC} min_sims=${BON_MCTS_MIN_SIMS}"
fi
if [[ "${METHODS}" == *"dts"* ]]; then
  echo "  dts: M=${DTS_M_ITER} lambda=${DTS_LAMBDA} pw=(C=${DTS_PW_C},a=${DTS_PW_ALPHA}) c_uct=${DTS_C_UCT} sde=${DTS_SDE_NOISE_SCALE}"
fi
echo "  use_qwen: ${USE_QWEN} (precompute=${PRECOMPUTE_REWRITES})"
echo "  rewrites_file: ${REWRITES_FILE}"
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

ensure_imagereward_runtime() {
  # External reward-server mode: reward backends run in a separate process/env.
  if [[ "${USE_REWARD_SERVER}" != "1" ]] && [[ -n "${REWARD_SERVER_URL:-}" ]]; then
    return 0
  fi
  local backend_lc
  backend_lc="$(echo "${REWARD_BACKEND}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${backend_lc}" != "imagereward" && "${backend_lc}" != "auto" && "${backend_lc}" != "blend" ]] && ! eval_backend_requested "imagereward"; then
    return 0
  fi
  local _stamp="${HOME}/.cache/sid_deps/reward_deps_ok_v2"
  if [[ "${FORCE_INSTALL_DEPS:-0}" != "1" ]] && [[ -f "${_stamp}" ]]; then
    return 0
  fi
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
    mkdir -p "$(dirname "${_stamp}")" && touch "${_stamp}"
    return 0
  fi
  echo "[deps] ImageReward runtime deps missing. Installing with install_reward_deps.sh ..."
  PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/install_reward_deps.sh"
  mkdir -p "$(dirname "${_stamp}")" && touch "${_stamp}"
}

ensure_imagereward_runtime

ensure_pickscore_runtime() {
  # External reward-server mode: reward backends run in a separate process/env.
  if [[ "${USE_REWARD_SERVER}" != "1" ]] && [[ -n "${REWARD_SERVER_URL:-}" ]]; then
    return 0
  fi
  local backend_lc _stamp
  backend_lc="$(echo "${REWARD_BACKEND}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${backend_lc}" != "pickscore" && "${backend_lc}" != "auto" ]] && ! eval_backend_requested "pickscore"; then
    return 0
  fi
  _stamp="${HOME}/.cache/sid_deps/reward_deps_ok_v2"
  if [[ "${FORCE_INSTALL_DEPS:-0}" != "1" ]] && [[ -f "${_stamp}" ]]; then return 0; fi
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import timm
from timm.data import ImageNetInfo
print(timm.__version__, ImageNetInfo.__name__)
PY
  then
    mkdir -p "$(dirname "${_stamp}")" && touch "${_stamp}"; return 0
  fi
  echo "[deps] PickScore runtime deps missing/incompatible (timm ImageNetInfo). Installing with install_reward_deps.sh ..."
  PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/install_reward_deps.sh"
  mkdir -p "$(dirname "${_stamp}")" && touch "${_stamp}"
}

ensure_pickscore_runtime

ensure_hpsv2_runtime() {
  # External reward-server mode: reward backends run in a separate process/env.
  if [[ "${USE_REWARD_SERVER}" != "1" ]] && [[ -n "${REWARD_SERVER_URL:-}" ]]; then
    return 0
  fi
  local backend_lc _stamp
  backend_lc="$(echo "${REWARD_BACKEND}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${backend_lc}" != "hpsv2" && "${backend_lc}" != "auto" && "${backend_lc}" != "blend" ]] && ! eval_backend_requested "hpsv2"; then
    return 0
  fi
  _stamp="${HOME}/.cache/sid_deps/reward_deps_ok_v2"
  if [[ "${FORCE_INSTALL_DEPS:-0}" != "1" ]] && [[ -f "${_stamp}" ]]; then return 0; fi
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
  local backend_lc _stamp hpsv3_impl
  backend_lc="$(echo "${REWARD_BACKEND}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${backend_lc}" != "hpsv3" && "${backend_lc}" != "auto" ]] && ! eval_backend_requested "hpsv3"; then
    return 0
  fi
  hpsv3_impl="$(echo "${SID_HPSV3_IMPL:-auto}" | tr '[:upper:]' '[:lower:]')"
  _stamp="${HOME}/.cache/sid_deps/reward_deps_ok_v2"
  if [[ "${FORCE_INSTALL_DEPS:-0}" != "1" ]] && [[ -f "${_stamp}" ]]; then return 0; fi
  if [[ "${hpsv3_impl}" == "imscore" || "${hpsv3_impl}" == "ims" ]]; then
    if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import imscore
from imscore.hpsv3.model import HPSv3
print(getattr(imscore, "__file__", "ok"), HPSv3.__name__)
PY
    then
      mkdir -p "$(dirname "${_stamp}")" && touch "${_stamp}"; return 0
    fi
    echo "[deps] HPSv3(imscore) runtime deps missing. Installing with install_reward_deps.sh ..."
    PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/install_reward_deps.sh"
    mkdir -p "$(dirname "${_stamp}")" && touch "${_stamp}"
    return 0
  fi
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import hpsv3
import omegaconf
import hydra
print(getattr(hpsv3, "__file__", "ok"), getattr(omegaconf, "__version__", "ok"), getattr(hydra, "__version__", "ok"))
PY
  then
    mkdir -p "$(dirname "${_stamp}")" && touch "${_stamp}"; return 0
  fi
  echo "[deps] HPSv3 runtime deps missing. Installing with install_reward_deps.sh ..."
  PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/install_reward_deps.sh"
  mkdir -p "$(dirname "${_stamp}")" && touch "${_stamp}"
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

ensure_qwen_precompute_runtime() {
  if [[ "${USE_QWEN}" != "1" || "${PRECOMPUTE_REWRITES}" != "1" ]]; then
    return 0
  fi
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import importlib.metadata as md
md.version("regex")
import transformers
print(transformers.__version__)
PY
  then
    return 0
  fi

  echo "[deps] Qwen precompute deps missing/broken (regex/transformers metadata). Installing regex ..."
  "${PYTHON_BIN}" -m pip install --no-cache-dir --upgrade "regex>=2024.11.6"

  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import importlib.metadata as md
md.version("regex")
import transformers
print(transformers.__version__)
PY
  then
    return 0
  fi

  echo "Error: Qwen precompute runtime still broken after regex repair." >&2
  echo "Try manual repair:" >&2
  echo "  ${PYTHON_BIN} -m pip install --no-cache-dir --upgrade regex transformers" >&2
  exit 1
}

precompute_rewrites_cache() {
  if [[ "${USE_QWEN}" != "1" ]]; then
    return 0
  fi
  if [[ "${PRECOMPUTE_REWRITES}" != "1" ]]; then
    return 0
  fi
  ensure_qwen_precompute_runtime
  echo "[rewrites] precomputing Qwen rewrites cache ..."
  local -a cmd=(
    "${PYTHON_BIN}" "-u" "${SCRIPT_DIR}/precompute_sd35_rewrites.py"
    --prompt_file "${PROMPT_FILE}"
    --rewrites_file "${REWRITES_FILE}"
    --start_index "${START_INDEX}"
    --end_index "${END_INDEX}"
    --n_variants "${N_VARIANTS}"
    --qwen_id "${QWEN_ID}"
    --qwen_dtype "${QWEN_DTYPE}"
    --device "${QWEN_PRECOMPUTE_DEVICE}"
    --batch_size "${QWEN_PRECOMPUTE_BATCH_SIZE}"
    --save_every_batches "${QWEN_PRECOMPUTE_SAVE_EVERY}"
    --max_new_tokens "${QWEN_PRECOMPUTE_MAX_NEW_TOKENS}"
    --temperature "${QWEN_PRECOMPUTE_TEMPERATURE}"
    --top_p "${QWEN_PRECOMPUTE_TOP_P}"
  )
  if [[ "${QWEN_PRECOMPUTE_CLEAR_CACHE}" == "1" ]]; then
    cmd+=(--clear_cache_each_batch)
  else
    cmd+=(--no-clear_cache_each_batch)
  fi
  if [[ "${REWRITES_OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u LOCAL_WORLD_SIZE -u NODE_RANK -u MASTER_ADDR -u MASTER_PORT \
    "${cmd[@]}"
  if [[ ! -s "${REWRITES_FILE}" ]]; then
    echo "Error: rewrite precompute finished but cache missing/empty: ${REWRITES_FILE}" >&2
    exit 1
  fi
  echo "[rewrites] cache ready: ${REWRITES_FILE}"
}

post_eval_best_images() {
  local method_out="$1"
  local method_name="$2"
  if [[ "${EVAL_BEST_IMAGES}" != "1" || ( "${SAVE_IMAGES}" != "1" && "${SAVE_BEST_IMAGES}" != "1" ) ]]; then
    return 0
  fi
  local -a cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_best_images_multi_reward.py"
    --layout sd35
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
  local eval_dev_lc
  eval_dev_lc="$(echo "${EVAL_REWARD_DEVICE}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${eval_dev_lc}" == cpu* ]]; then
    # Hard-disable CUDA visibility for eval when device=cpu.
    # This prevents heavyweight reward backends from grabbing GPU memory.
    CUDA_VISIBLE_DEVICES="" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
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

for log_path in glob.glob(os.path.join(method_out, "logs", "rank_*.jsonl")):
    if os.path.basename(log_path).endswith("_rewrite_examples.jsonl"):
        continue
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "score" not in row:
                continue
            score = float(row["score"])
            delta = float(row.get("delta_vs_base", 0.0))
            b = row.get("baseline_score")
            if b is None:
                b = score - delta
            baseline.append(float(b))
            search.append(score)
            deltas.append(delta)

if not baseline:
    raise RuntimeError(f"No rank logs found under {method_out}/logs")

mean_baseline = float(statistics.fmean(baseline))
mean_search = float(statistics.fmean(search))
mean_delta = float(statistics.fmean(deltas))

aggregate = {
    "method": method,
    "elapsed_sec": elapsed,
    "num_samples": len(search),
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

nfe_by_gen = defaultdict(list)
for hist_path in glob.glob(os.path.join(method_out, "ga_logs", "*ga_history.json")):
    with open(hist_path, encoding="utf-8") as f:
        payload = json.load(f)
    for row in payload.get("history", []):
        gen = int(row.get("generation", 0)) + 1
        nfe = float(row.get("nfe_per_generation", 0))
        nfe_by_gen[gen].append(nfe)

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
        writer.writerow(["method", "elapsed_sec", "num_samples", "mean_baseline", "mean_search", "mean_delta"] + [f"eval_{b}" for b in eval_backends])
    writer.writerow(
        [method, elapsed, len(search), f"{mean_baseline:.6f}", f"{mean_search:.6f}", f"{mean_delta:+.6f}"] + [eval_means[b] for b in eval_backends]
    )
PY
}

run_method() {
  local method="$1"
  local method_out="${RUN_DIR}/${method}"
  mkdir -p "${method_out}"
  chmod -R u+rwX "${method_out}" 2>/dev/null || true
  local mode_arg
  local runner_script="${SCRIPT_DIR}/sd35_ddp_experiment.py"
  local lookahead_mode_for_method="${LOOKAHEAD_METHOD_MODE}"
  case "${method}" in
    baseline) mode_arg="base" ;;
    greedy) mode_arg="greedy" ;;
    mcts) mode_arg="mcts" ;;
    noise|noiseinj|noise_inject) mode_arg="noise" ;;
    mcts_dynamiccfg_only)
      mode_arg="mcts"
      runner_script="${SCRIPT_DIR}/sd35_ddp_experiment_dynamic_cfg.py"
      ;;
    mcts_u_lookahead_only)
      mode_arg="mcts"
      runner_script="${SCRIPT_DIR}/sd35_ddp_experiment_lookahead_reweighting.py"
      lookahead_mode_for_method="rollout_tree_prior"
      ;;
    mcts_dynamiccfg_u_lookahead)
      mode_arg="mcts"
      runner_script="${SCRIPT_DIR}/sd35_ddp_experiment_lookahead_reweighting.py"
      # Respect LOOKAHEAD_METHOD_MODE env var; default to rollout_tree_prior
      lookahead_mode_for_method="${LOOKAHEAD_METHOD_MODE:-rollout_tree_prior}"
      ;;
    mcts_lookahead_dynamiccfg|mcts_lookahead)
      mode_arg="mcts"
      runner_script="${SCRIPT_DIR}/sd35_ddp_experiment_lookahead_reweighting.py"
      if [[ "${method}" == "mcts_lookahead_dynamiccfg" ]]; then
        lookahead_mode_for_method="rollout_tree_prior_adaptive_cfg"
      fi
      ;;
    ga) mode_arg="ga" ;;
    smc|smc_das) mode_arg="smc" ;;
    bon) mode_arg="bon" ;;
    bon_mcts)
      mode_arg="mcts"
      if [[ "${SD35_BACKEND}" == "sd35_base" ]]; then
        runner_script="${SCRIPT_DIR}/sd35_ddp_experiment_bon_mcts_sd35base.py"
      else
        runner_script="${SCRIPT_DIR}/sd35_ddp_experiment_bon_mcts.py"
      fi
      ;;
    beam) mode_arg="beam" ;;
    dts|dts_star)
      mode_arg="mcts"
      runner_script="${SCRIPT_DIR}/sd35_ddp_experiment_dts.py"
      ;;
    *)
      echo "Error: unsupported method '${method}' for SD3.5 suite." >&2
      exit 1
      ;;
  esac

  local -a extra=()
  if [[ -f "${REWRITES_FILE}" ]]; then
    extra+=(--rewrites_file "${REWRITES_FILE}")
  fi
  if [[ "${MCTS_FRESH_NOISE_KEY_STEPS}" == "1" ]]; then
    extra+=(--mcts_fresh_noise_key_steps)
  fi
  if [[ "${NOISE_INJECT_INCLUDE_NO_INJECT}" == "1" ]]; then
    extra+=(--noise_inject_include_no_inject)
  else
    extra+=(--no-noise_inject_include_no_inject)
  fi
  if [[ "${USE_QWEN}" == "1" ]]; then
    if [[ "${PRECOMPUTE_REWRITES}" == "1" && -f "${REWRITES_FILE}" ]]; then
      # Cache-only variant path: avoid loading Qwen in each DDP rank.
      extra+=(--no_qwen)
    elif [[ "${PRECOMPUTE_REWRITES}" == "1" && ! -f "${REWRITES_FILE}" ]]; then
      echo "Error: USE_QWEN=1 with PRECOMPUTE_REWRITES=1 requires rewrites cache file, but not found: ${REWRITES_FILE}" >&2
      exit 1
    fi
  else
    extra+=(--no_qwen)
  fi
  if [[ "${SAVE_IMAGES}" == "1" ]]; then
    extra+=(--save_images)
  fi
  if [[ "${SAVE_BEST_IMAGES}" == "1" ]]; then
    extra+=(--save_best_images)
  fi
  if [[ "${SAVE_VARIANTS}" == "1" ]]; then
    extra+=(--save_variants)
  fi
  if [[ "${GA_PHASE_CONSTRAINTS}" == "1" ]]; then
    extra+=(--ga_phase_constraints)
  fi
  if [[ -n "${REWARD_API_BASE}" ]]; then
    extra+=(--reward_api_base "${REWARD_API_BASE}")
  fi
  if [[ -n "${SD35_SIGMAS}" ]]; then
    extra+=(--sigmas ${SD35_SIGMAS})
  fi
  if [[ "${runner_script}" == "${SCRIPT_DIR}/sd35_ddp_experiment_dynamic_cfg.py" ]]; then
    extra+=(
      --mcts_cfg_mode "${MCTS_CFG_MODE}"
      --mcts_cfg_root_bank ${MCTS_CFG_ROOT_BANK}
      --mcts_cfg_anchors ${MCTS_CFG_ANCHORS}
      --mcts_cfg_step_anchor_count "${MCTS_CFG_STEP_ANCHOR_COUNT}"
      --mcts_cfg_min_parent_visits "${MCTS_CFG_MIN_PARENT_VISITS}"
      --mcts_cfg_round_ndigits "${MCTS_CFG_ROUND_NDIGITS}"
      --mcts_cfg_log_action_topk "${MCTS_CFG_LOG_ACTION_TOPK}"
      --mcts_key_mode "${MCTS_KEY_MODE}"
      --mcts_key_step_stride "${MCTS_KEY_STEP_STRIDE}"
      --mcts_key_default_count "${MCTS_KEY_DEFAULT_COUNT}"
    )
  fi
  if [[ "${runner_script}" == "${SCRIPT_DIR}/sd35_ddp_experiment_lookahead_reweighting.py" ]]; then
    extra+=(
      --lookahead_mode "${lookahead_mode_for_method}"
      --lookahead_u_t_def "${LOOKAHEAD_U_T_DEF}"
      --lookahead_tau "${LOOKAHEAD_TAU}"
      --lookahead_c_puct "${LOOKAHEAD_C_PUCT}"
      --lookahead_u_ref "${LOOKAHEAD_U_REF}"
      --lookahead_w_cfg "${LOOKAHEAD_W_CFG}"
      --lookahead_w_variant "${LOOKAHEAD_W_VARIANT}"
      --lookahead_w_cs "${LOOKAHEAD_W_CS}"
      --lookahead_w_q "${LOOKAHEAD_W_Q}"
      --lookahead_w_explore "${LOOKAHEAD_W_EXPLORE}"
      --lookahead_cfg_width_min "${LOOKAHEAD_CFG_WIDTH_MIN}"
      --lookahead_cfg_width_max "${LOOKAHEAD_CFG_WIDTH_MAX}"
      --lookahead_cfg_anchor_count "${LOOKAHEAD_CFG_ANCHOR_COUNT}"
      --lookahead_min_visits_for_center "${LOOKAHEAD_MIN_VISITS_FOR_CENTER}"
      --lookahead_log_action_topk "${LOOKAHEAD_LOG_ACTION_TOPK}"
    )
  fi
  if [[ "${runner_script}" == "${SCRIPT_DIR}/sd35_ddp_experiment_bon_mcts.py" || "${runner_script}" == "${SCRIPT_DIR}/sd35_ddp_experiment_bon_mcts_sd35base.py" ]]; then
    extra+=(
      --bon_mcts_n_seeds "${BON_MCTS_N_SEEDS}"
      --bon_mcts_topk "${BON_MCTS_TOPK}"
      --bon_mcts_seed_stride "${BON_MCTS_SEED_STRIDE}"
      --bon_mcts_seed_offset "${BON_MCTS_SEED_OFFSET}"
      --bon_mcts_sim_alloc "${BON_MCTS_SIM_ALLOC}"
      --bon_mcts_min_sims "${BON_MCTS_MIN_SIMS}"
    )
    if [[ -n "${BON_MCTS_PRESCREEN_CFG}" ]]; then
      extra+=(--bon_mcts_prescreen_cfg "${BON_MCTS_PRESCREEN_CFG}")
    fi
  fi
  if [[ "${runner_script}" == "${SCRIPT_DIR}/sd35_ddp_experiment_dts.py" ]]; then
    extra+=(
      --dts_method "${method}"
      --dts_m_iter "${DTS_M_ITER}"
      --dts_lambda "${DTS_LAMBDA}"
      --dts_pw_c "${DTS_PW_C}"
      --dts_pw_alpha "${DTS_PW_ALPHA}"
      --dts_c_uct "${DTS_C_UCT}"
      --dts_sde_noise_scale "${DTS_SDE_NOISE_SCALE}"
    )
  fi
  if [[ "${SMC_VARIANT_EXPANSION:-0}" == "1" ]]; then
    extra+=(--smc_variant_expansion)
    if [[ -n "${SMC_EXPANSION_VARIANTS:-}" ]]; then
      extra+=(--smc_expansion_variants ${SMC_EXPANSION_VARIANTS})
    fi
    if [[ -n "${SMC_EXPANSION_CFGS:-}" ]]; then
      extra+=(--smc_expansion_cfgs ${SMC_EXPANSION_CFGS})
    fi
    if [[ -n "${SMC_EXPANSION_CS:-}" ]]; then
      extra+=(--smc_expansion_cs ${SMC_EXPANSION_CS})
    fi
    if [[ -n "${SMC_EXPANSION_FACTOR:-}" ]]; then
      extra+=(--smc_expansion_factor "${SMC_EXPANSION_FACTOR}")
    fi
    if [[ -n "${SMC_EXPANSION_PROPOSAL:-}" ]]; then
      extra+=(--smc_expansion_proposal "${SMC_EXPANSION_PROPOSAL}")
    fi
    if [[ -n "${SMC_EXPANSION_TAU:-}" ]]; then
      extra+=(--smc_expansion_tau "${SMC_EXPANSION_TAU}")
    fi
    if [[ "${SMC_EXPANSION_LOOKAHEAD:-0}" == "1" ]]; then
      extra+=(--smc_expansion_lookahead)
    fi
  fi
  if [[ -n "${NOISE_INJECT_CANDIDATE_STEPS}" ]]; then
    extra+=(--noise_inject_candidate_steps "${NOISE_INJECT_CANDIDATE_STEPS}")
  fi
  if [[ -n "${NOISE_INJECT_CFG}" ]]; then
    extra+=(--noise_inject_cfg "${NOISE_INJECT_CFG}")
  fi
  # If snapshot paths were captured during preload, pass them directly so from_pretrained
  # reads from the exact local path instead of doing a cache lookup in offline mode.
  if [[ -n "${SD35_LOCAL_DIR:-}" ]]; then
    extra+=(--model_id "${SD35_LOCAL_DIR}")
  fi
  # Do NOT pass SENSEFLOW_LOCAL_DIR as --transformer_id.  The backend config
  # already sets transformer_id="domiso/SenseFlow" with the correct subfolder.
  # from_pretrained resolves the HF repo ID from cache in offline mode; passing
  # a local snapshot path breaks subfolder resolution for multi-model repos.
  local begin_ts
  begin_ts="$(date +%s)"
  echo "[$(date '+%F %T')] method=${method} start"

  "${PYTHON_BIN}" - <<'PY'
import os
import torch
cuda_ok = torch.cuda.is_available()
count = int(torch.cuda.device_count()) if cuda_ok else 0
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
print(f"[preflight] cuda_available={cuda_ok} device_count={count} CUDA_VISIBLE_DEVICES={cvd}")
if not cuda_ok:
    raise SystemExit("ERROR: CUDA unavailable before torchrun; refusing CPU fp16 SD3.5 run.")
PY

  torchrun --standalone --nproc_per_node "${NUM_GPUS}" "${runner_script}" \
    --backend "${SD35_BACKEND}" \
    --gen_batch_size "${GEN_BATCH_SIZE}" \
    --prompt_file "${PROMPT_FILE}" \
    --start_index "${START_INDEX}" \
    --end_index "${END_INDEX}" \
    --modes "${mode_arg}" \
    --steps "${STEPS}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --cfg_scales ${CFG_SCALES} \
    --baseline_cfg "${BASELINE_CFG}" \
    --n_variants "${N_VARIANTS}" \
    --qwen_id "${QWEN_ID}" \
    --qwen_dtype "${QWEN_DTYPE}" \
    --qwen_timeout_sec "${QWEN_TIMEOUT_SEC}" \
    --n_sims "${N_SIMS}" \
    --ucb_c "${UCB_C}" \
    --mcts_key_steps "${MCTS_KEY_STEPS}" \
    --mcts_key_step_count "${MCTS_KEY_STEP_COUNT}" \
    --mcts_fresh_noise_steps "${MCTS_FRESH_NOISE_STEPS}" \
    --mcts_fresh_noise_samples "${MCTS_FRESH_NOISE_SAMPLES}" \
    --mcts_fresh_noise_scale "${MCTS_FRESH_NOISE_SCALE}" \
    --noise_inject_mode "${NOISE_INJECT_MODE}" \
    --noise_inject_seed_budget "${NOISE_INJECT_SEED_BUDGET}" \
    --noise_inject_gamma_bank ${NOISE_INJECT_GAMMA_BANK} \
    --noise_inject_eps_samples "${NOISE_INJECT_EPS_SAMPLES}" \
    --noise_inject_steps_per_rollout "${NOISE_INJECT_STEPS_PER_ROLLOUT}" \
    --noise_inject_max_policies "${NOISE_INJECT_MAX_POLICIES}" \
    --noise_inject_variant_idx "${NOISE_INJECT_VARIANT_IDX}" \
    --smc_k "${SMC_K}" \
    --smc_gamma "${SMC_GAMMA}" \
    --ess_threshold "${ESS_THRESHOLD}" \
    --resample_start_frac "${RESAMPLE_START_FRAC}" \
    --smc_cfg_scale "${SMC_CFG_SCALE}" \
    --smc_variant_idx "${SMC_VARIANT_IDX}" \
    --seed "${SEED}" \
    --reward_backend "${REWARD_BACKEND}" \
    --reward_model "${REWARD_MODEL}" \
    --unifiedreward_model "${UNIFIEDREWARD_MODEL}" \
    --image_reward_model "${IMAGE_REWARD_MODEL}" \
    --pickscore_model "${PICKSCORE_MODEL}" \
    --reward_weights ${REWARD_WEIGHTS} \
    --reward_api_key "${REWARD_API_KEY}" \
    --reward_api_model "${REWARD_API_MODEL}" \
    --reward_max_new_tokens "${REWARD_MAX_NEW_TOKENS}" \
    --reward_prompt_mode "${REWARD_PROMPT_MODE}" \
    --ga_population "${GA_POPULATION}" \
    --ga_generations "${GA_GENERATIONS}" \
    --ga_elites "${GA_ELITES}" \
    --ga_mutation_prob "${GA_MUTATION_PROB}" \
    --ga_tournament_k "${GA_TOURNAMENT_K}" \
    --ga_selection "${GA_SELECTION}" \
    --ga_rank_pressure "${GA_RANK_PRESSURE}" \
    --ga_crossover "${GA_CROSSOVER}" \
    --ga_log_topk "${GA_LOG_TOPK}" \
    --ga_eval_batch "${GA_EVAL_BATCH}" \
    --bon_n "${BON_N}" \
    --beam_width "${BEAM_WIDTH}" \
    --correction_strengths ${CORRECTION_STRENGTHS} \
    --out_dir "${method_out}" \
    "${extra[@]}"

  local end_ts
  end_ts="$(date +%s)"
  local elapsed=$(( end_ts - begin_ts ))
  echo "[$(date '+%F %T')] method=${method} done elapsed=${elapsed}s"

  post_eval_best_images "${method_out}" "${mode_arg}"
  append_method_summary "${method_out}" "${method}" "${elapsed}"
}

precompute_rewrites_cache

# ── Start reward server (if two-env mode) ──────────────────────────────────────
start_reward_server

# ── Downgrade transformers for reward model compatibility ────────────────────
# Qwen3 precompute (above) needs transformers>=4.51.  ImageReward and hpsv3
# need transformers<=4.46 (removed APIs: additional_special_tokens_ids,
# DistributedTensorGatherer, etc.).  Now that precompute is done, pin 4.45.2.
# Skip the downgrade when using the reward server — the reward env has its own
# transformers==4.45.2, and the main env can keep transformers>=4.51.
HPSV3_IMPL_LC="$(echo "${SID_HPSV3_IMPL:-auto}" | tr '[:upper:]' '[:lower:]')"
if [[ "${USE_REWARD_SERVER}" == "1" ]]; then
  echo "[deps] Skipping transformers downgrade — reward server handles reward model deps."
elif [[ -n "${REWARD_SERVER_URL:-}" ]]; then
  echo "[deps] Skipping transformers downgrade — external REWARD_SERVER_URL is set."
elif [[ "${HPSV3_IMPL_LC}" == "imscore" || "${HPSV3_IMPL_LC}" == "ims" ]]; then
  echo "[deps] Skipping transformers downgrade — SID_HPSV3_IMPL=${SID_HPSV3_IMPL:-auto} uses single-env imscore path."
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
  run_method "${method}"
done

echo
echo "Suite summary: ${SUITE_TSV}"
cat "${SUITE_TSV}"
echo
echo "Outputs: ${RUN_DIR}"
