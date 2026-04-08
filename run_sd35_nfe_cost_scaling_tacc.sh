#!/usr/bin/env bash
# Run SD3.5 NFE cost-scaling benchmark on TACC:
# lookahead MCTS + reward correction vs SMC (ImageReward vs NFE curve).
#
# Usage:
#   bash run_sd35_nfe_cost_scaling_tacc.sh
#   NUM_PROMPTS=5 SIM_COSTS="5 10 20 35 50 80" bash run_sd35_nfe_cost_scaling_tacc.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SKIP_INSTALL="${SKIP_INSTALL:-1}"
source "${SCRIPT_DIR}/tacc_setup.sh"

_OUT_BASE="${SCRATCH:-${DATA_ROOT}}"
HPSV2_PROMPT_DIR="${HPSV2_PROMPT_DIR:-${_OUT_BASE}/hpsv2_prompt_cache}"
export HPSV2_PROMPT_DIR
RUN_TAG="${RUN_TAG:-sd35_nfe_cost_scaling}"
OUT_ROOT="${OUT_ROOT:-${_OUT_BASE}/hpsv2_all_models_runs/${RUN_TAG}}"
PROMPT_STYLE="${PROMPT_STYLE:-all}"
USE_SUBSET="${USE_SUBSET:-1}"
PROMPT_FILE="${PROMPT_FILE:-}"

START_INDEX="${START_INDEX:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
END_INDEX="${END_INDEX:-}"
NUM_GPUS="${NUM_GPUS:-$(${PYTHON_BIN} -c 'import torch; print(max(torch.cuda.device_count(),1))')}"

SD35_BACKEND="${SD35_BACKEND:-sid}"
STEPS="${STEPS:-4}"
SEED="${SEED:-42}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"

CFG_SCALES="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0}"
BASELINE_CFG="${BASELINE_CFG:-1.0}"
CORRECTION_STRENGTHS="${CORRECTION_STRENGTHS:-1.0}"
N_VARIANTS="${N_VARIANTS:-3}"
USE_QWEN="${USE_QWEN:-0}"
SIM_COSTS="${SIM_COSTS:-5 10 20 35 50}"
REWARD_NFE_WEIGHT="${REWARD_NFE_WEIGHT:-1.0}"

LOOKAHEAD_MODE="${LOOKAHEAD_MODE:-rollout_tree_prior}"
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

SMC_GAMMA="${SMC_GAMMA:-0.10}"
ESS_THRESHOLD="${ESS_THRESHOLD:-0.5}"
RESAMPLE_START_FRAC="${RESAMPLE_START_FRAC:-0.3}"
SMC_CFG_SCALE="${SMC_CFG_SCALE:-1.25}"
SMC_VARIANT_IDX="${SMC_VARIANT_IDX:-0}"

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

# Prefer NUM_PROMPTS over END_INDEX.
if (( NUM_PROMPTS > 0 )); then
  if [[ -n "${END_INDEX}" ]]; then
    echo "[nfe-scaling] both NUM_PROMPTS and END_INDEX are set; using NUM_PROMPTS (END_INDEX ignored)."
  fi
  END_INDEX="$((START_INDEX + NUM_PROMPTS))"
elif [[ -z "${END_INDEX}" ]]; then
  END_INDEX="-1"
fi

if [[ -z "${PROMPT_FILE}" ]]; then
  mkdir -p "${HPSV2_PROMPT_DIR}"
  OUT_DIR="${HPSV2_PROMPT_DIR}" STYLE="${PROMPT_STYLE}" bash "${SCRIPT_DIR}/get_hpsv2_prompts.sh"
  if [[ "${PROMPT_STYLE}" == "all" ]]; then
    PROMPT_FILE="${HPSV2_PROMPT_DIR}/hpsv2_prompts.txt"
  else
    PROMPT_FILE="${HPSV2_PROMPT_DIR}/hpsv2_prompts_${PROMPT_STYLE}.txt"
  fi
  if [[ "${USE_SUBSET}" == "1" ]]; then
    PROMPT_FILE="${SCRIPT_DIR}/hpsv2_subset.txt"
  fi
fi

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found: ${PROMPT_FILE}" >&2
  exit 1
fi

if (( NUM_GPUS < 1 )); then
  echo "Error: no visible GPUs (NUM_GPUS=${NUM_GPUS})" >&2
  exit 1
fi

read -r -a cfg_scales_arr <<< "${CFG_SCALES}"
read -r -a correction_arr <<< "${CORRECTION_STRENGTHS}"
read -r -a sim_costs_arr <<< "${SIM_COSTS}"
read -r -a reward_weights_arr <<< "${REWARD_WEIGHTS}"

if [[ "${#cfg_scales_arr[@]}" -eq 0 ]]; then
  echo "Error: CFG_SCALES is empty." >&2
  exit 1
fi
if [[ "${#correction_arr[@]}" -eq 0 ]]; then
  echo "Error: CORRECTION_STRENGTHS is empty." >&2
  exit 1
fi
if [[ "${#sim_costs_arr[@]}" -lt 5 ]]; then
  echo "Error: SIM_COSTS must contain at least 5 values." >&2
  exit 1
fi
if [[ "${#reward_weights_arr[@]}" -ne 2 ]]; then
  echo "Error: REWARD_WEIGHTS must contain exactly 2 values." >&2
  exit 1
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_ROOT}/${RUN_TAG}_${RUN_TS}"
mkdir -p "${RUN_DIR}"

case "${SD35_BACKEND}" in
  sid) PRELOAD_MODEL_ID="${PRELOAD_MODEL_ID:-YGu1998/SiD-DiT-SD3.5-large}" ;;
  sd35_base) PRELOAD_MODEL_ID="${PRELOAD_MODEL_ID:-stabilityai/stable-diffusion-3.5-large}" ;;
  senseflow_large) PRELOAD_MODEL_ID="${PRELOAD_MODEL_ID:-stabilityai/stable-diffusion-3.5-large}" ;;
  senseflow_medium) PRELOAD_MODEL_ID="${PRELOAD_MODEL_ID:-stabilityai/stable-diffusion-3.5-medium}" ;;
  *) PRELOAD_MODEL_ID="${PRELOAD_MODEL_ID:-YGu1998/SiD-DiT-SD3.5-large}" ;;
esac

echo "[nfe-scaling] run_dir=${RUN_DIR}"
echo "[nfe-scaling] prompt_file=${PROMPT_FILE} range=[${START_INDEX},${END_INDEX})"
echo "[nfe-scaling] backend=${SD35_BACKEND} reward_backend=${REWARD_BACKEND} num_gpus=${NUM_GPUS}"
echo "[nfe-scaling] sim_costs=[${SIM_COSTS}] lookahead_mode=${LOOKAHEAD_MODE}"

echo "[preload] caching model: ${PRELOAD_MODEL_ID}"
env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u LOCAL_WORLD_SIZE -u NODE_RANK \
  -u MASTER_ADDR -u MASTER_PORT \
  HF_HOME="${HF_HOME}" \
  _PRELOAD_MODEL_ID="${PRELOAD_MODEL_ID}" \
  "${PYTHON_BIN}" - <<'PY'
import os, sys
from huggingface_hub import snapshot_download

model_id = os.environ.get("_PRELOAD_MODEL_ID", "")
cache_dir = os.environ.get("HF_HOME") or None
try:
    snapshot_download(model_id, cache_dir=cache_dir)
    print(f"[preload] {model_id} OK")
except Exception as exc:
    print(f"[preload] warning: {exc}", file=sys.stderr)
PY

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if (( NUM_GPUS > 1 )); then
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS - 1)))"
fi

extra_args=()
if [[ "${USE_QWEN}" == "1" ]]; then
  extra_args+=(--use_qwen)
fi
if [[ -n "${REWRITES_FILE:-}" ]]; then
  extra_args+=(--rewrites_file "${REWRITES_FILE}")
fi
if [[ -n "${REWARD_API_BASE}" ]]; then
  extra_args+=(--reward_api_base "${REWARD_API_BASE}")
fi

IMAGEREWARD_CACHE="${IMAGEREWARD_CACHE}" \
HF_HOME="${HF_HOME}" \
HPS_ROOT="${HPS_ROOT}" \
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
"${PYTHON_BIN}" "${SCRIPT_DIR}/sd35_nfe_cost_scaling.py" \
  --prompt_file "${PROMPT_FILE}" \
  --start_index "${START_INDEX}" \
  --num_prompts "$((END_INDEX - START_INDEX))" \
  --sim_costs "${sim_costs_arr[@]}" \
  --out_dir "${RUN_DIR}" \
  --backend "${SD35_BACKEND}" \
  --steps "${STEPS}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --seed "${SEED}" \
  --cfg_scales "${cfg_scales_arr[@]}" \
  --baseline_cfg "${BASELINE_CFG}" \
  --correction_strengths "${correction_arr[@]}" \
  --n_variants "${N_VARIANTS}" \
  --reward_backend "${REWARD_BACKEND}" \
  --reward_model "${REWARD_MODEL}" \
  --unifiedreward_model "${UNIFIEDREWARD_MODEL}" \
  --image_reward_model "${IMAGE_REWARD_MODEL}" \
  --pickscore_model "${PICKSCORE_MODEL}" \
  --reward_weights "${reward_weights_arr[0]}" "${reward_weights_arr[1]}" \
  --reward_api_key "${REWARD_API_KEY}" \
  --reward_api_model "${REWARD_API_MODEL}" \
  --reward_max_new_tokens "${REWARD_MAX_NEW_TOKENS}" \
  --reward_prompt_mode "${REWARD_PROMPT_MODE}" \
  --lookahead_mode "${LOOKAHEAD_MODE}" \
  --lookahead_u_t_def "${LOOKAHEAD_U_T_DEF}" \
  --lookahead_tau "${LOOKAHEAD_TAU}" \
  --lookahead_c_puct "${LOOKAHEAD_C_PUCT}" \
  --lookahead_u_ref "${LOOKAHEAD_U_REF}" \
  --lookahead_w_cfg "${LOOKAHEAD_W_CFG}" \
  --lookahead_w_variant "${LOOKAHEAD_W_VARIANT}" \
  --lookahead_w_cs "${LOOKAHEAD_W_CS}" \
  --lookahead_w_q "${LOOKAHEAD_W_Q}" \
  --lookahead_w_explore "${LOOKAHEAD_W_EXPLORE}" \
  --lookahead_cfg_width_min "${LOOKAHEAD_CFG_WIDTH_MIN}" \
  --lookahead_cfg_width_max "${LOOKAHEAD_CFG_WIDTH_MAX}" \
  --lookahead_cfg_anchor_count "${LOOKAHEAD_CFG_ANCHOR_COUNT}" \
  --lookahead_min_visits_for_center "${LOOKAHEAD_MIN_VISITS_FOR_CENTER}" \
  --lookahead_log_action_topk "${LOOKAHEAD_LOG_ACTION_TOPK}" \
  --smc_gamma "${SMC_GAMMA}" \
  --ess_threshold "${ESS_THRESHOLD}" \
  --resample_start_frac "${RESAMPLE_START_FRAC}" \
  --smc_cfg_scale "${SMC_CFG_SCALE}" \
  --smc_variant_idx "${SMC_VARIANT_IDX}" \
  --reward_nfe_weight "${REWARD_NFE_WEIGHT}" \
  "${extra_args[@]}" \
  "$@"

echo
echo "[done] run_dir=${RUN_DIR}"
echo "[done] results=${RUN_DIR}/nfe_cost_scaling_results.json"
echo "[done] summary=${RUN_DIR}/nfe_cost_scaling_summary.tsv"
echo "[done] plot=${RUN_DIR}/imagereward_vs_nfe_reward_reweighted.png"
