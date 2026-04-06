#!/usr/bin/env bash
# Run SD3.5 MCTS lookahead-reweighting ablations on TACC.
#
# Usage:
#   bash run_sd35_lookahead_reweighting_tacc.sh
#   LOOKAHEAD_MODES="standard rollout_tree_prior adaptive_cfg_width" \
#   N_SIMS=20 bash run_sd35_lookahead_reweighting_tacc.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SKIP_INSTALL="${SKIP_INSTALL:-1}"
source "${SCRIPT_DIR}/tacc_setup.sh"

_OUT_BASE="${SCRATCH:-${DATA_ROOT}}"
HPSV2_PROMPT_DIR="${HPSV2_PROMPT_DIR:-${_OUT_BASE}/hpsv2_prompt_cache}"
export HPSV2_PROMPT_DIR
RUN_TAG="${RUN_TAG:-sd35_lookahead_reweighting}"
OUT_ROOT="${OUT_ROOT:-${_OUT_BASE}/hpsv2_all_models_runs/${RUN_TAG}}"
PROMPT_STYLE="${PROMPT_STYLE:-all}"
USE_SUBSET="${USE_SUBSET:-1}"
PROMPT_FILE="${PROMPT_FILE:-}"
START_INDEX="${START_INDEX:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
END_INDEX="${END_INDEX:-$((START_INDEX + NUM_PROMPTS))}"
NUM_GPUS="${NUM_GPUS:-$(${PYTHON_BIN} -c 'import torch; print(max(torch.cuda.device_count(),1))')}"

SD35_BACKEND="${SD35_BACKEND:-sid}"
SD35_SIGMAS="${SD35_SIGMAS:-}"
STEPS="${STEPS:-4}"
SEED="${SEED:-42}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1}"
BASELINE_CFG="${BASELINE_CFG:-1.0}"
CFG_SCALES="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0}"
N_SIMS="${N_SIMS:-50}"
UCB_C="${UCB_C:-1.41}"

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

CORRECTION_STRENGTHS="${CORRECTION_STRENGTHS:-0.0}"
N_VARIANTS="${N_VARIANTS:-3}"
USE_QWEN="${USE_QWEN:-1}"
if [[ "${CFG_ONLY:-0}" == "1" ]]; then
  N_VARIANTS=0
  CORRECTION_STRENGTHS="0.0"
  USE_QWEN=0
fi

LOOKAHEAD_MODES="${LOOKAHEAD_MODES:-standard instrumentation rollout_prior tree_prior rollout_tree_prior adaptive_cfg_width}"
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
read -r -a corr_strengths_arr <<< "${CORRECTION_STRENGTHS}"
read -r -a lookahead_modes_arr <<< "${LOOKAHEAD_MODES}"
if [[ "${#cfg_scales_arr[@]}" -eq 0 ]]; then
  echo "Error: CFG_SCALES is empty." >&2
  exit 1
fi
if [[ "${#lookahead_modes_arr[@]}" -eq 0 ]]; then
  echo "Error: LOOKAHEAD_MODES is empty." >&2
  exit 1
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_ROOT}/${RUN_TAG}_${RUN_TS}"
mkdir -p "${RUN_DIR}"

case "${SD35_BACKEND}" in
  sid) PRELOAD_MODEL_ID="${PRELOAD_MODEL_ID:-YGu1998/SiD-DiT-SD3.5-large}" ;;
  senseflow_large) PRELOAD_MODEL_ID="${PRELOAD_MODEL_ID:-stabilityai/stable-diffusion-3.5-large}" ;;
  senseflow_medium) PRELOAD_MODEL_ID="${PRELOAD_MODEL_ID:-stabilityai/stable-diffusion-3.5-medium}" ;;
  *) PRELOAD_MODEL_ID="${PRELOAD_MODEL_ID:-YGu1998/SiD-DiT-SD3.5-large}" ;;
esac

echo "[lookahead] run_dir=${RUN_DIR}"
echo "[lookahead] prompt_file=${PROMPT_FILE} range=[${START_INDEX},${END_INDEX})"
echo "[lookahead] backend=${SD35_BACKEND} reward_backend=${REWARD_BACKEND} num_gpus=${NUM_GPUS}"
echo "[lookahead] modes=[${LOOKAHEAD_MODES}] cfg_scales=[${CFG_SCALES}] n_variants=${N_VARIANTS}"

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

run_mode() {
  local lookahead_mode="$1"
  local mode_dir="${RUN_DIR}/lookahead_mode_${lookahead_mode}"
  mkdir -p "${mode_dir}"
  echo
  echo "[$(date '+%F %T')] lookahead_mode=${lookahead_mode} start -> ${mode_dir}"

  local -a extra_args=(--modes base mcts)
  if [[ "${USE_QWEN}" == "1" ]]; then
    extra_args+=(--qwen_id Qwen/Qwen3-4B)
  else
    extra_args+=(--no_qwen)
  fi
  if [[ "${SAVE_BEST_IMAGES:-1}" == "1" ]]; then
    extra_args+=(--save_best_images)
  fi
  if [[ -n "${REWARD_API_BASE}" ]]; then
    extra_args+=(--reward_api_base "${REWARD_API_BASE}")
  fi
  if [[ -n "${SD35_SIGMAS}" ]]; then
    extra_args+=(--sigmas ${SD35_SIGMAS})
  fi

  IMAGEREWARD_CACHE="${IMAGEREWARD_CACHE}" \
  HF_HOME="${HF_HOME}" \
  HPS_ROOT="${HPS_ROOT}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  torchrun --standalone --nproc_per_node "${NUM_GPUS}" \
    "${SCRIPT_DIR}/sd35_ddp_experiment_lookahead_reweighting.py" \
    --backend "${SD35_BACKEND}" \
    --prompt_file "${PROMPT_FILE}" \
    --start_index "${START_INDEX}" \
    --end_index "${END_INDEX}" \
    --gen_batch_size "${GEN_BATCH_SIZE}" \
    --steps "${STEPS}" \
    --cfg_scales "${cfg_scales_arr[@]}" \
    --baseline_cfg "${BASELINE_CFG}" \
    --n_variants "${N_VARIANTS}" \
    --correction_strengths "${corr_strengths_arr[@]}" \
    --n_sims "${N_SIMS}" \
    --ucb_c "${UCB_C}" \
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
    --out_dir "${mode_dir}" \
    --lookahead_mode "${lookahead_mode}" \
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
    "${extra_args[@]}"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_sd35_ddp.py" \
    --log_dir "${mode_dir}/logs" \
    --out_dir "${mode_dir}"
  echo "[$(date '+%F %T')] lookahead_mode=${lookahead_mode} done"
}

for mode in "${lookahead_modes_arr[@]}"; do
  run_mode "${mode}"
done

SUMMARY_TSV="${RUN_DIR}/lookahead_mode_summary.tsv"
"${PYTHON_BIN}" - <<'PY' "${RUN_DIR}" "${SUMMARY_TSV}" "${LOOKAHEAD_MODES}"
import json
import os
import sys

run_dir = sys.argv[1]
summary_tsv = sys.argv[2]
modes = [x for x in sys.argv[3].split() if x]

rows = []
for mode in modes:
    agg_path = os.path.join(run_dir, f"lookahead_mode_{mode}", "aggregate_summary.json")
    if not os.path.exists(agg_path):
        continue
    with open(agg_path, encoding="utf-8") as f:
        data = json.load(f)
    stats = data.get("mode_stats", {})
    base = stats.get("base", {})
    mcts = stats.get("mcts", {})
    rows.append(
        (
            mode,
            int(base.get("count", 0)),
            float(base.get("mean_score", 0.0)),
            int(mcts.get("count", 0)),
            float(mcts.get("mean_score", 0.0)),
            float(mcts.get("mean_delta_vs_base", 0.0)),
            float(mcts.get("std_delta_vs_base", 0.0)),
        )
    )

with open(summary_tsv, "w", encoding="utf-8") as f:
    f.write("lookahead_mode\tbase_count\tbase_mean\tmcts_count\tmcts_mean\tdelta_mean\tdelta_std\n")
    for row in rows:
        f.write(
            f"{row[0]}\t{row[1]}\t{row[2]:.6f}\t{row[3]}\t{row[4]:.6f}\t{row[5]:+.6f}\t{row[6]:.6f}\n"
        )

print("Lookahead mode summary:")
for row in rows:
    print(
        f"  {row[0]:<24} mcts_mean={row[4]:.4f} "
        f"delta_mean={row[5]:+.4f} n={row[3]}"
    )
print(f"\nWrote: {summary_tsv}")
PY

echo
echo "[done] run_dir=${RUN_DIR}"
echo "[done] summary=${SUMMARY_TSV}"
