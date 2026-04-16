#!/usr/bin/env bash
# Run SD3.5 MCTS with adaptive per-node CFG discretization on TACC.
#
# Usage:
#   bash run_sd35_dynamic_cfg_tacc.sh
#   MCTS_CFG_MODES="fixed adaptive" N_SIMS=20 bash run_sd35_dynamic_cfg_tacc.sh
#
# SLURM example:
#   sbatch --nodes=1 --ntasks=1 --cpus-per-task=8 \
#          --gres=gpu:a100:8 --time=24:00:00 \
#          --partition=gpu \
#          --wrap="bash ~/SiD-DiT_tts_dev/run_sd35_dynamic_cfg_tacc.sh"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SKIP_INSTALL="${SKIP_INSTALL:-1}"
source "${SCRIPT_DIR}/tacc_setup.sh"

# Prompt / output dirs (match run_tacc.sh layout)
_OUT_BASE="${SCRATCH:-${DATA_ROOT}}"
HPSV2_PROMPT_DIR="${HPSV2_PROMPT_DIR:-${_OUT_BASE}/hpsv2_prompt_cache}"
export HPSV2_PROMPT_DIR
RUN_TAG="${RUN_TAG:-sd35_dynamic_cfg}"
OUT_ROOT="${OUT_ROOT:-${_OUT_BASE}/hpsv2_all_models_runs/${RUN_TAG}}"
PROMPT_STYLE="${PROMPT_STYLE:-all}"
USE_SUBSET="${USE_SUBSET:-1}"
PROMPT_FILE="${PROMPT_FILE:-}"
START_INDEX="${START_INDEX:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
END_INDEX="${END_INDEX:-}"
NUM_GPUS="${NUM_GPUS:-$(${PYTHON_BIN} -c 'import torch; print(max(torch.cuda.device_count(),1))')}"

SD35_BACKEND="${SD35_BACKEND:-sd35_base}"
SD35_SIGMAS="${SD35_SIGMAS:-}"
STEPS="${STEPS:-28}"
SEED="${SEED:-42}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1}"
BASELINE_CFG="${BASELINE_CFG:-1.0}"
CFG_SCALES="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5}"
MCTS_CFG_MODES="${MCTS_CFG_MODES:-adaptive}"
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

CFG_ONLY="${CFG_ONLY:-0}"
N_VARIANTS="${N_VARIANTS:-4}"
CORRECTION_STRENGTHS="${CORRECTION_STRENGTHS:-0.0}"
USE_QWEN="${USE_QWEN:-1}"

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

SAVE_BEST_IMAGES="${SAVE_BEST_IMAGES:-1}"

# Prefer NUM_PROMPTS over END_INDEX so short test runs are predictable even if
# END_INDEX is already exported in the shell.
if (( NUM_PROMPTS > 0 )); then
  if [[ -n "${END_INDEX}" ]]; then
    echo "[dynamic-cfg] both NUM_PROMPTS and END_INDEX are set; using NUM_PROMPTS (END_INDEX ignored)."
  fi
  END_INDEX="$((START_INDEX + NUM_PROMPTS))"
elif [[ -z "${END_INDEX}" ]]; then
  END_INDEX="-1"
fi

# Build prompt file from HPSv2 cache unless PROMPT_FILE was explicitly given.
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

if [[ "${CFG_ONLY}" == "1" ]]; then
  N_VARIANTS=0
  CORRECTION_STRENGTHS="0.0"
  USE_QWEN=0
fi

# Backend-aware guidance defaults: keep SID scales for sid, use SD3.5-base scales otherwise.
if [[ -z "${CFG_SCALES:-}" || "${CFG_SCALES}" == "1.0 1.25 1.5 1.75 2.0 2.25 2.5" ]]; then
  if [[ "${SD35_BACKEND}" == "sid" ]]; then
    CFG_SCALES="1.0 1.25 1.5 1.75 2.0 2.25 2.5"
  else
    CFG_SCALES="3.5 4.0 4.5 5.0 5.5 6.0 7.0"
  fi
fi
if [[ -z "${BASELINE_CFG:-}" || "${BASELINE_CFG}" == "1.0" ]]; then
  if [[ "${SD35_BACKEND}" == "sid" ]]; then
    BASELINE_CFG="1.0"
  else
    BASELINE_CFG="4.5"
  fi
fi
if [[ -z "${MCTS_CFG_ROOT_BANK:-}" || "${MCTS_CFG_ROOT_BANK}" == "1.0 1.5 2.0 2.5" ]]; then
  if [[ "${SD35_BACKEND}" == "sid" ]]; then
    MCTS_CFG_ROOT_BANK="1.0 1.5 2.0 2.5"
  else
    MCTS_CFG_ROOT_BANK="4.0 4.5 5.0 5.5"
  fi
fi
if [[ -z "${MCTS_CFG_ANCHORS:-}" || "${MCTS_CFG_ANCHORS}" == "1.0 2.0" ]]; then
  if [[ "${SD35_BACKEND}" == "sid" ]]; then
    MCTS_CFG_ANCHORS="1.0 2.0"
  else
    MCTS_CFG_ANCHORS="3.5 7.0"
  fi
fi

read -r -a cfg_scales_arr <<< "${CFG_SCALES}"
read -r -a corr_strengths_arr <<< "${CORRECTION_STRENGTHS}"
read -r -a cfg_modes_arr <<< "${MCTS_CFG_MODES}"
read -r -a cfg_root_bank_arr <<< "${MCTS_CFG_ROOT_BANK}"
read -r -a cfg_anchors_arr <<< "${MCTS_CFG_ANCHORS}"

if [[ "${#cfg_scales_arr[@]}" -eq 0 ]]; then
  echo "Error: CFG_SCALES is empty." >&2
  exit 1
fi
if [[ "${#cfg_modes_arr[@]}" -eq 0 ]]; then
  echo "Error: MCTS_CFG_MODES is empty." >&2
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

echo "[dynamic-cfg] run_dir=${RUN_DIR}"
echo "[dynamic-cfg] prompt_file=${PROMPT_FILE} range=[${START_INDEX},${END_INDEX})"
echo "[dynamic-cfg] backend=${SD35_BACKEND} reward_backend=${REWARD_BACKEND} num_gpus=${NUM_GPUS}"
echo "[dynamic-cfg] cfg_scales=[${CFG_SCALES}] modes=[${MCTS_CFG_MODES}] cfg_only=${CFG_ONLY}"
echo "[dynamic-cfg] key_mode=${MCTS_KEY_MODE} key_steps='${MCTS_KEY_STEPS}' key_step_count=${MCTS_KEY_STEP_COUNT} key_step_stride=${MCTS_KEY_STEP_STRIDE}"
echo "[dynamic-cfg] fresh_noise_steps='${MCTS_FRESH_NOISE_STEPS}' samples=${MCTS_FRESH_NOISE_SAMPLES} scale=${MCTS_FRESH_NOISE_SCALE} key_steps=${MCTS_FRESH_NOISE_KEY_STEPS}"

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

run_cfg_mode() {
  local cfg_mode="$1"
  local mode_dir="${RUN_DIR}/cfg_mode_${cfg_mode}"
  mkdir -p "${mode_dir}"
  echo
  echo "[$(date '+%F %T')] cfg_mode=${cfg_mode} start -> ${mode_dir}"

  local -a extra_args=(--modes base mcts)
  if [[ "${USE_QWEN}" == "1" ]]; then
    extra_args+=(--qwen_id Qwen/Qwen3-4B)
  else
    extra_args+=(--no_qwen)
  fi
  if [[ "${SAVE_BEST_IMAGES}" == "1" ]]; then
    extra_args+=(--save_best_images)
  fi
  if [[ -n "${REWARD_API_BASE}" ]]; then
    extra_args+=(--reward_api_base "${REWARD_API_BASE}")
  fi
  if [[ -n "${SD35_SIGMAS}" ]]; then
    extra_args+=(--sigmas ${SD35_SIGMAS})
  fi
  extra_args+=(
    --mcts_key_mode "${MCTS_KEY_MODE}"
    --mcts_key_step_count "${MCTS_KEY_STEP_COUNT}"
    --mcts_key_step_stride "${MCTS_KEY_STEP_STRIDE}"
    --mcts_key_default_count "${MCTS_KEY_DEFAULT_COUNT}"
    --mcts_fresh_noise_steps "${MCTS_FRESH_NOISE_STEPS}"
    --mcts_fresh_noise_samples "${MCTS_FRESH_NOISE_SAMPLES}"
    --mcts_fresh_noise_scale "${MCTS_FRESH_NOISE_SCALE}"
  )
  if [[ -n "${MCTS_KEY_STEPS}" ]]; then
    extra_args+=(--mcts_key_steps "${MCTS_KEY_STEPS}")
  fi
  if [[ "${MCTS_FRESH_NOISE_KEY_STEPS}" == "1" ]]; then
    extra_args+=(--mcts_fresh_noise_key_steps)
  fi

  IMAGEREWARD_CACHE="${IMAGEREWARD_CACHE}" \
  HF_HOME="${HF_HOME}" \
  HPS_ROOT="${HPS_ROOT}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  torchrun --standalone --nproc_per_node "${NUM_GPUS}" \
    "${SCRIPT_DIR}/sd35_ddp_experiment_dynamic_cfg.py" \
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
    --mcts_cfg_mode "${cfg_mode}" \
    --mcts_cfg_root_bank "${cfg_root_bank_arr[@]}" \
    --mcts_cfg_anchors "${cfg_anchors_arr[@]}" \
    --mcts_cfg_step_anchor_count "${MCTS_CFG_STEP_ANCHOR_COUNT}" \
    --mcts_cfg_min_parent_visits "${MCTS_CFG_MIN_PARENT_VISITS}" \
    --mcts_cfg_round_ndigits "${MCTS_CFG_ROUND_NDIGITS}" \
    --mcts_cfg_log_action_topk "${MCTS_CFG_LOG_ACTION_TOPK}" \
    "${extra_args[@]}"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_sd35_ddp.py" \
    --log_dir "${mode_dir}/logs" \
    --out_dir "${mode_dir}"
  echo "[$(date '+%F %T')] cfg_mode=${cfg_mode} done"
}

for cfg_mode in "${cfg_modes_arr[@]}"; do
  run_cfg_mode "${cfg_mode}"
done

SUMMARY_TSV="${RUN_DIR}/cfg_mode_summary.tsv"
"${PYTHON_BIN}" - <<'PY' "${RUN_DIR}" "${SUMMARY_TSV}" "${MCTS_CFG_MODES}"
import json
import os
import sys

run_dir = sys.argv[1]
summary_tsv = sys.argv[2]
modes = [x for x in sys.argv[3].split() if x]

rows = []
for mode in modes:
    agg_path = os.path.join(run_dir, f"cfg_mode_{mode}", "aggregate_summary.json")
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
    f.write("cfg_mode\tbase_count\tbase_mean\tmcts_count\tmcts_mean\tdelta_mean\tdelta_std\n")
    for row in rows:
        f.write(
            f"{row[0]}\t{row[1]}\t{row[2]:.6f}\t{row[3]}\t{row[4]:.6f}\t{row[5]:+.6f}\t{row[6]:.6f}\n"
        )

print(f"[dynamic-cfg] wrote summary: {summary_tsv}")
PY

echo
echo "[dynamic-cfg] done. outputs: ${RUN_DIR}"
