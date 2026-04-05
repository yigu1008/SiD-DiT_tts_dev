#!/usr/bin/env bash
# Run SANA global-blend SPSA design ablation on TACC:
#   gb_spsa, gb_nlerp_spsa, gb_slerp_spsa
#
# Usage:
#   bash run_sana_design_spsa_tacc.sh
#   NUM_PROMPTS=10 N_SIMS=20 bash run_sana_design_spsa_tacc.sh
#
# SLURM:
#   sbatch --nodes=1 --ntasks=1 --cpus-per-task=8 \
#          --gres=gpu:a100:8 --time=24:00:00 \
#          --partition=gpu \
#          --wrap="bash ~/SiD-DiT_tts_dev/run_sana_design_spsa_tacc.sh"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SKIP_INSTALL="${SKIP_INSTALL:-1}"
source "${SCRIPT_DIR}/tacc_setup.sh"

# Prompt/output layout (aligned with run_tacc.sh style)
_OUT_BASE="${SCRATCH:-${DATA_ROOT}}"
HPSV2_PROMPT_DIR="${HPSV2_PROMPT_DIR:-${_OUT_BASE}/hpsv2_prompt_cache}"
export HPSV2_PROMPT_DIR
RUN_TAG="${RUN_TAG:-sana_design_spsa}"
OUT_ROOT="${OUT_ROOT:-${_OUT_BASE}/hpsv2_all_models_runs/${RUN_TAG}}"
PROMPT_STYLE="${PROMPT_STYLE:-all}"
USE_SUBSET="${USE_SUBSET:-1}"
PROMPT_FILE="${PROMPT_FILE:-}"
START_INDEX="${START_INDEX:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
END_INDEX="${END_INDEX:-$((START_INDEX + NUM_PROMPTS))}"

MODEL_ID="${MODEL_ID:-YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow}"
DTYPE="${DTYPE:-bf16}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
STEPS="${STEPS:-4}"
SEED="${SEED:-42}"
BASELINE_CFG="${BASELINE_CFG:-1.0}"
CFG_SCALES="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5}"
FAMILIES="${FAMILIES:-nlerp slerp}"
PREVIEW_EVERY="${PREVIEW_EVERY:--1}"
SAVE_IMAGES="${SAVE_IMAGES:-0}"
SAVE_FIRST_K="${SAVE_FIRST_K:-0}"

REWARD_TYPE="${REWARD_TYPE:-imagereward}"
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

NO_QWEN="${NO_QWEN:-1}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen3-4B}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
QWEN_DEVICE="${QWEN_DEVICE:-auto}"
QWEN_TIMEOUT_SEC="${QWEN_TIMEOUT_SEC:-240}"

# Core test toggles (this is the gb_spsa/gb_nlerp_spsa/gb_slerp_spsa test)
RUN_SWEEP="${RUN_SWEEP:-0}"
RUN_GA="${RUN_GA:-0}"
RUN_MCTS="${RUN_MCTS:-1}"
RUN_MCTS_DESIGN_ABLATION="${RUN_MCTS_DESIGN_ABLATION:-1}"
RUN_MCTS_FAMILY_SPSA_ABLATION="${RUN_MCTS_FAMILY_SPSA_ABLATION:-1}"
ENABLE_PROMPT_WEIGHT_SPSA="${ENABLE_PROMPT_WEIGHT_SPSA:-1}"
RUN_MCTS_WEIGHT_ABLATION="${RUN_MCTS_WEIGHT_ABLATION:-0}"

N_SIMS="${N_SIMS:-50}"
UCB_C="${UCB_C:-1.41}"
MCTS_LOG_EVERY="${MCTS_LOG_EVERY:-10}"
NUM_PROMPT_WEIGHTS="${NUM_PROMPT_WEIGHTS:-4}"
SPSA_BLOCK_ROLLOUTS="${SPSA_BLOCK_ROLLOUTS:-8}"
SPSA_C="${SPSA_C:-0.07}"
SPSA_ETA="${SPSA_ETA:-0.05}"
WEIGHT_CLIP_MIN="${WEIGHT_CLIP_MIN:-0.5}"
WEIGHT_CLIP_MAX="${WEIGHT_CLIP_MAX:-2.0}"

NUM_GPUS="${NUM_GPUS:-$(${PYTHON_BIN} -c 'import torch; print(max(torch.cuda.device_count(),1))')}"

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

read -r TOTAL_PROMPTS EFFECTIVE_END <<EOF
$("${PYTHON_BIN}" - <<'PY' "${PROMPT_FILE}" "${START_INDEX}" "${END_INDEX}"
import sys
path, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, encoding="utf-8") as f:
    prompts = [l.strip() for l in f if l.strip()]
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

GPU_IDS_STR="$("${PYTHON_BIN}" - <<'PY'
import os, torch
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if cvd:
    ids = [x.strip() for x in cvd.split(",") if x.strip()]
else:
    ids = [str(i) for i in range(torch.cuda.device_count())]
print(",".join(ids))
PY
)"
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
if (( ${#GPU_IDS[@]} == 0 )); then
  echo "Error: no visible GPUs." >&2
  exit 1
fi
if (( NUM_GPUS > ${#GPU_IDS[@]} )); then
  NUM_GPUS="${#GPU_IDS[@]}"
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_ROOT}/${RUN_TAG}_${RUN_TS}"
mkdir -p "${RUN_DIR}"

echo "[sana-design-spsa] run_dir=${RUN_DIR}"
echo "[sana-design-spsa] prompt_file=${PROMPT_FILE}"
echo "[sana-design-spsa] range=[${START_INDEX},${EFFECTIVE_END}) total=${RANGE_TOTAL}"
echo "[sana-design-spsa] gpus=${NUM_GPUS} ids=${GPU_IDS[*]}"
echo "[sana-design-spsa] reward_type=${REWARD_TYPE} families=[${FAMILIES}]"

echo "[preload] caching model: ${MODEL_ID}"
env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u LOCAL_WORLD_SIZE -u NODE_RANK \
  -u MASTER_ADDR -u MASTER_PORT \
  HF_HOME="${HF_HOME}" \
  _PRELOAD_MODEL_ID="${MODEL_ID}" \
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

read -r -a cfg_scales_arr <<< "${CFG_SCALES}"
read -r -a reward_weights_arr <<< "${REWARD_WEIGHTS}"
read -r -a families_arr <<< "${FAMILIES}"

if [[ "${#cfg_scales_arr[@]}" -eq 0 ]]; then
  echo "Error: CFG_SCALES is empty." >&2
  exit 1
fi
if [[ "${#reward_weights_arr[@]}" -ne 2 ]]; then
  echo "Error: REWARD_WEIGHTS must contain exactly 2 values." >&2
  exit 1
fi
if [[ "${#families_arr[@]}" -eq 0 ]]; then
  echo "Error: FAMILIES is empty." >&2
  exit 1
fi

algo_flags=()
if [[ "${RUN_SWEEP}" == "1" ]]; then
  algo_flags+=(--run_sweep)
else
  algo_flags+=(--no-run_sweep)
fi
if [[ "${RUN_GA}" == "1" ]]; then
  algo_flags+=(--run_ga)
else
  algo_flags+=(--no-run_ga)
fi
if [[ "${RUN_MCTS}" == "1" ]]; then
  algo_flags+=(--run_mcts)
else
  algo_flags+=(--no-run_mcts)
fi
if [[ "${RUN_MCTS_DESIGN_ABLATION}" == "1" ]]; then
  algo_flags+=(--run_mcts_design_ablation)
else
  algo_flags+=(--no-run_mcts_design_ablation)
fi
if [[ "${RUN_MCTS_FAMILY_SPSA_ABLATION}" == "1" ]]; then
  algo_flags+=(--run_mcts_family_spsa_ablation)
else
  algo_flags+=(--no-run_mcts_family_spsa_ablation)
fi
if [[ "${ENABLE_PROMPT_WEIGHT_SPSA}" == "1" ]]; then
  algo_flags+=(--enable_prompt_weight_spsa)
else
  algo_flags+=(--no-enable_prompt_weight_spsa)
fi
if [[ "${RUN_MCTS_WEIGHT_ABLATION}" == "1" ]]; then
  algo_flags+=(--run_mcts_weight_ablation)
else
  algo_flags+=(--no-run_mcts_weight_ablation)
fi

img_flag=(--no-save_images)
if [[ "${SAVE_IMAGES}" == "1" ]]; then
  img_flag=(--save_images)
fi

qwen_flags=()
if [[ "${NO_QWEN}" == "1" ]]; then
  qwen_flags+=(--no_qwen)
else
  qwen_flags+=(--qwen_id "${QWEN_ID}" --qwen_dtype "${QWEN_DTYPE}")
fi

extra_reward_flags=()
if [[ -n "${REWARD_API_BASE}" ]]; then
  extra_reward_flags+=(--reward_api_base "${REWARD_API_BASE}")
fi

chunk_size=$(( (RANGE_TOTAL + NUM_GPUS - 1) / NUM_GPUS ))
pids=()
launched=0

for (( rank=0; rank<NUM_GPUS; rank++ )); do
  shard_start=$(( START_INDEX + rank * chunk_size ))
  shard_end=$(( shard_start + chunk_size ))
  (( shard_start >= EFFECTIVE_END )) && continue
  (( shard_end > EFFECTIVE_END )) && shard_end="${EFFECTIVE_END}"

  gpu="${GPU_IDS[$rank]}"
  rank_dir="${RUN_DIR}/rank_${rank}"
  rank_prompt="${RUN_DIR}/prompts_rank_${rank}.txt"
  rank_log="${RUN_DIR}/rank_${rank}.log"
  mkdir -p "${rank_dir}"

  "${PYTHON_BIN}" - <<'PY' "${PROMPT_FILE}" "${rank_prompt}" "${shard_start}" "${shard_end}"
import sys
src, dst, start, end = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(src, encoding="utf-8") as f:
    prompts = [l.strip() for l in f if l.strip()]
with open(dst, "w", encoding="utf-8") as f:
    for p in prompts[start:end]:
        f.write(p + "\n")
PY

  launched=$(( launched + 1 ))
  CUDA_VISIBLE_DEVICES="${gpu}" \
  IMAGEREWARD_CACHE="${IMAGEREWARD_CACHE}" \
  HF_HOME="${HF_HOME}" \
  HPS_ROOT="${HPS_ROOT}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/sandbox_slerp_nlerp_unified_sana.py" \
    --prompt_file "${rank_prompt}" \
    --max_prompts -1 \
    --out_dir "${rank_dir}" \
    --model_id "${MODEL_ID}" \
    --dtype "${DTYPE}" \
    --steps "${STEPS}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --seed "${SEED}" \
    --guidance_scale "${BASELINE_CFG}" \
    --baseline_cfg "${BASELINE_CFG}" \
    --cfg_scales "${cfg_scales_arr[@]}" \
    --reward_type "${REWARD_TYPE}" \
    --reward_device "${REWARD_DEVICE}" \
    --reward_model "${REWARD_MODEL}" \
    --unifiedreward_model "${UNIFIEDREWARD_MODEL}" \
    --image_reward_model "${IMAGE_REWARD_MODEL}" \
    --pickscore_model "${PICKSCORE_MODEL}" \
    --reward_weights "${reward_weights_arr[0]}" "${reward_weights_arr[1]}" \
    --reward_api_key "${REWARD_API_KEY}" \
    --reward_api_model "${REWARD_API_MODEL}" \
    --reward_max_new_tokens "${REWARD_MAX_NEW_TOKENS}" \
    --reward_prompt_mode "${REWARD_PROMPT_MODE}" \
    "${extra_reward_flags[@]}" \
    --qwen_device "${QWEN_DEVICE}" \
    --qwen_timeout_sec "${QWEN_TIMEOUT_SEC}" \
    "${qwen_flags[@]}" \
    --families "${families_arr[@]}" \
    --preview_every "${PREVIEW_EVERY}" \
    --mcts_n_sims "${N_SIMS}" \
    --mcts_ucb_c "${UCB_C}" \
    --mcts_log_every "${MCTS_LOG_EVERY}" \
    --num_prompt_weights "${NUM_PROMPT_WEIGHTS}" \
    --spsa_block_rollouts "${SPSA_BLOCK_ROLLOUTS}" \
    --spsa_c "${SPSA_C}" \
    --spsa_eta "${SPSA_ETA}" \
    --weight_clip_min "${WEIGHT_CLIP_MIN}" \
    --weight_clip_max "${WEIGHT_CLIP_MAX}" \
    --save_first_k "${SAVE_FIRST_K}" \
    "${img_flag[@]}" \
    "${algo_flags[@]}" \
    >"${rank_log}" 2>&1 &
  pids+=("$!")
  echo "  rank=${rank} gpu=${gpu} range=[${shard_start},${shard_end}) log=${rank_log}"
done

if (( launched == 0 )); then
  echo "Error: no shards launched." >&2
  exit 1
fi

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if (( failed != 0 )); then
  echo "Error: at least one shard failed. See rank_*.log." >&2
  exit 1
fi

SUMMARY_TSV="${RUN_DIR}/design_spsa_summary.tsv"
"${PYTHON_BIN}" - <<'PY' "${RUN_DIR}" "${SUMMARY_TSV}"
import glob
import json
import os
import statistics
import sys

run_dir = sys.argv[1]
summary_tsv = sys.argv[2]

methods = {
    "baseline": [],
    "variant4": [],
    "gb_fixed": [],
    "gb_spsa": [],
    "gb_nlerp_spsa": [],
    "gb_slerp_spsa": [],
}

def _score(d):
    if not isinstance(d, dict):
        return None
    if "final_score" in d:
        try:
            return float(d["final_score"])
        except Exception:
            return None
    if "best_score" in d:
        try:
            return float(d["best_score"])
        except Exception:
            return None
    return None

for result_path in sorted(glob.glob(os.path.join(run_dir, "rank_*", "p*", "result.json"))):
    with open(result_path, encoding="utf-8") as f:
        row = json.load(f)
    base = row.get("baseline_score")
    if base is not None:
        methods["baseline"].append(float(base))

    da = row.get("mcts_design_ablation") or {}
    v = _score(da.get("variant4"))
    if v is not None:
        methods["variant4"].append(v)
    v = _score(da.get("global_blend_fixed"))
    if v is not None:
        methods["gb_fixed"].append(v)
    v = _score(da.get("global_blend_spsa"))
    if v is not None:
        methods["gb_spsa"].append(v)
    v = _score(da.get("global_blend_nlerp_spsa"))
    if v is not None:
        methods["gb_nlerp_spsa"].append(v)
    v = _score(da.get("global_blend_slerp_spsa"))
    if v is not None:
        methods["gb_slerp_spsa"].append(v)

with open(summary_tsv, "w", encoding="utf-8") as f:
    f.write("method\tmean\tn\n")
    for name in ["baseline", "variant4", "gb_fixed", "gb_spsa", "gb_nlerp_spsa", "gb_slerp_spsa"]:
        vals = methods[name]
        if vals:
            f.write(f"{name}\t{statistics.fmean(vals):.4f}\t{len(vals)}\n")
        else:
            f.write(f"{name}\tNA\t0\n")

print("Aggregate:")
for name in ["baseline", "variant4", "gb_fixed", "gb_spsa", "gb_nlerp_spsa", "gb_slerp_spsa"]:
    vals = methods[name]
    if vals:
        print(f"  {name:<20} {statistics.fmean(vals):>9.4f} {len(vals):>6d}")
    else:
        print(f"  {name:<20} {'NA':>9} {0:>6d}")
print(f"\nSummary TSV: {summary_tsv}")
PY

echo
echo "[sana-design-spsa] done. outputs: ${RUN_DIR}"
