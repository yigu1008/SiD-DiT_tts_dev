#!/usr/bin/env bash
# MCTS ablation study for SANA: prompt variants / CFG scale / reward correction
#
# Ablations (each adds one dimension vs the previous row):
#   none           – no search actions at all (pure baseline reference)
#   prompt         – prompt variants only          (V=3, CFG=1.0, C=off)
#   cfg            – CFG scale only                (V=1, CFG=all, C=off)
#   correction     – reward correction only        (V=1, CFG=1.0, C=[0,1])
#   prompt_cfg     – prompt + CFG                  (V=3, CFG=all, C=off)
#   prompt_corr    – prompt + correction           (V=3, CFG=1.0, C=[0,1])
#   cfg_corr       – CFG + correction              (V=1, CFG=all, C=[0,1])
#   full           – prompt + CFG + correction     (V=3, CFG=all, C=[0,1])
#
# Usage:
#   bash run_sana_tacc.sh
#   ABLATIONS="prompt_cfg full" bash run_sana_tacc.sh
#   N_SIMS=20 bash run_sana_tacc.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Use tacc_setup.sh (SKIP_INSTALL=1 = env vars only, no pip) to get HF_HOME,
# DATA_ROOT, IMAGEREWARD_CACHE, etc. pointing at the right model cache on $WORK.
export SKIP_INSTALL="${SKIP_INSTALL:-1}"
source "${SCRIPT_DIR}/tacc_setup.sh"

# ---------------------------------------------------------------------------
# Base config
# ---------------------------------------------------------------------------
MODEL_ID="${MODEL_ID:-YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/hpsv2_subset.txt}"
# Outputs go to $SCRATCH (large quota); model cache stays on $WORK via DATA_ROOT/HF_HOME.
OUT_ROOT="${OUT_ROOT:-${SCRATCH:-${DATA_ROOT}}/mcts_ablation_sana}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:--1}"

STEPS="${STEPS:-4}"
SEED="${SEED:-42}"
N_SIMS="${N_SIMS:-50}"
UCB_C="${UCB_C:-1.41}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"

REWARD_TYPE="${REWARD_TYPE:-imagereward}"
REWARD_DEVICE="${REWARD_DEVICE:-cpu}"
IMAGE_REWARD_MODEL="${IMAGE_REWARD_MODEL:-ImageReward-v1.0}"
PICKSCORE_MODEL="${PICKSCORE_MODEL:-yuvalkirstain/PickScore_v1}"
REWARD_MODEL="${REWARD_MODEL:-CodeGoat24/UnifiedReward-qwen-7b}"
UNIFIEDREWARD_MODEL="${UNIFIEDREWARD_MODEL:-${REWARD_MODEL}}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-1.0 1.0}"
REWARD_API_BASE="${REWARD_API_BASE:-}"
REWARD_API_KEY="${REWARD_API_KEY:-unifiedreward}"
REWARD_API_MODEL="${REWARD_API_MODEL:-UnifiedReward-7b-v1.5}"
REWARD_MAX_NEW_TOKENS="${REWARD_MAX_NEW_TOKENS:-512}"
REWARD_PROMPT_MODE="${REWARD_PROMPT_MODE:-standard}"

SAVE_FIRST_K="${SAVE_FIRST_K:-0}"

# Qwen rewrite precomputation (shared across prompt-using ablations)
PRECOMPUTE_REWRITES="${PRECOMPUTE_REWRITES:-1}"
REWRITES_OVERWRITE="${REWRITES_OVERWRITE:-0}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen3-4B}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
QWEN_PRECOMPUTE_DEVICE="${QWEN_PRECOMPUTE_DEVICE:-auto}"
QWEN_PRECOMPUTE_BATCH_SIZE="${QWEN_PRECOMPUTE_BATCH_SIZE:-16}"
QWEN_PRECOMPUTE_SAVE_EVERY="${QWEN_PRECOMPUTE_SAVE_EVERY:-1}"
QWEN_PRECOMPUTE_MAX_NEW_TOKENS="${QWEN_PRECOMPUTE_MAX_NEW_TOKENS:-120}"
QWEN_PRECOMPUTE_TEMPERATURE="${QWEN_PRECOMPUTE_TEMPERATURE:-0.6}"
QWEN_PRECOMPUTE_TOP_P="${QWEN_PRECOMPUTE_TOP_P:-0.9}"
QWEN_PRECOMPUTE_CLEAR_CACHE="${QWEN_PRECOMPUTE_CLEAR_CACHE:-1}"

ABLATIONS="${ABLATIONS:-none prompt cfg correction prompt_cfg prompt_corr cfg_corr full}"

# ---------------------------------------------------------------------------
# Ablation definitions: name → (N_VARIANTS, CFG_SCALES, CORRECTION_STRENGTHS, USE_QWEN)
# ---------------------------------------------------------------------------
ALL_CFG_SCALES="1.0 1.25 1.5 1.75 2.0 2.25 2.5"

declare -A ABLATION_N_VARIANTS=(
  [none]=1
  [prompt]=3
  [cfg]=1
  [correction]=1
  [prompt_cfg]=3
  [prompt_corr]=3
  [cfg_corr]=1
  [full]=3
)
declare -A ABLATION_CFG_SCALES=(
  [none]="1.0"
  [prompt]="1.0"
  [cfg]="${ALL_CFG_SCALES}"
  [correction]="1.0"
  [prompt_cfg]="${ALL_CFG_SCALES}"
  [prompt_corr]="1.0"
  [cfg_corr]="${ALL_CFG_SCALES}"
  [full]="${ALL_CFG_SCALES}"
)
declare -A ABLATION_CORRECTION_STRENGTHS=(
  [none]="0.0"
  [prompt]="0.0"
  [cfg]="0.0"
  [correction]="0.0 1.0"
  [prompt_cfg]="0.0"
  [prompt_corr]="0.0 1.0"
  [cfg_corr]="0.0 1.0"
  [full]="0.0 1.0"
)
declare -A ABLATION_USE_QWEN=(
  [none]=0
  [prompt]=1
  [cfg]=0
  [correction]=0
  [prompt_cfg]=1
  [prompt_corr]=1
  [cfg_corr]=0
  [full]=1
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found: ${PROMPT_FILE}" >&2
  exit 1
fi
mkdir -p "${OUT_ROOT}"

ABLATION_TS="$(date +%Y%m%d_%H%M%S)"
ABLATION_DIR="${OUT_ROOT}/ablation_${ABLATION_TS}"
mkdir -p "${ABLATION_DIR}"
SUMMARY_TSV="${ABLATION_DIR}/ablation_summary.tsv"

echo "SANA MCTS ablation study"
echo "  ablations: ${ABLATIONS}"
echo "  prompt_file: ${PROMPT_FILE}"
echo "  n_sims: ${N_SIMS}"
echo "  out: ${ABLATION_DIR}"
echo

# ---------------------------------------------------------------------------
# GPU discovery
# ---------------------------------------------------------------------------
GPU_IDS_STR="$("${PYTHON_BIN}" - <<'PY'
import os, torch
cvd = os.environ.get("CUDA_VISIBLE_DEVICES","").strip()
ids = [x.strip() for x in cvd.split(",") if x.strip()] if cvd else [str(i) for i in range(torch.cuda.device_count())]
print(",".join(ids))
PY
)"
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
NUM_GPUS="${#GPU_IDS[@]}"
if (( NUM_GPUS == 0 )); then
  echo "Error: no visible GPUs." >&2; exit 1
fi

# Prompt range
read -r TOTAL_PROMPTS EFFECTIVE_END <<EOF
$("${PYTHON_BIN}" - <<'PY' "${PROMPT_FILE}" "${START_INDEX}" "${END_INDEX}"
import sys
path, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, encoding="utf-8") as f:
    prompts = [l.strip() for l in f if l.strip()]
total = len(prompts)
if end < 0: end = total
end = min(end, total); start = max(0, min(start, end))
print(total, end)
PY
)
EOF
RANGE_TOTAL=$(( EFFECTIVE_END - START_INDEX ))
if (( RANGE_TOTAL <= 0 )); then
  echo "Error: empty prompt range start=${START_INDEX} end=${EFFECTIVE_END}" >&2; exit 1
fi

echo "  gpus(${NUM_GPUS}): ${GPU_IDS[*]}"
echo "  range=[${START_INDEX},${EFFECTIVE_END}) total=${RANGE_TOTAL}"
echo

# ---------------------------------------------------------------------------
# Pre-compute Qwen rewrites once (shared by prompt-using ablations)
# ---------------------------------------------------------------------------
SHARED_REWRITES="${ABLATION_DIR}/rewrites_cache.json"
precompute_shared_rewrites() {
  if [[ "${PRECOMPUTE_REWRITES}" != "1" ]]; then return 0; fi
  # Check if any selected ablation uses Qwen
  local needs_qwen=0
  for abl in ${ABLATIONS}; do
    if [[ "${ABLATION_USE_QWEN[${abl}]:-0}" == "1" ]]; then needs_qwen=1; break; fi
  done
  if [[ "${needs_qwen}" != "1" ]]; then return 0; fi

  if [[ -s "${SHARED_REWRITES}" && "${REWRITES_OVERWRITE}" != "1" ]]; then
    echo "[rewrites] using existing cache: ${SHARED_REWRITES}"
    return 0
  fi
  echo "[rewrites] precomputing shared Qwen rewrites (N_VARIANTS=3) ..."
  local -a cmd=(
    "${PYTHON_BIN}" "-u" "${SCRIPT_DIR}/precompute_sd35_rewrites.py"
    --prompt_file "${PROMPT_FILE}"
    --rewrites_file "${SHARED_REWRITES}"
    --start_index "${START_INDEX}"
    --end_index "${END_INDEX}"
    --n_variants 3
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
  fi
  env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u LOCAL_WORLD_SIZE -u NODE_RANK -u MASTER_ADDR -u MASTER_PORT \
    "${cmd[@]}"
  echo "[rewrites] cache ready: ${SHARED_REWRITES}"
}
precompute_shared_rewrites

# ---------------------------------------------------------------------------
# Run one ablation config (sharded across GPUs in parallel)
# ---------------------------------------------------------------------------
run_ablation() {
  local name="$1"
  if [[ -z "${ABLATION_N_VARIANTS[${name}]+x}" ]]; then
    echo "Error: unknown ablation '${name}'. Valid: ${!ABLATION_N_VARIANTS[*]}" >&2
    exit 1
  fi

  local n_var="${ABLATION_N_VARIANTS[${name}]}"
  local cfg_scales="${ABLATION_CFG_SCALES[${name}]}"
  local corr_strengths="${ABLATION_CORRECTION_STRENGTHS[${name}]}"
  local use_qwen="${ABLATION_USE_QWEN[${name}]}"
  local abl_dir="${ABLATION_DIR}/${name}"
  mkdir -p "${abl_dir}"

  echo "[$(date '+%F %T')] ablation=${name} start"
  echo "  n_variants=${n_var} cfg_scales=[${cfg_scales}] correction_strengths=[${corr_strengths}] use_qwen=${use_qwen}"

  # Build extra args
  local -a extra=()
  if [[ "${use_qwen}" == "1" && -s "${SHARED_REWRITES}" ]]; then
    extra+=(--rewrites_file "${SHARED_REWRITES}" --no_qwen)
  else
    extra+=(--no_qwen)
  fi
  if [[ -n "${REWARD_API_BASE}" ]]; then
    extra+=(--reward_api_base "${REWARD_API_BASE}")
  fi

  local chunk_size=$(( (RANGE_TOTAL + NUM_GPUS - 1) / NUM_GPUS ))
  local -a pids=()
  local launched=0
  local begin_ts
  begin_ts="$(date +%s)"

  for rank in "${!GPU_IDS[@]}"; do
    local shard_start=$(( START_INDEX + rank * chunk_size ))
    local shard_end=$(( shard_start + chunk_size ))
    (( shard_start >= EFFECTIVE_END )) && continue
    (( shard_end > EFFECTIVE_END )) && shard_end="${EFFECTIVE_END}"

    local rank_out="${abl_dir}/rank_${rank}"
    local rank_prompt="${abl_dir}/prompts_rank_${rank}.txt"
    local log_file="${abl_dir}/rank_${rank}.log"
    local gpu="${GPU_IDS[$rank]}"
    mkdir -p "${rank_out}"

    "${PYTHON_BIN}" - <<'PY' "${PROMPT_FILE}" "${rank_prompt}" "${shard_start}" "${shard_end}"
import sys
src, dst, start, end = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(src, encoding="utf-8") as f:
    prompts = [l.strip() for l in f if l.strip()]
with open(dst, "w", encoding="utf-8") as f:
    f.writelines(p + "\n" for p in prompts[start:end])
PY

    launched=$(( launched + 1 ))
    CUDA_VISIBLE_DEVICES="${gpu}" \
    IMAGEREWARD_CACHE="${IMAGEREWARD_CACHE}" \
    HF_HOME="${HF_HOME}" \
    HPS_ROOT="${HPS_ROOT}" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/sampling_unified.py" \
      --search_method mcts \
      --model_id "${MODEL_ID}" \
      --prompt_file "${rank_prompt}" \
      --steps "${STEPS}" \
      --width "${WIDTH}" \
      --height "${HEIGHT}" \
      --seed "${SEED}" \
      --dtype bf16 \
      --gpu_id 0 \
      --auto_select_gpu \
      --n_variants "${n_var}" \
      --cfg_scales ${cfg_scales} \
      --correction_strengths ${corr_strengths} \
      --n_sims "${N_SIMS}" \
      --ucb_c "${UCB_C}" \
      --reward_type "${REWARD_TYPE}" \
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
      --save_first_k "${SAVE_FIRST_K}" \
      --offload_text_encoder_after_encode \
      --decode_device auto \
      --decode_cpu_if_free_below_gb 16 \
      --empty_cache_after_decode \
      --out_dir "${rank_out}" \
      "${extra[@]}" \
      >"${log_file}" 2>&1 &
    pids+=("$!")
    echo "  rank=${rank} gpu=${gpu} range=[${shard_start},${shard_end}) log=${log_file}"
  done

  if (( launched == 0 )); then
    echo "Error: no shards launched for ablation=${name}." >&2; exit 1
  fi

  local failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  if (( failed != 0 )); then
    echo "Error: ablation=${name} failed on at least one shard." >&2; exit 1
  fi

  local end_ts elapsed
  end_ts="$(date +%s)"
  elapsed=$(( end_ts - begin_ts ))
  echo "[$(date '+%F %T')] ablation=${name} done elapsed=${elapsed}s"
}

# ---------------------------------------------------------------------------
# Run all selected ablations
# ---------------------------------------------------------------------------
for abl in ${ABLATIONS}; do
  run_ablation "${abl}"
done

# ---------------------------------------------------------------------------
# Compare results
# ---------------------------------------------------------------------------
echo
echo "[compare] collecting results ..."
"${PYTHON_BIN}" - <<'PY' "${ABLATION_DIR}" "${SUMMARY_TSV}" ${ABLATIONS}
import json, os, sys, glob, statistics

ablation_dir = sys.argv[1]
summary_tsv  = sys.argv[2]
ablation_names = sys.argv[3:]

rows = []
for name in ablation_names:
    abl_path = os.path.join(ablation_dir, name)
    baseline_scores, mcts_scores, deltas = [], [], []
    for summary_path in glob.glob(os.path.join(abl_path, "rank_*", "summary.json")):
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        samples = data if isinstance(data, list) else data.get("samples", [])
        for row in samples:
            for sample in (row.get("samples", []) if isinstance(row, dict) and "samples" in row else [row]):
                b = float(sample.get("baseline_score", 0.0))
                s = float(sample.get("search_score", b))
                d = float(sample.get("delta_score", s - b))
                baseline_scores.append(b)
                mcts_scores.append(s)
                deltas.append(d)
    if not mcts_scores:
        print(f"  [skip] {name}: no results")
        continue
    rows.append({
        "name": name,
        "n": len(mcts_scores),
        "base": statistics.fmean(baseline_scores) if baseline_scores else 0.0,
        "mcts": statistics.fmean(mcts_scores),
        "delta": statistics.fmean(deltas),
        "std":   statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
    })

if not rows:
    print("[compare] no results found.")
    sys.exit(0)

rows.sort(key=lambda r: r["delta"], reverse=True)
header = f"{'Ablation':<18} {'N':>5} {'Base':>8} {'MCTS':>8} {'Δ':>8} {'std':>7}"
sep    = "-" * len(header)
print(); print(header); print(sep)
for r in rows:
    print(f"{r['name']:<18} {r['n']:>5} {r['base']:>8.4f} {r['mcts']:>8.4f} {r['delta']:>+8.4f} {r['std']:>7.4f}")
print(sep)

with open(summary_tsv, "w", encoding="utf-8") as f:
    f.write("ablation\tn\tbase\tmcts\tdelta\tstd\n")
    for r in rows:
        f.write(f"{r['name']}\t{r['n']}\t{r['base']:.6f}\t{r['mcts']:.6f}\t{r['delta']:+.6f}\t{r['std']:.6f}\n")

print(f"\nSummary TSV: {summary_tsv}")
PY

echo
echo "Ablation outputs: ${ABLATION_DIR}"
