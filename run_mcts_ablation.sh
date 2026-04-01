#!/usr/bin/env bash
# MCTS ablation study: prompt variants / CFG scale / reward correction
#
# Runs baseline + MCTS for each ablation config, storing to labeled subdirs.
# Results are compared in a summary TSV at the end.
#
# Usage:
#   bash run_mcts_ablation.sh
#   ABLATIONS="prompt_cfg full" bash run_mcts_ablation.sh   # subset
#   N_SIMS=20 bash run_mcts_ablation.sh                      # quick test
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

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Use tacc_setup.sh (SKIP_INSTALL=1 = env vars only, no pip) to get HF_HOME,
# DATA_ROOT, IMAGEREWARD_CACHE, etc. pointing at the right model cache on home1/work.
export SKIP_INSTALL="${SKIP_INSTALL:-1}"
source "${SCRIPT_DIR}/tacc_setup.sh"

# ---------------------------------------------------------------------------
# Base config (all ablations inherit these; override per-ablation below)
# ---------------------------------------------------------------------------
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/hpsv2_subset.txt}"
# Outputs go to $SCRATCH (large quota); model cache stays on $WORK via DATA_ROOT/HF_HOME.
OUT_ROOT="${OUT_ROOT:-${SCRATCH:-${DATA_ROOT}}/mcts_ablation}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:--1}"
NUM_GPUS="${NUM_GPUS:-$(${PYTHON_BIN} -c 'import torch; print(max(torch.cuda.device_count(),1))')}"
SD35_BACKEND="${SD35_BACKEND:-sid}"
SD35_SIGMAS="${SD35_SIGMAS:-}"

STEPS="${STEPS:-4}"
SEED="${SEED:-42}"
BASELINE_CFG="${BASELINE_CFG:-1.0}"
N_SIMS="${N_SIMS:-50}"
UCB_C="${UCB_C:-1.41}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1}"

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

SAVE_BEST_IMAGES="${SAVE_BEST_IMAGES:-0}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen3-4B}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
QWEN_TIMEOUT_SEC="${QWEN_TIMEOUT_SEC:-240}"
PRECOMPUTE_REWRITES="${PRECOMPUTE_REWRITES:-1}"
REWRITES_OVERWRITE="${REWRITES_OVERWRITE:-0}"
QWEN_PRECOMPUTE_DEVICE="${QWEN_PRECOMPUTE_DEVICE:-auto}"
QWEN_PRECOMPUTE_BATCH_SIZE="${QWEN_PRECOMPUTE_BATCH_SIZE:-16}"
QWEN_PRECOMPUTE_SAVE_EVERY="${QWEN_PRECOMPUTE_SAVE_EVERY:-1}"
QWEN_PRECOMPUTE_MAX_NEW_TOKENS="${QWEN_PRECOMPUTE_MAX_NEW_TOKENS:-120}"
QWEN_PRECOMPUTE_TEMPERATURE="${QWEN_PRECOMPUTE_TEMPERATURE:-0.6}"
QWEN_PRECOMPUTE_TOP_P="${QWEN_PRECOMPUTE_TOP_P:-0.9}"
QWEN_PRECOMPUTE_CLEAR_CACHE="${QWEN_PRECOMPUTE_CLEAR_CACHE:-1}"

# All ablations — change with ABLATIONS="..." env var
ABLATIONS="${ABLATIONS:-none prompt cfg correction prompt_cfg prompt_corr cfg_corr full}"

# ---------------------------------------------------------------------------
# Ablation definitions: name → "N_VARIANTS CFG_SCALES CORRECTION_STRENGTHS USE_QWEN"
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

echo "MCTS ablation study"
echo "  ablations: ${ABLATIONS}"
echo "  sd35_backend: ${SD35_BACKEND} sd35_sigmas: ${SD35_SIGMAS:-<none>}"
echo "  prompt_file: ${PROMPT_FILE}"
echo "  n_sims: ${N_SIMS}"
echo "  out: ${ABLATION_DIR}"
echo

# ---------------------------------------------------------------------------
# Pre-cache model weights so torchrun workers never make outbound HF requests
# ---------------------------------------------------------------------------
echo "[preload] Caching SD3.5 model: YGu1998/SiD-DiT-SD3.5-large ..."
env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u LOCAL_WORLD_SIZE -u NODE_RANK \
  -u MASTER_ADDR -u MASTER_PORT \
  HF_HOME="${HF_HOME}" \
  "${PYTHON_BIN}" -c "
from huggingface_hub import snapshot_download
import os, sys
try:
    snapshot_download('YGu1998/SiD-DiT-SD3.5-large', cache_dir=os.environ.get('HF_HOME') or None)
    print('[preload] SD3.5 OK')
except Exception as e:
    print(f'[preload] warning: {e}', file=sys.stderr)
"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Pre-compute rewrites once (shared across all ablations that use Qwen)
SHARED_REWRITES="${ABLATION_DIR}/rewrites_cache.json"
precompute_shared_rewrites() {
  if [[ "${PRECOMPUTE_REWRITES}" != "1" ]]; then
    return 0
  fi
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
# Run one ablation
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

  local -a extra=(--modes base mcts)
  if [[ "${use_qwen}" == "1" && -s "${SHARED_REWRITES}" ]]; then
    extra+=(--no_qwen --rewrites_file "${SHARED_REWRITES}")
  else
    extra+=(--no_qwen)
  fi
  if [[ "${SAVE_BEST_IMAGES}" == "1" ]]; then
    extra+=(--save_best_images)
  fi
  if [[ -n "${REWARD_API_BASE}" ]]; then
    extra+=(--reward_api_base "${REWARD_API_BASE}")
  fi
  if [[ -n "${SD35_SIGMAS}" ]]; then
    extra+=(--sigmas ${SD35_SIGMAS})
  fi

  local begin_ts end_ts elapsed
  begin_ts="$(date +%s)"

  IMAGEREWARD_CACHE="${IMAGEREWARD_CACHE}" \
  HF_HOME="${HF_HOME}" \
  HPS_ROOT="${HPS_ROOT}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  torchrun --standalone --nproc_per_node "${NUM_GPUS}" \
    "${SCRIPT_DIR}/sd35_ddp_experiment.py" \
    --backend "${SD35_BACKEND}" \
    --prompt_file "${PROMPT_FILE}" \
    --start_index "${START_INDEX}" \
    --end_index "${END_INDEX}" \
    --gen_batch_size "${GEN_BATCH_SIZE}" \
    --steps "${STEPS}" \
    --cfg_scales ${cfg_scales} \
    --baseline_cfg "${BASELINE_CFG}" \
    --n_variants "${n_var}" \
    --correction_strengths ${corr_strengths} \
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
    --out_dir "${abl_dir}" \
    "${extra[@]}"

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
    log_dir  = os.path.join(abl_path, "logs")
    if not os.path.isdir(log_dir):
        print(f"  [skip] {name}: no logs dir")
        continue
    base_scores, mcts_scores, deltas = [], [], []
    for log_path in glob.glob(os.path.join(log_dir, "rank_*.jsonl")):
        if log_path.endswith("_rewrite_examples.jsonl"):
            continue
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("mode") == "base":
                    base_scores.append(float(r["score"]))
                elif r.get("mode") == "mcts":
                    mcts_scores.append(float(r["score"]))
                    deltas.append(float(r.get("delta_vs_base", 0.0)))
    if not mcts_scores:
        print(f"  [skip] {name}: no mcts rows")
        continue
    rows.append({
        "name": name,
        "n": len(mcts_scores),
        "base": statistics.fmean(base_scores) if base_scores else 0.0,
        "mcts": statistics.fmean(mcts_scores),
        "delta": statistics.fmean(deltas),
        "std":   statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
    })

if not rows:
    print("[compare] no results found.")
    sys.exit(0)

# Sort by delta descending
rows.sort(key=lambda r: r["delta"], reverse=True)

header = f"{'Ablation':<18} {'N':>5} {'Base':>8} {'MCTS':>8} {'Δ':>8} {'std':>7}"
sep    = "-" * len(header)
print()
print(header)
print(sep)
for r in rows:
    print(f"{r['name']:<18} {r['n']:>5} {r['base']:>8.4f} {r['mcts']:>8.4f} {r['delta']:>+8.4f} {r['std']:>7.4f}")
print(sep)

need_header = not os.path.exists(summary_tsv) or os.path.getsize(summary_tsv) == 0
with open(summary_tsv, "w", encoding="utf-8") as f:
    f.write("ablation\tn\tbase\tmcts\tdelta\tstd\n")
    for r in rows:
        f.write(f"{r['name']}\t{r['n']}\t{r['base']:.6f}\t{r['mcts']:.6f}\t{r['delta']:+.6f}\t{r['std']:.6f}\n")

print()
print(f"Summary TSV: {summary_tsv}")
PY

echo
echo "Ablation outputs: ${ABLATION_DIR}"
