#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

PROMPT_FILE="${PROMPT_FILE:-/data/ygu/hpsv2_prompts.txt}"
OUT_ROOT="${OUT_ROOT:-/data/ygu/hpsv2_sd35_sid_ddp}"
METHODS="${METHODS:-baseline greedy mcts ga}"

START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:--1}"
NUM_GPUS="${NUM_GPUS:-$("${PYTHON_BIN}" - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)}"

STEPS="${STEPS:-4}"
SEED="${SEED:-42}"
N_VARIANTS="${N_VARIANTS:-3}"
CFG_SCALES="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5}"
BASELINE_CFG="${BASELINE_CFG:-1.0}"
N_SIMS="${N_SIMS:-50}"
UCB_C="${UCB_C:-1.41}"
SMC_K="${SMC_K:-8}"
SMC_GAMMA="${SMC_GAMMA:-0.10}"
ESS_THRESHOLD="${ESS_THRESHOLD:-0.5}"
RESAMPLE_START_FRAC="${RESAMPLE_START_FRAC:-0.3}"
SMC_CFG_SCALE="${SMC_CFG_SCALE:-1.25}"
SMC_VARIANT_IDX="${SMC_VARIANT_IDX:-0}"
USE_QWEN="${USE_QWEN:-0}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen3-4B}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
SAVE_IMAGES="${SAVE_IMAGES:-0}"
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
REWARD_WEIGHTS="${REWARD_WEIGHTS:-1.0 1.0}"
REWARD_API_BASE="${REWARD_API_BASE:-}"
REWARD_API_KEY="${REWARD_API_KEY:-unifiedreward}"
REWARD_API_MODEL="${REWARD_API_MODEL:-UnifiedReward-7b-v1.5}"
REWARD_MAX_NEW_TOKENS="${REWARD_MAX_NEW_TOKENS:-512}"
REWARD_PROMPT_MODE="${REWARD_PROMPT_MODE:-standard}"

GA_POPULATION="${GA_POPULATION:-24}"
GA_GENERATIONS="${GA_GENERATIONS:-12}"
GA_ELITES="${GA_ELITES:-3}"
GA_MUTATION_PROB="${GA_MUTATION_PROB:-0.10}"
GA_TOURNAMENT_K="${GA_TOURNAMENT_K:-3}"
GA_SELECTION="${GA_SELECTION:-rank}"
GA_RANK_PRESSURE="${GA_RANK_PRESSURE:-1.7}"
GA_CROSSOVER="${GA_CROSSOVER:-uniform}"
GA_LOG_TOPK="${GA_LOG_TOPK:-3}"
GA_PHASE_CONSTRAINTS="${GA_PHASE_CONSTRAINTS:-1}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found: ${PROMPT_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_ROOT}/run_${RUN_TS}"
mkdir -p "${RUN_DIR}"
SUITE_TSV="${RUN_DIR}/suite_summary.tsv"
if [[ -z "${REWRITES_FILE}" ]]; then
  REWRITES_FILE="${RUN_DIR}/rewrites_cache.json"
fi

echo "SD3.5L SiD DDP suite"
echo "  prompt_file: ${PROMPT_FILE}"
echo "  modes: ${METHODS}"
echo "  nproc_per_node: ${NUM_GPUS}"
echo "  reward_backend: ${REWARD_BACKEND}"
echo "  use_qwen: ${USE_QWEN} (precompute=${PRECOMPUTE_REWRITES})"
echo "  rewrites_file: ${REWRITES_FILE}"
echo "  out: ${RUN_DIR}"

ensure_imagereward_runtime() {
  local backend_lc
  backend_lc="$(echo "${REWARD_BACKEND}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${backend_lc}" != "imagereward" && "${backend_lc}" != "auto" && "${backend_lc}" != "blend" ]]; then
    return 0
  fi
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import xxhash
import clip
import ImageReward as RM
print(xxhash.__version__, getattr(RM, "__file__", "ok"))
PY
  then
    return 0
  fi
  echo "[deps] ImageReward runtime deps missing. Installing with install_reward_deps.sh ..."
  PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/install_reward_deps.sh"
}

ensure_imagereward_runtime

precompute_rewrites_cache() {
  if [[ "${USE_QWEN}" != "1" ]]; then
    return 0
  fi
  if [[ "${PRECOMPUTE_REWRITES}" != "1" ]]; then
    return 0
  fi
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

append_method_summary() {
  local method_out="$1"
  local method_name="$2"
  local elapsed_sec="$3"
  "${PYTHON_BIN}" - <<'PY' "${method_out}" "${method_name}" "${elapsed_sec}" "${SUITE_TSV}"
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

baseline = []
search = []
deltas = []

for log_path in glob.glob(os.path.join(method_out, "logs", "rank_*.jsonl")):
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
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
        writer.writerow(
            ["method", "elapsed_sec", "num_samples", "mean_baseline", "mean_search", "mean_delta"]
        )
    writer.writerow(
        [method, elapsed, len(search), f"{mean_baseline:.6f}", f"{mean_search:.6f}", f"{mean_delta:+.6f}"]
    )
PY
}

run_method() {
  local method="$1"
  local method_out="${RUN_DIR}/${method}"
  mkdir -p "${method_out}"
  local mode_arg
  case "${method}" in
    baseline) mode_arg="base" ;;
    greedy) mode_arg="greedy" ;;
    mcts) mode_arg="mcts" ;;
    ga) mode_arg="ga" ;;
    smc) mode_arg="smc" ;;
    *)
      echo "Error: unsupported method '${method}' for SD3.5 suite." >&2
      exit 1
      ;;
  esac

  local -a extra=()
  if [[ -f "${REWRITES_FILE}" ]]; then
    extra+=(--rewrites_file "${REWRITES_FILE}")
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
  if [[ "${SAVE_VARIANTS}" == "1" ]]; then
    extra+=(--save_variants)
  fi
  if [[ "${GA_PHASE_CONSTRAINTS}" == "1" ]]; then
    extra+=(--ga_phase_constraints)
  fi
  if [[ -n "${REWARD_API_BASE}" ]]; then
    extra+=(--reward_api_base "${REWARD_API_BASE}")
  fi

  local begin_ts
  begin_ts="$(date +%s)"
  echo "[$(date '+%F %T')] method=${method} start"

  torchrun --standalone --nproc_per_node "${NUM_GPUS}" "${SCRIPT_DIR}/sd35_ddp_experiment.py" \
    --prompt_file "${PROMPT_FILE}" \
    --start_index "${START_INDEX}" \
    --end_index "${END_INDEX}" \
    --modes "${mode_arg}" \
    --steps "${STEPS}" \
    --cfg_scales ${CFG_SCALES} \
    --baseline_cfg "${BASELINE_CFG}" \
    --n_variants "${N_VARIANTS}" \
    --qwen_id "${QWEN_ID}" \
    --qwen_dtype "${QWEN_DTYPE}" \
    --n_sims "${N_SIMS}" \
    --ucb_c "${UCB_C}" \
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
    --out_dir "${method_out}" \
    "${extra[@]}"

  local end_ts
  end_ts="$(date +%s)"
  local elapsed=$(( end_ts - begin_ts ))
  echo "[$(date '+%F %T')] method=${method} done elapsed=${elapsed}s"

  append_method_summary "${method_out}" "${method}" "${elapsed}"
}

precompute_rewrites_cache

for method in ${METHODS}; do
  run_method "${method}"
done

echo
echo "Suite summary: ${SUITE_TSV}"
cat "${SUITE_TSV}"
echo
echo "Outputs: ${RUN_DIR}"
