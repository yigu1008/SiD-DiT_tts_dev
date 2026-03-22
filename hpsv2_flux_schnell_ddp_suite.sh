#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

PROMPT_FILE="${PROMPT_FILE:-/data/ygu/hpsv2_prompts.txt}"
OUT_ROOT="${OUT_ROOT:-/data/ygu/hpsv2_flux_schnell_ddp}"
METHODS="${METHODS:-baseline greedy mcts ga}"
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-schnell}"

START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:--1}"
SAVE_FIRST_K="${SAVE_FIRST_K:-10}"

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
GA_INIT_MODE="${GA_INIT_MODE:-random}"
GA_LOG_TOPK="${GA_LOG_TOPK:-3}"
GA_GUIDANCE_SCALES="${GA_GUIDANCE_SCALES:-1.0 1.25 1.5}"
N_VARIANTS="${N_VARIANTS:-5}"
CFG_SCALES="${CFG_SCALES:-1.0 1.25 1.5}"
N_SIMS="${N_SIMS:-50}"
UCB_C="${UCB_C:-1.41}"

SMC_K="${SMC_K:-12}"
SMC_GAMMA="${SMC_GAMMA:-0.10}"
SMC_GUIDANCE_SCALE="${SMC_GUIDANCE_SCALE:-1.25}"
SMC_CHUNK="${SMC_CHUNK:-4}"

NUM_GPUS="${NUM_GPUS:-0}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found: ${PROMPT_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"
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
echo "  reward_backend: ${REWARD_BACKEND} reward_device: ${REWARD_DEVICE}"
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
        )
    writer.writerow(
        [method, elapsed, len(baseline), f"{mean_baseline:.6f}", f"{mean_search:.6f}", f"{mean_delta:+.6f}", ""]
    )
PY
}

run_flux_sharded() {
  local method_name="$1"
  local flux_search_method="$2"
  shift 2
  local -a method_args=("$@")
  local method_out="${RUN_DIR}/${method_name}"
  local method_logs="${method_out}/logs"
  mkdir -p "${method_out}" "${method_logs}"
  local -a reward_extra=()
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
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SCRIPT_DIR}/sampling_flux_unified.py" \
      --search_method "${flux_search_method}" \
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

  append_method_summary "${method_out}" "${method_name}" "${elapsed}"
}

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
      run_flux_sharded "greedy" "greedy" \
        --n_variants "${N_VARIANTS}" \
        --cfg_scales ${CFG_SCALES}
      ;;
    mcts)
      run_flux_sharded "mcts" "mcts" \
        --n_variants "${N_VARIANTS}" \
        --cfg_scales ${CFG_SCALES} \
        --n_sims "${N_SIMS}" \
        --ucb_c "${UCB_C}"
      ;;
    smc)
      run_flux_sharded "smc" "smc" \
        --smc_k "${SMC_K}" \
        --smc_gamma "${SMC_GAMMA}" \
        --smc_guidance_scale "${SMC_GUIDANCE_SCALE}" \
        --smc_chunk "${SMC_CHUNK}"
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
