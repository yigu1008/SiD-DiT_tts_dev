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
QWEN_TIMEOUT_SEC="${QWEN_TIMEOUT_SEC:-240}"
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

GA_POPULATION="${GA_POPULATION:-24}"
GA_GENERATIONS="${GA_GENERATIONS:-8}"
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
echo "  eval_best_images: ${EVAL_BEST_IMAGES} eval_backends: ${EVAL_BACKENDS} eval_device: ${EVAL_REWARD_DEVICE}"
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
  local backend_lc
  backend_lc="$(echo "${REWARD_BACKEND}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${backend_lc}" != "imagereward" && "${backend_lc}" != "auto" && "${backend_lc}" != "blend" ]] && ! eval_backend_requested "imagereward"; then
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

ensure_pickscore_runtime() {
  local backend_lc
  backend_lc="$(echo "${REWARD_BACKEND}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${backend_lc}" != "pickscore" && "${backend_lc}" != "auto" ]] && ! eval_backend_requested "pickscore"; then
    return 0
  fi
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import timm
from timm.data import ImageNetInfo
print(timm.__version__, ImageNetInfo.__name__)
PY
  then
    return 0
  fi
  echo "[deps] PickScore runtime deps missing/incompatible (timm ImageNetInfo). Installing with install_reward_deps.sh ..."
  PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/install_reward_deps.sh"
}

ensure_pickscore_runtime

ensure_hpsv2_runtime() {
  local backend_lc
  backend_lc="$(echo "${REWARD_BACKEND}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${backend_lc}" != "hpsv2" && "${backend_lc}" != "auto" && "${backend_lc}" != "blend" ]] && ! eval_backend_requested "hpsv2"; then
    return 0
  fi
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
  local backend_lc
  backend_lc="$(echo "${REWARD_BACKEND}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${backend_lc}" != "hpsv3" && "${backend_lc}" != "auto" ]] && ! eval_backend_requested "hpsv3"; then
    return 0
  fi
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import hpsv3
import omegaconf
import hydra
print(getattr(hpsv3, "__file__", "ok"), getattr(omegaconf, "__version__", "ok"), getattr(hydra, "__version__", "ok"))
PY
  then
    return 0
  fi
  echo "[deps] HPSv3 runtime deps missing. Installing with install_reward_deps.sh ..."
  PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/install_reward_deps.sh"
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
  if [[ "${EVAL_BEST_IMAGES}" != "1" ]]; then
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

  torchrun --standalone --nproc_per_node "${NUM_GPUS}" "${SCRIPT_DIR}/sd35_ddp_experiment.py" \
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
    --out_dir "${method_out}" \
    "${extra[@]}"

  local end_ts
  end_ts="$(date +%s)"
  local elapsed=$(( end_ts - begin_ts ))
  echo "[$(date '+%F %T')] method=${method} done elapsed=${elapsed}s"

  post_eval_best_images "${method_out}" "${method}"
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
