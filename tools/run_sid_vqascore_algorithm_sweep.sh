#!/usr/bin/env bash
set -euo pipefail

# VQAScore-guided GenAI-Bench sweep for the methods in the ActDiff comparison:
#   Multi-step Baseline (optional SD3.5-Base), Distilled Baseline, DAS,
#   FK-Steering, BoN, Beam, SoP, GA, DTS, DTS*, Dynamic CFG, and ActDiff.
#
# Every method receives the exact same deterministic, stratum-balanced prompt
# subset and prompt-level base seed. Search uses VQAScore. Selected outputs are
# then evaluated against original c0 with ImageReward, HPSv3, PickScore, and
# VQAScore, one backend server at a time.
#
# Typical launch (four visible GPUs -> three generation + one reward):
#   CUDA_VISIBLE_DEVICES=4,5,6,7 \
#   HUMAN_EVAL_ROOT=/data/ygu/human_eval_genai40_v1 \
#   REWARD_ENV_CONDA_BASE=/home/ygu/miniconda3 \
#   VQA_REWARD_ENV_NAME=vqascore_reward \
#   STANDARD_REWARD_ENV_NAME=reward \
#   bash tools/run_sid_vqascore_algorithm_sweep.sh
#
# Useful controls:
#   SUBSET_SIZE=16              # positive multiple of 8
#   METHODS="baseline bon ..."  # SiD methods; keep das last
#   INCLUDE_MULTISTEP_BASELINE=0
#   USE_QWEN=1                  # c0 + three shared cached rewrites
#   POST_EVAL_ONLY=1
#   DRY_RUN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO}/shell_env.sh"

HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
PROMPTS_FILE="${PROMPTS_FILE:-${HUMAN_EVAL_ROOT}/prompts.csv}"
STUDY_ROOT="${STUDY_ROOT:-${HUMAN_EVAL_ROOT}/sid_vqascore_algorithm_sweep}"
RUN_ID="${RUN_ID:-v1}"
RUN_ROOT="${STUDY_ROOT}/${RUN_ID}"
STUDY_ID="${STUDY_ID:-sid_vqascore_algorithm_sweep_${RUN_ID}}"

SUBSET_SIZE="${SUBSET_SIZE:-16}"
SUBSET_SEED="${SUBSET_SEED:-20260728}"
GENERATION_BASE_SEED="${GENERATION_BASE_SEED:-12345}"
INCLUDE_MULTISTEP_BASELINE="${INCLUDE_MULTISTEP_BASELINE:-1}"
MULTISTEP_STEPS="${MULTISTEP_STEPS:-28}"
USE_QWEN="${USE_QWEN:-1}"
N_VARIANTS="${N_VARIANTS:-3}"
METHODS="${METHODS:-baseline fksteering bon beam sop ga dts dts_star dynamic_cfg_x0 bon_mcts das}"
POST_EVAL_ONLY="${POST_EVAL_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

VQASCORE_MODEL="${VQASCORE_MODEL:-clip-flant5-xxl}"
VQA_REWARD_ENV_NAME="${VQA_REWARD_ENV_NAME:-vqascore_reward}"
STANDARD_REWARD_ENV_NAME="${STANDARD_REWARD_ENV_NAME:-reward}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5160}"
POSTHOC_REWARD_SERVER_PORT="${POSTHOC_REWARD_SERVER_PORT:-5161}"

N_SIMS="${N_SIMS:-25}"
BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS:-8}"
BON_MCTS_TOPK="${BON_MCTS_TOPK:-2}"
BON_MCTS_MIN_SIMS="${BON_MCTS_MIN_SIMS:-8}"
BON_N="${BON_N:-16}"
BEAM_WIDTH="${BEAM_WIDTH:-4}"
SMC_K="${SMC_K:-8}"
SOP_INIT_PATHS="${SOP_INIT_PATHS:-8}"
SOP_BRANCH_FACTOR="${SOP_BRANCH_FACTOR:-4}"
SOP_KEEP_TOP="${SOP_KEEP_TOP:-4}"
GA_POPULATION="${GA_POPULATION:-24}"
GA_GENERATIONS="${GA_GENERATIONS:-8}"
DTS_M_ITER="${DTS_M_ITER:-64}"

if [[ ! -f "${PROMPTS_FILE}" ]]; then
  echo "Error: prepared GenAI-Bench prompts file not found: ${PROMPTS_FILE}" >&2
  exit 1
fi
if [[ "${RUN_ID}" == *"/"* ]]; then
  echo "Error: RUN_ID cannot contain '/': ${RUN_ID}" >&2
  exit 1
fi
if (( SUBSET_SIZE <= 0 || SUBSET_SIZE % 8 != 0 )); then
  echo "Error: SUBSET_SIZE must be a positive multiple of 8." >&2
  exit 1
fi
case "${INCLUDE_MULTISTEP_BASELINE}" in 0|1) ;; *)
  echo "Error: INCLUDE_MULTISTEP_BASELINE must be 0 or 1." >&2; exit 1 ;;
esac
case "${USE_QWEN}" in 0|1) ;; *)
  echo "Error: USE_QWEN must be 0 or 1." >&2; exit 1 ;;
esac

resolve_conda_base() {
  if [[ -n "${REWARD_ENV_CONDA_BASE:-}" ]]; then
    printf '%s\n' "${REWARD_ENV_CONDA_BASE}"
    return
  fi
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    cd "$(dirname "${CONDA_EXE}")/.." && pwd
    return
  fi
  local candidate
  for candidate in "${HOME}/miniconda3" "${HOME}/miniforge3" "/opt/conda"; do
    if [[ -x "${candidate}/bin/conda" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  echo "Error: cannot infer Conda base; export REWARD_ENV_CONDA_BASE." >&2
  exit 1
}

REWARD_ENV_CONDA_BASE="$(resolve_conda_base)"
VQASCORE_REWARD_PY="${REWARD_ENV_CONDA_BASE}/envs/${VQA_REWARD_ENV_NAME}/bin/python"
STANDARD_REWARD_PY="${REWARD_ENV_CONDA_BASE}/envs/${STANDARD_REWARD_ENV_NAME}/bin/python"

mkdir -p "${RUN_ROOT}"
SUBSET_CSV="${RUN_ROOT}/prompts_subset.csv"
SUBSET_TXT="${RUN_ROOT}/prompts_subset.txt"
SEED_MAP_FILE="${RUN_ROOT}/shared_root_seed_map.json"
SUBSET_MANIFEST="${RUN_ROOT}/subset_manifest.json"
STUDY_MANIFEST="${RUN_ROOT}/study_manifest.json"
REWRITES_FILE="${REWRITES_FILE:-${RUN_ROOT}/shared_rewrites_cache.json}"
SID_OUT_ROOT="${RUN_ROOT}/sid"
MULTISTEP_OUT_ROOT="${RUN_ROOT}/multi_step_baseline"

"${PYTHON_BIN}" "${REPO}/tools/prepare_genai_sweep_subset.py" \
  --input "${PROMPTS_FILE}" \
  --output-csv "${SUBSET_CSV}" \
  --output-txt "${SUBSET_TXT}" \
  --seed-map "${SEED_MAP_FILE}" \
  --manifest "${SUBSET_MANIFEST}" \
  --subset-size "${SUBSET_SIZE}" \
  --subset-seed "${SUBSET_SEED}" \
  --generation-base-seed "${GENERATION_BASE_SEED}" \
  --study-id "${STUDY_ID}"

METHODS="${METHODS}" \
STUDY_MANIFEST="${STUDY_MANIFEST}" \
SUBSET_MANIFEST="${SUBSET_MANIFEST}" \
RUN_ID="${RUN_ID}" \
USE_QWEN="${USE_QWEN}" \
N_VARIANTS="${N_VARIANTS}" \
INCLUDE_MULTISTEP_BASELINE="${INCLUDE_MULTISTEP_BASELINE}" \
MULTISTEP_STEPS="${MULTISTEP_STEPS}" \
VQASCORE_MODEL="${VQASCORE_MODEL}" \
REWRITES_FILE="${REWRITES_FILE}" \
N_SIMS="${N_SIMS}" \
BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS}" \
BON_MCTS_TOPK="${BON_MCTS_TOPK}" \
BON_MCTS_MIN_SIMS="${BON_MCTS_MIN_SIMS}" \
BON_N="${BON_N}" \
BEAM_WIDTH="${BEAM_WIDTH}" \
SMC_K="${SMC_K}" \
SOP_INIT_PATHS="${SOP_INIT_PATHS}" \
SOP_BRANCH_FACTOR="${SOP_BRANCH_FACTOR}" \
SOP_KEEP_TOP="${SOP_KEEP_TOP}" \
GA_POPULATION="${GA_POPULATION}" \
GA_GENERATIONS="${GA_GENERATIONS}" \
DTS_M_ITER="${DTS_M_ITER}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

subset = json.loads(Path(os.environ["SUBSET_MANIFEST"]).read_text(encoding="utf-8"))
methods = os.environ["METHODS"].split()
supported = {
    "baseline", "das", "fksteering", "bon", "beam", "sop", "ga",
    "dts", "dts_star", "dynamic_cfg_x0", "bon_mcts",
}
unknown = sorted(set(methods) - supported)
if unknown:
    raise SystemExit(f"unsupported sweep methods: {unknown}")
if len(methods) != len(set(methods)):
    raise SystemExit("METHODS contains duplicates")
if "das" in methods and methods[-1] != "das":
    raise SystemExit(
        "keep das last: its suite adapter intentionally disables prompt variants "
        "and that state would otherwise affect later methods"
    )

manifest = {
    "study_id": subset["study_id"],
    "run_id": os.environ["RUN_ID"],
    "search_reward": "vqascore",
    "vqascore_model": os.environ["VQASCORE_MODEL"],
    "evaluation_rewards": ["imagereward", "hpsv3", "pickscore", "vqascore"],
    "subset_manifest": str(Path(os.environ["SUBSET_MANIFEST"]).resolve()),
    "prompt_count": subset["subset_size"],
    "sid_model_name": "SiD-SD3.5",
    "sid_methods": methods,
    "method_labels": {
        "baseline": "Distilled Baseline",
        "das": "DAS",
        "fksteering": "FK-Steering",
        "bon": "BoN",
        "beam": "Beam",
        "sop": "SoP",
        "ga": "GA",
        "dts": "DTS",
        "dts_star": "DTS*",
        "dynamic_cfg_x0": "Dynamic CFG",
        "bon_mcts": "ActDiff",
    },
    "multi_step_baseline": {
        "included": bool(int(os.environ["INCLUDE_MULTISTEP_BASELINE"])),
        "model_name": "SD3.5-Base",
        "steps": int(os.environ["MULTISTEP_STEPS"]),
    },
    "prompt_action_bank": {
        "qwen_enabled": bool(int(os.environ["USE_QWEN"])),
        "rewrite_count": int(os.environ["N_VARIANTS"]),
        "rewrites_file": str(Path(os.environ["REWRITES_FILE"]).resolve()),
        "sharing_rule": "One cached rewrite bank is reused by every compatible method.",
    },
    "configured_budgets": {
        "bon_candidates": int(os.environ["BON_N"]),
        "beam_width": int(os.environ["BEAM_WIDTH"]),
        "smc_particles": int(os.environ["SMC_K"]),
        "sop_initial_paths": int(os.environ["SOP_INIT_PATHS"]),
        "sop_branch_factor": int(os.environ["SOP_BRANCH_FACTOR"]),
        "sop_keep_top": int(os.environ["SOP_KEEP_TOP"]),
        "ga_population": int(os.environ["GA_POPULATION"]),
        "ga_generations": int(os.environ["GA_GENERATIONS"]),
        "dts_iterations": int(os.environ["DTS_M_ITER"]),
        "actdiff_simulations": int(os.environ["N_SIMS"]),
        "actdiff_prescreen_roots": int(os.environ["BON_MCTS_N_SEEDS"]),
        "actdiff_prescreen_topk": int(os.environ["BON_MCTS_TOPK"]),
        "actdiff_minimum_sims_per_root": int(os.environ["BON_MCTS_MIN_SIMS"]),
        "actual_nfe": "Read from per-prompt rank JSONL logs and final summary CSV.",
    },
    "fairness_rules": [
        "Every method receives the same selected prompt IDs.",
        "Every method receives the same prompt-level base seed.",
        "Multi-root methods may select different winning roots; their outputs must not be labeled same-seed.",
        "Every online and post-hoc reward is evaluated against original c0.",
        "Search budgets are method-faithful, not forced to equal NFE; actual NFE is read from logs.",
    ],
    "method_implementation_map": {
        "DAS": "suite method das (continuous fixed trajectory action sampled in run_bon)",
        "FK-Steering": "suite method fksteering (SMC with reward-difference potential)",
        "DTS": "suite method dts",
        "DTS*": "suite method dts_star",
        "Dynamic CFG": "suite method dynamic_cfg_x0 with VQAScore evaluator",
        "ActDiff": "suite method bon_mcts",
    },
}
path = Path(os.environ["STUDY_MANIFEST"])
temporary = path.with_name(path.name + ".tmp")
temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
print(f"[study] manifest: {path}")
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] prepared subset and manifests; no models were run."
  echo "[dry-run] SiD methods: ${METHODS}"
  echo "[dry-run] output: ${RUN_ROOT}"
  exit 0
fi

require_executable() {
  local executable="$1"
  if [[ "${executable}" == */* ]]; then
    if [[ ! -x "${executable}" ]]; then
      echo "Error: required executable is unavailable: ${executable}" >&2
      exit 1
    fi
  elif ! command -v "${executable}" >/dev/null 2>&1; then
    echo "Error: required command is not on PATH: ${executable}" >&2
    exit 1
  fi
}

require_executable "${PYTHON_BIN}"
require_executable "${VQASCORE_REWARD_PY}"
require_executable "${STANDARD_REWARD_PY}"

# Keep CLIP-FlanT5 isolated and repair only its intentionally reduced registry.
if ! "${VQASCORE_REWARD_PY}" \
  "${REPO}/tools/patch_t2v_metrics_clip_flant5_only.py" --check \
  >/dev/null 2>&1
then
  "${VQASCORE_REWARD_PY}" \
    "${REPO}/tools/patch_t2v_metrics_clip_flant5_only.py"
fi
if ! PATH="$(dirname "${VQASCORE_REWARD_PY}"):${PATH}" \
  "${VQASCORE_REWARD_PY}" - <<'PY'
import shutil
import t2v_metrics
assert "clip-flant5-xxl" in set(t2v_metrics.list_all_models())
assert shutil.which("ffmpeg"), "ffmpeg is required by t2v-metrics"
print("[preflight] isolated VQAScore runtime OK")
PY
then
  echo "Error: ${VQA_REWARD_ENV_NAME} is not ready." >&2
  echo "Run: CONDA_BASE=${REWARD_ENV_CONDA_BASE} bash setup_vqascore_reward_env.sh" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import diffusers
import torch
import transformers
print(
    f"[preflight] generation runtime torch={torch.__version__} "
    f"transformers={transformers.__version__} diffusers={diffusers.__version__}"
)
PY

if [[ -z "${REWARD_CUDA_VISIBLE_DEVICES:-}" ]]; then
  visible="${CUDA_VISIBLE_DEVICES:-0}"
  REWARD_CUDA_VISIBLE_DEVICES="${visible##*,}"
fi

completed_method() {
  local root="$1"
  local method="$2"
  [[ -s "${root}/run_${RUN_ID}/${method}/aggregate_ddp.json" ]]
}

if [[ "${POST_EVAL_ONLY}" != "1" ]]; then
  pending=()
  for method in ${METHODS}; do
    if completed_method "${SID_OUT_ROOT}" "${method}"; then
      echo "[resume] complete, skipping SiD method=${method}"
    else
      pending+=("${method}")
    fi
  done

  if (( ${#pending[@]} > 0 )); then
    echo "[generation] SiD-SD3.5 methods: ${pending[*]}"
    env \
      "PROMPT_FILE=${SUBSET_TXT}" \
      "METHODS=${pending[*]}" \
      "SD35_BACKEND=sid" \
      "START_INDEX=0" \
      "END_INDEX=${SUBSET_SIZE}" \
      "SEED_MAP_FILE=${SEED_MAP_FILE}" \
      "RUN_TS=${RUN_ID}" \
      "OUT_ROOT=${SID_OUT_ROOT}" \
      "STEPS=4" \
      "N_VARIANTS=${N_VARIANTS}" \
      "USE_QWEN=${USE_QWEN}" \
      "PRECOMPUTE_REWRITES=1" \
      "REWRITES_FILE=${REWRITES_FILE}" \
      "REWRITES_OVERWRITE=0" \
      "REWARD_BACKEND=vqascore" \
      "VQASCORE_MODEL=${VQASCORE_MODEL}" \
      "USE_REWARD_SERVER=1" \
      "REWARD_SERVER_REQUIRE_ALL=1" \
      "REWARD_SERVER_BACKENDS=vqascore" \
      "REWARD_SERVER_PORT=${REWARD_SERVER_PORT}" \
      "REWARD_SERVER_MAX_WAIT=${REWARD_SERVER_MAX_WAIT:-1800}" \
      "REWARD_SERVER_SCORE_TIMEOUT=${REWARD_SERVER_SCORE_TIMEOUT:-300}" \
      "REWARD_ENV_CONDA_BASE=${REWARD_ENV_CONDA_BASE}" \
      "REWARD_ENV_NAME=${VQA_REWARD_ENV_NAME}" \
      "SAVE_IMAGES=0" \
      "SAVE_BEST_IMAGES=1" \
      "SAVE_VARIANTS=0" \
      "EVAL_BEST_IMAGES=0" \
      "DYNAMIC_CFG_X0_EVALUATORS=vqascore" \
      "DYNAMIC_CFG_X0_GRID=${DYNAMIC_CFG_X0_GRID:-1.0 1.5 2.0 2.5}" \
      "DYNAMIC_CFG_X0_SCORE_EVERY=${DYNAMIC_CFG_X0_SCORE_EVERY:-1}" \
      "DTS_CFG_BANK=${DTS_CFG_BANK:-1.0 1.5 2.0 2.5}" \
      "N_SIMS=${N_SIMS}" \
      "BON_MCTS_N_SEEDS=${BON_MCTS_N_SEEDS}" \
      "BON_MCTS_TOPK=${BON_MCTS_TOPK}" \
      "BON_MCTS_MIN_SIMS=${BON_MCTS_MIN_SIMS}" \
      "BON_N=${BON_N}" \
      "BEAM_WIDTH=${BEAM_WIDTH}" \
      "SMC_K=${SMC_K}" \
      "SOP_INIT_PATHS=${SOP_INIT_PATHS}" \
      "SOP_BRANCH_FACTOR=${SOP_BRANCH_FACTOR}" \
      "SOP_KEEP_TOP=${SOP_KEEP_TOP}" \
      "GA_POPULATION=${GA_POPULATION}" \
      "GA_GENERATIONS=${GA_GENERATIONS}" \
      "DTS_M_ITER=${DTS_M_ITER}" \
      bash "${REPO}/hpsv2_sd35_sid_ddp_suite.sh"
  fi

  if [[ "${INCLUDE_MULTISTEP_BASELINE}" == "1" ]]; then
    if completed_method "${MULTISTEP_OUT_ROOT}" baseline; then
      echo "[resume] complete, skipping Multi-step Baseline"
    else
      echo "[generation] SD3.5-Base Multi-step Baseline (${MULTISTEP_STEPS} steps)"
      env \
        "PROMPT_FILE=${SUBSET_TXT}" \
        "METHODS=baseline" \
        "SD35_BACKEND=sd35_base" \
        "START_INDEX=0" \
        "END_INDEX=${SUBSET_SIZE}" \
        "SEED_MAP_FILE=${SEED_MAP_FILE}" \
        "RUN_TS=${RUN_ID}" \
        "OUT_ROOT=${MULTISTEP_OUT_ROOT}" \
        "STEPS=${MULTISTEP_STEPS}" \
        "USE_QWEN=0" \
        "REWARD_BACKEND=vqascore" \
        "VQASCORE_MODEL=${VQASCORE_MODEL}" \
        "USE_REWARD_SERVER=1" \
        "REWARD_SERVER_REQUIRE_ALL=1" \
        "REWARD_SERVER_BACKENDS=vqascore" \
        "REWARD_SERVER_PORT=$((REWARD_SERVER_PORT + 1))" \
        "REWARD_SERVER_MAX_WAIT=${REWARD_SERVER_MAX_WAIT:-1800}" \
        "REWARD_ENV_CONDA_BASE=${REWARD_ENV_CONDA_BASE}" \
        "REWARD_ENV_NAME=${VQA_REWARD_ENV_NAME}" \
        "SAVE_IMAGES=0" \
        "SAVE_BEST_IMAGES=1" \
        "SAVE_VARIANTS=0" \
        "EVAL_BEST_IMAGES=0" \
        bash "${REPO}/hpsv2_sd35_sid_ddp_suite.sh"
    fi
  fi
fi

echo "[evaluation] ImageReward, HPSv3, PickScore, and VQAScore"
OUT_ROOT="${RUN_ROOT}" \
REWARD_PY="${STANDARD_REWARD_PY}" \
STANDARD_REWARD_PY="${STANDARD_REWARD_PY}" \
VQASCORE_REWARD_PY="${VQASCORE_REWARD_PY}" \
PYTHON_BIN="${PYTHON_BIN}" \
REWARD_SERVER_PORT="${POSTHOC_REWARD_SERVER_PORT}" \
REWARD_CUDA_VISIBLE_DEVICES="${REWARD_CUDA_VISIBLE_DEVICES}" \
POSTHOC_EVAL_BACKENDS="imagereward hpsv3 pickscore vqascore" \
POSTHOC_ALLOW_MISSING_BACKENDS=0 \
POSTHOC_LAYOUT=sd35 \
VQASCORE_MODEL="${VQASCORE_MODEL}" \
HEALTH_TIMEOUT_SECS="${HEALTH_TIMEOUT_SECS:-1800}" \
bash "${REPO}/post_eval_extra_rewards.sh"

"${PYTHON_BIN}" "${REPO}/tools/merge_posthoc_reward_evals.py" \
  --root "${RUN_ROOT}" \
  --backends imagereward hpsv3 pickscore vqascore \
  --summary-csv "${RUN_ROOT}/vqa_algorithm_sweep_summary.csv" \
  --expected-count "${SUBSET_SIZE}" \
  --strict

echo "[study] complete: ${RUN_ROOT}"
echo "[study] summary: ${RUN_ROOT}/vqa_algorithm_sweep_summary.csv"
