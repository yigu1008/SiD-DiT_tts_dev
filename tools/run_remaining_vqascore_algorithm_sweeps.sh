#!/usr/bin/env bash
set -euo pipefail

# Run the VQAScore-guided algorithm sweep on every model not covered by the
# completed SiD sweep. The exact SiD prompt subset, root seed map, and Qwen
# prompt bank are reused; SiD itself is never launched here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO}/shell_env.sh"

HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
SOURCE_SID_RUN_ROOT="${SOURCE_SID_RUN_ROOT:-${HUMAN_EVAL_ROOT}/sid_vqascore_algorithm_sweep/genai200_v1}"
RUN_ID="${RUN_ID:-genai200_v1}"
STUDY_ROOT="${STUDY_ROOT:-${HUMAN_EVAL_ROOT}/vqascore_remaining_models}"
RUN_ROOT="${STUDY_ROOT}/${RUN_ID}"

BACKENDS="${BACKENDS:-sd35_base senseflow_large senseflow_medium flux_schnell}"
METHODS="${METHODS:-baseline fksteering bon beam sop ga dts dts_star dynamic_cfg_x0 bon_mcts das}"
POST_EVAL_ONLY="${POST_EVAL_ONLY:-0}"
SKIP_POST_EVAL="${SKIP_POST_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"
FAIL_FAST="${FAIL_FAST:-1}"

VQASCORE_MODEL="${VQASCORE_MODEL:-clip-flant5-xxl}"
REWARD_ENV_CONDA_BASE="${REWARD_ENV_CONDA_BASE:-/home/ygu/miniconda3}"
VQA_REWARD_ENV_NAME="${VQA_REWARD_ENV_NAME:-vqascore_reward}"
STANDARD_REWARD_ENV_NAME="${STANDARD_REWARD_ENV_NAME:-reward}"
REWARD_SERVER_BASE_PORT="${REWARD_SERVER_BASE_PORT:-5300}"
POSTHOC_REWARD_SERVER_PORT="${POSTHOC_REWARD_SERVER_PORT:-5390}"
HEALTH_TIMEOUT_SECS="${HEALTH_TIMEOUT_SECS:-1800}"

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

for value in "${POST_EVAL_ONLY}" "${SKIP_POST_EVAL}" "${DRY_RUN}" "${FAIL_FAST}"; do
  case "${value}" in 0|1) ;; *) echo "Error: boolean controls must be 0 or 1: ${value}" >&2; exit 1 ;; esac
done
if [[ "${RUN_ID}" == *"/"* ]]; then
  echo "Error: RUN_ID cannot contain '/': ${RUN_ID}" >&2
  exit 1
fi
for backend in ${BACKENDS}; do
  case "${backend}" in
    sd35_base|senseflow_large|senseflow_medium|flux_schnell) ;;
    sid) echo "Error: SiD is already complete and is intentionally excluded." >&2; exit 1 ;;
    *) echo "Error: unsupported remaining backend '${backend}'" >&2; exit 1 ;;
  esac
done
for method in ${METHODS}; do
  case "${method}" in
    baseline|fksteering|bon|beam|sop|ga|dts|dts_star|dynamic_cfg_x0|bon_mcts|das) ;;
    *) echo "Error: unsupported method '${method}'" >&2; exit 1 ;;
  esac
done
if [[ " ${METHODS} " == *" das "* && "${METHODS##* }" != "das" ]]; then
  echo "Error: keep das last because the SD3.5 suite mutates its prompt-action state." >&2
  exit 1
fi

PROMPTS_TXT="${SOURCE_SID_RUN_ROOT}/prompts_subset.txt"
SEED_MAP_FILE="${SOURCE_SID_RUN_ROOT}/shared_root_seed_map.json"
SUBSET_MANIFEST="${SOURCE_SID_RUN_ROOT}/subset_manifest.json"
REWRITES_FILE="${SOURCE_SID_RUN_ROOT}/shared_rewrites_cache.json"
SOURCE_STUDY_MANIFEST="${SOURCE_SID_RUN_ROOT}/study_manifest.json"
for required in "${PROMPTS_TXT}" "${SEED_MAP_FILE}" "${SUBSET_MANIFEST}" "${SOURCE_STUDY_MANIFEST}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Error: completed SiD source artifact missing: ${required}" >&2
    exit 1
  fi
done
if [[ " ${METHODS} " != " baseline " && ! -s "${REWRITES_FILE}" ]]; then
  echo "Error: shared SiD rewrite cache missing: ${REWRITES_FILE}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}"
STUDY_MANIFEST="${RUN_ROOT}/study_manifest.json"
PROMPT_COUNT="$(${PYTHON_BIN} - <<'PY' "${SUBSET_MANIFEST}"
import json, sys
print(int(json.load(open(sys.argv[1], encoding="utf-8"))["subset_size"]))
PY
)"

SOURCE_SID_RUN_ROOT="${SOURCE_SID_RUN_ROOT}" RUN_ROOT="${RUN_ROOT}" \
BACKENDS="${BACKENDS}" METHODS="${METHODS}" \
PROMPT_COUNT="${PROMPT_COUNT}" VQASCORE_MODEL="${VQASCORE_MODEL}" \
STUDY_MANIFEST="${STUDY_MANIFEST}" "${PYTHON_BIN}" - <<'PY'
import json, os
from pathlib import Path

models = {
    "sd35_base": "SD3.5-Base",
    "senseflow_large": "SenseFlow-SD3.5-Large",
    "senseflow_medium": "SenseFlow-SD3.5-Medium",
    "flux_schnell": "Flux-Schnell",
}
backends = os.environ["BACKENDS"].split()
methods = os.environ["METHODS"].split()
payload = {
    "study_id": "remaining_models_vqascore_algorithm_sweep",
    "source_sid_run_root": str(Path(os.environ["SOURCE_SID_RUN_ROOT"]).resolve()),
    "run_root": str(Path(os.environ["RUN_ROOT"]).resolve()),
    "prompt_count": int(os.environ["PROMPT_COUNT"]),
    "search_reward": "vqascore",
    "vqascore_model": os.environ["VQASCORE_MODEL"],
    "models": [{"model_id": value, "model_name": models[value]} for value in backends],
    "methods_by_model": {
        value: methods for value in backends
    },
    "evaluation_rewards": ["imagereward", "hpsv3", "pickscore", "vqascore"],
    "sharing_rules": [
        "The exact completed-SiD prompt subset and prompt ordering are reused.",
        "The exact completed-SiD prompt-index root seed map is reused.",
        "SD3.5-family models reuse the completed-SiD c0 plus three-rewrite cache.",
        "FLUX uses its native prompt action bank because its action representation differs.",
        "Every online and post-hoc score is evaluated against original c0.",
    ],
}
path = Path(os.environ["STUDY_MANIFEST"])
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"[remaining-vqa] manifest={path}")
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] source_sid=${SOURCE_SID_RUN_ROOT}"
  echo "[dry-run] prompts=${PROMPT_COUNT} backends=${BACKENDS}"
  echo "[dry-run] methods=${METHODS}"
  echo "[dry-run] output=${RUN_ROOT}"
  exit 0
fi

VQASCORE_REWARD_PY="${REWARD_ENV_CONDA_BASE}/envs/${VQA_REWARD_ENV_NAME}/bin/python"
STANDARD_REWARD_PY="${REWARD_ENV_CONDA_BASE}/envs/${STANDARD_REWARD_ENV_NAME}/bin/python"
for executable in "${VQASCORE_REWARD_PY}" "${STANDARD_REWARD_PY}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "Error: required reward Python is not executable: ${executable}" >&2
    exit 1
  fi
done
export PATH="$(dirname "${VQASCORE_REWARD_PY}"):${PATH}"
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg is not visible from ${VQA_REWARD_ENV_NAME}." >&2
  exit 1
fi
if ! "${VQASCORE_REWARD_PY}" "${REPO}/tools/patch_t2v_metrics_clip_flant5_only.py" --check >/dev/null 2>&1; then
  "${VQASCORE_REWARD_PY}" "${REPO}/tools/patch_t2v_metrics_clip_flant5_only.py"
fi
PATH="$(dirname "${VQASCORE_REWARD_PY}"):${PATH}" "${VQASCORE_REWARD_PY}" - <<'PY'
import shutil, t2v_metrics
assert shutil.which("ffmpeg")
assert "clip-flant5-xxl" in set(t2v_metrics.list_all_models())
print("[remaining-vqa] isolated VQAScore runtime OK")
PY

cell_index=0
failed=()
for backend in ${BACKENDS}; do
  case "${backend}" in
    sd35_base)
      suite="${REPO}/hpsv2_sd35_sid_ddp_suite.sh"; layout=sd35; steps=28
      backend_methods="${METHODS}"; dynamic_grid="3.5 4.0 4.5 5.0 5.5" ;;
    senseflow_large|senseflow_medium)
      suite="${REPO}/hpsv2_sd35_sid_ddp_suite.sh"; layout=sd35; steps=4
      backend_methods="${METHODS}"; dynamic_grid="1.0 1.5 2.0 2.5" ;;
    flux_schnell)
      suite="${REPO}/hpsv2_flux_schnell_ddp_suite.sh"; layout=flux; steps=4
      backend_methods="${METHODS}"; dynamic_grid="0.0 1.0 1.5 2.0" ;;
  esac
  model_root="${RUN_ROOT}/${backend}"
  pending=()
  for method in ${backend_methods}; do
    if [[ -s "${model_root}/run_${RUN_ID}/${method}/aggregate_ddp.json" ]]; then
      echo "[resume] ${backend}/${method} complete"
    else
      pending+=("${method}")
    fi
  done
  if [[ "${POST_EVAL_ONLY}" == "1" || ${#pending[@]} -eq 0 ]]; then
    continue
  fi
  cell_index=$((cell_index + 1))
  echo "[remaining-vqa] backend=${backend} methods=${pending[*]} prompts=${PROMPT_COUNT}"
  common=(
    "PROMPT_FILE=${PROMPTS_TXT}" "METHODS=${pending[*]}" "START_INDEX=0"
    "END_INDEX=${PROMPT_COUNT}" "SEED_MAP_FILE=${SEED_MAP_FILE}"
    "RUN_TS=${RUN_ID}" "OUT_ROOT=${model_root}" "STEPS=${steps}"
    "WIDTH=1024" "HEIGHT=1024"
    "N_VARIANTS=3" "USE_QWEN=0" "PRECOMPUTE_REWRITES=0"
    "REWRITES_FILE=${REWRITES_FILE}" "SYNERGY_REWRITES_FILE=${REWRITES_FILE}"
    "REWARD_BACKEND=vqascore" "VQASCORE_MODEL=${VQASCORE_MODEL}"
    "USE_REWARD_SERVER=1" "REWARD_SERVER_REQUIRE_ALL=1"
    "REWARD_SERVER_BACKENDS=vqascore"
    "REWARD_SERVER_PORT=$((REWARD_SERVER_BASE_PORT + cell_index))"
    "REWARD_SERVER_MAX_WAIT=1800" "REWARD_SERVER_SCORE_TIMEOUT=300"
    "REWARD_ENV_CONDA_BASE=${REWARD_ENV_CONDA_BASE}"
    "REWARD_ENV_NAME=${VQA_REWARD_ENV_NAME}"
    "SAVE_IMAGES=0" "SAVE_BEST_IMAGES=1" "SAVE_VARIANTS=0"
    "SAVE_FIRST_K=-1" "EVAL_BEST_IMAGES=0"
    "DYNAMIC_CFG_X0_EVALUATORS=vqascore"
    "DYNAMIC_CFG_X0_GRID=${dynamic_grid}"
    "DYNAMIC_CFG_X0_SCORE_EVERY=1"
    "DTS_CFG_BANK=${dynamic_grid}"
    "N_SIMS=${N_SIMS}" "BON_MCTS_N_SEEDS=${BON_MCTS_N_SEEDS}"
    "BON_MCTS_TOPK=${BON_MCTS_TOPK}" "BON_MCTS_MIN_SIMS=${BON_MCTS_MIN_SIMS}"
    "BON_N=${BON_N}" "BEAM_WIDTH=${BEAM_WIDTH}" "SMC_K=${SMC_K}"
    "SOP_INIT_PATHS=${SOP_INIT_PATHS}" "SOP_BRANCH_FACTOR=${SOP_BRANCH_FACTOR}"
    "SOP_KEEP_TOP=${SOP_KEEP_TOP}" "GA_POPULATION=${GA_POPULATION}"
    "GA_GENERATIONS=${GA_GENERATIONS}" "DTS_M_ITER=${DTS_M_ITER}"
  )
  if [[ "${layout}" == "flux" ]]; then
    common+=("FLUX_BACKEND=flux" "BASELINE_GUIDANCE_SCALE=0.0" "BASELINE_CFG=0.0")
  else
    common+=("SD35_BACKEND=${backend}")
  fi
  if env "${common[@]}" bash "${suite}"; then
    echo "[remaining-vqa] generation complete: ${backend}"
  else
    rc=$?; failed+=("${backend}")
    if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
  fi
done
if (( ${#failed[@]} > 0 )); then
  echo "Error: failed backends: ${failed[*]}" >&2
  exit 1
fi

if [[ "${SKIP_POST_EVAL}" != "1" ]]; then
  if [[ -z "${REWARD_CUDA_VISIBLE_DEVICES:-}" ]]; then
    visible="${CUDA_VISIBLE_DEVICES:-0}"
    REWARD_CUDA_VISIBLE_DEVICES="${visible##*,}"
  fi
  for backend in ${BACKENDS}; do
    model_root="${RUN_ROOT}/${backend}"
    [[ -d "${model_root}" ]] || continue
    [[ "${backend}" == "flux_schnell" ]] && layout=flux || layout=sd35
    OUT_ROOT="${model_root}" REWARD_PY="${STANDARD_REWARD_PY}" \
    STANDARD_REWARD_PY="${STANDARD_REWARD_PY}" VQASCORE_REWARD_PY="${VQASCORE_REWARD_PY}" \
    PYTHON_BIN="${PYTHON_BIN}" REWARD_SERVER_PORT="${POSTHOC_REWARD_SERVER_PORT}" \
    REWARD_CUDA_VISIBLE_DEVICES="${REWARD_CUDA_VISIBLE_DEVICES}" \
    POSTHOC_EVAL_BACKENDS="imagereward hpsv3 pickscore vqascore" \
    POSTHOC_ALLOW_MISSING_BACKENDS=0 POSTHOC_LAYOUT="${layout}" \
    VQASCORE_MODEL="${VQASCORE_MODEL}" HEALTH_TIMEOUT_SECS="${HEALTH_TIMEOUT_SECS}" \
    bash "${REPO}/post_eval_extra_rewards.sh"
  done
  "${PYTHON_BIN}" "${REPO}/tools/merge_posthoc_reward_evals.py" \
    --root "${RUN_ROOT}" --backends imagereward hpsv3 pickscore vqascore \
    --summary-csv "${RUN_ROOT}/vqa_remaining_models_summary.csv" \
    --expected-count "${PROMPT_COUNT}" --strict
fi

audit_extra=()
if [[ "${SKIP_POST_EVAL}" == "1" ]]; then audit_extra+=(--no-strict); fi
"${PYTHON_BIN}" "${REPO}/tools/audit_vqascore_sweep_coverage.py" \
  --root "${RUN_ROOT}" --models ${BACKENDS} --methods ${METHODS} \
  --expected-prompts "${PROMPT_COUNT}" --run-id "${RUN_ID}" \
  --out-csv "${RUN_ROOT}/vqascore_coverage.csv" \
  "${audit_extra[@]}"

echo "[remaining-vqa] complete: ${RUN_ROOT}"
