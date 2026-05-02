#!/usr/bin/env bash
# NFE-vs-quality sweep for FLUX backends — sibling of nfe_sweep_sd35.sh.
# Currently covers: flux_schnell (4 steps, dyn_cfg_x0 grid like SID-3.5).
#
# Per-method NFE budgets and knob mapping mirror nfe_sweep_sd35.sh, dispatched
# through hpsv2_flux_schnell_ddp_suite.sh. The combiner (nfe_sweep_combine.py)
# handles both the SD3.5-tree and the FLUX-tree under one --inputs list.
#
# Usage:
#   bash nfe_sweep_flux.sh
#   FLUX_BACKEND_LIST=flux_schnell  NUM_PROMPTS=8  bash nfe_sweep_flux.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Heartbeat to prevent AMLT inactivity suspend.
if [[ -f "${SCRIPT_DIR}/_heartbeat.sh" ]]; then
  source "${SCRIPT_DIR}/_heartbeat.sh"
  start_heartbeat "nfe-sweep-flux"
fi

FLUX_BACKEND_LIST="${FLUX_BACKEND_LIST:-flux_schnell}"
SWEEP_METHODS="${SWEEP_METHODS:-baseline bon beam smc fksteering greedy ga dts dts_star dynamic_cfg_x0 sop}"

GA_GENERATIONS_SWEEP="${GA_GENERATIONS:-8}"
GA_ELITES_SWEEP="${GA_ELITES:-3}"
FKSTEERING_LAMBDA_SWEEP="${FKSTEERING_LAMBDA:-${SMC_LAMBDA:-10.0}}"
DYNAMIC_CFG_X0_SCORE_EVERY_SWEEP_SCHNELL="${DYNAMIC_CFG_X0_SCORE_EVERY_SWEEP_SCHNELL:-1}"
SOP_KM_SWEEP="${SOP_KM_SWEEP:-2:2 2:4 4:4 4:8 8:8 8:16 16:16}"

NUM_PROMPTS="${NUM_PROMPTS:-8}"
SEED="${SEED:-42}"
SAVE_BEST_IMAGES_SWEEP="${SAVE_BEST_IMAGES:-1}"
SAVE_IMAGES_SWEEP="${SAVE_IMAGES:-0}"
USE_QWEN_SWEEP="${USE_QWEN:-0}"
PRECOMPUTE_REWRITES_SWEEP="${PRECOMPUTE_REWRITES:-0}"

PROMPT_FILE_DEFAULT="${SCRIPT_DIR}/hpsv2_subset.txt"
PROMPT_FILE="${PROMPT_FILE:-${PROMPT_FILE_DEFAULT}}"
[[ -f "${PROMPT_FILE}" ]] || { echo "Error: PROMPT_FILE not found: ${PROMPT_FILE}" >&2; exit 1; }

REWARD_BACKEND="${REWARD_BACKEND:-imagereward}"
REWARD_BACKENDS="${REWARD_BACKENDS:-imagereward hpsv3}"
EVAL_BACKENDS="${EVAL_BACKENDS:-imagereward hpsv3}"

OUT_ROOT_BASE="${OUT_ROOT_BASE:-${OUT_ROOT:-/tmp/flux_nfe_sweep}}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SUITE_SCRIPT="${SCRIPT_DIR}/hpsv2_flux_schnell_ddp_suite.sh"
COMBINE_SCRIPT="${SCRIPT_DIR}/nfe_sweep_combine.py"

ceil_div() { local a="$1" b="$2"; echo $(( (a + b - 1) / b )); }
count_tokens() { local s="$1"; read -r -a _arr <<< "${s}"; echo "${#_arr[@]}"; }

resolve_backend_defaults() {
  local backend="$1"
  case "${backend}" in
    flux_schnell)
      STEPS="${STEPS_SCHNELL:-4}"
      NFE_BUDGETS="${NFE_BUDGETS_SCHNELL:-16 32 64 128 256 512}"
      BASELINE_CFG="${BASELINE_CFG_SCHNELL:-0.0}"
      BASELINE_GUIDANCE_SCALE="${BASELINE_GUIDANCE_SCALE_SCHNELL:-0.0}"
      MODEL_ID_OVERRIDE="${MODEL_ID:-black-forest-labs/FLUX.1-schnell}"
      BEAM_CFG_BANK="${BEAM_CFG_BANK_SCHNELL:-0.0}"
      DYNCFG_X0_GRID="${DYNCFG_X0_GRID_SCHNELL:-1.0 1.5 2.0 2.5}"
      DYNCFG_X0_START_FRAC="${DYNCFG_X0_START_FRAC_SCHNELL:-0.5}"
      DYNCFG_X0_SCORE_EVERY_SWEEP="${DYNAMIC_CFG_X0_SCORE_EVERY_SWEEP_SCHNELL}"
      ;;
    *)
      echo "Error: unsupported flux backend=${backend}" >&2
      return 1 ;;
  esac
}

run_one_config() {
  local backend="$1" method="$2" target_nfe="$3" sweep_root="$4"
  local label="${method}_nfe${target_nfe}"
  local config_root="${sweep_root}/${label}"
  mkdir -p "${config_root}"

  local -a env_pairs=(
    "METHODS=${method}"
    "OUT_ROOT=${config_root}"
    "STEPS=${STEPS}"
    "FLUX_BACKEND=flux"
    "MODEL_ID=${MODEL_ID_OVERRIDE}"
    "PROMPT_FILE=${PROMPT_FILE}"
    "START_INDEX=0" "END_INDEX=${NUM_PROMPTS}"
    "SEED=${SEED}"
    "BASELINE_CFG=${BASELINE_CFG}"
    "BASELINE_GUIDANCE_SCALE=${BASELINE_GUIDANCE_SCALE}"
    "USE_QWEN=${USE_QWEN_SWEEP}"
    "PRECOMPUTE_REWRITES=${PRECOMPUTE_REWRITES_SWEEP}"
    "SAVE_BEST_IMAGES=${SAVE_BEST_IMAGES_SWEEP}"
    "SAVE_IMAGES=${SAVE_IMAGES_SWEEP}"
    "REWARD_BACKEND=${REWARD_BACKEND}"
    "REWARD_BACKENDS=${REWARD_BACKENDS}"
    "EVAL_BACKENDS=${EVAL_BACKENDS}"
  )

  case "${method}" in
    baseline)
      env_pairs+=( "CFG_SCALES=${BASELINE_CFG}" ) ;;
    bon)
      local n; n="$(ceil_div "${target_nfe}" "${STEPS}")"; (( n < 1 )) && n=1
      env_pairs+=( "CFG_SCALES=${BASELINE_CFG}" "BON_N=${n}" ) ;;
    beam)
      local n_cfgs; n_cfgs="$(count_tokens "${BEAM_CFG_BANK}")"; (( n_cfgs < 1 )) && n_cfgs=1
      local denom=$(( STEPS * n_cfgs )); local w; w="$(ceil_div "${target_nfe}" "${denom}")"; (( w < 1 )) && w=1
      env_pairs+=( "CFG_SCALES=${BEAM_CFG_BANK}" "BEAM_WIDTH=${w}" ) ;;
    smc)
      local k; k="$(ceil_div "${target_nfe}" "${STEPS}")"; (( k < 2 )) && k=2
      env_pairs+=(
        "CFG_SCALES=${BASELINE_CFG}" "SMC_K=${k}"
        "SMC_GUIDANCE_SCALE=${BASELINE_GUIDANCE_SCALE}"
        "ESS_THRESHOLD=0.5" "RESAMPLE_START_FRAC=0.3"
        "SMC_POTENTIAL=tempering"
      ) ;;
    fksteering)
      local k; k="$(ceil_div "${target_nfe}" "${STEPS}")"; (( k < 2 )) && k=2
      env_pairs[0]="METHODS=smc"
      env_pairs+=(
        "CFG_SCALES=${BASELINE_CFG}" "SMC_K=${k}"
        "SMC_GUIDANCE_SCALE=${BASELINE_GUIDANCE_SCALE}"
        "ESS_THRESHOLD=0.5" "RESAMPLE_START_FRAC=0.3"
        "SMC_POTENTIAL=diff" "SMC_LAMBDA=${FKSTEERING_LAMBDA_SWEEP}"
      ) ;;
    greedy)
      local n; n="$(ceil_div "${target_nfe}" "${STEPS}")"; (( n < 1 )) && n=1
      env_pairs+=( "GREEDY_N_VARIANTS=${n}" "GREEDY_CFG_SCALES=${BASELINE_CFG}" ) ;;
    ga)
      local denom=$(( GA_GENERATIONS_SWEEP * STEPS )); local p; p="$(ceil_div "${target_nfe}" "${denom}")"; (( p < 2 )) && p=2
      env_pairs+=(
        "GA_POPULATION=${p}" "GA_GENERATIONS=${GA_GENERATIONS_SWEEP}" "GA_ELITES=${GA_ELITES_SWEEP}"
        "GA_GUIDANCE_SCALES=${BASELINE_GUIDANCE_SCALE}"
      ) ;;
    dts|dts_star)
      local m; m="$(ceil_div "${target_nfe}" "${STEPS}")"; (( m < 1 )) && m=1
      env_pairs+=( "CFG_SCALES=${BASELINE_CFG}" "DTS_M_ITER=${m}" ) ;;
    dynamic_cfg_x0)
      env_pairs+=(
        "CFG_SCALES=${BASELINE_CFG}"
        "DYNAMIC_CFG_X0_GRID=${DYNCFG_X0_GRID}"
        "DYNAMIC_CFG_X0_SCORE_START_FRAC=${DYNCFG_X0_START_FRAC}"
        "DYNAMIC_CFG_X0_SCORE_END_FRAC=1.0"
        "DYNAMIC_CFG_X0_SCORE_EVERY=${target_nfe}"
        "DYNAMIC_CFG_X0_EVALUATORS=${REWARD_BACKEND}"
        "DYNAMIC_CFG_X0_SMOOTH_WEIGHT=0.0"
        "DYNAMIC_CFG_X0_HIGH_CFG_PENALTY=0.0"
      ) ;;
    sop)
      local k_val="${target_nfe%:*}" m_val="${target_nfe##*:}"
      env_pairs+=(
        "CFG_SCALES=${BASELINE_CFG}"
        "SOP_INIT_PATHS=${k_val}" "SOP_KEEP_TOP=${k_val}" "SOP_BRANCH_FACTOR=${m_val}"
        "SOP_BRANCH_EVERY=1" "SOP_START_FRAC=${DYNCFG_X0_START_FRAC}" "SOP_END_FRAC=1.0"
        "SOP_SCORE_DECODE=x0_pred" "SOP_VARIANT_IDX=0"
      ) ;;
    *)
      echo "Error: unsupported method '${method}' in flux sweep" >&2; return 1 ;;
  esac

  echo "[sweep] config=${label} backend=${backend} target=${target_nfe} out=${config_root}"
  env "${env_pairs[@]}" bash "${SUITE_SCRIPT}"
}

declare -a SWEEP_ROOTS=()
for backend in ${FLUX_BACKEND_LIST}; do
  resolve_backend_defaults "${backend}"
  sweep_root="${OUT_ROOT_BASE}/${backend}/sweep_${RUN_TS}"
  mkdir -p "${sweep_root}"
  echo "[sweep] backend=${backend} steps=${STEPS} budgets=${NFE_BUDGETS}"
  for method in ${SWEEP_METHODS}; do
    case "${method}" in
      baseline)        sweep_values="${STEPS}" ;;
      dynamic_cfg_x0)  sweep_values="${DYNCFG_X0_SCORE_EVERY_SWEEP}" ;;
      sop)             sweep_values="${SOP_KM_SWEEP}" ;;
      *)               sweep_values="${NFE_BUDGETS}" ;;
    esac
    for nfe in ${sweep_values}; do
      run_one_config "${backend}" "${method}" "${nfe}" "${sweep_root}" || \
        echo "[sweep] WARN: backend=${backend} method=${method} nfe=${nfe} failed" >&2
    done
  done
  SWEEP_ROOTS+=("${sweep_root}")
done

echo "[sweep-flux] done. Roots:"; for r in "${SWEEP_ROOTS[@]}"; do echo "  ${r}"; done
