#!/usr/bin/env bash
# NFE-vs-quality sweep for SD3.5-SID and/or SD3.5-base across:
#   bon, beam, smc, bon_mcts
#
# How it works:
#   For each (backend, method, target_nfe) the driver picks a knob value
#   (BON_N, BEAM_WIDTH, SMC_K, BON_MCTS_N_SEEDS=N_SIMS) so the *expected*
#   number of transformer forwards lands on target_nfe, then invokes
#   hpsv2_sd35_sid_ddp_suite.sh with that single method and 1 image per GPU.
#
# NFE accounting (rounded; baseline assumes single variant + single CFG):
#   baseline   : STEPS
#   bon        : BON_N * STEPS
#   beam       : BEAM_WIDTH * |cfgs| * |variants| * STEPS
#   smc        : SMC_K * STEPS                     (expansion off for clean NFE)
#   bon_mcts   : (BON_MCTS_N_SEEDS + N_SIMS) * STEPS    (topk=1, sim_alloc=full)
#   greedy     : N_VARIANTS * STEPS                (per-step branching, pick best)
#   ga         : GA_POPULATION * GA_GENERATIONS * STEPS  (elite carryover ignored)
#   dts/dts*   : DTS_M_ITER * STEPS                (M trajectories × per-step calls)
#
# Per-config layout:
#   ${OUT_ROOT_BASE}/${SD35_BACKEND}/sweep_<TS>/${method}_nfe<N>/run_<TS>/<method>/
# Per-backend summary:
#   ${OUT_ROOT_BASE}/${SD35_BACKEND}/sweep_<TS>/sweep_summary.tsv
# Cross-backend merge (combined csv + plots):
#   ${OUT_ROOT_BASE}/merged_<TS>/combined.csv
#   ${OUT_ROOT_BASE}/merged_<TS>/<backend>_<eval>_vs_nfe.png
#   ${OUT_ROOT_BASE}/merged_<TS>/summary_<eval>_vs_nfe.png
#
# Usage:
#   # both backends sequentially (default), single trial:
#   bash nfe_sweep_sd35.sh
#   # one backend only:
#   SD35_BACKEND_LIST=sid       bash nfe_sweep_sd35.sh
#   SD35_BACKEND_LIST=sd35_base bash nfe_sweep_sd35.sh
#
# Common overrides:
#   SD35_BACKEND_LIST="sid sd35_base"
#   SWEEP_METHODS="bon beam smc bon_mcts"
#   NFE_BUDGETS_SID="16 32 64 128 256 512"
#   NFE_BUDGETS_BASE="56 112 224 448 896 1792"
#   NUM_PROMPTS=8                          # 1 image per visible GPU
#   BEAM_CFG_BANK_SID="1.0 2.0"  BEAM_CFG_BANK_BASE="3.5 7.0"
#   MCTS_CFG_BANK_SID="1.0 1.5 2.0"  MCTS_CFG_BANK_BASE="3.5 4.5 5.5 7.0"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SD35_BACKEND_LIST="${SD35_BACKEND_LIST:-${SD35_BACKEND:-sid sd35_base}}"
SWEEP_METHODS="${SWEEP_METHODS:-bon beam smc bon_mcts greedy ga dts dts_star}"

# GA defaults — generations fixed, population scaled to hit target_nfe.
GA_GENERATIONS_SWEEP="${GA_GENERATIONS:-8}"
GA_ELITES_SWEEP="${GA_ELITES:-3}"

NUM_PROMPTS="${NUM_PROMPTS:-8}"
GEN_BATCH_SIZE_SWEEP="${GEN_BATCH_SIZE:-1}"
SEED="${SEED:-42}"
SAVE_BEST_IMAGES_SWEEP="${SAVE_BEST_IMAGES:-1}"
SAVE_IMAGES_SWEEP="${SAVE_IMAGES:-0}"
USE_QWEN_SWEEP="${USE_QWEN:-0}"
PRECOMPUTE_REWRITES_SWEEP="${PRECOMPUTE_REWRITES:-0}"

BEAM_N_VARIANTS="${BEAM_N_VARIANTS:-1}"

PROMPT_FILE_DEFAULT="${SCRIPT_DIR}/hpsv2_subset.txt"
PROMPT_FILE="${PROMPT_FILE:-${PROMPT_FILE_DEFAULT}}"
if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found: ${PROMPT_FILE}" >&2
  exit 1
fi

REWARD_BACKEND="${REWARD_BACKEND:-imagereward}"
REWARD_BACKENDS="${REWARD_BACKENDS:-imagereward hpsv2 pickscore}"
EVAL_BACKENDS="${EVAL_BACKENDS:-imagereward hpsv2 pickscore}"

OUT_ROOT_BASE="${OUT_ROOT_BASE:-${OUT_ROOT:-/tmp/sd35_nfe_sweep}}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

PYTHON_BIN="${PYTHON_BIN:-python}"
SUITE_SCRIPT="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
COMBINE_SCRIPT="${SCRIPT_DIR}/nfe_sweep_combine.py"
if [[ ! -f "${SUITE_SCRIPT}" ]]; then
  echo "Error: suite not found at ${SUITE_SCRIPT}" >&2
  exit 1
fi

# ── Helpers ──────────────────────────────────────────────────────────────────
ceil_div() {
  local a="$1" b="$2"
  echo $(( (a + b - 1) / b ))
}

count_tokens() {
  local s="$1"
  read -r -a _arr <<< "${s}"
  echo "${#_arr[@]}"
}

resolve_backend_defaults() {
  local backend="$1"
  case "${backend}" in
    sid)
      STEPS="${STEPS_SID:-4}"
      NFE_BUDGETS="${NFE_BUDGETS_SID:-16 32 64 128 256 512}"
      BASELINE_CFG="${BASELINE_CFG_SID:-1.0}"
      BEAM_CFG_BANK="${BEAM_CFG_BANK_SID:-1.0 2.0}"
      MCTS_CFG_BANK="${MCTS_CFG_BANK_SID:-1.0 1.5 2.0}"
      BON_MCTS_KEY_STEP_COUNT="${BON_MCTS_KEY_STEP_COUNT_SID:-2}"
      ;;
    sd35_base)
      STEPS="${STEPS_BASE:-28}"
      NFE_BUDGETS="${NFE_BUDGETS_BASE:-56 112 224 448 896 1792}"
      BASELINE_CFG="${BASELINE_CFG_BASE:-4.5}"
      BEAM_CFG_BANK="${BEAM_CFG_BANK_BASE:-3.5 7.0}"
      MCTS_CFG_BANK="${MCTS_CFG_BANK_BASE:-3.5 4.5 5.5 7.0}"
      BON_MCTS_KEY_STEP_COUNT="${BON_MCTS_KEY_STEP_COUNT_BASE:-4}"
      ;;
    *)
      echo "Error: unsupported backend=${backend} (sid|sd35_base)" >&2
      return 1
      ;;
  esac
}

compute_knobs_for() {
  local method="$1" target_nfe="$2"
  local knob_label="" knob_value=0 nfe_actual=0
  case "${method}" in
    baseline)
      knob_label="steps"; knob_value="${STEPS}"; nfe_actual="${STEPS}" ;;
    bon)
      local n; n="$(ceil_div "${target_nfe}" "${STEPS}")"; (( n < 1 )) && n=1
      knob_label="bon_n"; knob_value="${n}"; nfe_actual=$(( n * STEPS )) ;;
    beam)
      local n_cfgs; n_cfgs="$(count_tokens "${BEAM_CFG_BANK}")"; (( n_cfgs < 1 )) && n_cfgs=1
      local denom=$(( STEPS * n_cfgs * BEAM_N_VARIANTS ))
      local w; w="$(ceil_div "${target_nfe}" "${denom}")"; (( w < 1 )) && w=1
      knob_label="beam_width"; knob_value="${w}"
      nfe_actual=$(( w * n_cfgs * BEAM_N_VARIANTS * STEPS )) ;;
    smc)
      local k; k="$(ceil_div "${target_nfe}" "${STEPS}")"; (( k < 2 )) && k=2
      knob_label="smc_k"; knob_value="${k}"; nfe_actual=$(( k * STEPS )) ;;
    bon_mcts)
      local x; x="$(ceil_div "${target_nfe}" $(( 2 * STEPS )))"; (( x < 1 )) && x=1
      knob_label="n_seeds_eq_n_sims"; knob_value="${x}"
      nfe_actual=$(( 2 * x * STEPS )) ;;
    greedy)
      local n; n="$(ceil_div "${target_nfe}" "${STEPS}")"; (( n < 1 )) && n=1
      knob_label="n_variants"; knob_value="${n}"; nfe_actual=$(( n * STEPS )) ;;
    ga)
      local denom=$(( GA_GENERATIONS_SWEEP * STEPS ))
      local p; p="$(ceil_div "${target_nfe}" "${denom}")"; (( p < 2 )) && p=2
      knob_label="ga_population"; knob_value="${p}"
      nfe_actual=$(( p * GA_GENERATIONS_SWEEP * STEPS )) ;;
    dts|dts_star)
      local m; m="$(ceil_div "${target_nfe}" "${STEPS}")"; (( m < 1 )) && m=1
      knob_label="dts_m_iter"; knob_value="${m}"; nfe_actual=$(( m * STEPS )) ;;
    *)
      echo "Error: unknown method ${method}" >&2; return 1 ;;
  esac
  printf '%s|%s|%s|%s\n' "${method}" "${target_nfe}" "${knob_label}=${knob_value}" "${nfe_actual}"
}

run_one_config() {
  local backend="$1" method="$2" target_nfe="$3" sweep_root="$4"
  local label="${method}_nfe${target_nfe}"
  local config_root="${sweep_root}/${label}"
  mkdir -p "${config_root}"

  local -a env_pairs=(
    "METHODS=${method}"
    "OUT_ROOT=${config_root}"
    "GEN_BATCH_SIZE=${GEN_BATCH_SIZE_SWEEP}"
    "STEPS=${STEPS}"
    "AUTO_BACKEND_STEPS=0"
    "SD35_BACKEND=${backend}"
    "PROMPT_FILE=${PROMPT_FILE}"
    "START_INDEX=0"
    "END_INDEX=${NUM_PROMPTS}"
    "SEED=${SEED}"
    "BASELINE_CFG=${BASELINE_CFG}"
    "CORRECTION_STRENGTHS=0.0"
    "USE_QWEN=${USE_QWEN_SWEEP}"
    "PRECOMPUTE_REWRITES=${PRECOMPUTE_REWRITES_SWEEP}"
    "SAVE_BEST_IMAGES=${SAVE_BEST_IMAGES_SWEEP}"
    "SAVE_IMAGES=${SAVE_IMAGES_SWEEP}"
    "REWARD_BACKEND=${REWARD_BACKEND}"
    "REWARD_BACKENDS=${REWARD_BACKENDS}"
    "EVAL_BACKENDS=${EVAL_BACKENDS}"
    "SMC_VARIANT_EXPANSION=0"
  )

  case "${method}" in
    bon)
      local n; n="$(ceil_div "${target_nfe}" "${STEPS}")"; (( n < 1 )) && n=1
      env_pairs+=( "N_VARIANTS=1" "CFG_SCALES=${BASELINE_CFG}" "BON_N=${n}" )
      ;;
    beam)
      local n_cfgs w
      n_cfgs="$(count_tokens "${BEAM_CFG_BANK}")"; (( n_cfgs < 1 )) && n_cfgs=1
      local denom=$(( STEPS * n_cfgs * BEAM_N_VARIANTS ))
      w="$(ceil_div "${target_nfe}" "${denom}")"; (( w < 1 )) && w=1
      env_pairs+=( "N_VARIANTS=${BEAM_N_VARIANTS}" "CFG_SCALES=${BEAM_CFG_BANK}" "BEAM_WIDTH=${w}" )
      ;;
    smc)
      local k; k="$(ceil_div "${target_nfe}" "${STEPS}")"; (( k < 2 )) && k=2
      env_pairs+=(
        "N_VARIANTS=1" "CFG_SCALES=${BASELINE_CFG}" "SMC_K=${k}"
        "SMC_CFG_SCALE=${BASELINE_CFG}" "SMC_VARIANT_IDX=0"
        "ESS_THRESHOLD=0.5" "RESAMPLE_START_FRAC=0.3"
      )
      ;;
    bon_mcts)
      local x; x="$(ceil_div "${target_nfe}" $(( 2 * STEPS )))"; (( x < 1 )) && x=1
      env_pairs+=(
        "N_VARIANTS=1" "CFG_SCALES=${MCTS_CFG_BANK}"
        "BON_MCTS_N_SEEDS=${x}" "BON_MCTS_TOPK=1"
        "BON_MCTS_SIM_ALLOC=full" "BON_MCTS_MIN_SIMS=1"
        "BON_MCTS_PRESCREEN_CFG=${BASELINE_CFG}"
        "N_SIMS=${x}" "MCTS_KEY_MODE=count"
        "MCTS_KEY_STEP_COUNT=${BON_MCTS_KEY_STEP_COUNT}"
      )
      ;;
    greedy)
      local n; n="$(ceil_div "${target_nfe}" "${STEPS}")"; (( n < 1 )) && n=1
      env_pairs+=( "N_VARIANTS=${n}" "CFG_SCALES=${BASELINE_CFG}" )
      ;;
    ga)
      local denom=$(( GA_GENERATIONS_SWEEP * STEPS ))
      local p; p="$(ceil_div "${target_nfe}" "${denom}")"; (( p < 2 )) && p=2
      env_pairs+=(
        "N_VARIANTS=1" "CFG_SCALES=${BASELINE_CFG}"
        "GA_POPULATION=${p}"
        "GA_GENERATIONS=${GA_GENERATIONS_SWEEP}"
        "GA_ELITES=${GA_ELITES_SWEEP}"
      )
      ;;
    dts|dts_star)
      local m; m="$(ceil_div "${target_nfe}" "${STEPS}")"; (( m < 1 )) && m=1
      env_pairs+=(
        "N_VARIANTS=1" "CFG_SCALES=${BASELINE_CFG}"
        "DTS_M_ITER=${m}"
        "DTS_LAMBDA=${DTS_LAMBDA:-1.0}"
        "DTS_PW_C=${DTS_PW_C:-1.0}"
        "DTS_PW_ALPHA=${DTS_PW_ALPHA:-0.5}"
        "DTS_C_UCT=${DTS_C_UCT:-1.0}"
        "DTS_SDE_NOISE_SCALE=${DTS_SDE_NOISE_SCALE:-0.0}"
        "DTS_CFG_BANK=${DTS_CFG_BANK:-}"
      )
      ;;
    *)
      echo "Error: unsupported method '${method}' in sweep" >&2; return 1 ;;
  esac

  echo
  echo "[sweep] ────────────────────────────────────────────────────────────"
  echo "[sweep] config=${label} backend=${backend} target_nfe=${target_nfe}"
  echo "[sweep]   knobs:"
  for kv in "${env_pairs[@]}"; do echo "[sweep]     ${kv}"; done
  echo "[sweep]   out=${config_root}"

  env "${env_pairs[@]}" bash "${SUITE_SCRIPT}"
}

emit_plan() {
  local sweep_root="$1"
  local plan_path="${sweep_root}/sweep_plan.tsv"
  {
    printf 'method\ttarget_nfe\tknob\tnfe_actual\n'
    for method in ${SWEEP_METHODS}; do
      for nfe in ${NFE_BUDGETS}; do
        local row; row="$(compute_knobs_for "${method}" "${nfe}")"
        IFS='|' read -r m n k a <<< "${row}"
        printf '%s\t%s\t%s\t%s\n' "${m}" "${n}" "${k}" "${a}"
      done
    done
  } > "${plan_path}"
  echo "[sweep] plan written: ${plan_path}"
  cat "${plan_path}"
}

aggregate_sweep() {
  local sweep_root="$1"
  local agg_path="${sweep_root}/sweep_summary.tsv"
  "${PYTHON_BIN}" - "${sweep_root}" "${agg_path}" <<'PY'
import csv
import glob
import json
import os
import sys

sweep_root = sys.argv[1]
agg_path = sys.argv[2]

rows = []
backends = []

for cfg_dir in sorted(glob.glob(os.path.join(sweep_root, "*_nfe*"))):
    label = os.path.basename(cfg_dir)
    if "_nfe" not in label:
        continue
    method, nfe_part = label.rsplit("_nfe", 1)
    try:
        target_nfe = int(nfe_part)
    except ValueError:
        continue

    method_dir = None
    for run_dir in sorted(glob.glob(os.path.join(cfg_dir, "run_*"))):
        for cand in os.listdir(run_dir):
            sub = os.path.join(run_dir, cand)
            if not os.path.isdir(sub):
                continue
            if os.path.exists(os.path.join(sub, "aggregate_ddp.json")):
                method_dir = sub
                break
        if method_dir:
            break
    if not method_dir:
        rows.append({"method": method, "target_nfe": target_nfe, "status": "missing"})
        continue

    with open(os.path.join(method_dir, "aggregate_ddp.json")) as f:
        agg = json.load(f)
    row = {
        "method": method,
        "target_nfe": target_nfe,
        "status": "ok",
        "num_samples": agg.get("num_samples"),
        "elapsed_sec": agg.get("elapsed_sec"),
        "mean_baseline": agg.get("mean_baseline_score"),
        "mean_search": agg.get("mean_search_score"),
        "mean_delta": agg.get("mean_delta_score"),
    }
    eval_path = os.path.join(method_dir, "best_images_multi_reward_aggregate.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_agg = json.load(f)
        for backend, stats in (eval_agg.get("backend_stats") or {}).items():
            mean_val = stats.get("mean")
            if mean_val is not None:
                col = f"eval_{backend}"
                row[col] = float(mean_val)
                if col not in backends:
                    backends.append(col)
    rows.append(row)

base_cols = ["method", "target_nfe", "status", "num_samples", "elapsed_sec",
             "mean_baseline", "mean_search", "mean_delta"]
cols = base_cols + sorted(backends)
with open(agg_path, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(cols)
    for row in rows:
        writer.writerow([row.get(c, "") for c in cols])
print(f"[sweep] aggregate written: {agg_path} (rows={len(rows)})")
PY
  if [[ -f "${agg_path}" ]]; then
    echo
    cat "${agg_path}"
  fi
}

run_one_backend() {
  local backend="$1"
  resolve_backend_defaults "${backend}"
  local sweep_root="${OUT_ROOT_BASE}/${backend}/sweep_${RUN_TS}"
  mkdir -p "${sweep_root}"

  echo
  echo "════════════════════════════════════════════════════════════════════"
  echo "[sweep] backend=${backend} steps=${STEPS}"
  echo "[sweep] methods=${SWEEP_METHODS}"
  echo "[sweep] nfe_budgets=${NFE_BUDGETS}"
  echo "[sweep] num_prompts=${NUM_PROMPTS} (1 per GPU expected) gen_batch_size=${GEN_BATCH_SIZE_SWEEP}"
  echo "[sweep] beam_cfg_bank='${BEAM_CFG_BANK}' (n_cfgs=$(count_tokens "${BEAM_CFG_BANK}")) beam_n_variants=${BEAM_N_VARIANTS}"
  echo "[sweep] mcts_cfg_bank='${MCTS_CFG_BANK}' bon_mcts_key_step_count=${BON_MCTS_KEY_STEP_COUNT}"
  echo "[sweep] sweep_root=${sweep_root}"
  echo "════════════════════════════════════════════════════════════════════"

  emit_plan "${sweep_root}"

  for method in ${SWEEP_METHODS}; do
    for nfe in ${NFE_BUDGETS}; do
      if ! run_one_config "${backend}" "${method}" "${nfe}" "${sweep_root}"; then
        echo "[sweep] WARNING: backend=${backend} method=${method} nfe=${nfe} failed" >&2
      fi
    done
  done

  aggregate_sweep "${sweep_root}"
  echo "${sweep_root}"
}

# ── Main ────────────────────────────────────────────────────────────────────
declare -a SWEEP_ROOTS=()
for backend in ${SD35_BACKEND_LIST}; do
  sweep_root="$(run_one_backend "${backend}" | tail -n 1)"
  if [[ -d "${sweep_root}" ]]; then
    SWEEP_ROOTS+=("${sweep_root}")
  fi
done

# ── Combine across backends ─────────────────────────────────────────────────
MERGED_DIR="${OUT_ROOT_BASE}/merged_${RUN_TS}"
COMBINE_INPUTS=("${SWEEP_ROOTS[@]}")
if [[ -n "${MERGE_WITH:-}" ]]; then
  if [[ -d "${MERGE_WITH}" || -f "${MERGE_WITH}" ]]; then
    COMBINE_INPUTS+=("${MERGE_WITH}")
  else
    echo "[sweep] MERGE_WITH not found, skipping: ${MERGE_WITH}" >&2
  fi
fi

if [[ ${#COMBINE_INPUTS[@]} -eq 0 ]]; then
  echo "[sweep] no successful sweep roots — nothing to merge." >&2
elif [[ ! -f "${COMBINE_SCRIPT}" ]]; then
  echo "[sweep] combiner not found at ${COMBINE_SCRIPT}, skipping merge." >&2
else
  echo
  echo "[sweep] combining ${#COMBINE_INPUTS[@]} sweep root(s) -> ${MERGED_DIR}"
  "${PYTHON_BIN}" "${COMBINE_SCRIPT}" \
    --inputs "${COMBINE_INPUTS[@]}" \
    --out_dir "${MERGED_DIR}" || echo "[sweep] WARNING: combiner failed" >&2
fi

echo
echo "[sweep] done."
for r in "${SWEEP_ROOTS[@]}"; do
  echo "[sweep]   per-backend: ${r}"
done
echo "[sweep]   merged: ${MERGED_DIR}"
