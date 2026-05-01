#!/usr/bin/env bash
# Single-bash entry: NFE sweep across all backends → canonical CSV → 6 figures.
#
# Pipeline:
#   1. nfe_sweep_sd35.sh          (sid + senseflow_large + sd35_base)
#   2. nfe_sweep_flux.sh          (flux_schnell)
#   3. aggregate_runs_to_csv.py   (one canonical CSV with the 3 NFE columns +
#                                  uses_reward_diff + mean_search per row)
#   4. plot_nfe_vs_reward_csv.py × 6 = 3 NFE flavors × 2 search rewards
#
# Outputs:
#   $OUT_ROOT_BASE_SD35/merged_<TS>/combined.csv   (sd3.5 family)
#   $OUT_ROOT_BASE_FLUX/merged_<TS>/combined.csv   (flux family)
#   $OUT_ROOT_BASE_PLOTS/all_combined.csv          (merged across families)
#   $OUT_ROOT_BASE_PLOTS/figures/{flavor}_{search_reward}/by_backend_*.png
#
# Override knobs (env vars):
#   NUM_PROMPTS         (default 8)
#   SEED                (default 42)
#   REWARD_BACKENDS     (default "imagereward hpsv3")
#   SD35_BACKEND_LIST   (default "sid senseflow_large sd35_base")
#   FLUX_BACKEND_LIST   (default "flux_schnell")
#   OUT_ROOT_BASE_SD35  (default /tmp/sd35_nfe_sweep)
#   OUT_ROOT_BASE_FLUX  (default /tmp/flux_nfe_sweep)
#   OUT_ROOT_BASE_PLOTS (default /tmp/nfe_plots_<TS>)
#   SKIP_SWEEPS=1       skip steps 1–2 (replot from existing CSV)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_PROMPTS="${NUM_PROMPTS:-8}"
SEED="${SEED:-42}"
REWARD_BACKENDS="${REWARD_BACKENDS:-imagereward hpsv3}"

OUT_ROOT_BASE_SD35="${OUT_ROOT_BASE_SD35:-/tmp/sd35_nfe_sweep}"
OUT_ROOT_BASE_FLUX="${OUT_ROOT_BASE_FLUX:-/tmp/flux_nfe_sweep}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT_BASE_PLOTS="${OUT_ROOT_BASE_PLOTS:-/tmp/nfe_plots_${RUN_TS}}"
mkdir -p "${OUT_ROOT_BASE_PLOTS}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# ── Step 1: SD3.5 family sweep (one pass per search_reward) ─────────────────
if [[ "${SKIP_SWEEPS:-0}" != "1" ]]; then
  for reward in ${REWARD_BACKENDS}; do
    echo "[pipeline] === SD3.5 sweep: search_reward=${reward} ==="
    NUM_PROMPTS="${NUM_PROMPTS}" SEED="${SEED}" \
    REWARD_BACKEND="${reward}" REWARD_BACKENDS="${reward}" EVAL_BACKENDS="imagereward hpsv3" \
    OUT_ROOT_BASE="${OUT_ROOT_BASE_SD35}/${reward}" \
      bash "${SCRIPT_DIR}/nfe_sweep_sd35.sh"
  done

  for reward in ${REWARD_BACKENDS}; do
    echo "[pipeline] === FLUX sweep: search_reward=${reward} ==="
    NUM_PROMPTS="${NUM_PROMPTS}" SEED="${SEED}" \
    REWARD_BACKEND="${reward}" REWARD_BACKENDS="${reward}" EVAL_BACKENDS="imagereward hpsv3" \
    OUT_ROOT_BASE="${OUT_ROOT_BASE_FLUX}/${reward}" \
      bash "${SCRIPT_DIR}/nfe_sweep_flux.sh"
  done
fi

# ── Step 2: build canonical CSV ─────────────────────────────────────────────
echo "[pipeline] === aggregating CSV ==="
declare -a AGG_ARGS=()
for reward in ${REWARD_BACKENDS}; do
  if [[ -d "${OUT_ROOT_BASE_SD35}/${reward}" ]]; then
    AGG_ARGS+=( --auto "${OUT_ROOT_BASE_SD35}/${reward}" )
  fi
  if [[ -d "${OUT_ROOT_BASE_FLUX}/${reward}" ]]; then
    AGG_ARGS+=( --auto "${OUT_ROOT_BASE_FLUX}/${reward}" )
  fi
done
COMBINED_CSV="${OUT_ROOT_BASE_PLOTS}/combined.csv"
"${PYTHON_BIN}" "${SCRIPT_DIR}/aggregate_runs_to_csv.py" "${AGG_ARGS[@]}" --out "${COMBINED_CSV}"
echo "[pipeline] CSV at: ${COMBINED_CSV}"

# ── Step 3: 6 figures (3 flavors × 2 search rewards) ────────────────────────
echo "[pipeline] === plotting ==="
for flavor in v1 v2 v3; do
  for reward in ${REWARD_BACKENDS}; do
    fig_dir="${OUT_ROOT_BASE_PLOTS}/figures/${flavor}_${reward}"
    mkdir -p "${fig_dir}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_nfe_vs_reward_csv.py" \
      --csv "${COMBINED_CSV}" \
      --out_dir "${fig_dir}" \
      --nfe_flavor "${flavor}" \
      --search_reward "${reward}" \
      || echo "[pipeline] WARN flavor=${flavor} reward=${reward} plot failed"
  done
done

echo
echo "[pipeline] DONE."
echo "  CSV:    ${COMBINED_CSV}"
echo "  Plots:  ${OUT_ROOT_BASE_PLOTS}/figures/{v1,v2,v3}_{${REWARD_BACKENDS// /,}}/"
