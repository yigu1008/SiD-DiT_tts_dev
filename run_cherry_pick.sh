#!/usr/bin/env bash
# Single-bash entry: cherry-pick best (prompt, seed) examples where bon_mcts wins.
#
# For ONE backend (passed via BACKEND env var):
#   1. cherry_pick_prompts.py samples 100 prompts from HPSv2 + DrawBench (seed
#      varies per backend → no overlap across backends).
#   2. Run the matching suite (sd35 or flux) at SEEDS={42,43,44,45} with
#      METHODS="bon smc fksteering dts_star bon_mcts".
#   3. cherry_pick_select.py reads best_images_multi_reward.json from each
#      method dir, finds (prompt, seed) where bon_mcts is rank-1 by
#      hpsv3+imagereward, and copies the top 8 (by margin) into winners/.
#
# Required env:
#   BACKEND               - one of {sid, senseflow_large, sd35_base, flux_schnell}
#   RUN_ROOT              - output dir parent
#   REWARD_SERVER_URL     - shared server (must host hpsv3 AND imagereward)
#
# Optional env:
#   N_PROMPTS             (default 100)
#   SEEDS                 (default "42 43 44 45")
#   N_WINNERS             (default 8)
#   PYTHON_BIN            (default python)
#   FAIL_FAST             (default 0)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${BACKEND:?BACKEND must be set (sid, senseflow_large, sd35_base, flux_schnell)}"
: "${RUN_ROOT:?RUN_ROOT must be set}"

N_PROMPTS="${N_PROMPTS:-100}"
SEEDS="${SEEDS:-42 43 44 45}"
N_WINNERS="${N_WINNERS:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FAIL_FAST="${FAIL_FAST:-0}"

PROMPTS_DIR="${RUN_ROOT}/_prompts"
mkdir -p "${PROMPTS_DIR}"
PROMPT_FILE="${PROMPTS_DIR}/backend_${BACKEND}.txt"

# ── Step 1: sample prompts (one-time per backend) ───────────────────────────
# Prompt download needs HF online — caller may have set HF_HUB_OFFLINE=1 for
# the sampling phase, so temporarily disable it here.
if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "[cherry] sampling prompts → ${PROMPT_FILE}"
  env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_prompts.py" \
    --n_prompts "${N_PROMPTS}" \
    --out_dir "${PROMPTS_DIR}" \
    --backends "${BACKEND}"
else
  echo "[cherry] reusing ${PROMPT_FILE}"
fi

# ── Step 2: per-seed suite runs ─────────────────────────────────────────────
case "${BACKEND}" in
  sid|senseflow_large|sd35_base)
    SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
    SUITE_KIND="sd35" ;;
  flux_schnell)
    SUITE="${SCRIPT_DIR}/hpsv2_flux_schnell_ddp_suite.sh"
    SUITE_KIND="flux" ;;
  *)
    echo "[cherry] ERROR unknown BACKEND=${BACKEND}" >&2
    exit 2 ;;
esac

# Default suite knobs — methods and basic config.
export METHODS="${METHODS:-bon smc fksteering dts_star bon_mcts}"
export PROMPT_FILE
export START_INDEX="${START_INDEX:-0}"
export END_INDEX="${END_INDEX:-${N_PROMPTS}}"
export N_VARIANTS="${N_VARIANTS:-1}"
export USE_QWEN="${USE_QWEN:-0}"
export PRECOMPUTE_REWRITES="${PRECOMPUTE_REWRITES:-0}"
export REWARDS_OVERWRITE="${REWARDS_OVERWRITE:-0}"
export CORRECTION_STRENGTHS="${CORRECTION_STRENGTHS:-0.0}"
export SAVE_BEST_IMAGES="${SAVE_BEST_IMAGES:-1}"
export SAVE_IMAGES="${SAVE_IMAGES:-0}"
export SAVE_VARIANTS="${SAVE_VARIANTS:-0}"
export EVAL_BACKENDS="${EVAL_BACKENDS:-imagereward hpsv3}"
export REWARD_BACKEND="${REWARD_BACKEND:-imagereward}"
export REWARD_BACKENDS="${REWARD_BACKENDS:-imagereward hpsv3}"

if [[ "${SUITE_KIND}" == "sd35" ]]; then
  export SD35_BACKEND="${BACKEND}"
fi

failed=()
for seed in ${SEEDS}; do
  seed_root="${RUN_ROOT}/${BACKEND}/seed${seed}"
  mkdir -p "${seed_root}"
  echo "[cherry] === ${BACKEND} seed=${seed} → ${seed_root} ==="
  if SEED="${seed}" OUT_ROOT="${seed_root}" bash "${SUITE}"; then
    echo "[cherry] OK ${BACKEND} seed=${seed}"
  else
    rc=$?
    echo "[cherry] FAIL ${BACKEND} seed=${seed} rc=${rc}" >&2
    failed+=("${seed}")
    if [[ "${FAIL_FAST}" == "1" ]]; then
      exit "${rc}"
    fi
  fi
done

# ── Step 3: aggregate winners across all seeds ──────────────────────────────
selector_root="${RUN_ROOT}/${BACKEND}"
selector_out="${RUN_ROOT}/${BACKEND}/_winners"
echo "[cherry] selecting top-${N_WINNERS} winners from ${selector_root}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_select.py" \
  --run_root "${selector_root}" \
  --out_dir "${selector_out}" \
  --n_winners "${N_WINNERS}"

echo
echo "[cherry] DONE backend=${BACKEND}"
echo "  prompts: ${PROMPT_FILE}"
echo "  winners: ${selector_out}/winners/"
echo "  manifest: ${selector_out}/winners.json"
if (( ${#failed[@]} > 0 )); then
  echo "[cherry] failures during seed runs: ${failed[*]}"
  exit 1
fi
