#!/usr/bin/env bash
# Generate diversity grids + metrics for SD3.5-SiD and SD3.5-large, then merge.
#
# Output:
#   ${OUT_ROOT}/sid/        per-prompt grids + diversity_sid.csv + bar chart
#   ${OUT_ROOT}/sd35_base/  per-prompt grids + diversity_sd35_base.csv + bar chart
#   ${OUT_ROOT}/compare/diversity_compare.png  -- both backends side-by-side
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/hpsv2_subset.txt}"
REWRITES_FILE="${REWRITES_FILE:-}"  # optional Qwen rewrites cache
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/diversity_out}"
N_PROMPTS="${N_PROMPTS:-4}"
N_SEEDS="${N_SEEDS:-4}"
N_VARIANTS="${N_VARIANTS:-4}"
SEED_BASE="${SEED_BASE:-42}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"

# ── SiD ──────────────────────────────────────────────────────────────────
if [[ "${RUN_SID:-1}" == "1" ]]; then
  CFG_SCALES_SID="${CFG_SCALES_SID:-1.0 1.5 2.0 2.5}"   # SiD is real-CFG ≈ 1.0
  STEPS_SID="${STEPS_SID:-4}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/diversity_visualization_sd35.py" \
    --backend sid --model_id YGu1998/SiD-DiT-SD3.5-large \
    --prompt_file "${PROMPT_FILE}" \
    ${REWRITES_FILE:+--rewrites_file "${REWRITES_FILE}"} \
    --n_prompts "${N_PROMPTS}" --n_seeds "${N_SEEDS}" --n_variants "${N_VARIANTS}" \
    --cfg_scales ${CFG_SCALES_SID} --seed_base "${SEED_BASE}" \
    --steps "${STEPS_SID}" --width "${WIDTH}" --height "${HEIGHT}" \
    --out_dir "${OUT_ROOT}/sid" --label sid
fi

# ── SD3.5-base ───────────────────────────────────────────────────────────
if [[ "${RUN_BASE:-1}" == "1" ]]; then
  CFG_SCALES_BASE="${CFG_SCALES_BASE:-3.5 4.5 5.5 7.0}"
  STEPS_BASE="${STEPS_BASE:-28}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/diversity_visualization_sd35.py" \
    --backend sd35_base --model_id stabilityai/stable-diffusion-3.5-large \
    --prompt_file "${PROMPT_FILE}" \
    ${REWRITES_FILE:+--rewrites_file "${REWRITES_FILE}"} \
    --n_prompts "${N_PROMPTS}" --n_seeds "${N_SEEDS}" --n_variants "${N_VARIANTS}" \
    --cfg_scales ${CFG_SCALES_BASE} --seed_base "${SEED_BASE}" \
    --steps "${STEPS_BASE}" --width "${WIDTH}" --height "${HEIGHT}" \
    --out_dir "${OUT_ROOT}/sd35_base" --label sd35_base
fi

# ── Cross-backend comparison ─────────────────────────────────────────────
"${PYTHON_BIN}" "${SCRIPT_DIR}/diversity_visualization_sd35.py" \
  --backend sid --model_id _ \
  --prompt_file "${PROMPT_FILE}" \
  --out_dir "${OUT_ROOT}/compare" \
  --compare_dirs "${OUT_ROOT}/sid" "${OUT_ROOT}/sd35_base"

echo "[diversity] done. Outputs under ${OUT_ROOT}"
