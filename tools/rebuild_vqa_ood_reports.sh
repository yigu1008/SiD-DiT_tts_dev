#!/usr/bin/env bash
set -euo pipefail

# Rebuild run-scoped OOD reward reports from existing per-method backend JSONs.
# This script never launches generation or a reward model. Incomplete methods
# remain visible in the coverage CSV while valid completed scores are retained
# in the partial CSV/JSON summary.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
RUN_ID="${RUN_ID:-genai200_v1}"
EXPECTED_PROMPTS="${EXPECTED_PROMPTS:-200}"
MODELS="${MODELS:-flux_schnell senseflow_large sd35_base}"
METHODS="${METHODS:-baseline fksteering bon beam sop ga dts dts_star dynamic_cfg_x0 bon_mcts}"
OOD_EVAL_BACKENDS="${OOD_EVAL_BACKENDS:-imagereward hpsv3 pickscore}"
STUDY_RUN_ROOT="${STUDY_RUN_ROOT:-${HUMAN_EVAL_ROOT}/vqascore_remaining_models/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-/home/ygu/miniconda3/envs/sid_dit/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${STUDY_RUN_ROOT}" ]]; then
  echo "Error: study run root is missing: ${STUDY_RUN_ROOT}" >&2
  exit 1
fi

for model in ${MODELS}; do
  model_root="${STUDY_RUN_ROOT}/${model}"
  run_root="${model_root}/run_${RUN_ID}"
  report_dir="${run_root}/reports"
  if [[ ! -d "${run_root}" ]]; then
    echo "[rebuild] WARN missing run root: ${run_root}" >&2
    continue
  fi
  mkdir -p "${report_dir}"
  echo "[rebuild] model=${model}"

  "${PYTHON_BIN}" "${REPO}/tools/merge_posthoc_reward_evals.py" \
    --root "${STUDY_RUN_ROOT}" \
    --include-models "${model}" \
    --run-id "${RUN_ID}" \
    --backends ${OOD_EVAL_BACKENDS} \
    --summary-csv "${report_dir}/vqa_ood_summary_partial.csv" \
    --expected-count "${EXPECTED_PROMPTS}" \
    --no-strict

  "${PYTHON_BIN}" "${REPO}/tools/audit_vqascore_sweep_coverage.py" \
    --root "${STUDY_RUN_ROOT}" \
    --models "${model}" \
    --methods ${METHODS} \
    --expected-prompts "${EXPECTED_PROMPTS}" \
    --eval-backends ${OOD_EVAL_BACKENDS} \
    --run-id "${RUN_ID}" \
    --out-csv "${report_dir}/vqascore_coverage.csv" \
    --no-strict

  cp "${report_dir}/vqa_ood_summary_partial.csv" \
    "${model_root}/vqa_ood_summary_partial.csv"
  cp "${report_dir}/vqa_ood_summary_partial.json" \
    "${model_root}/vqa_ood_summary_partial.json"
  cp "${report_dir}/vqascore_coverage.csv" \
    "${model_root}/vqascore_coverage.csv"
done

echo "[rebuild] reports are under each model's run_${RUN_ID}/reports directory"
