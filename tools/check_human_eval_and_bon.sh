#!/usr/bin/env bash
set -uo pipefail

# Run both non-mutating integrity audits. The only writes are JSON/CSV reports.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
HUMAN_EVAL_CONFIG="${HUMAN_EVAL_CONFIG:-${REPO}/configs/pairwise_human_eval.yaml}"
BON_RUN_ROOT="${BON_RUN_ROOT:-${HUMAN_EVAL_ROOT}/hpsv2_fixed_rewrite_bon8_reward_sweep/v1}"
HUMAN_ALLOW_INCOMPLETE="${HUMAN_ALLOW_INCOMPLETE:-0}"
BON_ALLOW_INCOMPLETE="${BON_ALLOW_INCOMPLETE:-0}"
BON_GENERATION_ONLY="${BON_GENERATION_ONLY:-0}"
REQUIRE_TASKS="${REQUIRE_TASKS:-0}"

human_args=(
  --root "${HUMAN_EVAL_ROOT}"
  --config "${HUMAN_EVAL_CONFIG}"
)
[[ "${HUMAN_ALLOW_INCOMPLETE}" == "1" ]] && human_args+=(--allow-incomplete)
[[ "${REQUIRE_TASKS}" == "1" ]] && human_args+=(--require-tasks)

bon_args=(--root "${BON_RUN_ROOT}")
[[ "${BON_ALLOW_INCOMPLETE}" == "1" ]] && bon_args+=(--allow-incomplete)
[[ "${BON_GENERATION_ONLY}" == "1" ]] && bon_args+=(--generation-only)

echo "[check] human-eval root: ${HUMAN_EVAL_ROOT}"
"${PYTHON_BIN}" "${REPO}/tools/audit_human_eval_integrity.py" "${human_args[@]}"
human_rc=$?

echo "[check] BoN-8 run root: ${BON_RUN_ROOT}"
"${PYTHON_BIN}" "${REPO}/tools/audit_hpsv2_bon8_results.py" "${bon_args[@]}"
bon_rc=$?

if (( human_rc != 0 || bon_rc != 0 )); then
  echo "[check] FAILED human_eval_rc=${human_rc} bon_rc=${bon_rc}" >&2
  exit 1
fi
echo "[check] OK"
