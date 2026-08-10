#!/usr/bin/env bash
set -euo pipefail

# Audit score-file coverage first, then summarize all valid OOD scores. This is
# metadata-only: no generation and no reward model are launched.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
RUN_ID="${RUN_ID:-genai200_v1}"
EXPECTED_PROMPTS="${EXPECTED_PROMPTS:-200}"
OOD_EVAL_BACKENDS="${OOD_EVAL_BACKENDS:-imagereward hpsv3 pickscore hpsv2}"
FLUX_METHODS="${FLUX_METHODS:-baseline fksteering bon beam sop ga dts dts_star dynamic_cfg_x0 bon_mcts}"
SENSE_METHODS="${SENSE_METHODS:-baseline fksteering bon beam sop ga dts dts_star dynamic_cfg_x0 bon_mcts}"
SD35_BASE_METHODS="${SD35_BASE_METHODS:-baseline fksteering bon beam sop ga dts dts_star dynamic_cfg_x0}"
PYTHON_BIN="${PYTHON_BIN:-/home/ygu/miniconda3/envs/sid_dit/bin/python}"
REQUIRE_COMPLETE="${REQUIRE_COMPLETE:-0}"

STUDY_RUN_ROOT="${STUDY_RUN_ROOT:-${HUMAN_EVAL_ROOT}/vqascore_remaining_models/${RUN_ID}}"
REPORT_ROOT="${REPORT_ROOT:-${STUDY_RUN_ROOT}/reports}"
COMBINED_CSV="${REPORT_ROOT}/vqa_ood_score_summary.csv"
COMBINED_JSON="${REPORT_ROOT}/vqa_ood_score_summary.json"
COMBINED_MD="${REPORT_ROOT}/vqa_ood_score_summary.md"

case "${REQUIRE_COMPLETE}" in 0|1) ;;
  *) echo "Error: REQUIRE_COMPLETE must be 0 or 1." >&2; exit 2 ;;
esac
if [[ " ${SD35_BASE_METHODS} " == *" bon_mcts "* ]]; then
  echo "Error: SD35_BASE_METHODS must exclude bon_mcts/ActDiff." >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${STUDY_RUN_ROOT}" ]]; then
  echo "Error: study root is missing: ${STUDY_RUN_ROOT}" >&2
  exit 1
fi
mkdir -p "${REPORT_ROOT}"

audit_model() {
  local model="$1"
  local methods="$2"
  local model_report_dir="${STUDY_RUN_ROOT}/${model}/run_${RUN_ID}/reports"
  mkdir -p "${model_report_dir}"
  echo "[ood-summary] auditing model=${model}"
  "${PYTHON_BIN}" "${REPO}/tools/audit_vqascore_sweep_coverage.py" \
    --root "${STUDY_RUN_ROOT}" --models "${model}" --methods ${methods} \
    --expected-prompts "${EXPECTED_PROMPTS}" \
    --eval-backends ${OOD_EVAL_BACKENDS} --run-id "${RUN_ID}" \
    --out-csv "${model_report_dir}/vqascore_coverage.csv" --no-strict
}

audit_model flux_schnell "${FLUX_METHODS}"
audit_model senseflow_large "${SENSE_METHODS}"
audit_model sd35_base "${SD35_BASE_METHODS}"

missing_count="$(${PYTHON_BIN} - "${STUDY_RUN_ROOT}" "${RUN_ID}" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_id = sys.argv[2]
missing = 0
for model in ("flux_schnell", "senseflow_large", "sd35_base"):
    path = root / model / f"run_{run_id}" / "reports" / "vqascore_coverage.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        missing += sum(row.get("status") != "OK" for row in csv.DictReader(handle))
print(missing)
PY
)"
echo "[ood-summary] incomplete model/method cells=${missing_count}"
if (( missing_count > 0 )) && [[ "${REQUIRE_COMPLETE}" == "1" ]]; then
  echo "Error: coverage is incomplete; inspect per-model vqascore_coverage.csv files." >&2
  exit 1
fi

summarize_model() {
  local model="$1"
  local methods="$2"
  local model_report_dir="${STUDY_RUN_ROOT}/${model}/run_${RUN_ID}/reports"
  echo "[ood-summary] summarizing model=${model}"
  "${PYTHON_BIN}" "${REPO}/tools/merge_posthoc_reward_evals.py" \
    --root "${STUDY_RUN_ROOT}" --include-models "${model}" \
    --include-methods ${methods} --run-id "${RUN_ID}" \
    --backends ${OOD_EVAL_BACKENDS} \
    --summary-csv "${model_report_dir}/vqa_ood_summary_partial.csv" \
    --expected-count "${EXPECTED_PROMPTS}" --no-strict
}

summarize_model flux_schnell "${FLUX_METHODS}"
summarize_model senseflow_large "${SENSE_METHODS}"
summarize_model sd35_base "${SD35_BASE_METHODS}"

STUDY_RUN_ROOT="${STUDY_RUN_ROOT}" RUN_ID="${RUN_ID}" \
COMBINED_CSV="${COMBINED_CSV}" COMBINED_JSON="${COMBINED_JSON}" \
COMBINED_MD="${COMBINED_MD}" MISSING_COUNT="${missing_count}" \
OOD_EVAL_BACKENDS="${OOD_EVAL_BACKENDS}" "${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["STUDY_RUN_ROOT"])
run_id = os.environ["RUN_ID"]
backends = os.environ["OOD_EVAL_BACKENDS"].split()
models = ("flux_schnell", "senseflow_large", "sd35_base")
rows = []
fieldnames = None
sources = []
for model in models:
    source = root / model / f"run_{run_id}" / "reports" / "vqa_ood_summary_partial.csv"
    sources.append(str(source))
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = fieldnames or reader.fieldnames
        rows.extend(reader)

csv_path = Path(os.environ["COMBINED_CSV"])
csv_path.parent.mkdir(parents=True, exist_ok=True)
temporary = csv_path.with_name(csv_path.name + ".tmp")
with temporary.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
temporary.replace(csv_path)

json_path = Path(os.environ["COMBINED_JSON"])
json_path.write_text(json.dumps({
    "run_id": run_id,
    "backends": backends,
    "row_count": len(rows),
    "complete_row_count": sum(str(row.get("ood_complete", "")).lower() == "true" for row in rows),
    "incomplete_coverage_cells": int(os.environ["MISSING_COUNT"]),
    "exclusions": [{
        "model_id": "sd35_base",
        "method": "bon_mcts",
        "reason": "SD3.5-Base ActDiff was explicitly excluded from this OOD summary.",
    }],
    "sources": sources,
    "rows": rows,
}, indent=2) + "\n", encoding="utf-8")

def number(row, name):
    value = row.get(name, "")
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"

headers = ["Model", "Method", "N", *backends, "Complete"]
lines = [
    "| " + " | ".join(headers) + " |",
    "|" + "|".join(["---", "---", "---:", *(["---:"] * len(backends)), "---:"]) + "|",
]
for row in rows:
    cells = [
        row.get("model_name", row.get("model_id", "")),
        row.get("method_label", row.get("method", "")),
        str(row.get("prompt_count", "")),
        *[number(row, f"eval_{backend}_mean") for backend in backends],
        "yes" if str(row.get("ood_complete", "")).lower() == "true" else "no",
    ]
    lines.append("| " + " | ".join(cells) + " |")
Path(os.environ["COMBINED_MD"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[ood-summary] combined rows={len(rows)} csv={csv_path}")
print(f"[ood-summary] markdown={os.environ['COMBINED_MD']}")
PY

echo "[ood-summary] audit first, summary second: complete"
