#!/usr/bin/env bash
set -euo pipefail

# Continue the existing GenAI-200 SD3.5-Base VQAScore sweep with ActDiff only.
# GA is complete by default; REQUIRE_GA_COMPLETE=0 explicitly preserves an
# interrupted GA and continues. DTS, DTS*, and Dynamic CFG are never dispatched.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${REPO}/tools/run_vqa_model_phase.sh"

HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
RUN_ID="${RUN_ID:-genai200_v1}"
EXPECTED_PROMPTS="${EXPECTED_PROMPTS:-200}"
GPUS="${GPUS:-4,5,6,7}"
PYTHON_BIN="${PYTHON_BIN:-/home/ygu/miniconda3/envs/sid_dit/bin/python}"
STUDY_ROOT="${STUDY_ROOT:-${HUMAN_EVAL_ROOT}/vqascore_remaining_models}"
MODEL_ROOT="${STUDY_ROOT}/${RUN_ID}/sd35_base"
RUN_ROOT="${MODEL_ROOT}/run_${RUN_ID}"
GA_AGGREGATE="${RUN_ROOT}/ga/aggregate_ddp.json"
ACTDIFF_AGGREGATE="${RUN_ROOT}/bon_mcts/aggregate_ddp.json"
LOG_DIR="${RUN_ROOT}/reports"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/actdiff_only_launcher.log}"
DRY_RUN="${DRY_RUN:-0}"
REQUIRE_GA_COMPLETE="${REQUIRE_GA_COMPLETE:-1}"

case "${DRY_RUN}" in 0|1) ;;
  *) echo "Error: DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac
case "${REQUIRE_GA_COMPLETE}" in 0|1) ;;
  *) echo "Error: REQUIRE_GA_COMPLETE must be 0 or 1." >&2; exit 2 ;;
esac
IFS=',' read -r -a gpu_array <<< "${GPUS}"
if (( ${#gpu_array[@]} != 4 )); then
  echo "Error: GPUS must contain exactly four GPUs (three generation + one reward)." >&2
  exit 2
fi

echo "[actdiff-only] model=sd35_base run_id=${RUN_ID}"
echo "[actdiff-only] GPUs=${GPUS}; reward GPU=${gpu_array[3]}"
echo "[actdiff-only] methods=bon_mcts"
echo "[actdiff-only] run_root=${RUN_ROOT}"
echo "[actdiff-only] log=${LOG_FILE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
ga_count="$(${PYTHON_BIN} - "${GA_AGGREGATE}" "${RUN_ROOT}/ga/logs" <<'PY'
import glob
import json
import sys
from pathlib import Path

aggregate = Path(sys.argv[1])
if aggregate.is_file():
    print(int(json.loads(aggregate.read_text(encoding="utf-8")).get("num_samples", 0)))
    raise SystemExit(0)

done = set()
for path in glob.glob(str(Path(sys.argv[2]) / "rank_[0-9][0-9][0-9].jsonl")):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("mode") == "ga" and "prompt_index" in row:
                done.add(int(row["prompt_index"]))
print(len(done))
PY
)"
if [[ "${REQUIRE_GA_COMPLETE}" == "1" && "${ga_count}" != "${EXPECTED_PROMPTS}" ]]; then
  echo "Error: GA coverage is ${ga_count}/${EXPECTED_PROMPTS}; refusing to continue." >&2
  echo "Set REQUIRE_GA_COMPLETE=0 to preserve partial GA results and run ActDiff now." >&2
  exit 1
fi
if [[ "${ga_count}" != "${EXPECTED_PROMPTS}" ]]; then
  echo "[actdiff-only] WARN: proceeding with partial GA coverage ${ga_count}/${EXPECTED_PROMPTS}."
  echo "[actdiff-only] Existing GA rank logs are preserved for a later resume."
fi
if [[ -s "${ACTDIFF_AGGREGATE}" ]]; then
  actdiff_count="$(${PYTHON_BIN} - "${ACTDIFF_AGGREGATE}" <<'PY'
import json
import sys
print(int(json.load(open(sys.argv[1], encoding="utf-8")).get("num_samples", 0)))
PY
)"
  if [[ "${actdiff_count}" == "${EXPECTED_PROMPTS}" ]]; then
    echo "[actdiff-only] ActDiff already complete (${actdiff_count}/${EXPECTED_PROMPTS}); nothing to run."
    exit 0
  fi
fi

mkdir -p "${LOG_DIR}"
echo "[actdiff-only] GA coverage=${ga_count}/${EXPECTED_PROMPTS}; starting ActDiff only."
echo "[actdiff-only] Make sure the previous suite and reward server have exited before this launch."

env \
  MODEL=sd35_base \
  PHASE=generate \
  GPUS="${GPUS}" \
  METHODS=bon_mcts \
  HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT}" \
  STUDY_ROOT="${STUDY_ROOT}" \
  RUN_ID="${RUN_ID}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  RESUME_PROMPTS=1 \
  bash "${RUNNER}" 2>&1 | tee -a "${LOG_FILE}"

echo "[actdiff-only] complete: ${ACTDIFF_AGGREGATE}"
