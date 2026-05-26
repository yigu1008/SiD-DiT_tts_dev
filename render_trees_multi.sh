#!/usr/bin/env bash
# Loop renderer: for each backend whose RUN_ROOT env var exists on disk,
# call render_trees_batch.py.  Skip silently if the path is missing.
#
# Caller env:
#   OUT_ROOT             - parent dir for per-backend subfolders
#   FLUX_ROOT            - FLUX-schnell run root (may be missing)
#   SID_ROOT             - SiD run root
#   SENSEFLOW_ROOT       - SenseFlow run root
#   SD35_BASE_ROOT       - SD3.5-base run root
#   PROMPT_RANGE         - e.g. "0:20"
#   PYTHON_BIN           - python interpreter (default: python)

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROMPT_RANGE="${PROMPT_RANGE:-0:20}"
: "${OUT_ROOT:?OUT_ROOT must be set}"
mkdir -p "${OUT_ROOT}"

# Keep AMLT/Singularity from killing the job for inactivity (the render
# loop can have multi-minute silences between prints when scanning large
# rank_*.jsonl files).
if [[ -f "${SCRIPT_DIR}/_heartbeat.sh" ]]; then
    source "${SCRIPT_DIR}/_heartbeat.sh"
    start_heartbeat "render-trees-multi"
fi
# Force unbuffered Python so each prompt's progress line flushes immediately.
export PYTHONUNBUFFERED=1

_render() {
    local label="$1" path="$2" title_label="$3"
    if [[ -z "${path}" || ! -d "${path}" ]]; then
        echo "[trees] SKIP ${label} -- not found: ${path}"
        return 0
    fi
    echo "[trees] rendering ${label} from ${path}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/render_trees_batch.py" \
        --run_root "${path}" \
        --method bon_mcts \
        --prompt_range "${PROMPT_RANGE}" \
        --out_dir "${OUT_ROOT}/${label}" \
        --title_prefix "ActDiff (${title_label})" || true
}

_render flux_schnell    "${FLUX_ROOT:-}"      "FLUX-schnell"
_render sid             "${SID_ROOT:-}"       "SiD-DiT-SD3.5L"
_render senseflow_large "${SENSEFLOW_ROOT:-}" "SenseFlow-large"
_render sd35_base       "${SD35_BASE_ROOT:-}" "SD3.5-base 28-step"

echo "[trees] all sources done. Browse: ${OUT_ROOT}"
find "${OUT_ROOT}" -name '*.png' | head -20 || true
