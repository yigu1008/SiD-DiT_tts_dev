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

echo "[trees] decision-tree PNGs done. Browse: ${OUT_ROOT}"
find "${OUT_ROOT}" -name '*.png' | head -20 || true

# ── Stage 2: per-step decoded x_0 images (winning trajectory replay) ──────
# Requires GPU + pipeline.  Gated on RENDER_STEP_IMAGES=1 (default ON).
if [[ "${RENDER_STEP_IMAGES:-1}" == "1" ]]; then
    echo
    echo "[trees] STAGE 2: replaying winning trajectories to dump per-step x_0 images"
    _replay() {
        local label="$1" path="$2" backend="$3"
        if [[ -z "${path}" || ! -d "${path}" ]]; then
            echo "[trees] SKIP step-images for ${label} -- run_root not found"
            return 0
        fi
        local out="${OUT_ROOT}/${label}_step_images"
        echo "[trees] ${label}: replaying -> ${out}"
        "${PYTHON_BIN}" "${SCRIPT_DIR}/replay_winner_step_images.py" \
            --run_root "${path}" \
            --method bon_mcts \
            --backend "${backend}" \
            --prompt_range "${PROMPT_RANGE}" \
            --out_dir "${out}" \
            --height 1024 --width 1024 || true
    }
    _replay flux_schnell    "${FLUX_ROOT:-}"      flux_schnell
    _replay sid             "${SID_ROOT:-}"       sid
    _replay senseflow_large "${SENSEFLOW_ROOT:-}" senseflow_large
    _replay sd35_base       "${SD35_BASE_ROOT:-}" sd35_base
    echo "[trees] step-image replay done."
fi

# ── Stage 3: text logs of winning action sequences ────────────────────────
if [[ "${RENDER_TEXT_LOGS:-1}" == "1" ]]; then
    _logs() {
        local label="$1" path="$2"
        if [[ -z "${path}" || ! -d "${path}" ]]; then return 0; fi
        local out="${OUT_ROOT}/${label}_logs"
        mkdir -p "${out}"
        "${PYTHON_BIN}" "${SCRIPT_DIR}/dump_winner_log.py" \
            --run_root "${path}" --method bon_mcts \
            --prompt_range "${PROMPT_RANGE}" \
            --out_dir "${out}" \
            --combined "${out}/_all.txt" || true
    }
    _logs flux_schnell    "${FLUX_ROOT:-}"
    _logs sid             "${SID_ROOT:-}"
    _logs senseflow_large "${SENSEFLOW_ROOT:-}"
    _logs sd35_base       "${SD35_BASE_ROOT:-}"
    echo "[trees] text logs done."
fi

echo
echo "[trees] ALL DONE. Output layout:"
echo "  ${OUT_ROOT}/<backend>/                  - decision-tree PNGs"
echo "  ${OUT_ROOT}/<backend>_step_images/      - per-step x_0 PNGs (one folder per prompt)"
echo "  ${OUT_ROOT}/<backend>_logs/             - per-prompt action-sequence text logs"
