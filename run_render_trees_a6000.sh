#!/usr/bin/env bash
# A6000-local version of the cluster's amlt/render_trees.yaml.
# Three-stage visualization from EXISTING bon_mcts run outputs:
#
#   Stage 1 (CPU-only): decision-tree PNGs per prompt per backend
#                       -> <OUT_ROOT>/<backend>/actdiff_*_p<N>_bon_mcts.png
#   Stage 2 (GPU):      replay winning trajectory, save per-step x_0 images
#                       -> <OUT_ROOT>/<backend>_step_images/prompt_NNNN/step_K_cfgX.XX.png
#   Stage 3 (CPU-only): per-prompt text logs of MCTS action sequences
#                       -> <OUT_ROOT>/<backend>_logs/prompt_NNNN.txt
#
# A6000 is 48GB; SD3.5L + T5XXL fits comfortably; FLUX-schnell barely fits
# at fp16 with text encoder on CPU after encoding.  We set
# OFFLOAD_TEXT_ENCODER_AFTER_ENCODE=1 by default for safety.
#
# Usage:
#   bash run_render_trees_a6000.sh
# or with overrides:
#   FLUX_ROOT=/data/ygu/my-flux-run SID_ROOT=/data/ygu/my-sid-run \
#     bash run_render_trees_a6000.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Heartbeat (keeps the shell process visible if you tail via ssh) ────────
if [[ -f "${SCRIPT_DIR}/_heartbeat.sh" ]]; then
    source "${SCRIPT_DIR}/_heartbeat.sh"
    start_heartbeat "render-trees-a6000"
fi
export PYTHONUNBUFFERED=1

# ── A6000-friendly defaults ────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

# Memory knobs (matter for Stage 2 only).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OFFLOAD_TEXT_ENCODER_AFTER_ENCODE="${OFFLOAD_TEXT_ENCODER_AFTER_ENCODE:-1}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"   # less T5 activation memory on A6000
export GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1}"

# Worker count for Stage 1 (parallel tree rendering).  Adjust to your CPU.
export TREE_WORKERS="${TREE_WORKERS:-4}"

# ── Per-backend run roots (override via env) ───────────────────────────────
# These should point at directories containing <method>/run_*/rank_*.jsonl.
FLUX_ROOT="${FLUX_ROOT:-/data/ygu/runs/flux-newcfg/composite/flux_schnell/seed42}"
SID_ROOT="${SID_ROOT:-/data/ygu/runs/step-reward-test}"
SENSEFLOW_ROOT="${SENSEFLOW_ROOT:-/data/ygu/runs/senseflow_large}"
SD35_BASE_ROOT="${SD35_BASE_ROOT:-/data/ygu/runs/sd35_base}"

OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/figures/tree_archive_$(date +%Y%m%d_%H%M%S)}"
PROMPT_RANGE="${PROMPT_RANGE:-0:10}"     # A6000 is slower than A100 cluster; default to 10 prompts
RENDER_STEP_IMAGES="${RENDER_STEP_IMAGES:-1}"
RENDER_TEXT_LOGS="${RENDER_TEXT_LOGS:-1}"
mkdir -p "${OUT_ROOT}"

echo "================================================================"
echo "render-trees-a6000"
echo "  PYTHON_BIN          = ${PYTHON_BIN}"
echo "  CUDA_VISIBLE_DEVICES= ${CUDA_VISIBLE_DEVICES}"
echo "  OUT_ROOT            = ${OUT_ROOT}"
echo "  PROMPT_RANGE        = ${PROMPT_RANGE}"
echo "  TREE_WORKERS        = ${TREE_WORKERS}"
echo "  RENDER_STEP_IMAGES  = ${RENDER_STEP_IMAGES}"
echo "  RENDER_TEXT_LOGS    = ${RENDER_TEXT_LOGS}"
echo "  FLUX_ROOT           = ${FLUX_ROOT}"
echo "  SID_ROOT            = ${SID_ROOT}"
echo "  SENSEFLOW_ROOT      = ${SENSEFLOW_ROOT}"
echo "  SD35_BASE_ROOT      = ${SD35_BASE_ROOT}"
echo "================================================================"

# ── Stage 1: decision-tree PNGs (CPU-only, parallel) ──────────────────────
_render_trees() {
    local label="$1" path="$2" title_label="$3"
    if [[ -z "${path}" || ! -d "${path}" ]]; then
        echo "[trees] SKIP ${label} -- not found: ${path}"
        return 0
    fi
    echo "[trees] STAGE 1: rendering ${label} trees from ${path}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/render_trees_batch.py" \
        --run_root "${path}" \
        --method bon_mcts \
        --prompt_range "${PROMPT_RANGE}" \
        --out_dir "${OUT_ROOT}/${label}" \
        --title_prefix "ActDiff (${title_label})" \
        --workers "${TREE_WORKERS}" || true
}
_render_trees flux_schnell    "${FLUX_ROOT}"      "FLUX-schnell"
_render_trees sid             "${SID_ROOT}"       "SiD-DiT-SD3.5L"
_render_trees senseflow_large "${SENSEFLOW_ROOT}" "SenseFlow-large"
_render_trees sd35_base       "${SD35_BASE_ROOT}" "SD3.5-base 28-step"
echo "[trees] STAGE 1 done."

# ── Stage 2: per-step x_0 image replay (GPU) ──────────────────────────────
if [[ "${RENDER_STEP_IMAGES}" == "1" ]]; then
    _replay() {
        local label="$1" path="$2" backend="$3"
        if [[ -z "${path}" || ! -d "${path}" ]]; then
            echo "[trees] SKIP step-images for ${label}"
            return 0
        fi
        local out="${OUT_ROOT}/${label}_step_images"
        echo "[trees] STAGE 2: replaying ${label} -> ${out}"
        "${PYTHON_BIN}" "${SCRIPT_DIR}/replay_winner_step_images.py" \
            --run_root "${path}" \
            --method bon_mcts \
            --backend "${backend}" \
            --prompt_range "${PROMPT_RANGE}" \
            --out_dir "${out}" \
            --height 1024 --width 1024 || true
    }
    _replay flux_schnell    "${FLUX_ROOT}"      flux_schnell
    _replay sid             "${SID_ROOT}"       sid
    _replay senseflow_large "${SENSEFLOW_ROOT}" senseflow_large
    _replay sd35_base       "${SD35_BASE_ROOT}" sd35_base
    echo "[trees] STAGE 2 done."
else
    echo "[trees] STAGE 2 skipped (RENDER_STEP_IMAGES=0)"
fi

# ── Stage 3: text logs ────────────────────────────────────────────────────
if [[ "${RENDER_TEXT_LOGS}" == "1" ]]; then
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
    _logs flux_schnell    "${FLUX_ROOT}"
    _logs sid             "${SID_ROOT}"
    _logs senseflow_large "${SENSEFLOW_ROOT}"
    _logs sd35_base       "${SD35_BASE_ROOT}"
    echo "[trees] STAGE 3 done."
else
    echo "[trees] STAGE 3 skipped (RENDER_TEXT_LOGS=0)"
fi

echo
echo "================================================================"
echo "ALL DONE."
echo "  Decision trees:  ${OUT_ROOT}/<backend>/*.png"
echo "  Step images:     ${OUT_ROOT}/<backend>_step_images/prompt_NNNN/*.png"
echo "  Text logs:       ${OUT_ROOT}/<backend>_logs/*.txt"
echo "================================================================"
ls -la "${OUT_ROOT}" 2>/dev/null
