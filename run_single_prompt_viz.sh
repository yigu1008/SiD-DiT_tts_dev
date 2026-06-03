#!/usr/bin/env bash
# Single-prompt focused visualization: run bon_mcts on ONE prompt and produce
# the complete viz set:
#   1. Decision tree PNG
#   2. Per-step x_0 trajectory (inline-saved, deterministic noise cache)
#   3. Action log text file (full step-by-step CFG/variant decisions)
#   4. Rank JSONL with diagnostics (for any further analysis)
#   5. Horizontal trajectory strip (composed at the end)
#
# Usage:
#   PROMPT="a detailed oil painting of ..." bash run_single_prompt_viz.sh
#   PROMPT_FILE=/path/to/single_prompt.txt bash run_single_prompt_viz.sh
#
# Or pass the prompt as the first positional argument:
#   bash run_single_prompt_viz.sh "a detailed oil painting of ..."
#
# Overrides:
#   BACKEND=sid|senseflow_large|sd35_base   (default: sid)
#   N_SIMS=64                                (default: 30 -- bump for richer tree)
#   SEED=42                                  (default: 42)
#   RUN_ROOT=/data/ygu/runs/focus_<ts>       (default auto-generated)

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Configuration (baked-in defaults; all overridable via env) ───────────
# Set BEFORE prompt-file assembly because we need RUN_ROOT to know where to
# write the baked-prompt scratch file.
BACKEND="${BACKEND:-sid}"
N_SIMS="${N_SIMS:-64}"                                    # rich tree for figures
SEED="${SEED:-42}"
RUN_ROOT="${RUN_ROOT:-/data/ygu/runs/raccoon_full_trace_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_ROOT}"

# Prompt rewriting + multi-variant exploration ON by default for the
# illustration run.  Override with USE_QWEN=0 N_VARIANTS=1 for CFG-only.
USE_QWEN="${USE_QWEN:-1}"
N_VARIANTS="${N_VARIANTS:-3}"

# Save EVERY MCTS-explored trajectory's final image (sortable by score).
# Toggle off for slim runs by exporting SAVE_ALL_ATTEMPTS=0 before launch.
export SAVE_ALL_ATTEMPTS="${SAVE_ALL_ATTEMPTS:-1}"

# ── Default prompt baked into the script ─────────────────────────────────
# This is the canonical "raccoon oil-painting" illustration prompt.  Caller
# can override via PROMPT="..." env, PROMPT_FILE=... env, or first positional
# arg.  Baking it here prevents shell line-wrapping from truncating the
# long compositional sentence when pasted into the terminal.
DEFAULT_PROMPT='a detailed oil painting that captures the essence of an elderly raccoon adorned with a distinguished black top hat. The raccoon'\''s fur is depicted with textured, swirling strokes reminiscent of Van Gogh'\''s signature style, and it clutches a bright red apple in its paws. The background swirls with vibrant colors, giving the impression of movement around the still figure of the raccoon.'

# ── Prompt input (priority: 1st arg > PROMPT env > PROMPT_FILE env > default) ─
if [[ -n "${1:-}" ]]; then
    PROMPT="$1"
fi
PROMPT="${PROMPT:-}"
PROMPT_FILE="${PROMPT_FILE:-}"

if [[ -z "${PROMPT_FILE}" ]]; then
    # If caller didn't pass PROMPT="..." or 1st arg, fall back to baked default.
    if [[ -z "${PROMPT}" ]]; then
        PROMPT="${DEFAULT_PROMPT}"
        echo "[focus] using baked-in default prompt (raccoon oil-painting)"
    fi
    PROMPT_DIR="${RUN_ROOT}/_baked_prompt"
    mkdir -p "${PROMPT_DIR}"
    PROMPT_FILE="${PROMPT_DIR}/prompt.txt"
    printf '%s\n' "${PROMPT}" > "${PROMPT_FILE}"
fi

if [[ ! -s "${PROMPT_FILE}" ]]; then
    echo "[FATAL] PROMPT_FILE empty or missing: ${PROMPT_FILE}" >&2
    exit 1
fi
PROMPT_TEXT="$(head -n1 "${PROMPT_FILE}")"

echo "================================================================"
echo "FOCUSED single-prompt viz"
echo "  BACKEND     = ${BACKEND}"
echo "  N_SIMS      = ${N_SIMS}"
echo "  SEED        = ${SEED}"
echo "  N_VARIANTS  = ${N_VARIANTS}"
echo "  USE_QWEN    = ${USE_QWEN}"
echo "  PROMPT_FILE = ${PROMPT_FILE}"
echo "  RUN_ROOT    = ${RUN_ROOT}"
echo "  PROMPT      = ${PROMPT_TEXT:0:120}..."
echo "================================================================"

# Hand the prompt + N=1 to the existing A6000 wrapper.  It already handles
# reward server, inline step images, decision tree render, text logs.
PROMPT_FILE="${PROMPT_FILE}" \
BACKEND="${BACKEND}" \
N_PROMPTS=1 \
N_SIMS="${N_SIMS}" \
SEED="${SEED}" \
USE_QWEN="${USE_QWEN}" \
N_VARIANTS="${N_VARIANTS}" \
RUN_ROOT="${RUN_ROOT}" \
  bash "${SCRIPT_DIR}/run_actdiff_render_a6000.sh" 2>&1 | tee "${RUN_ROOT}/_run.log"

echo
echo "[focus] composing horizontal trajectory strip"
python "${SCRIPT_DIR}/compose_trajectory_strips.py" \
    --in_dir "${RUN_ROOT}/step_images_inline" \
    --out_dir "${RUN_ROOT}/trajectory_strips" \
    --prompts_file "${PROMPT_FILE}" \
    --panel_size 384 --build_grid || \
  echo "[focus] WARN: strip composition failed"

echo
echo "================================================================"
echo "FOCUSED RUN DONE."
echo
echo "Key artifacts (relative to ${RUN_ROOT}):"
echo "  - prompts/backend_${BACKEND}.txt        the prompt actually used"
echo "  - run_*/bon_mcts/images/*.png           final MCTS-chosen image"
echo "  - run_*/bon_mcts/logs/rank_*.jsonl      raw MCTS rows + diagnostics"
echo "  - step_images_inline/prompt_0000/       per-step x_0 (file names encode cfg)"
echo "  - ${BACKEND}/actdiff_*_p0000_*.png      decision tree"
echo "  - ${BACKEND}_logs/prompt_0000.txt       text trace of chosen actions"
echo "  - trajectory_strips/prompt_0000.png     horizontal film strip"
echo "================================================================"

# Quick listing for the user
echo
ls -la "${RUN_ROOT}/run_"*/bon_mcts/logs/ 2>/dev/null | head
ls -la "${RUN_ROOT}/${BACKEND}/" 2>/dev/null | head
ls -la "${RUN_ROOT}/step_images_inline/prompt_0000/" 2>/dev/null | head
ls -la "${RUN_ROOT}/trajectory_strips/" 2>/dev/null | head
