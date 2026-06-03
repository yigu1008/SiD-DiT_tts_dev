#!/usr/bin/env bash
# Single-file, self-contained illustration run for ONE prompt.
# Combines: prompt baking, static rewrites (no Qwen on GPU), reward server,
# bon_mcts via the suite, inline x_0 step images, full deliberation trace
# (every MCTS-explored trajectory), decision tree render, per-prompt text log,
# horizontal trajectory strip.
#
# Just run it -- no env vars required:
#   bash run_single_prompt_viz.sh
#
# Common overrides:
#   PROMPT="..."          custom prompt (else uses baked raccoon)
#   BACKEND=sid|senseflow_large|sd35_base
#   N_SIMS=64             MCTS sim budget
#   SEED=42
#   RUN_ROOT=<path>       defaults to /data/ygu/runs/raccoon_full_trace_<ts>
#   USE_QWEN=1            run Qwen at runtime (default 0: hand-crafted rewrites)
#   N_VARIANTS=1          disable rewriting (CFG-only search)

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh" 2>/dev/null || true
type start_heartbeat >/dev/null 2>&1 && start_heartbeat "single-prompt-viz"
export PYTHONUNBUFFERED=1

# ── Configuration ─────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
BACKEND="${BACKEND:-sid}"
N_SIMS="${N_SIMS:-64}"
SEED="${SEED:-42}"
SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"

RUN_ROOT="${RUN_ROOT:-/data/ygu/runs/raccoon_full_trace_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${RUN_ROOT}}"
mkdir -p "${RUN_ROOT}" "${OUT_ROOT}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5318}"
REWARD_SERVER_URL="${REWARD_SERVER_URL:-http://localhost:${REWARD_SERVER_PORT}}"

# A6000 memory knobs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OFFLOAD_TEXT_ENCODER_AFTER_ENCODE="${OFFLOAD_TEXT_ENCODER_AFTER_ENCODE:-1}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"

# Rewriting: REAL Qwen rewrites, run as a STANDALONE step BEFORE the reward
# server boots, so Qwen has the full GPU.  After this step Qwen exits and
# all CUDA memory is released; the suite is then told USE_QWEN=0 and points
# at the cache.  No two heavy models on the GPU at the same time.
USE_QWEN="${USE_QWEN:-1}"
N_VARIANTS="${N_VARIANTS:-3}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen2.5-3B-Instruct}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
export SAVE_ALL_ATTEMPTS="${SAVE_ALL_ATTEMPTS:-1}"

# ── Default prompt baked into the script ─────────────────────────────────
DEFAULT_PROMPT='a detailed oil painting that captures the essence of an elderly raccoon adorned with a distinguished black top hat. The raccoon'\''s fur is depicted with textured, swirling strokes reminiscent of Van Gogh'\''s signature style, and it clutches a bright red apple in its paws. The background swirls with vibrant colors, giving the impression of movement around the still figure of the raccoon.'

if [[ -n "${1:-}" ]]; then PROMPT="$1"; fi
PROMPT="${PROMPT:-}"
PROMPT_FILE="${PROMPT_FILE:-}"
if [[ -z "${PROMPT_FILE}" ]]; then
    if [[ -z "${PROMPT}" ]]; then
        PROMPT="${DEFAULT_PROMPT}"
        echo "[viz] using baked-in default prompt (raccoon oil-painting)"
    fi
    PROMPT_DIR="${RUN_ROOT}/_baked_prompt"
    mkdir -p "${PROMPT_DIR}"
    PROMPT_FILE="${PROMPT_DIR}/prompt.txt"
    printf '%s\n' "${PROMPT}" > "${PROMPT_FILE}"
fi
if [[ ! -s "${PROMPT_FILE}" ]]; then
    echo "[FATAL] PROMPT_FILE empty or missing: ${PROMPT_FILE}" >&2; exit 1
fi
PROMPT_TEXT="$(head -n1 "${PROMPT_FILE}")"

# ── Rewrites cache ───────────────────────────────────────────────────────
# Two paths:
#   (A) USE_QWEN=1 (default): run Qwen STANDALONE here, before anything else
#       touches the GPU.  Qwen exits cleanly; CUDA memory fully released.
#   (B) USE_QWEN=0: fall back to a tiny hand-crafted JSON (raccoon-aware).
REWRITES_FILE="${REWRITES_FILE:-${RUN_ROOT}/rewrites.json}"

if [[ ! -s "${REWRITES_FILE}" && "${N_VARIANTS}" -gt 1 ]]; then
    mkdir -p "$(dirname "${REWRITES_FILE}")"
    if [[ "${USE_QWEN}" == "1" ]]; then
        echo "[rewrites] running Qwen STANDALONE (id=${QWEN_ID}) on GPU ${CUDA_DEVICE}"
        # n_variants is the TOTAL including canonical; Qwen needs to emit n-1 paraphrases.
        QWEN_N=$(( N_VARIANTS - 1 ))
        env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u LOCAL_WORLD_SIZE -u NODE_RANK -u MASTER_ADDR -u MASTER_PORT \
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
        "${PYTHON_BIN}" -u "${SCRIPT_DIR}/precompute_sd35_rewrites.py" \
            --prompt_file "${PROMPT_FILE}" \
            --rewrites_file "${REWRITES_FILE}" \
            --start_index 0 --end_index 1 \
            --n_variants "${QWEN_N}" \
            --qwen_id "${QWEN_ID}" --qwen_dtype "${QWEN_DTYPE}" \
            --device cuda:0 --batch_size 1 \
            --save_every_batches 1 \
            --max_new_tokens 256 --temperature 0.7 --top_p 0.9 \
            --no-clear_cache_each_batch \
            || echo "[rewrites] WARN Qwen precompute failed; will fall back to hand-crafted"
        # Make sure CUDA is fully released before reward server / SD3.5 loads.
        sleep 5
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo "[rewrites] GPU state after Qwen exit:"
            nvidia-smi --query-gpu=memory.used,memory.free --format=csv | head -2
        fi
    fi
    # Hand-crafted fallback (also covers USE_QWEN=0 case).
    if [[ ! -s "${REWRITES_FILE}" ]]; then
        python3 - "${PROMPT_TEXT}" "${REWRITES_FILE}" <<'PY'
import json, sys
prompt, out = sys.argv[1], sys.argv[2]
def derive(p):
    if "raccoon" in p.lower():
        return [
            "An expressive impressionist oil painting of a dignified old raccoon wearing a tall black top hat. Its bushy fur is rendered in thick, swirling Van-Gogh-style brushstrokes, and a vivid red apple is held tightly in its tiny paws. Around the raccoon, the canvas swirls with luminous colors that radiate motion while the animal itself remains perfectly still.",
            "A masterful Post-Impressionist portrait painted in oil: an elderly raccoon in a formal black top hat clasps a glossy crimson apple in its small hands. Every strand of its silvery, textured fur is laid down in churning Van Gogh brushwork, and the swirling, vividly hued background twists around the calm, stationary figure of the raccoon.",
        ]
    return [f"In a richly detailed scene: {p}",
            f"A carefully composed image where {p[0].lower()}{p[1:]}"]
with open(out, "w", encoding="utf-8") as f:
    json.dump({prompt: derive(prompt)}, f, indent=2, ensure_ascii=False)
print(f"[rewrites] (fallback) wrote {out}  variants_total={1+len(derive(prompt))}", flush=True)
PY
    fi
fi
# Inside the suite we DON'T want it to re-run Qwen -- we already have the cache.
export REWRITES_FILE
# These force the suite's runtime Qwen path off; suite will just read REWRITES_FILE.
USE_QWEN=0
export PRECOMPUTE_REWRITES=0

echo "================================================================"
echo "FOCUSED single-prompt viz (all-in-one)"
echo "  BACKEND       = ${BACKEND}"
echo "  N_SIMS        = ${N_SIMS}"
echo "  SEED          = ${SEED}"
echo "  N_VARIANTS    = ${N_VARIANTS}     USE_QWEN=${USE_QWEN}"
echo "  REWRITES_FILE = ${REWRITES_FILE}"
echo "  PROMPT_FILE   = ${PROMPT_FILE}"
echo "  RUN_ROOT      = ${RUN_ROOT}"
echo "  PROMPT        = ${PROMPT_TEXT:0:120}..."
echo "================================================================"

# ── Reward server ────────────────────────────────────────────────────────
SERVER_LOG="${RUN_ROOT}/reward_server.log"
SERVER_PID=""
if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then
    echo "[viz] reusing reward server at ${REWARD_SERVER_URL}"
else
    echo "[viz] booting ImageReward server on GPU ${CUDA_DEVICE}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
      "${PYTHON_BIN}" "${SCRIPT_DIR}/reward_server.py" \
        --port "${REWARD_SERVER_PORT}" --device cuda:0 \
        --backends imagereward --image_reward_model ImageReward-v1.0 \
        > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    trap 'kill "${SERVER_PID}" >/dev/null 2>&1 || true' EXIT
    for i in $(seq 1 60); do
        if curl -s --max-time 3 "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then break; fi
        kill -0 "${SERVER_PID}" 2>/dev/null || { echo "[FATAL] reward server died"; tail -n 50 "${SERVER_LOG}"; exit 1; }
        sleep 3
    done
fi
export REWARD_SERVER_URL

# ── Backend-specific defaults ────────────────────────────────────────────
case "${BACKEND}" in
    sid|senseflow_large)
        export SD35_BACKEND="${BACKEND}"; unset FLUX_BACKEND || true
        export STEPS=4; export BASELINE_CFG=1.0
        export CFG_SCALES="1.0 1.25 1.5 1.75 2.0 2.25 2.5"
        : "${MCTS_KEY_STEP_COUNT:=4}"
        SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
        ;;
    sd35_base)
        export SD35_BACKEND=sd35_base; unset FLUX_BACKEND || true
        export STEPS=28; export BASELINE_CFG=4.5
        export CFG_SCALES="3.5 4.0 4.5 5.0 5.5 6.0 7.0"
        : "${MCTS_KEY_STEP_COUNT:=8}"
        SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
        ;;
    *) echo "[FATAL] unsupported BACKEND=${BACKEND}"; exit 1 ;;
esac

# ── bon_mcts env ─────────────────────────────────────────────────────────
export METHODS="${METHODS:-bon_mcts}"
export PROMPT_FILE
export START_INDEX=0
export END_INDEX=1
export SEEDS="${SEED}"
export N_SIMS
export BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS:-8}"
export BON_MCTS_TOPK="${BON_MCTS_TOPK:-2}"
export BON_MCTS_MIN_SIMS="${BON_MCTS_MIN_SIMS:-8}"
export BON_MCTS_SIM_ALLOC=split
export BON_MCTS_REFINE_METHOD=ours_tree
export LOOKAHEAD_METHOD_MODE=rollout_tree_prior_adaptive_cfg
export N_VARIANTS USE_QWEN
export CORRECTION_STRENGTHS="0.0"
export UCB_C=1.0
export SAVE_BEST_IMAGES=1 SAVE_IMAGES=1
export EVAL_BACKENDS="imagereward"
export REWARD_BACKEND="${SEARCH_REWARD}"
export REWARD_TYPE="${SEARCH_REWARD}"
export REWARD_BACKENDS="${SEARCH_REWARD}"
export EVAL_BEST_IMAGES=1 EVAL_ALLOW_MISSING_BACKENDS=1 EVAL_REWARD_DEVICE=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export OUT_ROOT="${RUN_ROOT}"
export NUM_GPUS=1

# Inline saves: chosen-trajectory x_0 (per-step) + every-rollout trace.
export SAVE_BEST_STEP_IMAGES_DIR="${RUN_ROOT}/step_images_inline"
mkdir -p "${SAVE_BEST_STEP_IMAGES_DIR}"
if [[ "${SAVE_ALL_ATTEMPTS}" == "1" ]]; then
    export SAVE_ALL_ATTEMPTS_DIR="${RUN_ROOT}/all_attempts"
    mkdir -p "${SAVE_ALL_ATTEMPTS_DIR}"
    echo "[viz] full deliberation trace -> ${SAVE_ALL_ATTEMPTS_DIR}"
fi

# ── Stage A: bon_mcts ────────────────────────────────────────────────────
echo
echo "[viz] STAGE A: bon_mcts (backend=${BACKEND}, 1 prompt, N_SIMS=${N_SIMS})"
echo "[viz]   chosen trajectory x_0 -> ${SAVE_BEST_STEP_IMAGES_DIR}"
bash "${SUITE}" 2>&1 | tee "${RUN_ROOT}/_run.log"
SUITE_EC=${PIPESTATUS[0]}
[[ "${SUITE_EC}" -ne 0 ]] && echo "[viz] WARN suite exited ${SUITE_EC}; continuing to viz"

# ── Stage B: visualization ───────────────────────────────────────────────
echo
echo "[viz] STAGE B: decision tree + text log + strip"
export PROMPT_RANGE="0:1"

# Tree
"${PYTHON_BIN}" "${SCRIPT_DIR}/render_trees_batch.py" \
    --run_root "${RUN_ROOT}" --method bon_mcts \
    --prompt_range "${PROMPT_RANGE}" \
    --out_dir "${OUT_ROOT}/${BACKEND}" \
    --title_prefix "ActDiff (${BACKEND})" \
    --workers 1 || true

# Text log
"${PYTHON_BIN}" "${SCRIPT_DIR}/dump_winner_log.py" \
    --run_root "${RUN_ROOT}" --method bon_mcts \
    --prompt_range "${PROMPT_RANGE}" \
    --out_dir "${OUT_ROOT}/${BACKEND}_logs" \
    --combined "${OUT_ROOT}/${BACKEND}_logs/_all.txt" || true

# Horizontal trajectory strip
"${PYTHON_BIN}" "${SCRIPT_DIR}/compose_trajectory_strips.py" \
    --in_dir  "${SAVE_BEST_STEP_IMAGES_DIR}" \
    --out_dir "${OUT_ROOT}/trajectory_strips" \
    --prompts_file "${PROMPT_FILE}" \
    --panel_size 384 --build_grid || \
  echo "[viz] WARN strip composition failed"

# ── Summary ──────────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "DONE.  Artifacts under ${RUN_ROOT}/"
echo "  - run_*/bon_mcts/images/*.png         final MCTS-chosen image"
echo "  - run_*/bon_mcts/logs/rank_*.jsonl    raw MCTS rows + diagnostics"
echo "  - step_images_inline/prompt_0000/     per-step x_0 of winning path"
[[ "${SAVE_ALL_ATTEMPTS}" == "1" ]] && echo "  - all_attempts/prompt_0000/           every rollout's final image"
echo "  - ${BACKEND}/actdiff_*_p0000_*.png    decision tree"
echo "  - ${BACKEND}_logs/prompt_0000.txt     text trace of actions"
echo "  - trajectory_strips/prompt_0000.png   horizontal film strip"
echo "  - rewrites.json                       static prompt variants used"
echo "================================================================"
ls -la "${RUN_ROOT}/run_"*/bon_mcts/logs/ 2>/dev/null | head
ls -la "${RUN_ROOT}/${BACKEND}/" 2>/dev/null | head
ls -la "${RUN_ROOT}/step_images_inline/prompt_0000/" 2>/dev/null | head
ls -la "${RUN_ROOT}/trajectory_strips/" 2>/dev/null | head
