#!/usr/bin/env bash
# Self-contained noise ablation on A6000: bon_mcts WITHOUT prescreen, comparing
#   (A) FIXED (deterministic rollouts -- no per-step noise re-injection)
#   (B) FRESH (re-noise latents per step like run_baseline does)
#
# Each condition runs end-to-end inline (no chained wrappers):
#   1. boot ImageReward server
#   2. precompute Qwen rewrites (skipped unless USE_QWEN=1)
#   3. bon_mcts on N prompts via the suite
#   4. render decision trees + step-image strips + text logs
# Then we cross-compare: per-prompt side-by-side strips, tree pairs, SUMMARY.txt.
#
# Background run (disconnect-safe):
#   nohup bash run_noise_ablation_a6000.sh > /tmp/noise_abl.out 2>&1 &
#   disown
#
# Knobs (all env, all defaulted):
#   N_PROMPTS=4         BACKEND=sid          SEED=42
#   N_SIMS=30           PROMPT_FILE=/data/ygu/dpg_bench_prompts.txt
#   OUT_ROOT=/data/ygu/runs/noise_ablation_<ts>

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh" 2>/dev/null || true
type start_heartbeat >/dev/null 2>&1 && start_heartbeat "noise-ablation"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# ── Baked-in defaults ────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
BACKEND="${BACKEND:-sid}"
N_PROMPTS="${N_PROMPTS:-4}"
N_SIMS="${N_SIMS:-30}"
SEED="${SEED:-42}"
SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"

OUT_ROOT="${OUT_ROOT:-/data/ygu/runs/noise_ablation_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_ROOT}"

# Default prompts source: DPG-Bench (1065 dense compositional).  Override via
# PROMPT_FILE=... env or 1st positional arg.  USE_BAKED_PROMPT=1 focuses on
# the baked raccoon prompt.
DEFAULT_PROMPT_FILE="/data/ygu/dpg_bench_prompts.txt"
if [[ -n "${1:-}" ]]; then
    PROMPT_FILE="$1"
fi
PROMPT_FILE="${PROMPT_FILE:-${DEFAULT_PROMPT_FILE}}"

BAKED_RACCOON='a detailed oil painting that captures the essence of an elderly raccoon adorned with a distinguished black top hat. The raccoon'\''s fur is depicted with textured, swirling strokes reminiscent of Van Gogh'\''s signature style, and it clutches a bright red apple in its paws. The background swirls with vibrant colors, giving the impression of movement around the still figure of the raccoon.'

PROMPT="${PROMPT:-}"
if [[ -n "${PROMPT}" || "${USE_BAKED_PROMPT:-0}" == "1" ]]; then
    if [[ -z "${PROMPT}" ]]; then PROMPT="${BAKED_RACCOON}"; fi
    _BAKED_FILE="${OUT_ROOT}/_baked_prompt.txt"
    printf '%s\n' "${PROMPT}" > "${_BAKED_FILE}"
    PROMPT_FILE="${_BAKED_FILE}"
    N_PROMPTS=1
    echo "[ablation] focused mode -- using baked-in single prompt"
fi

# Memory knobs (A6000 48GB)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OFFLOAD_TEXT_ENCODER_AFTER_ENCODE="${OFFLOAD_TEXT_ENCODER_AFTER_ENCODE:-1}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"

# Rewriting OFF by default for the noise ablation (we want to isolate the
# noise axis; keep N_VARIANTS=1).  Set USE_QWEN=1 N_VARIANTS=3 to include
# rewrites in both conditions for a 2D ablation.
USE_QWEN="${USE_QWEN:-0}"
N_VARIANTS="${N_VARIANTS:-1}"

echo "================================================================"
echo "NOISE ABLATION (bon_mcts, no prescreen, ImageReward)"
echo "  BACKEND       = ${BACKEND}"
echo "  N_PROMPTS     = ${N_PROMPTS}"
echo "  N_SIMS        = ${N_SIMS}"
echo "  SEED          = ${SEED}"
echo "  N_VARIANTS    = ${N_VARIANTS}   USE_QWEN=${USE_QWEN}"
echo "  PROMPT_FILE   = ${PROMPT_FILE}"
echo "  OUT_ROOT      = ${OUT_ROOT}"
echo "  CUDA_DEVICE   = ${CUDA_DEVICE}"
echo "================================================================"

if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "[FATAL] PROMPT_FILE not found: ${PROMPT_FILE}" >&2
    exit 1
fi

# ── Backend-specific defaults (inlined from the old wrapper) ─────────────
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

# ── Per-condition execution (inline) ─────────────────────────────────────
_execute_condition() {
    local label="$1" fresh_rollout="$2"
    # IMPORTANT: save the script-level OUT_ROOT before we export a new value
    # below; otherwise the second call inherits the mutated value and nests
    # ${OUT_ROOT}/fresh inside ${OUT_ROOT}/fixed/.
    local _saved_out_root="${OUT_ROOT}"
    local run_root="${_saved_out_root}/${label}"
    mkdir -p "${run_root}"

    echo
    echo "================================================================"
    echo "[ablation] CONDITION = ${label}   MCTS_FRESH_ROLLOUT_NOISE=${fresh_rollout}"
    echo "  run_root      = ${run_root}"
    echo "  reward        = in-process ImageReward (no server)"
    echo "================================================================"

    # No reward server -- load ImageReward in-process to avoid stale-port /
    # broken-pipe issues that killed previous ablation runs mid-way.
    unset REWARD_SERVER_URL REWARD_SERVER_PORT
    local server_pid=""

    # bon_mcts via the suite
    export METHODS=bon_mcts
    export PROMPT_FILE
    export START_INDEX=0
    export END_INDEX="${N_PROMPTS}"
    export SEEDS="${SEED}"
    export N_SIMS
    export BON_MCTS_N_SEEDS=1     # no prescreen for this ablation
    export BON_MCTS_TOPK=1
    export BON_MCTS_MIN_SIMS="${N_SIMS}"
    export BON_MCTS_SIM_ALLOC=split
    export BON_MCTS_REFINE_METHOD=ours_tree
    export LOOKAHEAD_METHOD_MODE=rollout_tree_prior_adaptive_cfg
    export N_VARIANTS USE_QWEN
    export PRECOMPUTE_REWRITES=0
    export CORRECTION_STRENGTHS="0.0"
    export UCB_C=1.0
    export SAVE_BEST_IMAGES=1 SAVE_IMAGES=1
    export REWARD_BACKEND="${SEARCH_REWARD}"
    export REWARD_TYPE="${SEARCH_REWARD}"
    export REWARD_BACKENDS="${SEARCH_REWARD}"
    export EVAL_BACKENDS="${SEARCH_REWARD}"
    export EVAL_BEST_IMAGES=1 EVAL_ALLOW_MISSING_BACKENDS=1 EVAL_REWARD_DEVICE=cuda
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    export OUT_ROOT="${run_root}"
    export NUM_GPUS=1
    export SAVE_BEST_STEP_IMAGES_DIR="${run_root}/step_images_inline"
    export SAVE_ALL_ATTEMPTS_DIR="${run_root}/all_attempts"
    mkdir -p "${SAVE_BEST_STEP_IMAGES_DIR}" "${SAVE_ALL_ATTEMPTS_DIR}"
    export MCTS_FRESH_ROLLOUT_NOISE="${fresh_rollout}"

    echo "[ablation] STAGE A: running bon_mcts"
    bash "${SUITE}" > "${run_root}/_run.log" 2>&1 || \
      echo "[ablation] WARN suite exited non-zero (see ${run_root}/_run.log)"

    echo "[ablation] STAGE B: viz (trees + step images + logs)"
    local prompt_range="0:${N_PROMPTS}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/render_trees_batch.py" \
        --run_root "${run_root}" --method bon_mcts \
        --prompt_range "${prompt_range}" \
        --out_dir "${run_root}/${BACKEND}" \
        --title_prefix "ActDiff (${BACKEND} ${label})" \
        --workers 2 || true

    "${PYTHON_BIN}" "${SCRIPT_DIR}/dump_winner_log.py" \
        --run_root "${run_root}" --method bon_mcts \
        --prompt_range "${prompt_range}" \
        --out_dir "${run_root}/${BACKEND}_logs" \
        --combined "${run_root}/${BACKEND}_logs/_all.txt" || true

    "${PYTHON_BIN}" "${SCRIPT_DIR}/compose_trajectory_strips.py" \
        --in_dir "${run_root}/step_images_inline" \
        --out_dir "${run_root}/trajectory_strips" \
        --prompts_file "${PROMPT_FILE}" \
        --panel_size 384 --build_grid || \
      echo "[ablation] WARN strip composition failed for ${label}"

    # Restore parent-shell OUT_ROOT so the next condition (and the final
    # SUMMARY block) see the script-level root, not this per-condition one.
    export OUT_ROOT="${_saved_out_root}"
    echo "[ablation] condition ${label} DONE"
}

# Run sequentially.  fresh = MCTS_FRESH_ROLLOUT_NOISE=1.
_execute_condition fixed 0
sleep 30
# Clean up any zombie sampling procs (no reward server to worry about).
pkill -f sd35_ddp_experiment 2>/dev/null || true
pkill -f torchrun 2>/dev/null || true
sleep 10
_execute_condition fresh 1

# ── Side-by-side trajectory comparison ───────────────────────────────────
echo
echo "[ablation] composing fixed-vs-fresh side-by-side strips"
python3 - "${OUT_ROOT}" "${BACKEND}" <<'PY'
import sys, os, re
from pathlib import Path
out_root = Path(sys.argv[1]); backend = sys.argv[2]
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:
    print(f"[ablation] PIL unavailable: {exc}"); sys.exit(0)
fixed_dir = out_root / "fixed" / "trajectory_strips"
fresh_dir = out_root / "fresh" / "trajectory_strips"
cmp_dir = out_root / "comparison_strips"
cmp_dir.mkdir(parents=True, exist_ok=True)
if not (fixed_dir.is_dir() and fresh_dir.is_dir()):
    print(f"[ablation] missing strip dirs"); sys.exit(0)
font = ImageFont.load_default()
for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/Library/Fonts/Arial.ttf"):
    if os.path.isfile(cand):
        try: font = ImageFont.truetype(cand, 18); break
        except Exception: pass
for fp in sorted(fixed_dir.glob("prompt_*.png")):
    fr = fresh_dir / fp.name
    if not fr.exists(): continue
    a = Image.open(fp).convert("RGB"); b = Image.open(fr).convert("RGB")
    W = max(a.width, b.width); header_h = 28
    out = Image.new("RGB", (W, a.height + b.height + 2*header_h + 8), (255,255,255))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, W, header_h], fill=(60,60,60))
    d.text((10, 5), f"FIXED noise  ({fp.stem})", font=font, fill=(255,255,255))
    out.paste(a, ((W-a.width)//2, header_h))
    y2 = a.height + header_h + 4
    d.rectangle([0, y2, W, y2+header_h], fill=(20,90,140))
    d.text((10, y2+5), "FRESH noise", font=font, fill=(255,255,255))
    out.paste(b, ((W-b.width)//2, y2+header_h))
    out.save(cmp_dir / fp.name)
print(f"[ablation] comparison strips -> {cmp_dir}")
PY

# ── Side-by-side tree comparison ─────────────────────────────────────────
echo "[ablation] composing fixed-vs-fresh tree pairs"
python3 - "${OUT_ROOT}" "${BACKEND}" <<'PY'
import sys, os, re
from pathlib import Path
out_root = Path(sys.argv[1]); backend = sys.argv[2]
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    sys.exit(0)
fixed_dir = out_root / "fixed" / backend
fresh_dir = out_root / "fresh" / backend
cmp_dir = out_root / "comparison_trees"
cmp_dir.mkdir(parents=True, exist_ok=True)
if not (fixed_dir.is_dir() and fresh_dir.is_dir()): sys.exit(0)
font = ImageFont.load_default()
for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/Library/Fonts/Arial.ttf"):
    if os.path.isfile(cand):
        try: font = ImageFont.truetype(cand, 18); break
        except Exception: pass
def by_idx(d):
    out = {}
    for fp in d.glob("actdiff_*_p*_bon_mcts.png"):
        m = re.search(r"_p(\d+)_bon_mcts\.png$", fp.name)
        if m: out[int(m.group(1))] = fp
    return out
A = by_idx(fixed_dir); B = by_idx(fresh_dir)
for pi in sorted(set(A) & set(B)):
    a = Image.open(A[pi]).convert("RGB"); b = Image.open(B[pi]).convert("RGB")
    H = max(a.height, b.height); pad = 12; header_h = 28
    out = Image.new("RGB", (a.width + b.width + pad, H + header_h), (255,255,255))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, a.width, header_h], fill=(60,60,60))
    d.text((10, 5), f"FIXED (p={pi})", font=font, fill=(255,255,255))
    d.rectangle([a.width + pad, 0, a.width + pad + b.width, header_h], fill=(20,90,140))
    d.text((a.width + pad + 10, 5), "FRESH", font=font, fill=(255,255,255))
    out.paste(a, (0, header_h)); out.paste(b, (a.width + pad, header_h))
    out.save(cmp_dir / f"tree_p{pi:04d}.png")
print(f"[ablation] tree pairs -> {cmp_dir}")
PY

# ── SUMMARY ──────────────────────────────────────────────────────────────
SUMMARY="${OUT_ROOT}/SUMMARY.txt"
{
    echo "Noise ablation summary  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
    echo "================================================================"
    echo "BACKEND=${BACKEND}  N_PROMPTS=${N_PROMPTS}  N_SIMS=${N_SIMS}  SEED=${SEED}"
    echo "N_VARIANTS=${N_VARIANTS}  USE_QWEN=${USE_QWEN}"
    echo "Reward backend: ImageReward"
    echo
    for c in fixed fresh; do
        echo "--- condition: ${c} ---"
        eval_jsonl="$(ls "${OUT_ROOT}/${c}"/run_*/bon_mcts/logs/rank_*.jsonl 2>/dev/null | head -1)"
        if [[ -z "${eval_jsonl}" ]]; then
            echo "  (no rank file found)"; echo; continue
        fi
        python3 - "${eval_jsonl}" <<'PY'
import json, sys
fp = sys.argv[1]
scores, deltas, nfes = [], [], []
seen_modes, total_rows = set(), 0
for ln in open(fp):
    if not ln.strip(): continue
    try: r = json.loads(ln)
    except Exception: continue
    total_rows += 1
    if r.get("mode"): seen_modes.add(r["mode"])
    if r.get("score") is not None: scores.append(float(r["score"]))
    if r.get("delta_vs_base") is not None: deltas.append(float(r["delta_vs_base"]))
    if r.get("nfe") is not None: nfes.append(int(r["nfe"]))
def stat(xs): return f"n={len(xs)} mean={sum(xs)/len(xs):+.4f} min={min(xs):+.4f} max={max(xs):+.4f}" if xs else "(empty)"
print(f"  rank file: {fp}")
print(f"  rows scanned: {total_rows}   modes seen: {sorted(seen_modes)}")
print(f"  IR score:        {stat(scores)}")
print(f"  IR delta vs base:{stat(deltas)}")
print(f"  NFE per prompt:  {stat(nfes)}")
PY
        echo
    done
    echo "Per-prompt comparisons:"
    echo "  ${OUT_ROOT}/comparison_strips/  ${OUT_ROOT}/comparison_trees/"
} | tee "${SUMMARY}"

echo
echo "================================================================"
echo "DONE.  Summary: ${SUMMARY}"
echo "================================================================"
