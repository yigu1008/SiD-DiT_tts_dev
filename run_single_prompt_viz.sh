#!/usr/bin/env bash
# Single-prompt visualization, fully self-contained.
#
# What it does (in order):
#   1.  Bake the raccoon prompt to a file (overridable).
#   2.  Run Qwen STANDALONE in a clean subprocess to write rewrites.json
#       with strong-paraphrase styles (canonical + 3 rewrites).
#   3.  Verify the rewrites cache has the expected shape.
#   4.  Boot ImageReward server on its own port.
#   5.  Run bon_mcts via the suite with rewriting + per-step CFG + inline
#       step-image dump + all-rollout dump.
#   6.  Kill reward server, render decision tree, dump action log, compose
#       horizontal trajectory strip.
#   7.  Final verification: was MCTS actually exposed to variants > 0, and
#       did it ever pick one?
#
# Just run it:
#   bash run_single_prompt_viz.sh
#
# Override (env vars or 1st positional arg for prompt):
#   PROMPT="..." | PROMPT_FILE=/path | bash run_single_prompt_viz.sh "..."
#   BACKEND=sid|senseflow_large|sd35_base
#   N_SIMS=64   SEED=42   N_VARIANTS=3   USE_QWEN=1
#   RUN_ROOT=<path>

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh" 2>/dev/null || true
type start_heartbeat >/dev/null 2>&1 && start_heartbeat "single-prompt-viz"

# Prevent ~/.local site-packages (often a stale CPU-only torch) from
# shadowing the conda env.  Critical for both Qwen and reward server.
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# ── Configuration (all env-overridable) ──────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
BACKEND="${BACKEND:-sid}"
N_SIMS="${N_SIMS:-64}"
SEED="${SEED:-42}"
SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"
RUN_ROOT="${RUN_ROOT:-/data/ygu/runs/raccoon_full_trace_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_ROOT}"

# Rewriting + variant exploration.
USE_QWEN="${USE_QWEN:-1}"
N_VARIANTS="${N_VARIANTS:-3}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen2.5-3B-Instruct}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
QWEN_TEMP="${QWEN_TEMP:-1.0}"
QWEN_TOP_P="${QWEN_TOP_P:-0.95}"
QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-384}"

# Strong paraphrase instructions for the visualization run.
export REWRITE_STYLES_OVERRIDE="${REWRITE_STYLES_OVERRIDE:-Paraphrase this prompt for an image generator using completely different sentence structure and synonyms while preserving every visual detail.||Rewrite this prompt as if describing the same scene to a different illustrator, freely rearranging clauses and substituting words.||Restate the prompt with rich, vivid wording: invert the order of elements, replace verbs and adjectives, and add atmospheric detail without inventing new objects.}"

# A6000 memory knobs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OFFLOAD_TEXT_ENCODER_AFTER_ENCODE="${OFFLOAD_TEXT_ENCODER_AFTER_ENCODE:-1}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"

# Reward server intentionally disabled -- use in-process ImageReward.
# Keep these unset so the suite knows to load locally.
unset REWARD_SERVER_URL REWARD_SERVER_PORT 2>/dev/null || true

# ── Step 1: prompt input (priority: 1st arg > PROMPT env > PROMPT_FILE > default) ─
DEFAULT_PROMPT='a detailed oil painting that captures the essence of an elderly raccoon adorned with a distinguished black top hat. The raccoon'\''s fur is depicted with textured, swirling strokes reminiscent of Van Gogh'\''s signature style, and it clutches a bright red apple in its paws. The background swirls with vibrant colors, giving the impression of movement around the still figure of the raccoon.'

if [[ -n "${1:-}" ]]; then PROMPT="$1"; fi
PROMPT="${PROMPT:-}"
PROMPT_FILE="${PROMPT_FILE:-}"
if [[ -z "${PROMPT_FILE}" ]]; then
    if [[ -z "${PROMPT}" ]]; then
        PROMPT="${DEFAULT_PROMPT}"
        echo "[viz] using baked-in default prompt (raccoon oil-painting)"
    fi
    mkdir -p "${RUN_ROOT}/_baked_prompt"
    PROMPT_FILE="${RUN_ROOT}/_baked_prompt/prompt.txt"
    printf '%s\n' "${PROMPT}" > "${PROMPT_FILE}"
fi
[[ -s "${PROMPT_FILE}" ]] || { echo "[FATAL] PROMPT_FILE empty: ${PROMPT_FILE}" >&2; exit 1; }
PROMPT_TEXT="$(head -n1 "${PROMPT_FILE}")"

echo "================================================================"
echo "SINGLE-PROMPT VIZ"
echo "  BACKEND     = ${BACKEND}    N_SIMS = ${N_SIMS}    SEED = ${SEED}"
echo "  N_VARIANTS  = ${N_VARIANTS}    USE_QWEN = ${USE_QWEN}    QWEN_ID = ${QWEN_ID}"
echo "  PROMPT_FILE = ${PROMPT_FILE}"
echo "  RUN_ROOT    = ${RUN_ROOT}"
echo "  CUDA_DEVICE = ${CUDA_DEVICE}"
echo "  PROMPT      = ${PROMPT_TEXT:0:120}..."
echo "================================================================"

# ── Step 2: Qwen offline → rewrites.json ─────────────────────────────────
REWRITES_FILE="${REWRITES_FILE:-${RUN_ROOT}/rewrites.json}"
if [[ ! -s "${REWRITES_FILE}" && "${N_VARIANTS}" -gt 0 ]]; then
    if [[ "${USE_QWEN}" == "1" ]]; then
        echo
        echo "[step 2] Qwen STANDALONE (id=${QWEN_ID}) on GPU ${CUDA_DEVICE}"
        env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u LOCAL_WORLD_SIZE -u NODE_RANK -u MASTER_ADDR -u MASTER_PORT \
        PYTHONNOUSERSITE=1 \
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
          "${PYTHON_BIN}" -u "${SCRIPT_DIR}/precompute_sd35_rewrites.py" \
            --prompt_file "${PROMPT_FILE}" \
            --rewrites_file "${REWRITES_FILE}" \
            --start_index 0 --end_index 1 \
            --n_variants "${N_VARIANTS}" \
            --qwen_id "${QWEN_ID}" --qwen_dtype "${QWEN_DTYPE}" \
            --device cuda:0 --batch_size 1 \
            --save_every_batches 1 \
            --max_new_tokens "${QWEN_MAX_NEW_TOKENS}" \
            --temperature "${QWEN_TEMP}" --top_p "${QWEN_TOP_P}" \
            --no-clear_cache_each_batch \
            || echo "[step 2] WARN Qwen precompute failed; will use hand-crafted fallback"
        sleep 5
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo "[step 2] GPU after Qwen exit:"
            nvidia-smi --query-gpu=memory.used,memory.free --format=csv | head -2
        fi
    fi
    # Hand-crafted fallback if Qwen unavailable/failed.
    if [[ ! -s "${REWRITES_FILE}" ]]; then
        echo "[step 2] writing hand-crafted rewrites fallback"
        python3 - "${PROMPT_TEXT}" "${REWRITES_FILE}" "${N_VARIANTS}" <<'PY'
import json, sys
prompt, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
RACCOON_POOL = [
  "An expressive impressionist oil painting of a dignified old raccoon wearing a tall black top hat, holding a vivid red apple. Heavy Van-Gogh brushwork swirls across textured fur while a kaleidoscope of color eddies around the still creature.",
  "A masterful Post-Impressionist portrait of an elderly raccoon in formal black top hat clasping a glossy crimson apple. Every strand of its silvery fur is laid down in churning brushwork, the background a vivid whirl of motion around the calm animal.",
  "Oil on canvas, Van-Gogh inspired: a stately senior raccoon crowned with a polished black top hat grasps a luscious scarlet apple. Rhythmic brushstrokes weave colorful curls of motion that orbit the perfectly still subject.",
  "Ornate Post-Impressionist canvas: aged raccoon in dignified black top hat firmly holding a deep crimson apple. Tufts of rich silver-charcoal fur in expressive strokes; swirling kaleidoscope of color animates the scene around the serene composition.",
]
def variants(p, k):
    if "raccoon" in p.lower():
        pool = RACCOON_POOL
    else:
        pool = [f"In a richly detailed scene: {p}", f"A carefully composed image where {p[0].lower()}{p[1:]}",
                f"An evocative rendering: {p}", f"A masterful depiction. {p}"]
    while len(pool) < k:
        pool.append(p + " (variation " + str(len(pool)+1) + ")")
    return [prompt] + pool[:k]
vs = variants(prompt, n)
json.dump({prompt: vs}, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"[step 2] wrote {out}  entries={len(vs)} (1 canonical + {len(vs)-1} rewrites)", flush=True)
PY
    fi
fi
export REWRITES_FILE

# ── Step 3: verify rewrites cache before we sink hours into sampling ─────
echo
echo "[step 3] verifying rewrites cache"
python3 - "${REWRITES_FILE}" "${PROMPT_TEXT}" "${N_VARIANTS}" <<'PY'
import json, sys
fp, key, n_req = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    d = json.load(open(fp, encoding="utf-8"))
except Exception as exc:
    print(f"[step 3] FATAL: cannot load {fp}: {exc}"); sys.exit(1)
if key not in d:
    print(f"[step 3] FATAL: prompt key MISSING from cache -- exact-match lookup will fail!")
    print(f"        expected: {key[:80]}...")
    print(f"        keys present: {[k[:60] for k in list(d.keys())[:3]]}")
    sys.exit(1)
entries = d[key]
expected = 1 + n_req
print(f"[step 3] cache file: {fp}")
print(f"[step 3] entries: {len(entries)}  (expected {expected} = 1 canonical + {n_req} rewrites)")
dupes = sum(1 for v in entries[1:] if v.strip() == entries[0].strip())
print(f"[step 3] duplicates of canonical: {dupes}")
for i, v in enumerate(entries):
    label = "canonical" if i == 0 else "rewrite  "
    print(f"        v{i} ({label}): {v[:100]}{'...' if len(v)>100 else ''}")
if len(entries) < expected:
    print(f"[step 3] WARN: fewer entries than requested.  MCTS will see {len(entries)} variants.")
print("[step 3] PASS" if len(entries) >= 2 else "[step 3] FAIL -- no rewrites in cache")
PY
[[ $? -eq 0 ]] || exit 1

# ── Step 4: LOCAL reward model (no server, no port reuse, no broken pipes) ─
echo
echo "[step 4] using in-process ImageReward (no server)"
# Critical: clear REWARD_SERVER_URL so the suite loads ImageReward locally.
unset REWARD_SERVER_URL
SERVER_PID=""

# ── Step 5: backend defaults + bon_mcts run ──────────────────────────────
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

export METHODS=bon_mcts
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
# Inside the suite: don't re-run Qwen, we already have the cache.
export USE_QWEN=0
export PRECOMPUTE_REWRITES=0
export N_VARIANTS
export CORRECTION_STRENGTHS="0.0"
export UCB_C=1.0
export SAVE_BEST_IMAGES=1 SAVE_IMAGES=1
export SAVE_VARIANTS=1
export REWARD_BACKEND="${SEARCH_REWARD}"
export REWARD_TYPE="${SEARCH_REWARD}"
export REWARD_BACKENDS="${SEARCH_REWARD}"
export EVAL_BACKENDS="${SEARCH_REWARD}"
export EVAL_BEST_IMAGES=1 EVAL_ALLOW_MISSING_BACKENDS=1 EVAL_REWARD_DEVICE=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export OUT_ROOT="${RUN_ROOT}"
export NUM_GPUS=1
export SAVE_BEST_STEP_IMAGES_DIR="${RUN_ROOT}/step_images_inline"
export SAVE_ALL_ATTEMPTS_DIR="${RUN_ROOT}/all_attempts"
mkdir -p "${SAVE_BEST_STEP_IMAGES_DIR}" "${SAVE_ALL_ATTEMPTS_DIR}"

echo
echo "[step 5] STAGE A: bon_mcts on 1 prompt (N_SIMS=${N_SIMS})"
bash "${SUITE}" 2>&1 | tee "${RUN_ROOT}/_run.log"

# No reward server to kill — running in-process.

# ── Step 6: viz ──────────────────────────────────────────────────────────
echo
echo "[step 6] STAGE B: decision tree + action log + trajectory strip"
"${PYTHON_BIN}" "${SCRIPT_DIR}/render_trees_batch.py" \
    --run_root "${RUN_ROOT}" --method bon_mcts \
    --prompt_range "0:1" \
    --out_dir "${RUN_ROOT}/${BACKEND}" \
    --title_prefix "ActDiff (${BACKEND})" \
    --workers 1 || true

"${PYTHON_BIN}" "${SCRIPT_DIR}/dump_winner_log.py" \
    --run_root "${RUN_ROOT}" --method bon_mcts \
    --prompt_range "0:1" \
    --out_dir "${RUN_ROOT}/${BACKEND}_logs" \
    --combined "${RUN_ROOT}/${BACKEND}_logs/_all.txt" || true

"${PYTHON_BIN}" "${SCRIPT_DIR}/compose_trajectory_strips.py" \
    --in_dir "${SAVE_BEST_STEP_IMAGES_DIR}" \
    --out_dir "${RUN_ROOT}/trajectory_strips" \
    --prompts_file "${PROMPT_FILE}" \
    --panel_size 384 --build_grid || true

# ── Step 7: final verification ───────────────────────────────────────────
echo
echo "[step 7] verifying MCTS actually explored variants"
python3 - "${RUN_ROOT}" "${BACKEND}" "${N_VARIANTS}" <<'PY'
import glob, json, sys, re
run_root, backend, n_req = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(f"[step 7] run_root: {run_root}")

# (a) runner saw variants?
vfiles = glob.glob(f"{run_root}/run_*/bon_mcts/p*_variants.txt") + \
         glob.glob(f"{run_root}/run_*/p*_variants.txt")
if vfiles:
    lines = [l.strip() for l in open(vfiles[0]).read().splitlines() if l.strip()]
    print(f"[step 7] runner saw {len(lines)} variants (expected {n_req+1}):")
    for ln in lines[:n_req+1]:
        print(f"           {ln[:100]}")
else:
    print(f"[step 7] WARN no *_variants.txt found -- SAVE_VARIANTS may not have plumbed through")

# (b) chosen variant per step
ranks = (glob.glob(f"{run_root}/run_*/bon_mcts/logs/rank_*.jsonl") +
         glob.glob(f"{run_root}/run_*/rank_*.jsonl"))
if not ranks:
    print(f"[step 7] FAIL no rank file -- bon_mcts didn't complete"); sys.exit(0)
for ln in open(ranks[0]):
    if not ln.strip(): continue
    r = json.loads(ln)
    if r.get("mode") not in ("mcts", "bon_mcts"): continue
    acts = r.get("actions", [])
    vs = [int(a[0]) for a in acts]
    cfgs = [float(a[1]) for a in acts]
    print(f"[step 7] CHOSEN variant_idx per step: {vs}")
    print(f"[step 7] CHOSEN cfg          per step: {cfgs}")
    nz = [i for i, v in enumerate(vs) if v > 0]
    if nz:
        print(f"[step 7] PASS: variants > 0 chosen at step(s) {nz}")
    else:
        print(f"[step 7] note: only v=0 was chosen (canonical won at every step)")
    break

# (c) how many of EACH variant did MCTS evaluate?
attempts = glob.glob(f"{run_root}/all_attempts/prompt_0000/*.png")
print(f"[step 7] all_attempts/ contains {len(attempts)} explored trajectories")
if attempts:
    counts = {}
    for fp in attempts:
        m = re.search(r"_v([0-9-]+)_", fp)
        if m:
            for tok in m.group(1).split("-"):
                counts[tok] = counts.get(tok, 0) + 1
    print(f"[step 7] variant-step-evaluation counts: {dict(sorted(counts.items()))}")
PY

# ── Final summary ────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "DONE.  Artifacts under ${RUN_ROOT}/"
echo "  rewrites.json                   prompt + 3 paraphrases"
echo "  run_*/bon_mcts/images/*.png     final MCTS-chosen image"
echo "  run_*/bon_mcts/logs/rank_*.jsonl raw MCTS rows + diagnostics"
echo "  step_images_inline/prompt_0000/ per-step x_0 of chosen path"
echo "  all_attempts/prompt_0000/       every MCTS-explored trajectory"
echo "  ${BACKEND}/actdiff_*_p0000_*.png decision tree"
echo "  ${BACKEND}_logs/prompt_0000.txt full per-step action trace"
echo "  trajectory_strips/prompt_0000.png horizontal film strip"
echo "================================================================"
