#!/usr/bin/env bash
# Controlled experiment on A6000: bon_mcts WITHOUT prescreen, comparing
#   (A) FIXED noise   -- deterministic noise_step_cache (current default)
#   (B) FRESH noise   -- torch.randn_like() per step (matches baselines)
#
# Same prompts, same seed, same action space -- only the noise source differs.
# Reward backend: ImageReward.
#
# Designed to be run with `nohup ... &` so the user can disconnect.  All
# stdout/stderr lands in <out>/{fixed,fresh}/_run.log.  A short summary is
# written to <out>/SUMMARY.txt at the end.
#
# Usage (foreground):
#   bash run_noise_ablation_a6000.sh
# Usage (background, disconnect-safe):
#   nohup bash run_noise_ablation_a6000.sh > /tmp/noise_ablation.out 2>&1 &
#   tail -f /tmp/noise_ablation.out
#
# Knobs (env vars):
#   N_PROMPTS=20     # how many prompts to evaluate
#   N_SIMS=30        # MCTS sims per prompt
#   SEED=42          # base seed (same across both conditions)
#   BACKEND=sid      # sid|senseflow_large|sd35_base
#   PROMPT_FILE=/data/ygu/dpg_bench_prompts.txt
#   OUT_ROOT=/data/ygu/runs/noise_ablation_<ts>

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

N_PROMPTS="${N_PROMPTS:-20}"
N_SIMS="${N_SIMS:-30}"
SEED="${SEED:-42}"
BACKEND="${BACKEND:-sid}"
PROMPT_FILE="${PROMPT_FILE:-/data/ygu/dpg_bench_prompts.txt}"
OUT_ROOT="${OUT_ROOT:-/data/ygu/runs/noise_ablation_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_ROOT}"

echo "================================================================"
echo "CONTROLLED NOISE ABLATION (bon_mcts, no prescreen, ImageReward)"
echo "  BACKEND     = ${BACKEND}"
echo "  N_PROMPTS   = ${N_PROMPTS}"
echo "  N_SIMS      = ${N_SIMS}"
echo "  SEED        = ${SEED}"
echo "  PROMPT_FILE = ${PROMPT_FILE}"
echo "  OUT_ROOT    = ${OUT_ROOT}"
echo "================================================================"

if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "[FATAL] PROMPT_FILE not found: ${PROMPT_FILE}" >&2
    exit 1
fi

_run_cond() {
    local label="$1" mcts_fixed_noise="$2"
    local run_root="${OUT_ROOT}/${label}"
    mkdir -p "${run_root}"
    echo
    echo "================================================================"
    echo "[ablation] CONDITION = ${label}   MCTS_FIXED_NOISE=${mcts_fixed_noise}"
    echo "================================================================"
    # Disable prescreen entirely: 1 seed, top-K=1.
    PROMPT_FILE="${PROMPT_FILE}" \
    BACKEND="${BACKEND}" \
    N_PROMPTS="${N_PROMPTS}" \
    N_SIMS="${N_SIMS}" \
    SEED="${SEED}" \
    BON_MCTS_N_SEEDS=1 \
    BON_MCTS_TOPK=1 \
    BON_MCTS_MIN_SIMS="${N_SIMS}" \
    BON_MCTS_SIM_ALLOC=split \
    USE_QWEN=0 N_VARIANTS=1 \
    SEARCH_REWARD=imagereward \
    MCTS_FIXED_NOISE="${mcts_fixed_noise}" \
    RUN_ROOT="${run_root}" \
      bash "${SCRIPT_DIR}/run_actdiff_render_a6000.sh" \
      > "${run_root}/_run.log" 2>&1
    echo "[ablation] condition ${label} finished -- log: ${run_root}/_run.log"
}

# Run sequentially so the two conditions don't fight for GPU.  Reward server
# is killed between conditions inside run_actdiff_render_a6000.sh (Stage B.2).
_run_cond fixed 1
# Give CUDA a moment to release between runs.
sleep 30
pkill -f reward_server.py 2>/dev/null || true
pkill -f sd35_ddp_experiment 2>/dev/null || true
sleep 10
_run_cond fresh 0

# ── Summarize ─────────────────────────────────────────────────────────────
SUMMARY="${OUT_ROOT}/SUMMARY.txt"
{
    echo "Noise ablation summary  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
    echo "================================================================"
    echo "BACKEND=${BACKEND}  N_PROMPTS=${N_PROMPTS}  N_SIMS=${N_SIMS}  SEED=${SEED}"
    echo "Reward backend: ImageReward"
    echo
    for c in fixed fresh; do
        echo "--- condition: ${c} ---"
        eval_jsonl="$(ls "${OUT_ROOT}/${c}"/run_*/bon_mcts/logs/rank_*.jsonl 2>/dev/null | head -1)"
        if [[ -z "${eval_jsonl}" ]]; then
            echo "  (no rank file found)"
            continue
        fi
        python3 - "${eval_jsonl}" <<'PY'
import json, sys
fp = sys.argv[1]
scores = []
deltas = []
nfes = []
for ln in open(fp):
    if not ln.strip(): continue
    try:
        r = json.loads(ln)
    except Exception:
        continue
    if r.get("mode") not in ("mcts", "bon_mcts"):
        continue
    if r.get("score") is not None:
        scores.append(float(r["score"]))
    if r.get("delta_vs_base") is not None:
        deltas.append(float(r["delta_vs_base"]))
    if r.get("nfe") is not None:
        nfes.append(int(r["nfe"]))
def stats(xs):
    if not xs: return "(empty)"
    return f"n={len(xs)} mean={sum(xs)/len(xs):+.4f} min={min(xs):+.4f} max={max(xs):+.4f}"
print(f"  rank file: {fp}")
print(f"  IR score:        {stats(scores)}")
print(f"  IR delta vs base:{stats(deltas)}")
print(f"  NFE per prompt:  {stats(nfes)}")
PY
        echo
    done
    echo "Full logs and per-prompt images under:"
    echo "  ${OUT_ROOT}/fixed/"
    echo "  ${OUT_ROOT}/fresh/"
} | tee "${SUMMARY}"

echo
echo "================================================================"
echo "DONE.  Summary: ${SUMMARY}"
echo "================================================================"
