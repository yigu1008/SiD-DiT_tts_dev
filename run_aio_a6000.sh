#!/usr/bin/env bash
# A6000-local port of amlt/all_in_one.yaml's "composite" stage.
#
# Cross-method comparison on ONE backend (sid by default) at single-NFE budget,
# with MULTI-REWARD evaluation: imagereward + hpsv3 + HPSv2.
#
# Methods (9 like the cluster aio):
#   baseline bon beam smc fksteering dts dts_star sop bon_mcts
#
# Reward server hosts ALL THREE backends (imagereward + hpsv3 + hpsv2) so
# every method's final image gets scored on all 3 metrics simultaneously.
#
# Just run:
#   bash run_aio_a6000.sh
# Override:
#   N_PROMPTS=200   N_SIMS=30   SEED=42   BACKEND=sid
#   SEARCH_REWARD=imagereward     (or composite_hpsv3_ir; what MCTS optimizes)
#   METHODS="baseline bon bon_mcts"  (subset for fast test)
#   TOTAL_GPUS=8

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_a6000_common.sh"
source "${SCRIPT_DIR}/_a6000_common_multigpu.sh"
source "${SCRIPT_DIR}/_heartbeat.sh" 2>/dev/null || true
type start_heartbeat >/dev/null 2>&1 && start_heartbeat "aio-a6000"

# ── Pinned defaults ─────────────────────────────────────────────────────
BACKEND="${BACKEND:-sid}"
N_PROMPTS="${N_PROMPTS:-200}"
N_SIMS="${N_SIMS:-30}"
SEED="${SEED:-42}"
SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"   # what methods optimize
EVAL_BACKENDS="${EVAL_BACKENDS:-imagereward hpsv3 hpsv2}"   # ← multi-reward eval
METHODS="${METHODS:-baseline bon beam smc fksteering dts dts_star sop bon_mcts}"
TOTAL_GPUS="${TOTAL_GPUS:-8}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/prompts.txt}"
OUT_ROOT="${OUT_ROOT:-/data/ygu/runs/aio_a6000_$(date +%Y%m%d_%H%M%S)}"
export SLIM_MODE="${SLIM_MODE:-1}"
mkdir -p "${OUT_ROOT}"

# Auto-fetch prompts
if [[ ! -s "${PROMPT_FILE}" ]] || [[ $(grep -c . "${PROMPT_FILE}" 2>/dev/null || echo 0) -lt "${N_PROMPTS}" ]]; then
    echo "[fetch] building ${PROMPT_FILE} via HPSv2"
    PYTHONNOUSERSITE=1 python3 "${SCRIPT_DIR}/fetch_hpsv2.py" \
        --out_file "${PROMPT_FILE}" --n_prompts "${N_PROMPTS}" \
        --shuffle --seed "${SEED}" || { echo "[FATAL] fetch failed"; exit 1; }
fi
export PROMPT_FILE

a6000_setup_backend

# Reward server loads ALL 3 backends (this is the multi-reward enabler).
# Adds ~3 GB GPU memory (HPSv2 ~ 1.5GB, HPSv3 ~ 2GB, ImageReward ~ 1.5GB).
mgpu_boot_reward_server "${OUT_ROOT}/reward_server.log" "imagereward hpsv3 hpsv2" || exit 1
trap 'mgpu_kill_reward_server' EXIT
mgpu_setup_sampling_gpus "${TOTAL_GPUS}"

echo "================================================================"
echo "AIO A6000  (multi-reward eval: imagereward + hpsv3 + hpsv2)"
echo "  BACKEND        = ${BACKEND}"
echo "  N_PROMPTS      = ${N_PROMPTS}    N_SIMS = ${N_SIMS}    SEED = ${SEED}"
echo "  SEARCH_REWARD  = ${SEARCH_REWARD}     (what methods optimize)"
echo "  EVAL_BACKENDS  = ${EVAL_BACKENDS}     (multi-reward scoring)"
echo "  METHODS        = ${METHODS}"
echo "  OUT_ROOT       = ${OUT_ROOT}"
echo "================================================================"

_run() {
    local m="$1"
    local rr="${OUT_ROOT}/${m}"
    echo; echo "[aio] ${m}"
    (
        a6000_setup_bon_mcts_env "${rr}" "${N_PROMPTS}"
        export METHODS="${m}"
        export SEARCH_REWARD="${SEARCH_REWARD}"
        export REWARD_BACKEND="${SEARCH_REWARD}"
        export REWARD_TYPE="${SEARCH_REWARD}"
        export REWARD_BACKENDS="${SEARCH_REWARD}"
        # CRITICAL: override eval backends to include hpsv2.
        export EVAL_BACKENDS="${EVAL_BACKENDS}"
        a6000_run_bon_mcts "${rr}"
    )
    pkill -f sd35_ddp_experiment 2>/dev/null || true
    pkill -f torchrun 2>/dev/null || true
    sleep 5
}

for m in ${METHODS}; do
    _run "${m}"
done

# ── Aggregate: per-method × per-reward table ────────────────────────────
SUMMARY="${OUT_ROOT}/multi_reward_summary.csv"
{
    echo "method,n_prompts,search_score,baseline,delta_vs_base,eval_imagereward,eval_hpsv3,eval_hpsv2"
    for m in ${METHODS}; do
        rf=$(ls "${OUT_ROOT}/${m}"/run_*/*/logs/rank_*.jsonl 2>/dev/null | head -1)
        [[ -f "${rf}" ]] || continue
        python3 - "${rf}" "${m}" <<'PY'
import json, sys
fp, m = sys.argv[1], sys.argv[2]
sc, base, dv = [], [], []
eval_ir, eval_h3, eval_h2 = [], [], []
for ln in open(fp):
    if not ln.strip(): continue
    try: r = json.loads(ln)
    except Exception: continue
    if r.get("score") is not None: sc.append(float(r["score"]))
    if r.get("baseline_score") is not None: base.append(float(r["baseline_score"]))
    if r.get("delta_vs_base") is not None: dv.append(float(r["delta_vs_base"]))
    ev = r.get("eval") or r.get("eval_rewards") or {}
    if isinstance(ev, dict):
        for k, v in ev.items():
            if "imagereward" in k.lower() and isinstance(v, (int, float)):
                eval_ir.append(float(v))
            elif "hpsv3" in k.lower() and isinstance(v, (int, float)):
                eval_h3.append(float(v))
            elif "hpsv2" in k.lower() and isinstance(v, (int, float)):
                eval_h2.append(float(v))
    # also try flat fields
    for k_in, dst in (("eval_imagereward", eval_ir), ("eval_hpsv3", eval_h3), ("eval_hpsv2", eval_h2)):
        v = r.get(k_in)
        if isinstance(v, (int, float)): dst.append(float(v))
def m_(xs): return f"{sum(xs)/len(xs):.4f}" if xs else ""
print(f"{m},{len(sc)},{m_(sc)},{m_(base)},{m_(dv)},{m_(eval_ir)},{m_(eval_h3)},{m_(eval_h2)}")
PY
    done
} > "${SUMMARY}"

# Also try to pull from the per-method aggregate JSON if it exists
SUMMARY2="${OUT_ROOT}/multi_reward_summary_v2.csv"
{
    echo "method,eval_imagereward,eval_hpsv3,eval_hpsv2"
    for m in ${METHODS}; do
        agg=$(find "${OUT_ROOT}/${m}" -name 'best_images_multi_reward_aggregate.json' 2>/dev/null | head -1)
        [[ -f "${agg}" ]] || continue
        python3 - "${agg}" "${m}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
m = sys.argv[2]
def get(k):
    v = d.get(k, {})
    if isinstance(v, dict): return v.get("mean", "")
    return ""
print(f"{m},{get('imagereward')},{get('hpsv3')},{get('hpsv2')}")
PY
    done
} > "${SUMMARY2}"

echo
echo "================================================================"
echo "DONE.  ${OUT_ROOT}/"
echo
echo "Per-prompt rank-file derived:"
column -t -s, "${SUMMARY}"
echo
echo "From best_images aggregate (if eval was run):"
column -t -s, "${SUMMARY2}" 2>/dev/null
echo "================================================================"
