#!/usr/bin/env bash
# MCTS-internal ablation on sd3.5_base (28-step) to dissect which MCTS
# components actually contribute.
#
# Components ablated (existing suite aliases):
#   baseline                  : single forward (reference)
#   bon                       : noise-only BoN (no search algo, no prescreen)
#   bon_mcts_singleseed       : MCTS only, NO prescreen (N_SEEDS=1, TOPK=1)
#   bon_mcts_static_cfg       : prescreen + mcts, no adaptive cfg, no rewrite
#   bon_mcts_adaptive_cfg     : prescreen + ours_tree + adaptive cfg, no rewrite
#   bon_mcts_rewrite_only     : prescreen + mcts, no adaptive cfg, WITH rewrites
#   bon_mcts_full             : prescreen + ours_tree + adaptive cfg + rewrites
#   bon_mcts                  : same as bon_mcts_full (the headline number)
#
# Reads: WHICH component drives MCTS's lift over BoN?
#   - prescreen layer:       bon_mcts_singleseed vs bon_mcts_static_cfg
#   - adaptive cfg:          static_cfg vs adaptive_cfg
#   - prompt rewrites:       adaptive_cfg vs full   (and rewrite_only vs static)
#   - both axes (full):      static_cfg vs full
#
# Just run:
#   bash run_mcts_ablation_sd35base.sh
# Override:
#   N_PROMPTS=10   N_SIMS=120   SEED=42

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_a6000_common.sh"
source "${SCRIPT_DIR}/_a6000_common_multigpu.sh"
source "${SCRIPT_DIR}/_heartbeat.sh" 2>/dev/null || true
type start_heartbeat >/dev/null 2>&1 && start_heartbeat "mcts-ablation-sd35base"

# ── Pinned defaults (sd35_base 28-step) ─────────────────────────────────
export BACKEND=sd35_base
export N_PROMPTS="${N_PROMPTS:-5}"
export SEED="${SEED:-42}"
export SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"
export N_SIMS="${N_SIMS:-120}"
export TOTAL_GPUS="${TOTAL_GPUS:-8}"
export PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/prompts.txt}"
export SLIM_MODE="${SLIM_MODE:-1}"

# The 8 methods of the ablation (in evaluation order).
METHODS="${METHODS:-baseline bon bon_mcts_singleseed bon_mcts_static_cfg bon_mcts_adaptive_cfg bon_mcts_rewrite_only bon_mcts_full bon_mcts}"

OUT_ROOT="${OUT_ROOT:-/data/ygu/runs/mcts_ablation_sd35base_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_ROOT}"

# Auto-fetch prompts
if [[ ! -s "${PROMPT_FILE}" ]] || [[ $(grep -c . "${PROMPT_FILE}" 2>/dev/null || echo 0) -lt "${N_PROMPTS}" ]]; then
    PYTHONNOUSERSITE=1 python3 "${SCRIPT_DIR}/fetch_hpsv2.py" \
        --out_file "${PROMPT_FILE}" --n_prompts "${N_PROMPTS}" \
        --shuffle --seed "${SEED}" || { echo "[FATAL] fetch failed"; exit 1; }
fi

a6000_setup_backend
mgpu_boot_reward_server "${OUT_ROOT}/reward_server.log" "imagereward" || exit 1
trap 'mgpu_kill_reward_server' EXIT
mgpu_setup_sampling_gpus "${TOTAL_GPUS}"

echo "================================================================"
echo "MCTS ABLATION on sd3.5_base (28-step)"
echo "  N_PROMPTS    = ${N_PROMPTS}"
echo "  N_SIMS       = ${N_SIMS}"
echo "  SEED         = ${SEED}"
echo "  METHODS      = ${METHODS}"
echo "  OUT_ROOT     = ${OUT_ROOT}"
echo "================================================================"

_run() {
    local m="$1"
    local rr="${OUT_ROOT}/${m}"
    echo; echo "[ablation] ${m}"
    (
        a6000_setup_bon_mcts_env "${rr}" "${N_PROMPTS}"
        export METHODS="${m}"
        a6000_run_bon_mcts "${rr}"
    )
    pkill -f sd35_ddp_experiment 2>/dev/null || true
    pkill -f torchrun 2>/dev/null || true
    sleep 5
}

for m in ${METHODS}; do _run "${m}"; done

# Aggregate
SUMMARY="${OUT_ROOT}/mcts_ablation_summary.csv"
{
    echo "method,n_prompts,mean_baseline,mean_search,mean_delta,eval_imagereward,eval_hpsv3,seconds"
    for m in ${METHODS}; do
        rf=$(ls "${OUT_ROOT}/${m}"/run_*/*/logs/rank_*.jsonl 2>/dev/null | head -1)
        [[ -f "${rf}" ]] || continue
        python3 - "${rf}" "${m}" <<'PY'
import json, sys, os, time
fp, m = sys.argv[1], sys.argv[2]
sc, base, dv, eir, eh3 = [], [], [], [], []
for ln in open(fp):
    if not ln.strip(): continue
    try: r = json.loads(ln)
    except Exception: continue
    if r.get("score") is not None: sc.append(float(r["score"]))
    if r.get("baseline_score") is not None: base.append(float(r["baseline_score"]))
    if r.get("delta_vs_base") is not None: dv.append(float(r["delta_vs_base"]))
    for k in ("eval_imagereward", "imagereward"):
        v = r.get(k);
        if isinstance(v, (int, float)): eir.append(float(v)); break
    for k in ("eval_hpsv3", "hpsv3"):
        v = r.get(k);
        if isinstance(v, (int, float)): eh3.append(float(v)); break
if not sc: sys.exit(0)
def m_(xs): return f"{sum(xs)/len(xs):.4f}" if xs else ""
print(f"{m},{len(sc)},{m_(base)},{m_(sc)},{m_(dv)},{m_(eir)},{m_(eh3)},")
PY
    done
} > "${SUMMARY}"

echo
echo "================================================================"
echo "DONE.  ${OUT_ROOT}/"
column -t -s, "${SUMMARY}"
echo "================================================================"
