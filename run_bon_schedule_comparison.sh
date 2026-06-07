#!/usr/bin/env bash
# BoN-over-CFG-schedule comparison.
# Compares three flavors of BoN at matched compute (BON_N noise samples):
#   bon              : fixed cfg=baseline, only noise varies                 [single point]
#   bon_actdiff_cfg  : per-trajectory cfg from CFG bank (T uses same cfg)    [|bank| points]
#   bon_schedule     : per-STEP cfg drawn iid from a small CFG bank          [|bank|^T points]
#
# Small CFG bank by default ({1.0, 1.5, 2.0, 2.5} = 4 values).  With T=4 steps,
# the schedule space is 4^4 = 256 — much larger than the 7-cfg per-trajectory
# bank.  This lets BoN access dynamic CFG schedules (the regime MCTS dominates in).
#
# Just run:
#   bash run_bon_schedule_comparison.sh
# Override:
#   BON_N=16            (default 16)
#   CFG_SCALES="1.0 1.5 2.0 2.5"   (small bank, default; T=4 → 256 schedules)
#   N_PROMPTS=200
#   PROMPT_FILE=./prompts.txt

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_a6000_common.sh"
source "${SCRIPT_DIR}/_a6000_common_multigpu.sh"
source "${SCRIPT_DIR}/_heartbeat.sh" 2>/dev/null || true
type start_heartbeat >/dev/null 2>&1 && start_heartbeat "bon-schedule"

# ── Pinned config ────────────────────────────────────────────────────────
export BACKEND="${BACKEND:-sid}"
export N_PROMPTS="${N_PROMPTS:-200}"
export SEED="${SEED:-42}"
export SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"
export BON_N="${BON_N:-16}"
export TOTAL_GPUS="${TOTAL_GPUS:-4}"
export SLIM_MODE="${SLIM_MODE:-1}"
export PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/prompts.txt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# Override the suite-default 7-cfg bank to a smaller, well-spaced 4-cfg pool.
export CFG_SCALES="${CFG_SCALES:-1.0 1.5 2.0 2.5}"
OUT_ROOT="${OUT_ROOT:-/data/ygu/runs/bon_schedule_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_ROOT}"

# Auto-fetch prompts if missing/short
if [[ ! -s "${PROMPT_FILE}" ]] || [[ $(grep -c . "${PROMPT_FILE}" 2>/dev/null || echo 0) -lt "${N_PROMPTS}" ]]; then
    echo "[fetch] (re-)building ${PROMPT_FILE} via HPSv2"
    PYTHONNOUSERSITE=1 python3 "${SCRIPT_DIR}/fetch_hpsv2.py" \
        --out_file "${PROMPT_FILE}" --n_prompts "${N_PROMPTS}" \
        --shuffle --seed "${SEED}"
fi

a6000_setup_backend
# Override the backend-default CFG bank with our small bank
export CFG_SCALES

mgpu_boot_reward_server "${OUT_ROOT}/reward_server.log" "imagereward" || exit 1
trap 'mgpu_kill_reward_server' EXIT
mgpu_setup_sampling_gpus "${TOTAL_GPUS}"

echo "================================================================"
echo "BoN-SCHEDULE COMPARISON  (per-step CFG schedule BoN)"
echo "  BACKEND        = ${BACKEND}"
echo "  N_PROMPTS      = ${N_PROMPTS}    BON_N = ${BON_N}"
echo "  CFG_SCALES     = ${CFG_SCALES}  (|bank|=$(echo "${CFG_SCALES}" | wc -w | tr -d ' '))"
echo "  T (steps)      = ${STEPS}"
echo "  schedule space = |bank|^T"
echo "  OUT_ROOT       = ${OUT_ROOT}"
echo "================================================================"

_run() {
    local label="$1" dyn_flag="$2" action_diverse="$3"
    local rr="${OUT_ROOT}/${label}"
    echo; echo "[run] ${label}  BON_CFG_SCHEDULE=${dyn_flag}  action_diverse=${action_diverse}"
    (
        a6000_setup_bon_mcts_env "${rr}" "${N_PROMPTS}"
        export METHODS=bon
        export BON_N
        export BON_ACTION_DIVERSE="${action_diverse}"
        export BON_CFG_SCHEDULE="${dyn_flag}"
        a6000_run_bon_mcts "${rr}"
    )
    pkill -f sd35_ddp_experiment 2>/dev/null || true
    pkill -f torchrun 2>/dev/null || true
    sleep 5
}

# 1) baseline reference (single sample)
echo; echo "[run] baseline"
(
    a6000_setup_bon_mcts_env "${OUT_ROOT}/baseline" "${N_PROMPTS}"
    export METHODS=baseline
    a6000_run_bon_mcts "${OUT_ROOT}/baseline"
)

# 2) Three flavors of BoN at matched BON_N
_run bon              0 0    # vanilla: noise only
_run bon_actdiff_cfg  0 1    # per-trajectory cfg
_run bon_schedule     1 1    # per-step cfg + variant schedule  ← NEW

# Summary
SUMMARY="${OUT_ROOT}/summary.csv"
{
    echo "method,n_prompts,mean_search,mean_baseline,mean_delta,eval_imagereward"
    for m in baseline bon bon_actdiff_cfg bon_schedule; do
        rf=$(ls "${OUT_ROOT}/${m}"/run_*/*/logs/rank_*.jsonl 2>/dev/null | head -1)
        [[ -f "${rf}" ]] || continue
        python3 - "${rf}" "${m}" <<'PY'
import json, sys
fp, m = sys.argv[1], sys.argv[2]
sc, base, dv = [], [], []
for ln in open(fp):
    if not ln.strip(): continue
    r = json.loads(ln)
    if r.get("score") is not None: sc.append(float(r["score"]))
    if r.get("baseline_score") is not None: base.append(float(r["baseline_score"]))
    if r.get("delta_vs_base") is not None: dv.append(float(r["delta_vs_base"]))
if not sc: sys.exit(0)
ms = sum(sc)/len(sc); mb = sum(base)/len(base) if base else ms
md = sum(dv)/len(dv) if dv else (ms-mb)
print(f"{m},{len(sc)},{ms:.6f},{mb:.6f},{md:+.6f},{ms:.6f}")
PY
    done
} > "${SUMMARY}"

echo
echo "================================================================"
echo "DONE.  ${OUT_ROOT}/"
column -t -s, "${SUMMARY}"
echo "================================================================"
