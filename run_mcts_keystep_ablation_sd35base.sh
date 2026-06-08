#!/usr/bin/env bash
# Ablation: WHERE in the 28-step denoising trajectory should bon_mcts apply
# its action-axis search (key steps)?
#
# sd35_base has T=28 steps.  MCTS doesn't branch at every step — it branches
# at "key" steps and uses fixed CFG between them.  The choice of which steps
# matters: early steps shape composition, late steps shape detail.
#
# Cells (MCTS_KEY_STEPS):
#   early4     : 0 1 2 3                                (decisions only in first 4 steps)
#   early8     : 0 1 2 3 4 5 6 7                        (decisions only in first 8)
#   late4      : 24 25 26 27                            (decisions only in last 4)
#   late8      : 20 21 22 23 24 25 26 27                (decisions only in last 8)
#   spread4    : 0 9 18 27                              (4 evenly-spaced)
#   spread8    : 0 4 8 12 16 20 24 27                   (8 evenly-spaced, current default)
#   sparse3    : 0 14 27                                (just 3)
#   dense      : 0 1 2 ... 27                           (every step)
#   mid8       : 10 12 14 16 18 20 22 24                (8 concentrated in middle)
#
# Each cell runs bon_mcts_full on sd35_base with the specified key steps.
# Compare delta_vs_base / eval_IR / eval_HPSv3 across cells.
#
# Just run:
#   bash run_mcts_keystep_ablation_sd35base.sh
# Override:
#   N_PROMPTS=10     N_SIMS=120
#   CELLS="early4 spread8 late4"     (subset)
#   TOTAL_GPUS=4

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_a6000_common.sh"
source "${SCRIPT_DIR}/_a6000_common_multigpu.sh"
source "${SCRIPT_DIR}/_heartbeat.sh" 2>/dev/null || true
type start_heartbeat >/dev/null 2>&1 && start_heartbeat "mcts-keystep-ablation"

# ── Pinned defaults ─────────────────────────────────────────────────────
export BACKEND=sd35_base
export N_PROMPTS="${N_PROMPTS:-5}"
export SEED="${SEED:-42}"
export N_SIMS="${N_SIMS:-120}"
export SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"
export TOTAL_GPUS="${TOTAL_GPUS:-8}"
export PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/prompts.txt}"
export SLIM_MODE="${SLIM_MODE:-1}"

# Cells to evaluate (named -> KEY_STEPS string).
declare -A CELL_STEPS=(
    [early4]="0 1 2 3"
    [early8]="0 1 2 3 4 5 6 7"
    [late4]="24 25 26 27"
    [late8]="20 21 22 23 24 25 26 27"
    [spread4]="0 9 18 27"
    [spread8]="0 4 8 12 16 20 24 27"
    [sparse3]="0 14 27"
    [dense]="$(seq -s' ' 0 27)"
    [mid8]="10 12 14 16 18 20 22 24"
)

CELLS="${CELLS:-early4 early8 late4 late8 spread4 spread8 sparse3 mid8}"
OUT_ROOT="${OUT_ROOT:-/data/ygu/runs/mcts_keystep_sd35base_$(date +%Y%m%d_%H%M%S)}"
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
echo "MCTS KEY-STEP ABLATION on sd35_base (28-step)"
echo "  N_PROMPTS    = ${N_PROMPTS}    N_SIMS = ${N_SIMS}"
echo "  CELLS        = ${CELLS}"
echo "  OUT_ROOT     = ${OUT_ROOT}"
echo "================================================================"

# Add baseline + standard bon_mcts (default key steps) as reference
echo; echo "[abl] baseline reference"
(
    a6000_setup_bon_mcts_env "${OUT_ROOT}/_baseline" "${N_PROMPTS}"
    export METHODS=baseline
    a6000_run_bon_mcts "${OUT_ROOT}/_baseline"
)

for cell in ${CELLS}; do
    steps="${CELL_STEPS[${cell}]:-}"
    [[ -z "${steps}" ]] && { echo "[abl] WARN unknown cell '${cell}'"; continue; }
    n_keysteps=$(echo "${steps}" | wc -w | tr -d ' ')
    echo; echo "================================================================"
    echo "[abl] cell=${cell}   key_steps='${steps}'   n_keysteps=${n_keysteps}"
    echo "================================================================"
    (
        rr="${OUT_ROOT}/${cell}"
        a6000_setup_bon_mcts_env "${rr}" "${N_PROMPTS}"
        export METHODS=bon_mcts_full
        export MCTS_KEY_STEPS="${steps}"
        export MCTS_KEY_STEP_COUNT="${n_keysteps}"
        a6000_run_bon_mcts "${rr}"
    )
    pkill -f sd35_ddp_experiment 2>/dev/null || true
    pkill -f torchrun 2>/dev/null || true
    sleep 5
done

# Aggregate
SUMMARY="${OUT_ROOT}/keystep_ablation_summary.csv"
{
    echo "cell,key_steps,n_keysteps,n_prompts,mean_baseline,mean_search,mean_delta,eval_imagereward,eval_hpsv3"
    # Baseline row
    rf=$(ls "${OUT_ROOT}/_baseline"/run_*/baseline/logs/rank_*.jsonl 2>/dev/null | head -1)
    if [[ -f "${rf}" ]]; then
        python3 - "${rf}" "_baseline" "" "0" <<'PY'
import json, sys
fp, cell, steps, nk = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sc, base, dv, eir, eh3 = [], [], [], [], []
for ln in open(fp):
    if not ln.strip(): continue
    try: r = json.loads(ln)
    except Exception: continue
    if r.get("score") is not None: sc.append(float(r["score"]))
    if r.get("baseline_score") is not None: base.append(float(r["baseline_score"]))
    if r.get("delta_vs_base") is not None: dv.append(float(r["delta_vs_base"]))
    v=r.get("eval_imagereward") or r.get("imagereward")
    if isinstance(v,(int,float)): eir.append(float(v))
    v=r.get("eval_hpsv3") or r.get("hpsv3")
    if isinstance(v,(int,float)): eh3.append(float(v))
def m_(xs): return f"{sum(xs)/len(xs):.4f}" if xs else ""
print(f"{cell},{steps},{nk},{len(sc)},{m_(base)},{m_(sc)},{m_(dv)},{m_(eir)},{m_(eh3)}")
PY
    fi
    for cell in ${CELLS}; do
        steps="${CELL_STEPS[${cell}]:-}"
        n_keysteps=$(echo "${steps}" | wc -w | tr -d ' ')
        rf=$(ls "${OUT_ROOT}/${cell}"/run_*/*/logs/rank_*.jsonl 2>/dev/null | head -1)
        [[ -f "${rf}" ]] || continue
        python3 - "${rf}" "${cell}" "${steps}" "${n_keysteps}" <<'PY'
import json, sys
fp, cell, steps, nk = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sc, base, dv, eir, eh3 = [], [], [], [], []
for ln in open(fp):
    if not ln.strip(): continue
    try: r = json.loads(ln)
    except Exception: continue
    if r.get("score") is not None: sc.append(float(r["score"]))
    if r.get("baseline_score") is not None: base.append(float(r["baseline_score"]))
    if r.get("delta_vs_base") is not None: dv.append(float(r["delta_vs_base"]))
    v=r.get("eval_imagereward") or r.get("imagereward")
    if isinstance(v,(int,float)): eir.append(float(v))
    v=r.get("eval_hpsv3") or r.get("hpsv3")
    if isinstance(v,(int,float)): eh3.append(float(v))
def m_(xs): return f"{sum(xs)/len(xs):.4f}" if xs else ""
# Quote the steps if it contains spaces.
import csv, io
buf = io.StringIO()
csv.writer(buf).writerow([cell, steps, nk, len(sc), m_(base), m_(sc), m_(dv), m_(eir), m_(eh3)])
print(buf.getvalue().rstrip())
PY
    done
} > "${SUMMARY}"

echo
echo "================================================================"
echo "DONE.  ${OUT_ROOT}/"
column -t -s, "${SUMMARY}"
echo "================================================================"
