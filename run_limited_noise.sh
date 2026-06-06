#!/usr/bin/env bash
# Limited-noise comparison (no sweep).
# Hypothesis: when each method can only use a SMALL number of unique noise
# seeds, ActDiff's CFG/prompt-axis search starts to dominate plain BoN.
#
# Three methods at the SAME noise seed budget (BON_N):
#   bon                  : ${BON_N} unique noise seeds, vanilla
#   bon_actdiff_cfg      : ${BON_N} unique noise seeds x CFG bank
#   bon_actdiff_full     : ${BON_N} unique noise seeds x (CFG x prompt) bank
#
# Just run:
#   bash run_limited_noise.sh
# Override:
#   BON_N=8               (default; small => stresses noise-paucity regime)
#   N_PROMPTS=200
#   PROMPT_FILE=./prompts.txt

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_a6000_common.sh"
source "${SCRIPT_DIR}/_a6000_common_multigpu.sh"
source "${SCRIPT_DIR}/_heartbeat.sh" 2>/dev/null || true
type start_heartbeat >/dev/null 2>&1 && start_heartbeat "limited-noise"

# ── Pinned configuration (small noise budget) ─────────────────────────────
BACKEND="${BACKEND:-sid}"
N_PROMPTS="${N_PROMPTS:-200}"
SEED="${SEED:-42}"
SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/prompts.txt}"
TOTAL_GPUS="${TOTAL_GPUS:-8}"
BON_N="${BON_N:-8}"                     # ← noise seed budget (small on purpose)
METHODS_LIMITED="${METHODS_LIMITED:-bon bon_actdiff_cfg bon_actdiff_full}"
OUT_ROOT="${OUT_ROOT:-/data/ygu/runs/limited_noise_N${BON_N}_$(date +%Y%m%d_%H%M%S)}"
export SLIM_MODE="${SLIM_MODE:-1}"
mkdir -p "${OUT_ROOT}"

# Auto-fetch prompts if missing/short
if [[ ! -s "${PROMPT_FILE}" ]] || [[ $(grep -c . "${PROMPT_FILE}" 2>/dev/null || echo 0) -lt "${N_PROMPTS}" ]]; then
    echo "[fetch] (re-)building ${PROMPT_FILE} via HPSv2"
    PYTHONNOUSERSITE=1 python3 "${SCRIPT_DIR}/fetch_hpsv2.py" \
        --out_file "${PROMPT_FILE}" --n_prompts "${N_PROMPTS}" \
        --shuffle --seed "${SEED}" || { echo "[FATAL] fetch failed"; exit 1; }
fi
export PROMPT_FILE

a6000_setup_backend
mgpu_boot_reward_server "${OUT_ROOT}/reward_server.log" "imagereward" || exit 1
trap 'mgpu_kill_reward_server' EXIT
mgpu_setup_sampling_gpus "${TOTAL_GPUS}"

echo "================================================================"
echo "LIMITED-NOISE COMPARISON"
echo "  BON_N (noise seeds) = ${BON_N}"
echo "  N_PROMPTS           = ${N_PROMPTS}"
echo "  BACKEND             = ${BACKEND}"
echo "  METHODS             = ${METHODS_LIMITED}"
echo "  OUT_ROOT            = ${OUT_ROOT}"
echo "================================================================"

# Baseline (single)
echo; echo "[run] baseline"
(
    rr="${OUT_ROOT}/baseline"
    a6000_setup_bon_mcts_env "${rr}" "${N_PROMPTS}"
    export METHODS=baseline
    a6000_run_bon_mcts "${rr}"
)

# Each method at the same small noise budget
for METHOD in ${METHODS_LIMITED}; do
    echo; echo "================================================================"
    echo "[run] ${METHOD}   BON_N=${BON_N}"
    echo "================================================================"
    (
        rr="${OUT_ROOT}/${METHOD}"
        a6000_setup_bon_mcts_env "${rr}" "${N_PROMPTS}"
        export METHODS="${METHOD}"
        export BON_N="${BON_N}"
        a6000_run_bon_mcts "${rr}"
    )
    pkill -f sd35_ddp_experiment 2>/dev/null || true
    pkill -f torchrun 2>/dev/null || true
    sleep 8
done

# ── Aggregate ────────────────────────────────────────────────────────────
SUMMARY="${OUT_ROOT}/limited_noise_summary.csv"
{
    echo "method,n_prompts,mean_search,mean_baseline,mean_delta,eval_imagereward"
    for m in baseline ${METHODS_LIMITED}; do
        rf=$(ls "${OUT_ROOT}/${m}"/run_*/${m}/logs/rank_*.jsonl 2>/dev/null | head -1)
        [[ -z "${rf}" ]] && rf=$(ls "${OUT_ROOT}/${m}"/run_*/*/logs/rank_*.jsonl 2>/dev/null | head -1)
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
mean_s = sum(sc)/len(sc)
mean_b = sum(base)/len(base) if base else mean_s
mean_d = sum(dv)/len(dv) if dv else (mean_s - mean_b)
print(f"{m},{len(sc)},{mean_s:.6f},{mean_b:.6f},{mean_d:+.6f},{mean_s:.6f}")
PY
    done
} > "${SUMMARY}"

echo
echo "================================================================"
echo "[done] ${OUT_ROOT}"
column -t -s, "${SUMMARY}"
echo "================================================================"
