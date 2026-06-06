#!/usr/bin/env bash
# Noise-budget sweep: does ActDiff's CFG/prompt-axis search help when noise
# samples are scarce?
#
# At each BON_N value in {4, 8, 16, 32, 76}, run four methods on the same
# prompts and compare their delta-vs-baseline:
#   bon                 = vary noise only
#   bon_actdiff_cfg     = vary noise + CFG
#   bon_actdiff_full    = vary noise + (CFG x prompt)
#   bon_mcts            = full ActDiff with MCTS refine (upper bound)
#
# Hypothesis: at small BON_N (4-16), bon_actdiff_* and bon_mcts widen the
# gap vs bon because spreading the budget across action axes compensates
# for the noise paucity.  At large BON_N (76+), bon ~ bon_actdiff because
# noise diversity alone is sufficient.
#
# Just run (after fixing N_PROMPTS and BON_N_LIST below if needed):
#   bash run_noise_budget_sweep.sh
#
# Override:
#   N_PROMPTS=20         (default 30; smaller => faster turnaround)
#   BON_N_LIST="4 8 16 32 76"  (override the budget grid)
#   BACKEND=sid
#   PROMPT_FILE=./prompts.txt
#
# ETA on 8 GPUs at default settings:
#   5 budgets x 4 methods x ~30 prompts x ~0.5 min/prompt-batch ~= 4-6 hours

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_a6000_common.sh"
source "${SCRIPT_DIR}/_a6000_common_multigpu.sh"
source "${SCRIPT_DIR}/_heartbeat.sh" 2>/dev/null || true
type start_heartbeat >/dev/null 2>&1 && start_heartbeat "noise-budget-sweep"

# ── Defaults ─────────────────────────────────────────────────────────────
BACKEND="${BACKEND:-sid}"
N_PROMPTS="${N_PROMPTS:-30}"
SEED="${SEED:-42}"
SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/prompts.txt}"
TOTAL_GPUS="${TOTAL_GPUS:-8}"
BON_N_LIST="${BON_N_LIST:-4 8 16 32 76}"
METHODS_BUDGET="${METHODS_BUDGET:-bon bon_actdiff_cfg bon_actdiff_full}"
OUT_ROOT="${OUT_ROOT:-/data/ygu/runs/noise_budget_sweep_$(date +%Y%m%d_%H%M%S)}"
export SLIM_MODE="${SLIM_MODE:-1}"      # no per-prompt images
mkdir -p "${OUT_ROOT}"

# Auto-fetch prompts if missing/short
if [[ ! -s "${PROMPT_FILE}" ]] || [[ $(grep -c . "${PROMPT_FILE}" 2>/dev/null || echo 0) -lt "${N_PROMPTS}" ]]; then
    echo "[fetch] (re-)building ${PROMPT_FILE} via HPSv2"
    PYTHONNOUSERSITE=1 python3 "${SCRIPT_DIR}/fetch_hpsv2.py" \
        --out_file "${PROMPT_FILE}" --n_prompts "${N_PROMPTS}" \
        --shuffle --seed "${SEED}" || { echo "[FATAL] fetch failed"; exit 1; }
fi
export PROMPT_FILE

# In-process reward setup wouldn't work with 8 GPUs (we want DDP).  Use
# the 8-GPU pattern: reward server on GPU 0, sampling on GPUs 1..N-1.
a6000_setup_backend
mgpu_boot_reward_server "${OUT_ROOT}/reward_server.log" "imagereward" || exit 1
trap 'mgpu_kill_reward_server' EXIT
mgpu_setup_sampling_gpus "${TOTAL_GPUS}"

# Always need to also include baseline once (NFE-cheap, doesn't sweep).
echo
echo "================================================================"
echo "[sweep] launching baseline (single run, no budget axis)"
echo "================================================================"
(
    rr="${OUT_ROOT}/_baseline"
    a6000_setup_bon_mcts_env "${rr}" "${N_PROMPTS}"
    export METHODS=baseline
    a6000_run_bon_mcts "${rr}"
)

# Sweep: budget x method
for BON_N in ${BON_N_LIST}; do
    for METHOD in ${METHODS_BUDGET}; do
        echo
        echo "================================================================"
        echo "[sweep] BON_N=${BON_N}   METHOD=${METHOD}"
        echo "================================================================"
        rr="${OUT_ROOT}/N${BON_N}/${METHOD}"
        (
            a6000_setup_bon_mcts_env "${rr}" "${N_PROMPTS}"
            export METHODS="${METHOD}"
            export BON_N="${BON_N}"
            a6000_run_bon_mcts "${rr}"
        )
        # Free any per-method allocations between runs.
        pkill -f sd35_ddp_experiment 2>/dev/null || true
        pkill -f torchrun 2>/dev/null || true
        sleep 8
    done
done

# ── Aggregate to a single CSV: budget, method, mean_score, mean_delta ────
SUMMARY="${OUT_ROOT}/sweep_summary.csv"
{
    echo "budget,method,n_prompts,mean_search,mean_baseline,mean_delta,eval_imagereward"
    # baseline single row (budget=0 placeholder)
    bf=$(ls "${OUT_ROOT}/_baseline"/run_*/baseline/logs/rank_*.jsonl 2>/dev/null | head -1)
    if [[ -f "${bf}" ]]; then
        python3 - "${bf}" <<'PY'
import json, sys
sc, base = [], []
for ln in open(sys.argv[1]):
    if not ln.strip(): continue
    r = json.loads(ln)
    if r.get("score") is not None: sc.append(float(r["score"]))
    if r.get("baseline_score") is not None: base.append(float(r["baseline_score"]))
mean = sum(sc)/len(sc) if sc else 0.0
b = sum(base)/len(base) if base else mean
print(f"0,baseline,{len(sc)},{mean:.6f},{b:.6f},{mean-b:+.6f},{mean:.6f}")
PY
    fi
    for BON_N in ${BON_N_LIST}; do
        for METHOD in ${METHODS_BUDGET}; do
            rf=$(ls "${OUT_ROOT}/N${BON_N}/${METHOD}"/run_*/${METHOD}/logs/rank_*.jsonl 2>/dev/null | head -1)
            [[ -z "${rf}" ]] && rf=$(ls "${OUT_ROOT}/N${BON_N}/${METHOD}"/run_*/*/logs/rank_*.jsonl 2>/dev/null | head -1)
            [[ -z "${rf}" ]] && continue
            python3 - "${rf}" "${BON_N}" "${METHOD}" <<'PY'
import json, sys
fp, bn, m = sys.argv[1], int(sys.argv[2]), sys.argv[3]
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
print(f"{bn},{m},{len(sc)},{mean_s:.6f},{mean_b:.6f},{mean_d:+.6f},{mean_s:.6f}")
PY
        done
    done
} > "${SUMMARY}"

echo
echo "================================================================"
echo "[sweep] DONE.  Output: ${OUT_ROOT}/"
echo "  sweep_summary.csv:"
column -t -s, "${SUMMARY}" | head -40
echo "================================================================"

# ── Quick plot: delta-vs-baseline as function of BON_N per method ─────
"${PYTHON_BIN:-python}" - "${SUMMARY}" "${OUT_ROOT}/budget_sweep.pdf" <<'PY'
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

summary, out_path = sys.argv[1], sys.argv[2]
data = defaultdict(list)  # method -> [(budget, delta), ...]
for row in csv.DictReader(open(summary)):
    m = row["method"]; b = int(row["budget"]); d = float(row["mean_delta"])
    if m == "baseline": continue
    data[m].append((b, d))

if not data:
    print("[plot] no rows; skipping"); sys.exit(0)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
colors = {"bon": "#1f77b4", "bon_actdiff_cfg": "#2ca02c",
          "bon_actdiff_full": "#d62728"}
labels = {"bon": "BoN", "bon_actdiff_cfg": "BoN+ActDiff (CFG)",
          "bon_actdiff_full": "BoN+ActDiff (CFG+Prompt)"}
for m in ("bon", "bon_actdiff_cfg", "bon_actdiff_full"):
    pts = sorted(data.get(m, []))
    if not pts: continue
    xs, ys = zip(*pts)
    lw = 3.0 if m == "bon_actdiff_full" else 1.8
    ax.plot(xs, ys, marker="o", color=colors.get(m, "#777"),
            label=labels.get(m, m), linewidth=lw, markersize=8)
ax.set_xscale("log", base=2)
ax.set_xlabel("Noise-sample budget BON_N (log2)")
ax.set_ylabel(r"$\Delta$ ImageReward vs baseline")
ax.set_title("Does ActDiff help when noise budget is limited?")
ax.grid(True, alpha=0.3)
ax.legend(loc="best", framealpha=0.92)
plt.tight_layout()
plt.savefig(out_path, dpi=180, bbox_inches="tight")
plt.savefig(out_path.replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
print(f"[plot] saved {out_path} and {out_path.replace('.pdf', '.png')}")
PY

echo
echo "Plot: ${OUT_ROOT}/budget_sweep.pdf"
