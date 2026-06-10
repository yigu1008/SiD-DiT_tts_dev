#!/usr/bin/env bash
# Qualitative Exp 1: test-time scaling via budget sweep on ACTDIFF with fixed
# seed (sid).
#
# Demonstrates that with the SAME base seed, growing the BoN-ActDiff budget
# (N=1,2,4,8,16,32) progressively yields better images.  All budget cells
# share base SEED so at N=k, samples use seeds {seed, seed+1, ..., seed+k-1}
# — i.e. the high-N cell can only improve over the low-N cell (best-of-k ⊇
# best-of-j for k≥j).  Method is bon_actdiff_cfg: each sample draws a distinct
# (cfg) action from the discrete CFG bank, paired with a fresh noise seed.
#
# Output layout:
#   /data/ygu/runs/qual_demo_exp1_<ts>/
#     actdiff_n1/run_*/bon_actdiff_cfg/best_images/
#     actdiff_n2/run_*/bon_actdiff_cfg/best_images/
#     ...
#     actdiff_n32/run_*/bon_actdiff_cfg/best_images/
#     gallery.html                       ← simple 5×6 visual grid
#
# Just:    bash run_qual_exp1_budget_sid.sh
# Override:
#   PROMPT_FILE=./my5prompts.txt   N_PROMPTS=5
#   BUDGETS="1 4 16 64"            (override budget ladder)
#   SEED=123                       (different base seed)

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_a6000_common.sh"
source "${SCRIPT_DIR}/_a6000_common_multigpu.sh"
source "${SCRIPT_DIR}/_heartbeat.sh" 2>/dev/null || true
type start_heartbeat >/dev/null 2>&1 && start_heartbeat "qual-exp1-budget"

# ── Pinned config ────────────────────────────────────────────────────────
export BACKEND="${BACKEND:-sid}"
export N_PROMPTS="${N_PROMPTS:-5}"
export SEED="${SEED:-42}"
export SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"
export TOTAL_GPUS="${TOTAL_GPUS:-4}"
export PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/prompts_qual_exp1.txt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# Qualitative demo: KEEP image dumps (no SLIM_MODE).
export SLIM_MODE=0
export SAVE_BEST_IMAGES=1
export SAVE_IMAGES=1
BUDGETS="${BUDGETS:-1 2 4 8 16 32}"

OUT_ROOT="${OUT_ROOT:-/data/ygu/runs/qual_demo_exp1_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_ROOT}"

# Auto-fetch 5 prompts if missing/short
if [[ ! -s "${PROMPT_FILE}" ]] || [[ $(grep -c . "${PROMPT_FILE}" 2>/dev/null || echo 0) -lt "${N_PROMPTS}" ]]; then
    echo "[qual-exp1] (re-)building ${PROMPT_FILE} via HPSv2 (n=${N_PROMPTS})"
    PYTHONNOUSERSITE=1 python3 "${SCRIPT_DIR}/fetch_hpsv2.py" \
        --out_file "${PROMPT_FILE}" --n_prompts "${N_PROMPTS}" \
        --shuffle --seed "${SEED}"
fi

a6000_setup_backend
mgpu_boot_reward_server "${OUT_ROOT}/reward_server.log" "imagereward" || exit 1
trap 'mgpu_kill_reward_server' EXIT
mgpu_setup_sampling_gpus "${TOTAL_GPUS}"

echo "================================================================"
echo "QUAL EXP 1 — actdiff test-time scaling (fixed base seed, budget sweep)"
echo "  BACKEND     = ${BACKEND}"
echo "  METHOD      = bon_actdiff_cfg  (noise × CFG bank)"
echo "  N_PROMPTS   = ${N_PROMPTS}    SEED = ${SEED}  (fixed across cells)"
echo "  BUDGETS     = ${BUDGETS}"
echo "  OUT_ROOT    = ${OUT_ROOT}"
echo "================================================================"

for n in ${BUDGETS}; do
    cell="actdiff_n${n}"
    rr="${OUT_ROOT}/${cell}"
    echo; echo "================================================================"
    echo "[exp1] budget=${n}  cell=${cell}"
    echo "================================================================"
    (
        a6000_setup_bon_mcts_env "${rr}" "${N_PROMPTS}"
        export METHODS=bon_actdiff_cfg
        export BON_N="${n}"
        export BON_ACTION_DIVERSE=1       # actdiff: vary CFG across samples
        export BON_CFG_SCHEDULE=0
        export DAS_CONTINUOUS=0
        a6000_run_bon_mcts "${rr}"
    )
    pkill -f sd35_ddp_experiment 2>/dev/null || true
    pkill -f torchrun 2>/dev/null || true
    sleep 3
done

# ── Assemble a simple HTML gallery: rows = prompts, cols = budget ─────────
GALLERY="${OUT_ROOT}/gallery.html"
python3 - "${OUT_ROOT}" "${PROMPT_FILE}" "${BUDGETS}" > "${GALLERY}" <<'PY'
import os, sys, glob, html
out_root, prompt_file, budgets = sys.argv[1], sys.argv[2], sys.argv[3].split()
prompts = [ln.strip() for ln in open(prompt_file) if ln.strip()]
def find_imgs(cell):
    pats = [
        os.path.join(out_root, cell, "run_*", "bon_actdiff_cfg", "best_images", "*.png"),
        os.path.join(out_root, cell, "run_*", "bon_actdiff_cfg", "best_images", "*", "*.png"),
    ]
    found = []
    for p in pats:
        found.extend(sorted(glob.glob(p)))
    return found
print("<html><head><style>")
print("body{font-family:sans-serif;background:#111;color:#eee;padding:20px;}")
print("table{border-collapse:collapse;}")
print("td{padding:4px;text-align:center;vertical-align:top;border:1px solid #333;}")
print("img{max-width:180px;max-height:180px;display:block;}")
print("th{background:#222;padding:8px;}")
print(".prompt{max-width:200px;font-size:12px;text-align:left;}")
print("</style></head><body>")
print(f"<h2>Qual Exp 1: ActDiff TTS budget sweep (fixed seed, sid)</h2>")
print("<table><tr><th>prompt</th>")
for n in budgets: print(f"<th>N={n}</th>")
print("</tr>")
for i, prompt in enumerate(prompts):
    print(f"<tr><td class='prompt'>{html.escape(prompt[:120])}</td>")
    for n in budgets:
        imgs = find_imgs(f"actdiff_n{n}")
        # heuristic: pick the i-th image for the i-th prompt
        img = imgs[i] if i < len(imgs) else None
        if img:
            rel = os.path.relpath(img, out_root)
            print(f"<td><img src='{html.escape(rel)}'/></td>")
        else:
            print("<td style='color:#888'>missing</td>")
    print("</tr>")
print("</table></body></html>")
PY

echo
echo "================================================================"
echo "DONE.  ${OUT_ROOT}/"
echo "  HTML gallery:  ${GALLERY}"
echo "  Open with:     python3 -m http.server --directory ${OUT_ROOT}"
echo "================================================================"
