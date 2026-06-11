#!/usr/bin/env bash
# Qualitative Exp 2: action-axis ablation within ACTDIFF (sid).
#
# Compares 2 actdiff cells at MATCHED BoN budget — same N, same base seed —
# so the only thing that differs is whether the prompt-rewrite axis is added
# on top of the CFG axis:
#   bon_actdiff_cfg  : noise × CFG bank
#   bon_actdiff_full : noise × CFG × prompt-rewrite bank
#
# Demonstrates: adding the prompt axis on top of CFG enriches the candidate
# distribution further → better best-of-N image at the same total budget.
#
# Output layout:
#   /data/ygu/runs/qual_demo_exp2_<ts>/
#     bon/run_*/bon/best_images/
#     bon_actdiff_cfg/run_*/bon_actdiff_cfg/best_images/
#     bon_actdiff_full/run_*/bon_actdiff_full/best_images/
#     gallery.html
#
# Just:    bash run_qual_exp2_axes_sid.sh
# Override:
#   PROMPT_FILE=./my5prompts.txt   N_PROMPTS=5
#   BON_N=16                       (per-method budget; default 16)
#   SEED=123

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_a6000_common.sh"
source "${SCRIPT_DIR}/_a6000_common_multigpu.sh"
source "${SCRIPT_DIR}/_heartbeat.sh" 2>/dev/null || true
type start_heartbeat >/dev/null 2>&1 && start_heartbeat "qual-exp2-axes"

# ── Pinned config ────────────────────────────────────────────────────────
export BACKEND="${BACKEND:-sid}"
export N_PROMPTS="${N_PROMPTS:-5}"
export SEED="${SEED:-42}"
export SEARCH_REWARD="${SEARCH_REWARD:-composite_ir_ps}"
export TOTAL_GPUS="${TOTAL_GPUS:-2}"
export PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/prompts_qual_exp2.txt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export BON_N="${BON_N:-16}"
export SLIM_MODE=0
export SAVE_BEST_IMAGES=1
export SAVE_IMAGES=1

OUT_ROOT="${OUT_ROOT:-/data/ygu/runs/qual_demo_exp2_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_ROOT}"

# Auto-fetch 5 prompts if missing/short
if [[ ! -s "${PROMPT_FILE}" ]] || [[ $(grep -c . "${PROMPT_FILE}" 2>/dev/null || echo 0) -lt "${N_PROMPTS}" ]]; then
    echo "[qual-exp2] (re-)building ${PROMPT_FILE} via HPSv2 (n=${N_PROMPTS})"
    PYTHONNOUSERSITE=1 python3 "${SCRIPT_DIR}/fetch_hpsv2.py" \
        --out_file "${PROMPT_FILE}" --n_prompts "${N_PROMPTS}" \
        --shuffle --seed "${SEED}"
fi

# Composite rewards need their component models loaded on the reward server.
case "${SEARCH_REWARD}" in
    composite_ir_ps)    REWARD_SERVER_BACKENDS="imagereward pickscore" ;;
    composite_hpsv3_ir) REWARD_SERVER_BACKENDS="imagereward hpsv3" ;;
    composite_all4)     REWARD_SERVER_BACKENDS="imagereward pickscore hpsv2 hpsv3" ;;
    *)                  REWARD_SERVER_BACKENDS="${SEARCH_REWARD}" ;;
esac

a6000_setup_backend
mgpu_boot_reward_server "${OUT_ROOT}/reward_server.log" "${REWARD_SERVER_BACKENDS}" || exit 1
trap 'mgpu_kill_reward_server' EXIT
mgpu_setup_sampling_gpus "${TOTAL_GPUS}"

echo "================================================================"
echo "QUAL EXP 2 — action-axis ablation (matched BoN budget, same seed)"
echo "  BACKEND       = ${BACKEND}"
echo "  N_PROMPTS     = ${N_PROMPTS}    SEED = ${SEED}"
echo "  BON_N         = ${BON_N}        (matched across cells)"
echo "  OUT_ROOT      = ${OUT_ROOT}"
echo "================================================================"

_run_cell() {
    local cell="$1" action_diverse="$2" use_rewrites="$3"
    local rr="${OUT_ROOT}/${cell}"
    echo; echo "================================================================"
    echo "[exp2] cell=${cell}  action_diverse=${action_diverse}  rewrites=${use_rewrites}"
    echo "================================================================"
    (
        a6000_setup_bon_mcts_env "${rr}" "${N_PROMPTS}"
        # Final output only: keep best_images, drop the step/attempt trace dumps.
        export SAVE_BEST_IMAGES=1 SAVE_IMAGES=0 SAVE_VARIANTS=0
        unset SAVE_BEST_STEP_IMAGES_DIR SAVE_ALL_ATTEMPTS_DIR SAVE_ALL_STEP_IMAGES_DIR
        export METHODS="${cell}"
        export BON_N
        export BON_ACTION_DIVERSE="${action_diverse}"
        export BON_CFG_SCHEDULE=0
        export DAS_CONTINUOUS=0
        # The bon_actdiff_full method enables rewrites internally via N_VARIANTS;
        # bon and bon_actdiff_cfg leave N_VARIANTS=1.  Suite handles the dispatch.
        a6000_run_bon_mcts "${rr}"
    )
    pkill -f sd35_ddp_experiment 2>/dev/null || true
    pkill -f torchrun 2>/dev/null || true
    sleep 3
}

_run_cell bon_actdiff_cfg  1 0   # noise × CFG
_run_cell bon_actdiff_full 1 1   # noise × CFG × prompt

# ── Assemble a simple HTML gallery: rows = prompts, cols = method ─────────
GALLERY="${OUT_ROOT}/gallery.html"
python3 - "${OUT_ROOT}" "${PROMPT_FILE}" > "${GALLERY}" <<'PY'
import os, sys, glob, html
out_root, prompt_file = sys.argv[1], sys.argv[2]
prompts = [ln.strip() for ln in open(prompt_file) if ln.strip()]
cells = [
    ("bon_actdiff_cfg",  "ActDiff (CFG axis)"),
    ("bon_actdiff_full", "ActDiff (CFG + prompt)"),
]
def find_imgs(cell):
    pats = [
        os.path.join(out_root, cell, "run_*", cell, "best_images", "*.png"),
        os.path.join(out_root, cell, "run_*", cell, "best_images", "*", "*.png"),
    ]
    found = []
    for p in pats:
        found.extend(sorted(glob.glob(p)))
    return found
print("<html><head><style>")
print("body{font-family:sans-serif;background:#111;color:#eee;padding:20px;}")
print("table{border-collapse:collapse;}")
print("td{padding:4px;text-align:center;vertical-align:top;border:1px solid #333;}")
print("img{max-width:220px;max-height:220px;display:block;}")
print("th{background:#222;padding:8px;}")
print(".prompt{max-width:240px;font-size:12px;text-align:left;}")
print("</style></head><body>")
print("<h2>Qual Exp 2: ActDiff axis ablation (CFG vs CFG+prompt, matched BoN, sid)</h2>")
print("<table><tr><th>prompt</th>")
for _, label in cells: print(f"<th>{html.escape(label)}</th>")
print("</tr>")
for i, prompt in enumerate(prompts):
    print(f"<tr><td class='prompt'>{html.escape(prompt[:150])}</td>")
    for cell, _ in cells:
        imgs = find_imgs(cell)
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
