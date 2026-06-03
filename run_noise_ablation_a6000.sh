#!/usr/bin/env bash
# Noise ablation on A6000: bon_mcts WITHOUT prescreen, comparing
#   fixed = MCTS_FRESH_ROLLOUT_NOISE=0  (deterministic rollouts)
#   fresh = MCTS_FRESH_ROLLOUT_NOISE=1  (re-noise each step like baselines do)
#
# Reward runs in-process (no server -> no port-reuse bugs).
#
# Just run (or backgroundable with nohup):
#   bash run_noise_ablation_a6000.sh
# Override:
#   N_PROMPTS=4 | N_SIMS=30 | SEED=42 | BACKEND=sid
#   PROMPT_FILE=/path | USE_BAKED_PROMPT=1 (focused single-prompt mode)

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_a6000_common.sh"
source "${SCRIPT_DIR}/_heartbeat.sh" 2>/dev/null || true
type start_heartbeat >/dev/null 2>&1 && start_heartbeat "noise-ablation"

# Defaults
BACKEND="${BACKEND:-sid}"
N_PROMPTS="${N_PROMPTS:-4}"
N_SIMS="${N_SIMS:-30}"
SEED="${SEED:-42}"
N_VARIANTS="${N_VARIANTS:-1}"      # rewriting off -- isolate noise axis
USE_QWEN="${USE_QWEN:-0}"
OUT_ROOT="${OUT_ROOT:-/data/ygu/runs/noise_ablation_$(date +%Y%m%d_%H%M%S)}"
export SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"
mkdir -p "${OUT_ROOT}"

# Prompt input: PROMPT_FILE env wins; else 1st arg; else DPG-Bench; else baked.
DEFAULT_PROMPT_FILE="/data/ygu/dpg_bench_prompts.txt"
PROMPT_FILE="${PROMPT_FILE:-${1:-${DEFAULT_PROMPT_FILE}}}"
if [[ "${USE_BAKED_PROMPT:-0}" == "1" ]]; then
    a6000_bake_prompt "${OUT_ROOT}/_baked_prompt"
    N_PROMPTS=1
fi
export PROMPT_FILE

echo "================================================================"
echo "NOISE ABLATION  (fixed vs fresh noise, in-process ImageReward)"
echo "  BACKEND=${BACKEND}  N_PROMPTS=${N_PROMPTS}  N_SIMS=${N_SIMS}  SEED=${SEED}"
echo "  N_VARIANTS=${N_VARIANTS}  USE_QWEN=${USE_QWEN}"
echo "  PROMPT_FILE=${PROMPT_FILE}"
echo "  OUT_ROOT=${OUT_ROOT}"
echo "================================================================"

# Common setup
a6000_use_inprocess_reward
a6000_setup_backend

# Reduce prescreen (we want pure refine MCTS comparison)
export BON_MCTS_N_SEEDS=1 BON_MCTS_TOPK=1
export BON_MCTS_MIN_SIMS="${N_SIMS}"

# Run each condition (use a subshell so OUT_ROOT mutation doesn't leak).
_run_condition() {
    local label="$1" fresh="$2"
    local rr="${OUT_ROOT}/${label}"
    echo; echo "[ablation] CONDITION=${label}  MCTS_FRESH_ROLLOUT_NOISE=${fresh}"
    (
        a6000_setup_bon_mcts_env "${rr}" "${N_PROMPTS}"
        export MCTS_FRESH_ROLLOUT_NOISE="${fresh}"
        a6000_run_bon_mcts "${rr}"
        a6000_render_viz "${rr}" "${N_PROMPTS}"
    )
    echo "[ablation] ${label} DONE"
}

_run_condition fixed 0
sleep 20
pkill -f sd35_ddp_experiment 2>/dev/null || true
pkill -f torchrun 2>/dev/null || true
sleep 5
_run_condition fresh 1

# Side-by-side comparison (strips + trees)
echo; echo "[ablation] building comparison views"
python3 - "${OUT_ROOT}" "${BACKEND}" <<'PY'
import sys, os, re
from pathlib import Path
out_root = Path(sys.argv[1]); backend = sys.argv[2]
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:
    print(f"[compare] PIL unavailable: {exc}"); sys.exit(0)
font = ImageFont.load_default()
for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/Library/Fonts/Arial.ttf"):
    if os.path.isfile(cand):
        try: font = ImageFont.truetype(cand, 18); break
        except Exception: pass

def pair_dirs(sub):
    a = out_root / "fixed" / sub
    b = out_root / "fresh" / sub
    return (a, b) if a.is_dir() and b.is_dir() else (None, None)

# Strips
a, b = pair_dirs("trajectory_strips")
if a is not None:
    cmp_dir = out_root / "comparison_strips"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    for fp in sorted(a.glob("prompt_*.png")):
        fr = b / fp.name
        if not fr.exists(): continue
        ia = Image.open(fp).convert("RGB"); ib = Image.open(fr).convert("RGB")
        W = max(ia.width, ib.width); H = 28
        out = Image.new("RGB", (W, ia.height + ib.height + 2*H + 8), (255,255,255))
        d = ImageDraw.Draw(out)
        d.rectangle([0,0,W,H], fill=(60,60,60)); d.text((10,5), f"FIXED ({fp.stem})", font=font, fill=(255,255,255))
        out.paste(ia, ((W-ia.width)//2, H))
        y2 = ia.height + H + 4
        d.rectangle([0,y2,W,y2+H], fill=(20,90,140)); d.text((10,y2+5), "FRESH", font=font, fill=(255,255,255))
        out.paste(ib, ((W-ib.width)//2, y2+H))
        out.save(cmp_dir / fp.name)
    print(f"[compare] strips -> {cmp_dir}")

# Trees
a, b = pair_dirs(backend)
if a is not None:
    cmp_dir = out_root / "comparison_trees"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    def by_idx(dd):
        out_d = {}
        for fp in dd.glob("actdiff_*_p*_bon_mcts.png"):
            m = re.search(r"_p(\d+)_bon_mcts\.png$", fp.name)
            if m: out_d[int(m.group(1))] = fp
        return out_d
    A = by_idx(a); B = by_idx(b)
    for pi in sorted(set(A) & set(B)):
        ia = Image.open(A[pi]).convert("RGB"); ib = Image.open(B[pi]).convert("RGB")
        H = max(ia.height, ib.height); pad = 12; hh = 28
        out = Image.new("RGB", (ia.width + ib.width + pad, H + hh), (255,255,255))
        d = ImageDraw.Draw(out)
        d.rectangle([0,0,ia.width,hh], fill=(60,60,60)); d.text((10,5), f"FIXED (p={pi})", font=font, fill=(255,255,255))
        d.rectangle([ia.width+pad,0,ia.width+pad+ib.width,hh], fill=(20,90,140))
        d.text((ia.width+pad+10,5), "FRESH", font=font, fill=(255,255,255))
        out.paste(ia, (0,hh)); out.paste(ib, (ia.width+pad, hh))
        out.save(cmp_dir / f"tree_p{pi:04d}.png")
    print(f"[compare] trees -> {cmp_dir}")
PY

# Summary
SUMMARY="${OUT_ROOT}/SUMMARY.txt"
{
    echo "Noise ablation  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
    echo "BACKEND=${BACKEND}  N_PROMPTS=${N_PROMPTS}  N_SIMS=${N_SIMS}  SEED=${SEED}"
    echo "N_VARIANTS=${N_VARIANTS}  USE_QWEN=${USE_QWEN}  Reward=ImageReward"
    echo
    for c in fixed fresh; do
        echo "--- ${c} ---"
        rf="$(ls "${OUT_ROOT}/${c}"/run_*/bon_mcts/logs/rank_*.jsonl 2>/dev/null | head -1)"
        if [[ -z "${rf}" ]]; then echo "  (no rank file)"; echo; continue; fi
        python3 - "${rf}" <<'PY'
import json, sys
fp=sys.argv[1]; sc=[]; de=[]; nf=[]
for ln in open(fp):
    if not ln.strip(): continue
    try: r=json.loads(ln)
    except Exception: continue
    if r.get("score") is not None: sc.append(float(r["score"]))
    if r.get("delta_vs_base") is not None: de.append(float(r["delta_vs_base"]))
    if r.get("nfe") is not None: nf.append(int(r["nfe"]))
def s(xs): return f"n={len(xs)} mean={sum(xs)/len(xs):+.4f} min={min(xs):+.4f} max={max(xs):+.4f}" if xs else "(empty)"
print(f"  rank: {fp}")
print(f"  IR:    {s(sc)}")
print(f"  delta: {s(de)}")
print(f"  NFE:   {s(nf)}")
PY
        echo
    done
} | tee "${SUMMARY}"

echo
echo "================================================================"
echo "DONE.  ${OUT_ROOT}/"
echo "  fixed/  fresh/  comparison_strips/  comparison_trees/  SUMMARY.txt"
echo "================================================================"
