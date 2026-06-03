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
OUT_ROOT="${OUT_ROOT:-/data/ygu/runs/noise_ablation_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_ROOT}"

# Default prompts source: DPG-Bench (1065 dense compositional).  Override
# via PROMPT_FILE=... env or the 1st positional arg to use anything else.
DEFAULT_PROMPT_FILE="/data/ygu/dpg_bench_prompts.txt"
if [[ -n "${1:-}" ]]; then
    PROMPT_FILE="$1"
fi
PROMPT_FILE="${PROMPT_FILE:-${DEFAULT_PROMPT_FILE}}"

# Optional: bake in a single illustration prompt if the user wants to focus
# (PROMPT="..." or USE_BAKED_PROMPT=1 to use the raccoon).
BAKED_RACCOON='a detailed oil painting that captures the essence of an elderly raccoon adorned with a distinguished black top hat. The raccoon'\''s fur is depicted with textured, swirling strokes reminiscent of Van Gogh'\''s signature style, and it clutches a bright red apple in its paws. The background swirls with vibrant colors, giving the impression of movement around the still figure of the raccoon.'

PROMPT="${PROMPT:-}"
if [[ -n "${PROMPT}" || "${USE_BAKED_PROMPT:-0}" == "1" ]]; then
    if [[ -z "${PROMPT}" ]]; then PROMPT="${BAKED_RACCOON}"; fi
    _BAKED_FILE="${OUT_ROOT}/_baked_prompt.txt"
    printf '%s\n' "${PROMPT}" > "${_BAKED_FILE}"
    PROMPT_FILE="${_BAKED_FILE}"
    N_PROMPTS=1
    echo "[ablation] focused mode -- using baked-in single prompt"
fi

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

# ── Compose trajectory strips per condition ─────────────────────────────
echo
echo "[ablation] composing horizontal trajectory strips for each condition"
for c in fixed fresh; do
    INDIR="${OUT_ROOT}/${c}/step_images_inline"
    OUTDIR="${OUT_ROOT}/${c}/trajectory_strips"
    PFILE="${OUT_ROOT}/${c}/_prompts/backend_${BACKEND}.txt"
    if [[ -d "${INDIR}" ]]; then
        python "${SCRIPT_DIR}/compose_trajectory_strips.py" \
            --in_dir  "${INDIR}" \
            --out_dir "${OUTDIR}" \
            --prompts_file "${PFILE}" \
            --panel_size 384 --build_grid \
            2>&1 | tail -10 || echo "[ablation] WARN strip composition failed for ${c}"
    else
        echo "[ablation] no step_images_inline for ${c}; skipping strips"
    fi
done

# ── Side-by-side comparison (fixed on top row, fresh on bottom row) ────
echo
echo "[ablation] building per-prompt side-by-side comparison PNGs"
python3 - "${OUT_ROOT}" "${BACKEND}" <<'PY'
import sys, glob, os
from pathlib import Path
out_root = Path(sys.argv[1]); backend = sys.argv[2]
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:
    print(f"[ablation] PIL unavailable: {exc}"); sys.exit(0)
fixed_dir = out_root / "fixed" / "trajectory_strips"
fresh_dir = out_root / "fresh" / "trajectory_strips"
cmp_dir   = out_root / "comparison"
cmp_dir.mkdir(parents=True, exist_ok=True)
if not (fixed_dir.is_dir() and fresh_dir.is_dir()):
    print(f"[ablation] missing strip dirs ({fixed_dir.is_dir()}, {fresh_dir.is_dir()})"); sys.exit(0)
font = None
for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/Library/Fonts/Arial.ttf"):
    if os.path.isfile(cand):
        try: font = ImageFont.truetype(cand, 18); break
        except Exception: pass
if font is None: font = ImageFont.load_default()
prompt_files = sorted(fixed_dir.glob("prompt_*.png"))
print(f"[ablation] {len(prompt_files)} prompts to compare")
for fp in prompt_files:
    name = fp.name
    fr = fresh_dir / name
    if not fr.exists(): continue
    a = Image.open(fp).convert("RGB"); b = Image.open(fr).convert("RGB")
    W = max(a.width, b.width); header_h = 28
    out = Image.new("RGB", (W, a.height + b.height + 2*header_h + 8), (255,255,255))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, W, header_h], fill=(60,60,60))
    d.text((10, 5), f"FIXED noise  ({name.replace('.png','')})", font=font, fill=(255,255,255))
    out.paste(a, ((W-a.width)//2, header_h))
    y2 = a.height + header_h + 4
    d.rectangle([0, y2, W, y2+header_h], fill=(20,90,140))
    d.text((10, y2+5), f"FRESH noise", font=font, fill=(255,255,255))
    out.paste(b, ((W-b.width)//2, y2+header_h))
    out.save(cmp_dir / name)
print(f"[ablation] comparison PNGs -> {cmp_dir}")
PY

# ── Side-by-side DECISION-TREE comparison (fixed vs fresh) ────────────
echo
echo "[ablation] building per-prompt decision-tree comparison PNGs"
python3 - "${OUT_ROOT}" "${BACKEND}" <<'PY'
import sys, glob, re
from pathlib import Path
out_root = Path(sys.argv[1]); backend = sys.argv[2]
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:
    print(f"[ablation] PIL unavailable for tree compare: {exc}"); sys.exit(0)
fixed_dir = out_root / "fixed" / backend
fresh_dir = out_root / "fresh" / backend
cmp_dir   = out_root / "comparison_trees"
cmp_dir.mkdir(parents=True, exist_ok=True)
if not (fixed_dir.is_dir() and fresh_dir.is_dir()):
    print("[ablation] no tree dirs to compare"); sys.exit(0)
font = ImageFont.load_default()
for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/Library/Fonts/Arial.ttf"):
    try: font = ImageFont.truetype(cand, 18); break
    except Exception: pass
# Pair by prompt index extracted from filename actdiff_*_p<N>_bon_mcts.png
def by_idx(d):
    out = {}
    for fp in d.glob("actdiff_*_p*_bon_mcts.png"):
        m = re.search(r"_p(\d+)_bon_mcts\.png$", fp.name)
        if m: out[int(m.group(1))] = fp
    return out
fixed_map = by_idx(fixed_dir); fresh_map = by_idx(fresh_dir)
common = sorted(set(fixed_map.keys()) & set(fresh_map.keys()))
print(f"[ablation] {len(common)} prompt pairs to render")
for pi in common:
    a = Image.open(fixed_map[pi]).convert("RGB")
    b = Image.open(fresh_map[pi]).convert("RGB")
    H = max(a.height, b.height); pad = 12; header_h = 28
    out = Image.new("RGB", (a.width + b.width + pad, H + header_h),
                    (255, 255, 255))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, a.width, header_h], fill=(60, 60, 60))
    d.text((10, 5), f"FIXED noise (p={pi})", font=font, fill=(255,255,255))
    d.rectangle([a.width + pad, 0, a.width + pad + b.width, header_h], fill=(20,90,140))
    d.text((a.width + pad + 10, 5), "FRESH noise", font=font, fill=(255,255,255))
    out.paste(a, (0, header_h))
    out.paste(b, (a.width + pad, header_h))
    out.save(cmp_dir / f"tree_p{pi:04d}.png")
print(f"[ablation] tree comparisons -> {cmp_dir}")
PY

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
seen_modes = set()
total_rows = 0
for ln in open(fp):
    if not ln.strip(): continue
    try:
        r = json.loads(ln)
    except Exception:
        continue
    total_rows += 1
    m = r.get("mode")
    if m: seen_modes.add(m)
    # Accept any row that has a score field; the search method's identity
    # is enforced upstream (only bon_mcts ran), so don't gate on mode name.
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
print(f"  rows scanned: {total_rows}   modes seen: {sorted(seen_modes)}")
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
