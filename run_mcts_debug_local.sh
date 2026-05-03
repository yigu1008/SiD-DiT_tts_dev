#!/usr/bin/env bash
# Local single-GPU MCTS debug + upper-bound run.
# Tested config: 48GB A100, SD3.5-large, 10 prompts.
#
# What this does:
#   - Runs single-GPU (no DDP, no torchrun multi-rank).
#   - Loads ImageReward INLINE on the same GPU (no separate reward server,
#     no isolated conda env). Saves the GPU-0-dedicated-to-server cost.
#   - Defaults to bon_mcts at AGGRESSIVE settings to probe upper-bound
#     reward at 10-prompt scale: N_SIMS=120, topk=4, full alloc, n_seeds=16.
#   - Runs `baseline` first as the lower-bound reference.
#
# Memory budget on 48GB A100 (rough):
#   SD3.5-large pipeline (fp16 transformer + VAE + 3 text encoders) :  ~17 GB
#   ImageReward (BLIP-large)                                        :   ~3 GB
#   HPSv3 (HPSv3 model + CLIP + extras) — ONLY if you enable        :   ~5 GB
#   Activations + KV cache during sampling                          :  4-8 GB
#   MCTS tree state (K trees × ~30 nodes × few MB latent)           :  ~2 GB
#   ────────────────────────────────────────────────────────────────────────
#   Headroom with imagereward only                                  : ~22 GB ✓
#   Headroom with imagereward + hpsv3 (composite)                   : ~17 GB ✓ tight
#
# Usage:
#   bash run_mcts_debug_local.sh                            # defaults below
#   SD35_BACKEND=senseflow_large bash run_mcts_debug_local.sh
#   N_SIMS=200 BON_MCTS_TOPK=8 bash run_mcts_debug_local.sh  # crank further
#   REWARD_BACKEND=composite_hpsv3_ir bash run_mcts_debug_local.sh
#
# Override anything via env. The defaults below target sid (4-step), the
# fastest backend — plenty of room to crank knobs and see upper bound.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Backend (default sid for fastest iteration; flip to sd35_base if you have time) ──
SD35_BACKEND="${SD35_BACKEND:-sid}"
case "${SD35_BACKEND}" in
    sid)             STEPS="${STEPS:-4}";  BASELINE_CFG="${BASELINE_CFG:-1.0}"; CFG_SCALES="${CFG_SCALES:-1.0 1.5 2.0 2.5}" ;;
    senseflow_large) STEPS="${STEPS:-4}";  BASELINE_CFG="${BASELINE_CFG:-1.0}"; CFG_SCALES="${CFG_SCALES:-1.0 1.5 2.0 2.5}" ;;
    sd35_base)       STEPS="${STEPS:-28}"; BASELINE_CFG="${BASELINE_CFG:-4.5}"; CFG_SCALES="${CFG_SCALES:-3.5 4.5 5.5 7.0}" ;;
    *) echo "Unknown SD35_BACKEND=${SD35_BACKEND}" >&2; exit 1 ;;
esac

# ── Single-GPU setup ────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GPUS=1
# Importantly: NOT setting REWARD_SERVER_URL → reward loads inline on the
# same GPU. Saves GPU-0-dedicated-to-server overhead.
unset REWARD_SERVER_URL || true

# ── Memory + perf hygiene for 48GB A100 ─────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8
export SID_FORCE_WANDB_STUB=1
export WANDB_DISABLED=true

# ── Reward (default imagereward — small + fast). Override to composite if desired. ──
REWARD_BACKEND="${REWARD_BACKEND:-imagereward}"
EVAL_BACKENDS="${EVAL_BACKENDS:-imagereward}"   # only imagereward by default; add hpsv3 if wanted

# ── Run knobs ───────────────────────────────────────────────────────────────
NUM_PROMPTS="${NUM_PROMPTS:-10}"
SEED="${SEED:-42}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/hpsv2_subset.txt}"
OUT_ROOT_BASE="${OUT_ROOT_BASE:-/tmp/mcts_debug}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${OUT_ROOT_BASE}/${SD35_BACKEND}/run_${RUN_TS}"
mkdir -p "${RUN_ROOT}"

# ── Aggressive upper-bound bon_mcts knobs ───────────────────────────────────
# Total per-prompt NFE ~= n_seeds * STEPS + topk * n_sims * STEPS
# sid     (STEPS=4):  16*4 + 4*120*4   = 1984 forwards/prompt
# sd35_base (STEPS=28): 16*28 + 4*120*28 = 13888 forwards/prompt   (~6h for 10 prompts)
N_SIMS="${N_SIMS:-120}"
UCB_C="${UCB_C:-1.0}"
BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS:-16}"
BON_MCTS_TOPK="${BON_MCTS_TOPK:-4}"
BON_MCTS_SIM_ALLOC="${BON_MCTS_SIM_ALLOC:-full}"     # each top-K seed gets full N_SIMS
BON_MCTS_MIN_SIMS="${BON_MCTS_MIN_SIMS:-30}"
BON_MCTS_REFINE_METHOD="${BON_MCTS_REFINE_METHOD:-mcts}"  # vanilla — winner from prescreen ablation

# ── Methods: baseline first (warmup) then bon_mcts (upper bound) ───────────
METHODS="${METHODS:-baseline bon_mcts}"
N_VARIANTS="${N_VARIANTS:-1}"      # set to 3 with USE_QWEN=1 if you want Qwen rewrites in the action space
USE_QWEN="${USE_QWEN:-0}"
PRECOMPUTE_REWRITES="${PRECOMPUTE_REWRITES:-0}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen3-4B}"
CORRECTION_STRENGTHS="${CORRECTION_STRENGTHS:-0.0}"
SAVE_BEST_IMAGES="${SAVE_BEST_IMAGES:-1}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"     # save ALL per-step images locally for debugging
SAVE_VARIANTS="${SAVE_VARIANTS:-1}"
EVAL_BEST_IMAGES="${EVAL_BEST_IMAGES:-1}"
EVAL_REWARD_DEVICE="${EVAL_REWARD_DEVICE:-cuda}"
EVAL_ALLOW_MISSING_BACKENDS="${EVAL_ALLOW_MISSING_BACKENDS:-1}"

# ── Tag the run for clarity ─────────────────────────────────────────────────
echo "================================================================"
echo "[mcts-debug-local]"
echo "  backend:     ${SD35_BACKEND}  (steps=${STEPS} cfg_scales='${CFG_SCALES}')"
echo "  reward:      ${REWARD_BACKEND}  (eval=${EVAL_BACKENDS})"
echo "  prompts:     ${NUM_PROMPTS} from ${PROMPT_FILE}  seed=${SEED}"
echo "  methods:     ${METHODS}"
echo "  bon_mcts:    n_seeds=${BON_MCTS_N_SEEDS} topk=${BON_MCTS_TOPK} alloc=${BON_MCTS_SIM_ALLOC} min_sims=${BON_MCTS_MIN_SIMS}"
echo "               n_sims=${N_SIMS}  refine=${BON_MCTS_REFINE_METHOD}  ucb_c=${UCB_C}"
echo "  output:      ${RUN_ROOT}"
echo "  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "================================================================"

# Quick GPU sanity print.
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>/dev/null || true

# ── Export everything the suite reads ───────────────────────────────────────
export SD35_BACKEND STEPS BASELINE_CFG CFG_SCALES
export PROMPT_FILE START_INDEX=0 END_INDEX="${NUM_PROMPTS}"
export SEED
export METHODS N_VARIANTS USE_QWEN PRECOMPUTE_REWRITES QWEN_ID
export CORRECTION_STRENGTHS
export SAVE_BEST_IMAGES SAVE_IMAGES SAVE_VARIANTS
export EVAL_BACKENDS EVAL_BEST_IMAGES EVAL_REWARD_DEVICE EVAL_ALLOW_MISSING_BACKENDS
export REWARD_BACKEND
export REWARD_TYPE="${REWARD_BACKEND}"
export REWARD_BACKENDS="${REWARD_BACKEND}"
export N_SIMS UCB_C
export BON_MCTS_N_SEEDS BON_MCTS_TOPK BON_MCTS_SIM_ALLOC BON_MCTS_MIN_SIMS BON_MCTS_REFINE_METHOD
export OUT_ROOT="${RUN_ROOT}"
export NUM_GPUS

# ── Run via the SD3.5 suite (handles all dispatch + DDP=1 single process) ──
bash "${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"

echo
echo "================================================================"
echo "[mcts-debug-local] DONE."
echo "  Outputs:    ${RUN_ROOT}"
echo "  Suite TSV:  ${RUN_ROOT}/run_*/suite_summary.tsv"
echo "  Aggregate:  find ${RUN_ROOT} -name 'aggregate_ddp.json'"
echo "================================================================"
