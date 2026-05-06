#!/usr/bin/env bash
# A6000-tuned cherry-pick driver (sibling of run_cherry_pick.sh).
#
# Same flow as the cluster script — sample prompts → run suite at multiple
# seeds with METHODS="bon smc fksteering dts_star bon_mcts" → cherry_pick_select
# winners — but with no reward server. ImageReward (search reward) + HPSv3
# (eval reward) load inline. Lighter defaults for single-box A6000.
#
# Differences from run_cherry_pick.sh:
#   - No REWARD_SERVER_URL requirement; suite uses inline reward backends.
#   - DDP-aware: NUM_GPUS auto-derived from CUDA_VISIBLE_DEVICES.
#   - Defaults: N_PROMPTS=30, SEEDS="42 43" (vs cluster's 100 × 4 seeds).
#
# Required:
#   BACKEND   - one of {sid, senseflow_large, sd35_base, flux_schnell}
#   RUN_ROOT  - output dir parent
#
# Optional:
#   N_PROMPTS  (default 30)
#   SEEDS      (default "42 43")
#   N_WINNERS  (default 8)
#   METHODS    (default "bon smc fksteering dts_star bon_mcts")
#   FAIL_FAST  (default 0)
#
# Memory note (A6000 48GB, 4-step distilled): SD3.5 ~17 + IR ~3 + HPSv3 ~5
# + activations ~5 ≈ 30 GB. Comfortable.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "cherry-pick-a6000"

: "${BACKEND:?BACKEND must be set (sid, senseflow_large, sd35_base, flux_schnell)}"
: "${RUN_ROOT:?RUN_ROOT must be set}"

# ── A6000-friendly defaults ─────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -z "${NUM_GPUS:-}" ]]; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')
fi
export NUM_GPUS
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SID_FORCE_WANDB_STUB=1
export WANDB_DISABLED=true
# SD3.5 prompt-encoding memory: keep encoders on GPU only during encode_prompt
# and push them back to CPU after. Transformer stays on GPU (the swap was
# unsafe — some submodules don't follow .to(device) cleanly under DDP).
export OFFLOAD_TEXT_ENCODER_AFTER_ENCODE="${OFFLOAD_TEXT_ENCODER_AFTER_ENCODE:-1}"
# Lower T5 max_sequence_length from 256 → 128 to halve T5 attention/MLP memory
# (most prompts <128 tokens; minor truncation at the very long ones).
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"

# Inline mode — no reward server.
unset REWARD_SERVER_URL || true
echo "[cherry-a6000] REWARD_SERVER_URL unset → ImageReward + HPSv3 load inline"

N_PROMPTS="${N_PROMPTS:-30}"
SEEDS="${SEEDS:-42 43}"
N_WINNERS="${N_WINNERS:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FAIL_FAST="${FAIL_FAST:-0}"

# Search reward is also the prompt-subset tag → distinct (backend, reward)
# combinations get NON-OVERLAPPING prompt subsets.
SEARCH_REWARD="${SEARCH_REWARD:-imagereward}"
case "${SEARCH_REWARD}" in
    imagereward|hpsv3) : ;;
    *) echo "[cherry-a6000] ERROR unknown SEARCH_REWARD='${SEARCH_REWARD}'" >&2; exit 1 ;;
esac

PROMPTS_DIR="${RUN_ROOT}/_prompts"
mkdir -p "${PROMPTS_DIR}"
PROMPT_FILE="${PROMPTS_DIR}/backend_${BACKEND}_${SEARCH_REWARD}.txt"

# ── Step 1: sample prompts (per (backend, search_reward) → unique subset) ───
if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "[cherry-a6000] sampling prompts → ${PROMPT_FILE}"
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_prompts.py" \
        --n_prompts "${N_PROMPTS}" \
        --out_dir "${PROMPTS_DIR}" \
        --backends "${BACKEND}" \
        --tag "${SEARCH_REWARD}"
else
    echo "[cherry-a6000] reusing ${PROMPT_FILE}"
fi

# ── Step 2: per-seed suite runs ─────────────────────────────────────────────
case "${BACKEND}" in
    sid)
        SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"; SUITE_KIND="sd35"
        export SD35_BACKEND=sid; unset FLUX_BACKEND || true
        export STEPS="${STEPS:-4}"; export BASELINE_CFG="${BASELINE_CFG:-1.0}"
        export CFG_SCALES="${CFG_SCALES:-1.0 1.5 2.0 2.5}"
        ;;
    senseflow_large)
        SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"; SUITE_KIND="sd35"
        export SD35_BACKEND=senseflow_large; unset FLUX_BACKEND || true
        export STEPS="${STEPS:-4}"; export BASELINE_CFG="${BASELINE_CFG:-1.0}"
        export CFG_SCALES="${CFG_SCALES:-1.0 1.5 2.0 2.5}"
        ;;
    sd35_base)
        SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"; SUITE_KIND="sd35"
        export SD35_BACKEND=sd35_base; unset FLUX_BACKEND || true
        export STEPS="${STEPS:-28}"; export BASELINE_CFG="${BASELINE_CFG:-4.5}"
        export CFG_SCALES="${CFG_SCALES:-3.5 4.0 4.5 5.0 5.5 6.0 7.0}"
        echo "[cherry-a6000] WARN sd35_base on a single A6000 is slow (~5-10× sid); expect long wallclock"
        ;;
    flux_schnell)
        SUITE="${SCRIPT_DIR}/hpsv2_flux_schnell_ddp_suite.sh"; SUITE_KIND="flux"
        export FLUX_BACKEND=flux; unset SD35_BACKEND || true
        export STEPS="${STEPS:-4}"
        export MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-schnell}"
        # FLUX.1-schnell is CFG-distilled at 0; mild quality degrades at higher
        # CFG, but we need a non-degenerate bank to give MCTS / fksteering / SoP
        # a real action space. Baseline still runs at cfg=0 (the natural mode).
        export BASELINE_GUIDANCE_SCALE="${BASELINE_GUIDANCE_SCALE:-0.0}"
        export BASELINE_CFG="${BASELINE_CFG:-0.0}"
        export CFG_SCALES="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0}"
        # DTS at Euler-mode flux is deterministic without SDE noise → tree
        # collapses at non-root nodes. Enable mild SDE noise so dts_star can
        # actually branch.
        export DTS_SDE_NOISE_SCALE="${DTS_SDE_NOISE_SCALE:-0.1}"
        ;;
    *)
        echo "[cherry-a6000] ERROR unknown BACKEND=${BACKEND}" >&2
        exit 2 ;;
esac

# Default suite knobs. Narrow comparison: bon_mcts vs base + fksteering + dts*.
export METHODS="${METHODS:-baseline fksteering dts_star bon_mcts}"
export PROMPT_FILE
export START_INDEX="${START_INDEX:-0}"
export END_INDEX="${END_INDEX:-${N_PROMPTS}}"
export N_VARIANTS="${N_VARIANTS:-1}"
export USE_QWEN="${USE_QWEN:-0}"
export PRECOMPUTE_REWRITES="${PRECOMPUTE_REWRITES:-0}"
export REWARDS_OVERWRITE="${REWARDS_OVERWRITE:-0}"
export CORRECTION_STRENGTHS="${CORRECTION_STRENGTHS:-0.0}"
export SAVE_BEST_IMAGES="${SAVE_BEST_IMAGES:-1}"
export SAVE_IMAGES="${SAVE_IMAGES:-0}"
export SAVE_VARIANTS="${SAVE_VARIANTS:-0}"
export EVAL_BACKENDS="${EVAL_BACKENDS:-imagereward hpsv3}"
export REWARD_BACKEND="${REWARD_BACKEND:-${SEARCH_REWARD}}"
export REWARD_TYPE="${REWARD_TYPE:-${SEARCH_REWARD}}"
export REWARD_BACKENDS="${REWARD_BACKENDS:-${SEARCH_REWARD}}"
export EVAL_BEST_IMAGES="${EVAL_BEST_IMAGES:-1}"
export EVAL_REWARD_DEVICE="${EVAL_REWARD_DEVICE:-cuda}"
export EVAL_ALLOW_MISSING_BACKENDS="${EVAL_ALLOW_MISSING_BACKENDS:-1}"

# ── Banner ──────────────────────────────────────────────────────────────────
echo "================================================================"
echo "[cherry-a6000]   cherry-pick on A6000 (no reward server)"
echo "  BACKEND=${BACKEND}  RUN_ROOT=${RUN_ROOT}"
echo "  N_PROMPTS=${N_PROMPTS}  SEEDS=${SEEDS}  N_WINNERS=${N_WINNERS}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  NUM_GPUS=${NUM_GPUS}"
echo "  METHODS=${METHODS}"
echo "  reward=${REWARD_BACKEND}  eval_backends=${EVAL_BACKENDS}"
echo "================================================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>/dev/null || true

failed=()
for seed in ${SEEDS}; do
    seed_root="${RUN_ROOT}/${BACKEND}_${SEARCH_REWARD}/seed${seed}"
    mkdir -p "${seed_root}"
    echo
    echo "================================================================"
    echo "[cherry-a6000] backend=${BACKEND}  seed=${seed} → ${seed_root}"
    echo "================================================================"
    if SEED="${seed}" OUT_ROOT="${seed_root}" bash "${SUITE}"; then
        echo "[cherry-a6000] OK ${BACKEND} seed=${seed}"
    else
        rc=$?
        echo "[cherry-a6000] FAIL ${BACKEND} seed=${seed} rc=${rc}" >&2
        failed+=("${seed}")
        if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
    fi
done

# ── Step 3: aggregate winners across all seeds ──────────────────────────────
selector_root="${RUN_ROOT}/${BACKEND}_${SEARCH_REWARD}"
selector_out="${RUN_ROOT}/${BACKEND}_${SEARCH_REWARD}/_winners"
echo
echo "================================================================"
echo "[cherry-a6000] selecting top-${N_WINNERS} winners from ${selector_root}"
echo "================================================================"
"${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_select.py" \
    --run_root "${selector_root}" \
    --out_dir "${selector_out}" \
    --n_winners "${N_WINNERS}" \
    --methods ${METHODS} || \
    echo "[cherry-a6000] WARN selector exited non-zero (no winners?)"

echo
echo "[cherry-a6000] DONE backend=${BACKEND}"
echo "  prompts:   ${PROMPT_FILE}"
echo "  winners:   ${selector_out}/winners/"
echo "  manifest:  ${selector_out}/winners.json"
if (( ${#failed[@]} > 0 )); then
    echo "[cherry-a6000] WARN seed failures: ${failed[*]}"
    exit 1
fi
