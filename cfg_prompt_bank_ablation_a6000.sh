#!/usr/bin/env bash
# A6000-tuned cfg-bank + prompt-bank MCTS ablation (sibling of
# cfg_prompt_bank_ablation.sh; don't touch the cluster version).
#
# Differences from the cluster sibling:
#   - No reward server. ImageReward loads inline.
#   - Phase-2 posthoc HPSv3 + PickScore runs inline on saved best_images.
#   - Lighter defaults: 30 prompts, seed 42, default backends sid+flux_schnell.
#   - DDP-aware: NUM_GPUS auto-derived from CUDA_VISIBLE_DEVICES.
#   - sd35_base excluded by default (28-step is too slow on a single A6000).
#
# Two ablations in one bash:
#   A) cfg_bank   : vary CFG bank size {1, 4, 7}; N_VARIANTS=1 fixed
#   B) prompt_bank: vary N_VARIANTS {1, 3, 5}; CFG bank size = 4 fixed
#
# flux_schnell is CFG-distilled → cfg_bank phase auto-skipped for it
# (only prompt_bank cells run).
#
# Optional env (caller-overridable):
#   RUN_ROOT          required (output dir; auto-created)
#   BACKENDS          (default "sid senseflow_large flux_schnell")
#   CELLS_CFG         (default "cfg_bank_1 cfg_bank_4 cfg_bank_7")
#   CELLS_PB          (default "prompt_bank_1 prompt_bank_3 prompt_bank_5")
#   SEEDS             (default "42")
#   N_PROMPTS         (default 30)
#   FAIL_FAST         (default 0)
#   SKIP_CFG          (default 0; 1 to skip ablation A)
#   SKIP_PB           (default 0; 1 to skip ablation B)
#   RUN_POSTHOC       (default 1; 0 to skip phase-2 hpsv3+pickscore eval)
#   QWEN_ID           (default Qwen/Qwen3-4B)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "cfg-prompt-bank-a6000"

: "${RUN_ROOT:?RUN_ROOT must be set}"

# ── A6000-friendly defaults ────────────────────────────────────────────────
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

# Inline ImageReward — no reward server.
unset REWARD_SERVER_URL || true
echo "[cpb-a6000] REWARD_SERVER_URL not set → ImageReward loads inline (local mode)"

BACKENDS="${BACKENDS:-sid senseflow_large flux_schnell}"
CELLS_CFG="${CELLS_CFG:-cfg_bank_1 cfg_bank_4 cfg_bank_7}"
CELLS_PB="${CELLS_PB:-prompt_bank_1 prompt_bank_3 prompt_bank_5}"
FAIL_FAST="${FAIL_FAST:-0}"
SKIP_CFG="${SKIP_CFG:-0}"
SKIP_PB="${SKIP_PB:-0}"

N_PROMPTS="${N_PROMPTS:-30}"
SEEDS="${SEEDS:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ── Shared knobs ────────────────────────────────────────────────────────────
export METHODS="baseline bon_mcts"
export START_INDEX=0
export END_INDEX="${N_PROMPTS}"
export REWARDS_OVERWRITE=0
export CORRECTION_STRENGTHS="0.0"
export SAVE_BEST_IMAGES=1
export SAVE_IMAGES=0
export SAVE_VARIANTS=1
export EVAL_BACKENDS="imagereward"
export REWARD_BACKEND="imagereward"
export REWARD_TYPE="imagereward"
export REWARD_BACKENDS="imagereward"
export EVAL_ALLOW_MISSING_BACKENDS=1
export EVAL_BEST_IMAGES=1
export EVAL_REWARD_DEVICE=cuda

# Anchored bon_mcts knobs (matches default cell of mcts_param ablation).
export N_SIMS="${N_SIMS:-30}"
export UCB_C="${UCB_C:-1.0}"
export BON_MCTS_N_SEEDS=8
export BON_MCTS_TOPK=2
export BON_MCTS_SIM_ALLOC=split
export BON_MCTS_MIN_SIMS=8
export BON_MCTS_REFINE_METHOD=mcts          # vanilla refine

PROMPTS_DIR="${RUN_ROOT}/_prompts"
mkdir -p "${PROMPTS_DIR}"

# ── Per-backend env: suite + CFG family ────────────────────────────────────
_apply_backend() {
    case "$1" in
        sid)
            export SD35_BACKEND=sid; unset FLUX_BACKEND || true
            export STEPS=4; export BASELINE_CFG=1.0
            export _SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
            export _CFG_FAMILY=sd35_small
            ;;
        senseflow_large)
            export SD35_BACKEND=senseflow_large; unset FLUX_BACKEND || true
            export STEPS=4; export BASELINE_CFG=1.0
            export _SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
            export _CFG_FAMILY=sd35_small
            ;;
        sd35_base)
            export SD35_BACKEND=sd35_base; unset FLUX_BACKEND || true
            export STEPS=28; export BASELINE_CFG=4.5
            export _SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
            export _CFG_FAMILY=sd35_base
            echo "[cpb-a6000] WARN sd35_base on A6000 is ~5-10× slower than sid; expect long wallclock"
            ;;
        flux_schnell)
            export FLUX_BACKEND=flux; unset SD35_BACKEND || true
            export STEPS=4
            export MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-schnell}"
            export BASELINE_GUIDANCE_SCALE=0.0; export BASELINE_CFG=0.0
            export _SUITE="${SCRIPT_DIR}/hpsv2_flux_schnell_ddp_suite.sh"
            export _CFG_FAMILY=flux_distilled
            ;;
        *) echo "[cpb-a6000] ERROR unknown backend '$1'" >&2; return 1 ;;
    esac
}

_sample_prompts() {
    local backend="$1"
    local prompt_file="${PROMPTS_DIR}/backend_${backend}.txt"
    if [[ ! -f "${prompt_file}" ]]; then
        echo "[cpb-a6000] sampling prompts → ${prompt_file}"
        env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
            "${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_prompts.py" \
            --n_prompts "${N_PROMPTS}" \
            --out_dir "${PROMPTS_DIR}" \
            --backends "${backend}" || \
            echo "[cpb-a6000] WARN sampling failed; falling back to ${SCRIPT_DIR}/hpsv2_subset.txt"
    fi
    if [[ -f "${prompt_file}" ]]; then
        export PROMPT_FILE="${prompt_file}"
    else
        export PROMPT_FILE="${SCRIPT_DIR}/hpsv2_subset.txt"
    fi
}

_apply_cfg_cell() {
    export USE_QWEN=0
    export PRECOMPUTE_REWRITES=0
    export N_VARIANTS=1
    case "${_CFG_FAMILY}" in
        sd35_small)
            case "$1" in
                cfg_bank_1) export CFG_SCALES="1.5" ;;
                cfg_bank_4) export CFG_SCALES="1.0 1.5 2.0 2.5" ;;
                cfg_bank_7) export CFG_SCALES="1.0 1.25 1.5 1.75 2.0 2.25 2.5" ;;
                *) echo "[cpb-a6000] ERROR unknown cfg cell '$1'" >&2; return 1 ;;
            esac
            ;;
        sd35_base)
            case "$1" in
                cfg_bank_1) export CFG_SCALES="4.5" ;;
                cfg_bank_4) export CFG_SCALES="3.5 4.5 5.5 7.0" ;;
                cfg_bank_7) export CFG_SCALES="3.5 4.0 4.5 5.0 5.5 6.0 7.0" ;;
                *) echo "[cpb-a6000] ERROR unknown cfg cell '$1'" >&2; return 1 ;;
            esac
            ;;
        flux_distilled)
            echo "[cpb-a6000] WARN cfg cell on flux_distilled is degenerate; skipping"
            return 2
            ;;
    esac
}

_apply_pb_cell() {
    export USE_QWEN=1
    export PRECOMPUTE_REWRITES=1
    case "${_CFG_FAMILY}" in
        sd35_small)     export CFG_SCALES="1.0 1.5 2.0 2.5" ;;
        sd35_base)      export CFG_SCALES="3.5 4.5 5.5 7.0" ;;
        flux_distilled) export CFG_SCALES="0.0" ;;
    esac
    case "$1" in
        prompt_bank_1) export N_VARIANTS=1 ;;
        prompt_bank_3) export N_VARIANTS=3 ;;
        prompt_bank_5) export N_VARIANTS=5 ;;
        *) echo "[cpb-a6000] ERROR unknown prompt_bank cell '$1'" >&2; return 1 ;;
    esac
}

# Verify Qwen rewrite model in cache (HF_HOME/hub/...).
_ensure_qwen_cached() {
    local repo="${QWEN_ID:-Qwen/Qwen3-4B}"
    local cache_root="${HF_HOME:-${HOME}/.cache/huggingface}"
    local hub_dir="${cache_root}/hub/models--${repo//\//--}"
    if ls "${hub_dir}/snapshots/"*/config.json >/dev/null 2>&1; then
        return 0
    fi
    echo "[cpb-a6000] cache MISS for ${repo} → fetching to ${cache_root}/hub/"
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_HOME="${cache_root}" \
        "${PYTHON_BIN}" -c "
import os, sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1],
                  token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'),
                  resume_download=True, max_workers=2)
print('[cpb-a6000] Qwen prefetch OK:', sys.argv[1])
" "${repo}"
    if ! ls "${hub_dir}/snapshots/"*/config.json >/dev/null 2>&1; then
        echo "[cpb-a6000] FATAL: ${repo} still not at ${hub_dir}/snapshots/*/config.json" >&2
        return 1
    fi
}

# ── Banner ──────────────────────────────────────────────────────────────────
echo "================================================================"
echo "[cpb-a6000]   cfg-bank + prompt-bank ablation (A6000, IR inline)"
echo "  backends=${BACKENDS}  N_PROMPTS=${N_PROMPTS}  SEEDS=${SEEDS}"
echo "  CELLS_CFG=${CELLS_CFG}  SKIP_CFG=${SKIP_CFG}"
echo "  CELLS_PB=${CELLS_PB}    SKIP_PB=${SKIP_PB}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  NUM_GPUS=${NUM_GPUS}"
echo "  reward=imagereward (inline)"
echo "================================================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>/dev/null || true

failed=()
for backend in ${BACKENDS}; do
    _apply_backend "${backend}"
    _sample_prompts "${backend}"

    # ── Ablation A: vary cfg bank, hold prompt bank fixed (N_VARIANTS=1) ────
    if [[ "${SKIP_CFG}" != "1" && "${_CFG_FAMILY}" != "flux_distilled" ]]; then
        for cell in ${CELLS_CFG}; do
            if ! _apply_cfg_cell "${cell}"; then
                failed+=("${backend}/${cell}/bad-cell"); continue
            fi
            for seed in ${SEEDS}; do
                cell_root="${RUN_ROOT}/${backend}/${cell}/seed${seed}"
                mkdir -p "${cell_root}"
                export OUT_ROOT="${cell_root}"

                echo
                echo "================================================================"
                echo "[cpb-a6000] (A) backend=${backend}  cell=${cell}  seed=${seed}"
                echo "  CFG_SCALES='${CFG_SCALES}'  N_VARIANTS=${N_VARIANTS}  USE_QWEN=${USE_QWEN}"
                echo "================================================================"

                if SEED="${seed}" bash "${_SUITE}"; then
                    echo "[cpb-a6000] OK ${backend}/${cell}/seed${seed}"
                else
                    rc=$?
                    echo "[cpb-a6000] FAIL ${backend}/${cell}/seed${seed} rc=${rc}" >&2
                    failed+=("${backend}/${cell}/seed${seed}")
                    if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
                fi
            done
        done
    elif [[ "${_CFG_FAMILY}" == "flux_distilled" ]]; then
        echo "[cpb-a6000] backend=${backend}: skipping cfg_bank phase (flux is CFG-distilled)"
    fi

    # ── Ablation B: vary prompt bank, hold cfg bank fixed (size 4) ──────────
    if [[ "${SKIP_PB}" != "1" ]]; then
        if ! _ensure_qwen_cached; then
            echo "[cpb-a6000] FAIL Qwen prefetch — skipping prompt_bank for ${backend}" >&2
            failed+=("${backend}/prompt_bank/qwen-prefetch")
            continue
        fi
        export QWEN_ID="${QWEN_ID:-Qwen/Qwen3-4B}"
        export REWRITES_FILE="${PROMPTS_DIR}/${backend}_qwen_rewrites.json"

        for cell in ${CELLS_PB}; do
            if ! _apply_pb_cell "${cell}"; then
                failed+=("${backend}/${cell}/bad-cell"); continue
            fi
            for seed in ${SEEDS}; do
                cell_root="${RUN_ROOT}/${backend}/${cell}/seed${seed}"
                mkdir -p "${cell_root}"
                export OUT_ROOT="${cell_root}"

                echo
                echo "================================================================"
                echo "[cpb-a6000] (B) backend=${backend}  cell=${cell}  seed=${seed}"
                echo "  CFG_SCALES='${CFG_SCALES}'  N_VARIANTS=${N_VARIANTS}  QWEN_ID=${QWEN_ID}"
                echo "  rewrites_cache=${REWRITES_FILE}"
                echo "================================================================"

                if SEED="${seed}" bash "${_SUITE}"; then
                    echo "[cpb-a6000] OK ${backend}/${cell}/seed${seed}"
                else
                    rc=$?
                    echo "[cpb-a6000] FAIL ${backend}/${cell}/seed${seed} rc=${rc}" >&2
                    failed+=("${backend}/${cell}/seed${seed}")
                    if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
                fi
            done
        done
    fi
done

if (( ${#failed[@]} > 0 )); then
    echo "[cpb-a6000] phase-1 DONE with failures: ${failed[*]}"
fi
echo "[cpb-a6000] phase-1 DONE."

# ── Phase-2 posthoc: HPSv3 + PickScore on saved best_images ─────────────────
if [[ "${RUN_POSTHOC:-1}" == "1" ]]; then
    echo
    echo "================================================================"
    echo "[cpb-a6000] phase-2 posthoc: hpsv3 + pickscore on saved best_images"
    echo "================================================================"
    posthoc_failed=()
    for backend in ${BACKENDS}; do
        case "${backend}" in
            flux*) layout=flux ;;
            *)     layout=sd35 ;;
        esac
        for cell in ${CELLS_CFG} ${CELLS_PB}; do
            for seed in ${SEEDS}; do
                cell_root="${RUN_ROOT}/${backend}/${cell}/seed${seed}"
                [[ -d "${cell_root}" ]] || continue
                for method_dir in $(find "${cell_root}" -maxdepth 3 -type d -name 'bon_mcts' 2>/dev/null); do
                    echo "[posthoc] eval ${method_dir} (layout=${layout})"
                    if "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_best_images_multi_reward.py" \
                        --layout "${layout}" \
                        --method_out "${method_dir}" \
                        --method bon_mcts \
                        --backends hpsv3 pickscore \
                        --reward_device cuda \
                        --out_json "${method_dir}/best_images_posthoc.json" \
                        --out_aggregate "${method_dir}/best_images_posthoc_aggregate.json" \
                        --allow_missing_backends; then
                        :
                    else
                        echo "[posthoc] WARN failed: ${method_dir}" >&2
                        posthoc_failed+=("${cell}/seed${seed}")
                    fi
                done
            done
        done
    done
    if (( ${#posthoc_failed[@]} > 0 )); then
        echo "[cpb-a6000] phase-2 posthoc WARN failures: ${posthoc_failed[*]}"
    else
        echo "[cpb-a6000] phase-2 posthoc DONE."
    fi
fi

echo "[cpb-a6000] ALL DONE."
echo "  Output: ${RUN_ROOT}"
if (( ${#failed[@]} > 0 )); then exit 1; fi
