#!/usr/bin/env bash
# Two MCTS branching-factor ablations in one bash:
#
#   A) cfg_bank        : vary CFG bank size, hold prompt bank fixed (N_VARIANTS=1)
#                        cells: cfg_bank_1, cfg_bank_4, cfg_bank_7
#   B) prompt_bank     : vary N_VARIANTS, hold CFG bank fixed (size 4: 1.0 1.5 2.0 2.5)
#                        cells: prompt_bank_1, prompt_bank_3, prompt_bank_5
#
# Each cell × backend × seed dispatches one suite invocation. One bash, one
# AMLT job, one reward server boot. ImageReward is the search reward;
# HPSv3+PickScore are added in phase-2 posthoc.
#
# Caller env (typically AMLT yaml):
#   RUN_ROOT, REWARD_SERVER_URL
#
# Optional:
#   BACKENDS  (default "sid sd35_base")
#   CELLS_CFG (default "cfg_bank_1 cfg_bank_4 cfg_bank_7")
#   CELLS_PB  (default "prompt_bank_1 prompt_bank_3 prompt_bank_5")
#   SEEDS     (default "42")
#   N_PROMPTS (default 100)
#   FAIL_FAST (default 0)
#   SKIP_CFG  (default 0; 1 to skip ablation A)
#   SKIP_PB   (default 0; 1 to skip ablation B — handy if Qwen unavailable)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "cfg-prompt-bank-ablation"

: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${REWARD_SERVER_URL:?REWARD_SERVER_URL must be set}"

BACKENDS="${BACKENDS:-sid senseflow_large sd35_base flux_schnell}"
CELLS_CFG="${CELLS_CFG:-cfg_bank_1 cfg_bank_4 cfg_bank_7}"
CELLS_PB="${CELLS_PB:-prompt_bank_1 prompt_bank_3 prompt_bank_5}"
FAIL_FAST="${FAIL_FAST:-0}"
SKIP_CFG="${SKIP_CFG:-0}"
SKIP_PB="${SKIP_PB:-0}"

N_PROMPTS="${N_PROMPTS:-50}"
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
# Phase-1 eval: HPSv3 + PickScore are added posthoc; phase-1 only IR (cheaper).
export EVAL_BACKENDS="imagereward"
export REWARD_BACKEND="imagereward"
export REWARD_TYPE="imagereward"
export REWARD_BACKENDS="imagereward"
export EVAL_ALLOW_MISSING_BACKENDS=1
export EVAL_BEST_IMAGES=1
export EVAL_REWARD_DEVICE=cuda

# Anchored bon_mcts knobs (matches default cell of mcts_param ablation).
export N_SIMS=30
export UCB_C=1.0
export BON_MCTS_N_SEEDS=8
export BON_MCTS_TOPK=2
export BON_MCTS_SIM_ALLOC=split
export BON_MCTS_MIN_SIMS=8
export BON_MCTS_REFINE_METHOD=mcts          # vanilla refine

PROMPTS_DIR="${RUN_ROOT}/_prompts"
mkdir -p "${PROMPTS_DIR}"

# ── Helpers ────────────────────────────────────────────────────────────────
# Per-backend env: which suite, model knobs, CFG family.
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
            ;;
        flux_schnell)
            export FLUX_BACKEND=flux; unset SD35_BACKEND || true
            export STEPS=4
            export MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-schnell}"
            export BASELINE_GUIDANCE_SCALE=0.0
            export BASELINE_CFG=0.0
            export _SUITE="${SCRIPT_DIR}/hpsv2_flux_schnell_ddp_suite.sh"
            export _CFG_FAMILY=flux_distilled    # CFG-distilled, no CFG bank to vary
            ;;
        *) echo "[cpb-ablation] ERROR unknown backend '$1'" >&2; return 1 ;;
    esac
}

_sample_prompts() {
    local backend="$1"
    local prompt_file="${PROMPTS_DIR}/backend_${backend}.txt"
    if [[ ! -f "${prompt_file}" ]]; then
        echo "[cpb-ablation] sampling prompts → ${prompt_file}"
        env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
            "${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_prompts.py" \
            --n_prompts "${N_PROMPTS}" \
            --out_dir "${PROMPTS_DIR}" \
            --backends "${backend}"
    fi
    export PROMPT_FILE="${prompt_file}"
}

_apply_cfg_cell() {
    # Hold prompt bank fixed at 1 (no Qwen needed); vary CFG bank size.
    # NOTE: flux_schnell is CFG-distilled (CFG=0 fixed) → cfg-bank ablation
    # is degenerate for it; the main loop skips this phase for flux_schnell.
    export USE_QWEN=0
    export PRECOMPUTE_REWRITES=0
    export N_VARIANTS=1
    case "${_CFG_FAMILY}" in
        sd35_small)
            case "$1" in
                cfg_bank_1) export CFG_SCALES="1.5" ;;
                cfg_bank_4) export CFG_SCALES="1.0 1.5 2.0 2.5" ;;
                cfg_bank_7) export CFG_SCALES="1.0 1.25 1.5 1.75 2.0 2.25 2.5" ;;
                *) echo "[cpb-ablation] ERROR unknown cfg cell '$1'" >&2; return 1 ;;
            esac
            ;;
        sd35_base)
            case "$1" in
                cfg_bank_1) export CFG_SCALES="4.5" ;;
                cfg_bank_4) export CFG_SCALES="3.5 4.5 5.5 7.0" ;;
                cfg_bank_7) export CFG_SCALES="3.5 4.0 4.5 5.0 5.5 6.0 7.0" ;;
                *) echo "[cpb-ablation] ERROR unknown cfg cell '$1'" >&2; return 1 ;;
            esac
            ;;
        flux_distilled)
            echo "[cpb-ablation] WARN cfg cell on flux_distilled is degenerate; skipping"
            return 2
            ;;
    esac
}

_apply_pb_cell() {
    # Hold CFG bank fixed at size 4; vary N_VARIANTS (prompt bank).
    export USE_QWEN=1
    export PRECOMPUTE_REWRITES=1
    case "${_CFG_FAMILY}" in
        sd35_small)     export CFG_SCALES="1.0 1.5 2.0 2.5" ;;
        sd35_base)      export CFG_SCALES="3.5 4.5 5.5 7.0" ;;
        flux_distilled) export CFG_SCALES="0.0" ;;   # flux: no real cfg dim, single-action cfg
    esac
    case "$1" in
        prompt_bank_1) export N_VARIANTS=1 ;;
        prompt_bank_3) export N_VARIANTS=3 ;;
        prompt_bank_5) export N_VARIANTS=5 ;;
        *) echo "[cpb-ablation] ERROR unknown prompt_bank cell '$1'" >&2; return 1 ;;
    esac
}

# Verify Qwen rewrite model is in cache (transformers reads $HF_HOME/hub/...).
_ensure_qwen_cached() {
    local repo="${QWEN_ID:-Qwen/Qwen3-4B}"
    local cache_root="${HF_HOME:-${HOME}/.cache/huggingface}"
    local hub_dir="${cache_root}/hub/models--${repo//\//--}"
    if ls "${hub_dir}/snapshots/"*/config.json >/dev/null 2>&1; then
        return 0
    fi
    echo "[cpb-ablation] cache MISS for ${repo} → fetching to ${cache_root}/hub/"
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_HOME="${cache_root}" \
        "${PYTHON_BIN}" -c "
import os, sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1],
                  token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'),
                  resume_download=True, max_workers=2)
print('[cpb-ablation] Qwen prefetch OK:', sys.argv[1])
" "${repo}"
    if ! ls "${hub_dir}/snapshots/"*/config.json >/dev/null 2>&1; then
        echo "[cpb-ablation] FATAL: ${repo} still not at ${hub_dir}/snapshots/*/config.json" >&2
        return 1
    fi
}

# ── Main loop ───────────────────────────────────────────────────────────────
echo "================================================================"
echo "[cpb-ablation]  cfg-bank + prompt-bank branching-factor ablation"
echo "  backends=${BACKENDS}  N_PROMPTS=${N_PROMPTS}  SEEDS=${SEEDS}"
echo "  CELLS_CFG=${CELLS_CFG}  SKIP_CFG=${SKIP_CFG}"
echo "  CELLS_PB=${CELLS_PB}    SKIP_PB=${SKIP_PB}"
echo "  reward=imagereward (search), eval=imagereward (phase-1)"
echo "================================================================"

failed=()
for backend in ${BACKENDS}; do
    _apply_backend "${backend}"
    _sample_prompts "${backend}"

    # ── Ablation A: vary cfg bank, hold prompt bank fixed (N_VARIANTS=1) ────
    # Skipped for flux_schnell (CFG-distilled → cfg-bank degenerate).
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
                echo "[cpb-ablation] (A) backend=${backend}  cell=${cell}  seed=${seed}"
                echo "  CFG_SCALES='${CFG_SCALES}'  N_VARIANTS=${N_VARIANTS}  USE_QWEN=${USE_QWEN}"
                echo "================================================================"

                if SEED="${seed}" bash "${_SUITE}"; then
                    echo "[cpb-ablation] OK ${backend}/${cell}/seed${seed}"
                else
                    rc=$?
                    echo "[cpb-ablation] FAIL ${backend}/${cell}/seed${seed} rc=${rc}" >&2
                    failed+=("${backend}/${cell}/seed${seed}")
                    if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
                fi
            done
        done
    elif [[ "${_CFG_FAMILY}" == "flux_distilled" ]]; then
        echo "[cpb-ablation] backend=${backend}: skipping cfg_bank phase (flux is CFG-distilled)"
    fi

    # ── Ablation B: vary prompt bank, hold cfg bank fixed (size 4) ──────────
    if [[ "${SKIP_PB}" != "1" ]]; then
        if ! _ensure_qwen_cached; then
            echo "[cpb-ablation] FAIL Qwen prefetch — skipping prompt_bank for ${backend}" >&2
            failed+=("${backend}/prompt_bank/qwen-prefetch")
            continue
        fi
        # Per-backend rewrites cache file (5 variants is the largest cell;
        # smaller cells will use the first N).
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
                echo "[cpb-ablation] (B) backend=${backend}  cell=${cell}  seed=${seed}"
                echo "  CFG_SCALES='${CFG_SCALES}'  N_VARIANTS=${N_VARIANTS}  QWEN_ID=${QWEN_ID}"
                echo "  rewrites_cache=${REWRITES_FILE}"
                echo "================================================================"

                if SEED="${seed}" bash "${_SUITE}"; then
                    echo "[cpb-ablation] OK ${backend}/${cell}/seed${seed}"
                else
                    rc=$?
                    echo "[cpb-ablation] FAIL ${backend}/${cell}/seed${seed} rc=${rc}" >&2
                    failed+=("${backend}/${cell}/seed${seed}")
                    if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
                fi
            done
        done
    fi
done

if (( ${#failed[@]} > 0 )); then
    echo "[cpb-ablation] phase-1 DONE with failures: ${failed[*]}"
fi
echo "[cpb-ablation] phase-1 DONE."

# ── Phase-2 posthoc: HPSv3 + PickScore on saved best_images ─────────────────
if [[ "${RUN_POSTHOC:-1}" == "1" ]]; then
    echo
    echo "================================================================"
    echo "[cpb-ablation] phase-2 posthoc: hpsv3 + pickscore on saved best_images"
    echo "================================================================"
    posthoc_failed=()
    for backend in ${BACKENDS}; do
        for cell in ${CELLS_CFG} ${CELLS_PB}; do
            for seed in ${SEEDS}; do
                cell_root="${RUN_ROOT}/${backend}/${cell}/seed${seed}"
                [[ -d "${cell_root}" ]] || continue
                for method_dir in $(find "${cell_root}" -maxdepth 3 -type d -name 'bon_mcts' 2>/dev/null); do
                    echo "[posthoc] eval ${method_dir}"
                    if "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_best_images_multi_reward.py" \
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
        echo "[cpb-ablation] phase-2 posthoc WARN failures: ${posthoc_failed[*]}"
    else
        echo "[cpb-ablation] phase-2 posthoc DONE."
    fi
fi

echo "[cpb-ablation] ALL DONE."
echo "  Output: ${RUN_ROOT}"
echo "  Compare: python3 mcts_param_compare.py --root ${RUN_ROOT} --cells ${CELLS_CFG} ${CELLS_PB}"
if (( ${#failed[@]} > 0 )); then exit 1; fi
