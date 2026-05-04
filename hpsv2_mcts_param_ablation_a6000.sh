#!/usr/bin/env bash
# A6000-tuned MCTS hyperparam ablation (sibling of hpsv2_mcts_param_ablation.sh).
#
# Differences from the A100 version (don't touch that one):
#   - Lighter N_SIMS sweep (15/30/60/90 instead of 15/30/60/120) — A6000 is
#     roughly 3-4× slower than A100 per transformer call, so cap the heaviest
#     cell at 90 to keep wallclock reasonable.
#   - Single-GPU default (no DDP); user overrides CUDA_VISIBLE_DEVICES.
#   - Default to N_PROMPTS=30 (vs A100's 100) to keep one full ablation
#     under ~6h on a single A6000.
#   - Same phase-1 (ImageReward inline, no reward server) + phase-2 posthoc
#     (HPSv3 + PickScore inline) layout as the A100 version.
#
# Anchors (held fixed across cells, unchanged from A100):
#   refine_method = mcts (vanilla, the prescreen-ablation winner)
#   reward        = imagereward (search + phase-1 eval)
#   N_SEEDS       = 8
#   refine_alloc  = split, min_sims=8

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "mcts-param-ablation-a6000"

: "${RUN_ROOT:?RUN_ROOT must be set}"

# REWARD_SERVER_URL is optional; unset → reward_unified loads ImageReward inline.
if [[ -z "${REWARD_SERVER_URL:-}" ]]; then
    echo "[a6000-ablation] REWARD_SERVER_URL not set → ImageReward loads inline (local mode)"
    unset REWARD_SERVER_URL || true
fi

BACKENDS="${BACKENDS:-sid}"                                        # default sid only (4-step, fastest on A6000)
CELLS="${CELLS:-default n_sims_15 n_sims_60 n_sims_90 topk_1 topk_4 ucb_c_0.5 ucb_c_2.0}"
FAIL_FAST="${FAIL_FAST:-0}"

N_PROMPTS="${N_PROMPTS:-30}"                                       # lighter than A100's 100
SEEDS="${SEEDS:-42 43}"                                            # 2 seeds default (vs A100's 4)
PYTHON_BIN="${PYTHON_BIN:-python}"

# ── A6000-friendly defaults ────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -z "${NUM_GPUS:-}" ]]; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')
fi
export NUM_GPUS
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

# ── Shared run knobs ────────────────────────────────────────────────────────
export METHODS="baseline bon_mcts"
export START_INDEX=0
export END_INDEX="${N_PROMPTS}"
export N_VARIANTS=1
export USE_QWEN=0
export PRECOMPUTE_REWRITES=0
export REWARDS_OVERWRITE=0
export CORRECTION_STRENGTHS="0.0"
export SAVE_BEST_IMAGES="${SAVE_BEST_IMAGES:-1}"
export SAVE_IMAGES="${SAVE_IMAGES:-0}"
export SAVE_VARIANTS="${SAVE_VARIANTS:-0}"
export EVAL_BACKENDS="${EVAL_BACKENDS:-imagereward}"
export REWARD_BACKEND="${REWARD_BACKEND:-imagereward}"
export REWARD_TYPE="${REWARD_TYPE:-imagereward}"
export REWARD_BACKENDS="${REWARD_BACKENDS:-imagereward}"
export EVAL_ALLOW_MISSING_BACKENDS="${EVAL_ALLOW_MISSING_BACKENDS:-1}"
export EVAL_BEST_IMAGES="${EVAL_BEST_IMAGES:-1}"
export EVAL_REWARD_DEVICE="${EVAL_REWARD_DEVICE:-cuda}"

# Anchored bon_mcts knobs.
export BON_MCTS_N_SEEDS=8
export BON_MCTS_SIM_ALLOC=split
export BON_MCTS_REFINE_METHOD=mcts          # vanilla refine

PROMPTS_DIR="${RUN_ROOT}/_prompts"
mkdir -p "${PROMPTS_DIR}"

# ── Per-cell knob assignment (A6000-tuned: caps heaviest n_sims at 90) ─────
_apply_cell() {
    export N_SIMS=30
    export UCB_C=1.0
    export BON_MCTS_TOPK=2
    export BON_MCTS_MIN_SIMS=8

    case "$1" in
        default)        : ;;
        n_sims_15)      export N_SIMS=15 ;;
        n_sims_60)      export N_SIMS=60 ;;
        n_sims_90)      export N_SIMS=90 ;;            # A6000 cap (vs A100's 120)
        topk_1)         export BON_MCTS_TOPK=1; export BON_MCTS_MIN_SIMS=8 ;;
        topk_4)         export BON_MCTS_TOPK=4; export BON_MCTS_MIN_SIMS=4 ;;
        ucb_c_0.5)      export UCB_C=0.5 ;;
        ucb_c_2.0)      export UCB_C=2.0 ;;
        *)
            echo "[a6000-ablation] ERROR unknown cell '$1'" >&2; return 1 ;;
    esac
}

_apply_backend() {
    export SD35_BACKEND="$1"
    case "$1" in
        sid)               export STEPS=4;  export BASELINE_CFG=1.0; export CFG_SCALES="1.0" ;;
        senseflow_large)   export STEPS=4;  export BASELINE_CFG=1.0; export CFG_SCALES="1.0" ;;
        sd35_base)         export STEPS=28; export BASELINE_CFG=4.5; export CFG_SCALES="3.5 4.0 4.5 5.0 5.5 6.0 7.0"
                           echo "[a6000-ablation] WARN sd35_base on A6000 is ~5-10× slower than sid; expect long wallclock" ;;
        *) echo "[a6000-ablation] ERROR unknown backend '$1'" >&2; return 1 ;;
    esac
}

_sample_prompts() {
    local backend="$1"
    local prompt_file="${PROMPTS_DIR}/backend_${backend}.txt"
    if [[ ! -f "${prompt_file}" ]]; then
        echo "[a6000-ablation] sampling prompts → ${prompt_file}"
        env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
            "${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_prompts.py" \
            --n_prompts "${N_PROMPTS}" \
            --out_dir "${PROMPTS_DIR}" \
            --backends "${backend}"
    fi
    export PROMPT_FILE="${prompt_file}"
}

# ── Banner ──────────────────────────────────────────────────────────────────
echo "================================================================"
echo "[a6000-ablation]  (lighter defaults than A100 sibling script)"
echo "  backends=${BACKENDS}  cells=${CELLS}"
echo "  N_PROMPTS=${N_PROMPTS}  SEEDS=${SEEDS}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  NUM_GPUS=${NUM_GPUS}"
echo "  reward=imagereward (inline; no server)"
echo "================================================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>/dev/null || true

# ── Main loop ───────────────────────────────────────────────────────────────
failed=()
for backend in ${BACKENDS}; do
    _apply_backend "${backend}"
    _sample_prompts "${backend}"

    for cell in ${CELLS}; do
        if ! _apply_cell "${cell}"; then
            failed+=("${backend}/${cell}/bad-cell")
            continue
        fi

        for seed in ${SEEDS}; do
            cell_root="${RUN_ROOT}/${backend}/${cell}/seed${seed}"
            mkdir -p "${cell_root}"
            export OUT_ROOT="${cell_root}"

            echo
            echo "================================================================"
            echo "[a6000-ablation] backend=${backend}  cell=${cell}  seed=${seed}"
            echo "  N_SIMS=${N_SIMS} UCB_C=${UCB_C} TOPK=${BON_MCTS_TOPK} MIN_SIMS=${BON_MCTS_MIN_SIMS}"
            echo "================================================================"

            if SEED="${seed}" bash "${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"; then
                echo "[a6000-ablation] OK ${backend}/${cell}/seed${seed}"
            else
                rc=$?
                echo "[a6000-ablation] FAIL ${backend}/${cell}/seed${seed} rc=${rc}" >&2
                failed+=("${backend}/${cell}/seed${seed}")
                if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
            fi
        done
    done
done

if (( ${#failed[@]} > 0 )); then
    echo "[a6000-ablation] DONE with phase-1 failures: ${failed[*]}"
fi
echo "[a6000-ablation] DONE all (${BACKENDS} × ${CELLS}) cells phase-1."

# ── Phase-2 posthoc: HPSv3 + PickScore inline on saved best_images ──────────
if [[ "${RUN_POSTHOC:-1}" == "1" ]]; then
    echo
    echo "================================================================"
    echo "[a6000-ablation] phase-2 posthoc: hpsv3 + pickscore on saved best_images"
    echo "================================================================"
    posthoc_failed=()
    for backend_name in ${BACKENDS}; do
        for cell in ${CELLS}; do
            for seed in ${SEEDS}; do
                cell_root="${RUN_ROOT}/${backend_name}/${cell}/seed${seed}"
                [[ -d "${cell_root}" ]] || continue
                for method_dir in $(find "${cell_root}" -maxdepth 3 -type d -name 'bon_mcts' 2>/dev/null); do
                    echo "[posthoc] eval ${method_dir}"
                    if "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_best_images_multi_reward.py" \
                        --layout sd35 \
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
        echo "[a6000-ablation] phase-2 posthoc WARN failures: ${posthoc_failed[*]}"
    else
        echo "[a6000-ablation] phase-2 posthoc DONE."
    fi
fi

echo "[a6000-ablation] Compare: python3 mcts_param_compare.py --root ${RUN_ROOT}"
if (( ${#failed[@]} > 0 )); then exit 1; fi
