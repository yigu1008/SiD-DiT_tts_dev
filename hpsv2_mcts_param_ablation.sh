#!/usr/bin/env bash
# Single-bash driver for the MCTS-hyperparam ablation around bon_mcts.
#
# Anchors:
#   - refine_method = mcts (vanilla — strongest in the prescreen ablation)
#   - search_reward = imagereward
#   - N_PROMPTS=100, SEEDS="42 43 44 45"
#   - BON_MCTS_N_SEEDS=8 (fixed; not ablated this round)
#
# Per-cell knobs (one-at-a-time around `default`):
#   default      : N_SIMS=30 TOPK=2 UCB_C=1.0 MIN_SIMS=8
#   n_sims_15    : N_SIMS=15
#   n_sims_60    : N_SIMS=60
#   n_sims_120   : N_SIMS=120
#   topk_1       : TOPK=1                           (rest = default)
#   topk_4       : TOPK=4 MIN_SIMS=4                (rest = default)
#   ucb_c_0.5    : UCB_C=0.5                        (rest = default)
#   ucb_c_2.0    : UCB_C=2.0                        (rest = default)
#
# Caller env (typically AMLT yaml):
#   RUN_ROOT, REWARD_SERVER_URL
#
# Optional:
#   BACKENDS  (default "sid senseflow_large")
#   CELLS     (default all 8)
#   FAIL_FAST (default 0)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "mcts-param-ablation"

: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${REWARD_SERVER_URL:?REWARD_SERVER_URL must be set}"

BACKENDS="${BACKENDS:-sid senseflow_large}"
CELLS="${CELLS:-default n_sims_15 n_sims_60 n_sims_120 topk_1 topk_4 ucb_c_0.5 ucb_c_2.0}"
FAIL_FAST="${FAIL_FAST:-0}"

N_PROMPTS="${N_PROMPTS:-100}"
SEEDS="${SEEDS:-42 43 44 45}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ── Shared run knobs ────────────────────────────────────────────────────────
export METHODS="baseline bon_mcts"
export START_INDEX=0
export END_INDEX="${N_PROMPTS}"
export N_VARIANTS=1
export USE_QWEN=0
export PRECOMPUTE_REWRITES=0
export REWARDS_OVERWRITE=0
export CORRECTION_STRENGTHS="0.0"
export SAVE_BEST_IMAGES=1
export SAVE_IMAGES=0
export SAVE_VARIANTS=0
export EVAL_BACKENDS="imagereward hpsv3"
export REWARD_BACKEND="imagereward"
export REWARD_TYPE="imagereward"
export REWARD_BACKENDS="imagereward"

# Anchored bon_mcts knobs.
export BON_MCTS_N_SEEDS=8
export BON_MCTS_SIM_ALLOC=split
export BON_MCTS_REFINE_METHOD=mcts          # vanilla refine (winner from prescreen ablation)

# Anchored prompt file; sample once per backend (reuse cherry-pick prompts).
PROMPTS_DIR="${RUN_ROOT}/_prompts"
mkdir -p "${PROMPTS_DIR}"

# ── Per-cell knob assignment (env vars only) ────────────────────────────────
_apply_cell() {
    # Reset to defaults first.
    export N_SIMS=30
    export UCB_C=1.0
    export BON_MCTS_TOPK=2
    export BON_MCTS_MIN_SIMS=8

    case "$1" in
        default)        : ;;  # no override
        n_sims_15)      export N_SIMS=15 ;;
        n_sims_60)      export N_SIMS=60 ;;
        n_sims_120)     export N_SIMS=120 ;;
        topk_1)         export BON_MCTS_TOPK=1; export BON_MCTS_MIN_SIMS=8 ;;
        topk_4)         export BON_MCTS_TOPK=4; export BON_MCTS_MIN_SIMS=4 ;;
        ucb_c_0.5)      export UCB_C=0.5 ;;
        ucb_c_2.0)      export UCB_C=2.0 ;;
        *)
            echo "[ablation] ERROR unknown cell '$1'" >&2; return 1 ;;
    esac
}

_apply_backend() {
    export SD35_BACKEND="$1"
    case "$1" in
        sid)               export STEPS=4;  export BASELINE_CFG=1.0; export CFG_SCALES="1.0" ;;
        senseflow_large)   export STEPS=4;  export BASELINE_CFG=1.0; export CFG_SCALES="1.0" ;;
        sd35_base)         export STEPS=28; export BASELINE_CFG=4.5; export CFG_SCALES="3.5 4.0 4.5 5.0 5.5 6.0 7.0" ;;
        *) echo "[ablation] ERROR unknown backend '$1'" >&2; return 1 ;;
    esac
}

_sample_prompts() {
    local backend="$1"
    local prompt_file="${PROMPTS_DIR}/backend_${backend}.txt"
    if [[ ! -f "${prompt_file}" ]]; then
        echo "[ablation] sampling prompts → ${prompt_file}"
        env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
            "${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_prompts.py" \
            --n_prompts "${N_PROMPTS}" \
            --out_dir "${PROMPTS_DIR}" \
            --backends "${backend}"
    fi
    export PROMPT_FILE="${prompt_file}"
}

# ── Main loop ───────────────────────────────────────────────────────────────
echo "[ablation] backends=${BACKENDS}  cells=${CELLS}  N_PROMPTS=${N_PROMPTS}  SEEDS=${SEEDS}"
echo "[ablation] anchor: refine=mcts reward=imagereward N_SEEDS=8"

failed=()
for backend in ${BACKENDS}; do
    _apply_backend "${backend}"
    _sample_prompts "${backend}"

    for cell in ${CELLS}; do
        if ! _apply_cell "${cell}"; then
            failed+=("${backend}/${cell}/bad-cell")
            continue
        fi

        # All seeds run inside one suite invocation (suite handles its own seed=SEED).
        for seed in ${SEEDS}; do
            cell_root="${RUN_ROOT}/${backend}/${cell}/seed${seed}"
            mkdir -p "${cell_root}"
            export OUT_ROOT="${cell_root}"

            echo
            echo "================================================================"
            echo "[ablation] backend=${backend}  cell=${cell}  seed=${seed}"
            echo "  N_SIMS=${N_SIMS} UCB_C=${UCB_C} TOPK=${BON_MCTS_TOPK} MIN_SIMS=${BON_MCTS_MIN_SIMS}"
            echo "================================================================"

            if SEED="${seed}" bash "${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"; then
                echo "[ablation] OK ${backend}/${cell}/seed${seed}"
            else
                rc=$?
                echo "[ablation] FAIL ${backend}/${cell}/seed${seed} rc=${rc}" >&2
                failed+=("${backend}/${cell}/seed${seed}")
                if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
            fi
        done
    done
done

if (( ${#failed[@]} > 0 )); then
    echo "[ablation] DONE with failures: ${failed[*]}"
    exit 1
fi
echo "[ablation] DONE all (${BACKENDS} × ${CELLS}) cells OK."
echo "[ablation] Compare: python3 mcts_param_compare.py --root ${RUN_ROOT}"
