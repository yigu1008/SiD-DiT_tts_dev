#!/usr/bin/env bash
# Exploration constant (UCB c) ablation for MCTS, across all 4 backends.
#
# 3 cells × 4 backends × 1 seed = 12 cells, 50 prompts each.
#
# Cells (ucb_c bracketed around the default 1.0):
#   ucb_c_0.5   : c = 0.5  (more greedy)
#   ucb_c_1.0   : c = 1.0  (default; what mcts_param ablation uses)
#   ucb_c_2.0   : c = 2.0  (more exploratory)
#
# Anchored knobs (held fixed across cells):
#   refine_method = mcts (vanilla)
#   N_SIMS=30  TOPK=2  N_SEEDS=8  MIN_SIMS=8
#   reward = imagereward (search). phase-2 posthoc = hpsv3 + pickscore.
#   N_VARIANTS=1, USE_QWEN=0 (no rewrites — clean control).
#
# Caller env (typically AMLT yaml):
#   RUN_ROOT, REWARD_SERVER_URL
#
# Optional:
#   BACKENDS  (default "sid senseflow_large sd35_base flux_schnell")
#   CELLS     (default "ucb_c_0.5 ucb_c_1.0 ucb_c_2.0")
#   SEEDS     (default "42")
#   N_PROMPTS (default 50)
#   FAIL_FAST (default 0)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "ucb-c-ablation"

: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${REWARD_SERVER_URL:?REWARD_SERVER_URL must be set}"

BACKENDS="${BACKENDS:-sid senseflow_large sd35_base flux_schnell}"
CELLS="${CELLS:-ucb_c_0.5 ucb_c_1.0 ucb_c_2.0}"
FAIL_FAST="${FAIL_FAST:-0}"

N_PROMPTS="${N_PROMPTS:-50}"
SEEDS="${SEEDS:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ── Shared knobs ────────────────────────────────────────────────────────────
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
export EVAL_BACKENDS="imagereward"
export REWARD_BACKEND="imagereward"
export REWARD_TYPE="imagereward"
export REWARD_BACKENDS="imagereward"
export EVAL_ALLOW_MISSING_BACKENDS=1
export EVAL_BEST_IMAGES=1
export EVAL_REWARD_DEVICE=cuda

# Anchored bon_mcts knobs (matches default cell of mcts_param ablation).
export N_SIMS=30
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
            export STEPS=4; export BASELINE_CFG=1.0; export CFG_SCALES="1.0 1.5 2.0 2.5"
            export _SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
            ;;
        senseflow_large)
            export SD35_BACKEND=senseflow_large; unset FLUX_BACKEND || true
            export STEPS=4; export BASELINE_CFG=1.0; export CFG_SCALES="1.0 1.5 2.0 2.5"
            export _SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
            ;;
        sd35_base)
            export SD35_BACKEND=sd35_base; unset FLUX_BACKEND || true
            export STEPS=28; export BASELINE_CFG=4.5; export CFG_SCALES="3.5 4.5 5.5 7.0"
            export _SUITE="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"
            ;;
        flux_schnell)
            export FLUX_BACKEND=flux; unset SD35_BACKEND || true
            export STEPS=4
            export MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-schnell}"
            export BASELINE_GUIDANCE_SCALE=0.0; export BASELINE_CFG=0.0; export CFG_SCALES="0.0"
            export _SUITE="${SCRIPT_DIR}/hpsv2_flux_schnell_ddp_suite.sh"
            ;;
        *) echo "[ucb-c-ablation] ERROR unknown backend '$1'" >&2; return 1 ;;
    esac
}

_sample_prompts() {
    local backend="$1"
    local prompt_file="${PROMPTS_DIR}/backend_${backend}.txt"
    if [[ ! -f "${prompt_file}" ]]; then
        echo "[ucb-c-ablation] sampling prompts → ${prompt_file}"
        env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
            "${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_prompts.py" \
            --n_prompts "${N_PROMPTS}" \
            --out_dir "${PROMPTS_DIR}" \
            --backends "${backend}"
    fi
    export PROMPT_FILE="${prompt_file}"
}

_apply_cell() {
    case "$1" in
        ucb_c_0.5) export UCB_C=0.5 ;;
        ucb_c_1.0) export UCB_C=1.0 ;;
        ucb_c_2.0) export UCB_C=2.0 ;;
        *) echo "[ucb-c-ablation] ERROR unknown cell '$1'" >&2; return 1 ;;
    esac
}

# ── Banner ──────────────────────────────────────────────────────────────────
echo "================================================================"
echo "[ucb-c-ablation]   exploration constant ablation"
echo "  backends=${BACKENDS}  N_PROMPTS=${N_PROMPTS}  SEEDS=${SEEDS}"
echo "  cells=${CELLS}"
echo "  reward=imagereward (search), phase-2 posthoc=hpsv3+pickscore"
echo "================================================================"

failed=()
for backend in ${BACKENDS}; do
    _apply_backend "${backend}"
    _sample_prompts "${backend}"

    for cell in ${CELLS}; do
        if ! _apply_cell "${cell}"; then
            failed+=("${backend}/${cell}/bad-cell"); continue
        fi
        for seed in ${SEEDS}; do
            cell_root="${RUN_ROOT}/${backend}/${cell}/seed${seed}"
            mkdir -p "${cell_root}"
            export OUT_ROOT="${cell_root}"

            echo
            echo "================================================================"
            echo "[ucb-c-ablation] backend=${backend}  cell=${cell}  seed=${seed}"
            echo "  UCB_C=${UCB_C}  N_SIMS=${N_SIMS} TOPK=${BON_MCTS_TOPK}"
            echo "  CFG_SCALES='${CFG_SCALES}'"
            echo "================================================================"

            if SEED="${seed}" bash "${_SUITE}"; then
                echo "[ucb-c-ablation] OK ${backend}/${cell}/seed${seed}"
            else
                rc=$?
                echo "[ucb-c-ablation] FAIL ${backend}/${cell}/seed${seed} rc=${rc}" >&2
                failed+=("${backend}/${cell}/seed${seed}")
                if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
            fi
        done
    done
done

if (( ${#failed[@]} > 0 )); then
    echo "[ucb-c-ablation] phase-1 DONE with failures: ${failed[*]}"
fi
echo "[ucb-c-ablation] phase-1 DONE."

# ── Phase-2 posthoc: HPSv3 + PickScore on saved best_images ─────────────────
if [[ "${RUN_POSTHOC:-1}" == "1" ]]; then
    echo
    echo "================================================================"
    echo "[ucb-c-ablation] phase-2 posthoc: hpsv3 + pickscore on saved best_images"
    echo "================================================================"
    posthoc_failed=()
    for backend in ${BACKENDS}; do
        for cell in ${CELLS}; do
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
        echo "[ucb-c-ablation] phase-2 posthoc WARN failures: ${posthoc_failed[*]}"
    else
        echo "[ucb-c-ablation] phase-2 posthoc DONE."
    fi
fi

echo "[ucb-c-ablation] ALL DONE."
echo "  Output:  ${RUN_ROOT}"
echo "  Compare: python3 mcts_param_compare.py --root ${RUN_ROOT} --cells ${CELLS}"
if (( ${#failed[@]} > 0 )); then exit 1; fi
