#!/usr/bin/env bash
# Ablation: Qwen3-4B vs Qwen3-8B for prompt rewriting, with bon_mcts downstream.
#
# Anchors (held fixed across both cells):
#   refine_method   = mcts (vanilla)
#   reward          = imagereward
#   N_VARIANTS      = 3   (3 rewrites per original prompt; +1 if original kept)
#   N_SIMS          = 30  (matched to default cell of mcts_param ablation)
#   BON_MCTS_TOPK   = 2
#   BON_MCTS_N_SEEDS= 8
#   UCB_C           = 1.0
#
# Cells:
#   qwen_4b   QWEN_ID=Qwen/Qwen3-4B  (current default — your baseline rewrite quality)
#   qwen_8b   QWEN_ID=Qwen/Qwen3-8B  (next larger Qwen3 size — Qwen3-7B does NOT
#                                     exist on HF; the lineup is 0.6/1.7/4/8/14/32B)
#
# Caller env (typically AMLT yaml):
#   RUN_ROOT, REWARD_SERVER_URL
#
# Optional:
#   BACKENDS  (default "sid senseflow_large")
#   CELLS     (default "qwen_4b qwen_8b")
#   FAIL_FAST (default 0)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "qwen-rewrite-ablation"

: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${REWARD_SERVER_URL:?REWARD_SERVER_URL must be set}"

BACKENDS="${BACKENDS:-sid senseflow_large}"
CELLS="${CELLS:-qwen_4b qwen_8b}"
FAIL_FAST="${FAIL_FAST:-0}"

N_PROMPTS="${N_PROMPTS:-100}"
SEEDS="${SEEDS:-42 43 44 45}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ── Anchored shared knobs ───────────────────────────────────────────────────
export METHODS="baseline bon_mcts"
export START_INDEX=0
export END_INDEX="${N_PROMPTS}"
export N_VARIANTS=3                         # 3 Qwen rewrites per prompt
export USE_QWEN=1                           # rewrites ON
export PRECOMPUTE_REWRITES=1                # precompute once per cell (avoids per-rank Qwen load)
export REWARDS_OVERWRITE=0
export CORRECTION_STRENGTHS="0.0"
export SAVE_BEST_IMAGES=1
export SAVE_IMAGES=0
export SAVE_VARIANTS=1                       # save the rewrite text — useful for sanity-checking
export EVAL_BACKENDS="imagereward hpsv3"
export REWARD_BACKEND="imagereward"
export REWARD_TYPE="imagereward"
export REWARD_BACKENDS="imagereward"

# Anchored MCTS / bon_mcts knobs (matches `default` cell of mcts_param ablation).
export N_SIMS=30
export UCB_C=1.0
export BON_MCTS_N_SEEDS=8
export BON_MCTS_TOPK=2
export BON_MCTS_SIM_ALLOC=split
export BON_MCTS_MIN_SIMS=8
export BON_MCTS_REFINE_METHOD=mcts

# Anchored prompts (HF download once per backend; reuse across both cells).
PROMPTS_DIR="${RUN_ROOT}/_prompts"
mkdir -p "${PROMPTS_DIR}"

_apply_cell() {
    case "$1" in
        qwen_4b)
            export QWEN_ID="Qwen/Qwen3-4B"
            ;;
        qwen_8b)
            export QWEN_ID="Qwen/Qwen3-8B"
            ;;
        *)
            echo "[ablation] ERROR unknown cell '$1'" >&2; return 1 ;;
    esac
}

_apply_backend() {
    export SD35_BACKEND="$1"
    case "$1" in
        sid)             export STEPS=4;  export BASELINE_CFG=1.0; export CFG_SCALES="1.0" ;;
        senseflow_large) export STEPS=4;  export BASELINE_CFG=1.0; export CFG_SCALES="1.0" ;;
        sd35_base)       export STEPS=28; export BASELINE_CFG=4.5; export CFG_SCALES="3.5 4.0 4.5 5.0 5.5 6.0 7.0" ;;
        *) echo "[ablation] ERROR unknown backend '$1'" >&2; return 1 ;;
    esac
}

_sample_prompts() {
    local backend="$1"
    local prompt_file="${PROMPTS_DIR}/backend_${backend}.txt"
    if [[ ! -f "${prompt_file}" ]]; then
        echo "[qwen-ablation] sampling prompts → ${prompt_file}"
        env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
            "${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_prompts.py" \
            --n_prompts "${N_PROMPTS}" \
            --out_dir "${PROMPTS_DIR}" \
            --backends "${backend}"
    fi
    export PROMPT_FILE="${prompt_file}"
}

# ── Main loop ───────────────────────────────────────────────────────────────
echo "[qwen-ablation] backends=${BACKENDS}  cells=${CELLS}  N_PROMPTS=${N_PROMPTS}  SEEDS=${SEEDS}"
echo "[qwen-ablation] anchor: refine=mcts reward=imagereward N_VARIANTS=${N_VARIANTS} N_SIMS=${N_SIMS}"

failed=()
for backend in ${BACKENDS}; do
    _apply_backend "${backend}"
    _sample_prompts "${backend}"

    for cell in ${CELLS}; do
        if ! _apply_cell "${cell}"; then
            failed+=("${backend}/${cell}/bad-cell")
            continue
        fi

        # Per-cell rewrites cache file (keyed by Qwen model). The suite expects
        # REWRITES_FILE; we set a unique path per (backend, cell) so 4B and 7B
        # don't clobber each other's caches.
        rewrite_tag="${cell}"
        export REWRITES_FILE="${PROMPTS_DIR}/${backend}_${rewrite_tag}_rewrites.json"

        for seed in ${SEEDS}; do
            cell_root="${RUN_ROOT}/${backend}/${cell}/seed${seed}"
            mkdir -p "${cell_root}"
            export OUT_ROOT="${cell_root}"

            echo
            echo "================================================================"
            echo "[qwen-ablation] backend=${backend}  cell=${cell}  seed=${seed}"
            echo "  QWEN_ID=${QWEN_ID}  N_VARIANTS=${N_VARIANTS}"
            echo "  rewrites_cache=${REWRITES_FILE}"
            echo "================================================================"

            if SEED="${seed}" bash "${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"; then
                echo "[qwen-ablation] OK ${backend}/${cell}/seed${seed}"
            else
                rc=$?
                echo "[qwen-ablation] FAIL ${backend}/${cell}/seed${seed} rc=${rc}" >&2
                failed+=("${backend}/${cell}/seed${seed}")
                if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
            fi
        done
    done
done

if (( ${#failed[@]} > 0 )); then
    echo "[qwen-ablation] DONE with failures: ${failed[*]}"
    exit 1
fi
echo "[qwen-ablation] DONE all (${BACKENDS} × ${CELLS}) cells OK."
echo "[qwen-ablation] Compare: python3 mcts_param_compare.py --root ${RUN_ROOT} --cells qwen_4b qwen_8b"
