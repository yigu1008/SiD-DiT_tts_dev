#!/usr/bin/env bash
# Single-bash: bon_mcts + baseline at composite_hpsv3_ir reward, across all
# 4 backends (sid, senseflow_large, sd35_base, flux_schnell).
#
# One reward server boot, one env build (paid by the AMLT yaml). The bash
# loops sequentially over backends and dispatches to the matching suite.
#
# Caller env (from AMLT yaml):
#   RUN_ROOT           - output dir parent
#   REWARD_SERVER_URL  - shared reward server URL (must host hpsv3 + imagereward)
#
# Optional:
#   BACKENDS           (default: all 4)
#   METHODS            (default: full kit — baseline, bon, beam, smc, fksteering,
#                       dts, dts_star, sop, bon_mcts)
#   N_PROMPTS          (default 200)
#   SEEDS              (default "42")
#   N_SIMS             (default 30)
#   FAIL_FAST          (default 0)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "composite-all-backends"

: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${REWARD_SERVER_URL:?REWARD_SERVER_URL must be set}"

BACKENDS="${BACKENDS:-sid senseflow_large sd35_base flux_schnell}"
N_PROMPTS="${N_PROMPTS:-200}"
SEEDS="${SEEDS:-42}"
N_SIMS="${N_SIMS:-60}"
FAIL_FAST="${FAIL_FAST:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ── 80GB A100 speed/budget knobs ───────────────────────────────────────────
# Keep fragmentation low when 9 methods cycle through different alloc patterns.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Encoders STAY on GPU on 80GB — we have headroom and avoid re-load cost
# per prompt. Override to 1 only if you OOM (mainly FLUX peaks).
export OFFLOAD_TEXT_ENCODER_AFTER_ENCODE="${OFFLOAD_TEXT_ENCODER_AFTER_ENCODE:-0}"
# 256 is enough for HPSv2-style prompts; longer adds T5 activations linearly.
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
# CFG doubles the forward batch. 80GB → GEN_BATCH_SIZE=2 is safe and ~1.5×
# faster per prompt vs the 40GB-safe default of 1.
export GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-2}"
# SMC / FK-Steering particle count.
export SMC_K="${SMC_K:-8}"

# ── sd35_base lightweight mode ─────────────────────────────────────────────
# sd35_base is 28-step (vs 4-step for the other 3 backends) — it's the
# wallclock long pole. With LIGHTWEIGHT_SD35_BASE=1 we run it with a reduced
# method kit + halved N_SIMS so the full composite job finishes in ~half the
# time. The other backends still get the full 9-method kit.
LIGHTWEIGHT_SD35_BASE="${LIGHTWEIGHT_SD35_BASE:-1}"
SD35_BASE_METHODS="${SD35_BASE_METHODS:-baseline bon bon_mcts}"
SD35_BASE_N_SIMS="${SD35_BASE_N_SIMS:-120}"

# ── Shared run knobs ────────────────────────────────────────────────────────
# Full search-method kit: baseline (no search reference) + all baselines
# (bon, beam, smc/DAS, FK-Steering, DTS, DTS*, SoP) + ours (bon_mcts).
export METHODS="${METHODS:-baseline bon beam smc fksteering dts dts_star sop bon_mcts}"
export START_INDEX=0
export END_INDEX="${N_PROMPTS}"
export N_VARIANTS=1
export USE_QWEN=0
export PRECOMPUTE_REWRITES=0
export REWARDS_OVERWRITE=0
# Reward-gradient correction bank.  "0.0" = off (paper-canonical for SMC/FK).
# Setting non-zero values turns on classifier-guidance-style ∇r pull on the
# velocity at each step — adds an extra action axis for search methods
# (bon_mcts, dts, sop, smc, fksteering, …) and triples the per-step action
# space when the bank has 3 entries.  "0.0 0.5 1.0" matches Universal-Guidance
# style ranges; expect ~2–3× slower per cell.
export CORRECTION_STRENGTHS="${CORRECTION_STRENGTHS:-0.0 0.5 1.0}"
export SAVE_BEST_IMAGES=1
export SAVE_IMAGES=0
export SAVE_VARIANTS=0
export EVAL_BEST_IMAGES=1
export EVAL_REWARD_DEVICE=cuda
export EVAL_ALLOW_MISSING_BACKENDS=0

# ── Composite reward (the half-half normalized hpsv3+imagereward) ────────────
# Default to composite_hpsv3_ir for backwards compat, but honor any caller
# override (e.g. actdiff_grid_*_ir.yaml passes SEARCH_REWARD=imagereward).
export REWARD_BACKEND="${REWARD_BACKEND:-${SEARCH_REWARD:-composite_hpsv3_ir}}"
export REWARD_TYPE="${REWARD_TYPE:-${REWARD_BACKEND}}"
export REWARD_BACKENDS="${REWARD_BACKENDS:-${REWARD_BACKEND}}"
# Eval covers both raw rewards so we can decompose the composite gain.
export EVAL_BACKENDS="imagereward hpsv3"

# ── Anchored bon_mcts knobs (matches default cell of mcts_param ablation) ───
export N_SIMS UCB_C="${UCB_C:-1.0}"
export BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS:-16}"
export BON_MCTS_TOPK="${BON_MCTS_TOPK:-4}"
export BON_MCTS_SIM_ALLOC=split
export BON_MCTS_MIN_SIMS=8
export BON_MCTS_REFINE_METHOD=mcts        # vanilla refine

# Reuse cherry_pick prompt sampler (HPSv2 + DrawBench).
PROMPTS_DIR="${RUN_ROOT}/_prompts"
mkdir -p "${PROMPTS_DIR}"

# ── Per-backend dispatch ────────────────────────────────────────────────────
_run_one_backend() {
    local backend="$1"
    local prompt_file="${PROMPTS_DIR}/backend_${backend}.txt"

    if [[ ! -f "${prompt_file}" ]]; then
        echo "[composite] sampling prompts → ${prompt_file}"
        env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
            "${PYTHON_BIN}" "${SCRIPT_DIR}/cherry_pick_prompts.py" \
            --n_prompts "${N_PROMPTS}" \
            --out_dir "${PROMPTS_DIR}" \
            --backends "${backend}"
    fi
    export PROMPT_FILE="${prompt_file}"

    local suite suite_kind
    case "${backend}" in
        sid)
            suite="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"; suite_kind="sd35"
            export SD35_BACKEND=sid; export STEPS=4
            export BASELINE_CFG=1.0; export CFG_SCALES="1.0 1.25 1.5 1.75 2.0 2.25 2.5"
            export MCTS_KEY_STEP_COUNT="${MCTS_KEY_STEP_COUNT:-4}"
            unset FLUX_BACKEND || true ;;
        senseflow_large)
            suite="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"; suite_kind="sd35"
            export SD35_BACKEND=senseflow_large; export STEPS=4
            export BASELINE_CFG=1.0; export CFG_SCALES="1.0 1.25 1.5 1.75 2.0 2.25 2.5"
            export MCTS_KEY_STEP_COUNT="${MCTS_KEY_STEP_COUNT:-4}"
            unset FLUX_BACKEND || true ;;
        sd35_base)
            suite="${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh"; suite_kind="sd35"
            export SD35_BACKEND=sd35_base; export STEPS=28
            export BASELINE_CFG=4.5; export CFG_SCALES="3.5 4.0 4.5 5.0 5.5 6.0 7.0"
            export MCTS_KEY_STEP_COUNT="${MCTS_KEY_STEP_COUNT:-8}"
            unset FLUX_BACKEND || true ;;
        flux_schnell)
            suite="${SCRIPT_DIR}/hpsv2_flux_schnell_ddp_suite.sh"; suite_kind="flux"
            export FLUX_BACKEND=flux; export STEPS=4
            export MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-schnell}"
            # FLUX.1-schnell is CFG-distilled at 0.  Pragmatic bank: include
            # the canonical 0.0 so MCTS can fall back to the distilled
            # setting, and drop the noisy 2.25/2.5 tail where quality degrades.
            export BASELINE_GUIDANCE_SCALE=0.0; export BASELINE_CFG=0.0
            export CFG_SCALES="0.0 1.0 1.25 1.5 1.75 2.0"
            export MCTS_KEY_STEP_COUNT="${MCTS_KEY_STEP_COUNT:-4}"
            unset SD35_BACKEND || true ;;
        *)
            echo "[composite] ERROR unknown backend '${backend}'" >&2; return 2 ;;
    esac

    # Per-backend method/n_sims override (sd35_base is the long pole).
    local backend_methods="${METHODS}"
    local backend_n_sims="${N_SIMS}"
    if [[ "${backend}" == "sd35_base" && "${LIGHTWEIGHT_SD35_BASE}" == "1" ]]; then
        backend_methods="${SD35_BASE_METHODS}"
        backend_n_sims="${SD35_BASE_N_SIMS}"
        echo "[composite] sd35_base lightweight: methods='${backend_methods}' n_sims=${backend_n_sims}"
    fi

    local seed_failed=()
    for seed in ${SEEDS}; do
        local cell_root="${RUN_ROOT}/${backend}/seed${seed}"
        mkdir -p "${cell_root}"
        export OUT_ROOT="${cell_root}"
        echo
        echo "================================================================"
        echo "[composite] backend=${backend}  seed=${seed}  reward=composite_hpsv3_ir"
        echo "  methods='${backend_methods}'"
        echo "  N_SIMS=${backend_n_sims} TOPK=${BON_MCTS_TOPK} N_SEEDS=${BON_MCTS_N_SEEDS}"
        echo "  out=${cell_root}"
        echo "================================================================"
        if METHODS="${backend_methods}" N_SIMS="${backend_n_sims}" SEED="${seed}" bash "${suite}"; then
            echo "[composite] OK ${backend}/seed${seed}"
        else
            local rc=$?
            echo "[composite] FAIL ${backend}/seed${seed} rc=${rc}" >&2
            seed_failed+=("${seed}")
            if [[ "${FAIL_FAST}" == "1" ]]; then return "${rc}"; fi
        fi
    done

    if (( ${#seed_failed[@]} > 0 )); then
        echo "[composite] WARN ${backend} seed failures: ${seed_failed[*]}"
        return 1
    fi
    return 0
}

# ── Loop ────────────────────────────────────────────────────────────────────
echo "[composite] backends=${BACKENDS}  reward=composite_hpsv3_ir"
echo "[composite] N_PROMPTS=${N_PROMPTS}  SEEDS=${SEEDS}  N_SIMS=${N_SIMS}"

backend_failed=()
for backend in ${BACKENDS}; do
    if _run_one_backend "${backend}"; then
        echo "[composite] DONE OK ${backend}"
    else
        rc=$?
        echo "[composite] FAIL backend=${backend} rc=${rc}" >&2
        backend_failed+=("${backend}")
        if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
    fi
done

if (( ${#backend_failed[@]} > 0 )); then
    echo "[composite] DONE with backend failures: ${backend_failed[*]}"
    exit 1
fi
echo "[composite] DONE all backends OK"
echo "  Compare aggregate_ddp.json files under ${RUN_ROOT}/<backend>/seed*/"
