#!/usr/bin/env bash
# Shared helpers for A6000 bon_mcts runs.  Sourced by
# run_single_prompt_viz.sh and run_noise_ablation_a6000.sh.

# Prevent ~/.local/lib torch from shadowing conda env.
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
# Make CUDA enumerate GPUs by PCI bus id so cuda:N == nvidia-smi GPU N.
# Without this, FASTEST_FIRST ordering can permute the mapping.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ── In-process reward (no server) ────────────────────────────────────────
a6000_use_inprocess_reward() {
    unset REWARD_SERVER_URL REWARD_SERVER_PORT
}

# ── Backend setup: CFG bank, steps, baseline ─────────────────────────────
a6000_setup_backend() {
    local backend="${BACKEND:-sid}"
    case "${backend}" in
        sid|senseflow_large)
            export SD35_BACKEND="${backend}"; unset FLUX_BACKEND 2>/dev/null || true
            export STEPS=4; export BASELINE_CFG=1.0
            export CFG_SCALES="1.0 1.25 1.5 1.75 2.0 2.25 2.5"
            : "${MCTS_KEY_STEP_COUNT:=4}"
            ;;
        sd35_base)
            export SD35_BACKEND=sd35_base; unset FLUX_BACKEND 2>/dev/null || true
            export STEPS=28; export BASELINE_CFG=4.5
            export CFG_SCALES="3.5 4.0 4.5 5.0 5.5 6.0 7.0"
            : "${MCTS_KEY_STEP_COUNT:=8}"
            ;;
        *) echo "[FATAL] unsupported BACKEND=${backend}" >&2; exit 1 ;;
    esac
    export SUITE="${A6000_SUITE:-${SCRIPT_DIR}/hpsv2_sd35_sid_ddp_suite.sh}"
}

# ── Bake prompt to a file (raccoon default) ──────────────────────────────
a6000_bake_prompt() {
    local out_dir="$1" prompt_text="${2:-}"
    mkdir -p "${out_dir}"
    PROMPT_FILE="${out_dir}/prompt.txt"
    : "${prompt_text:=${PROMPT:-}}"
    if [[ -z "${prompt_text}" ]]; then
        prompt_text='a detailed oil painting that captures the essence of an elderly raccoon adorned with a distinguished black top hat. The raccoon'\''s fur is depicted with textured, swirling strokes reminiscent of Van Gogh'\''s signature style, and it clutches a bright red apple in its paws. The background swirls with vibrant colors, giving the impression of movement around the still figure of the raccoon.'
    fi
    printf '%s\n' "${prompt_text}" > "${PROMPT_FILE}"
    export PROMPT_FILE
}

# ── Run Qwen standalone to write rewrites cache ──────────────────────────
# args:  $1 prompt_file   $2 rewrites_file   $3 n_variants
a6000_run_qwen_rewrites() {
    local prompt_file="$1" rewrites_file="$2" n_variants="$3"
    local qwen_id="${QWEN_ID:-Qwen/Qwen2.5-3B-Instruct}"
    local qwen_dtype="${QWEN_DTYPE:-bfloat16}"
    local cuda_device="${CUDA_VISIBLE_DEVICES:-0}"
    [[ -s "${rewrites_file}" ]] && { echo "[qwen] reusing ${rewrites_file}"; return 0; }
    [[ "${n_variants}" -le 0 ]] && return 0
    mkdir -p "$(dirname "${rewrites_file}")"
    echo "[qwen] running ${qwen_id} standalone on GPU ${cuda_device}"
    env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u LOCAL_WORLD_SIZE -u NODE_RANK -u MASTER_ADDR -u MASTER_PORT \
    PYTHONNOUSERSITE=1 \
    CUDA_VISIBLE_DEVICES="${cuda_device}" \
      "${PYTHON_BIN:-python}" -u "${SCRIPT_DIR}/precompute_sd35_rewrites.py" \
        --prompt_file "${prompt_file}" \
        --rewrites_file "${rewrites_file}" \
        --start_index 0 --end_index 1 \
        --n_variants "${n_variants}" \
        --qwen_id "${qwen_id}" --qwen_dtype "${qwen_dtype}" \
        --device cuda:0 --batch_size 1 \
        --save_every_batches 1 \
        --max_new_tokens "${QWEN_MAX_NEW_TOKENS:-384}" \
        --temperature "${QWEN_TEMP:-1.0}" \
        --top_p "${QWEN_TOP_P:-0.95}" \
        --no-clear_cache_each_batch \
        || echo "[qwen] WARN precompute failed"
    sleep 5
}

# ── Verify rewrites cache structure ──────────────────────────────────────
a6000_verify_rewrites() {
    local rewrites_file="$1" prompt_text="$2" n_variants="$3"
    [[ -s "${rewrites_file}" ]] || { echo "[verify] no rewrites file"; return 0; }
    python3 - "${rewrites_file}" "${prompt_text}" "${n_variants}" <<'PY'
import json, sys
fp, key, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(fp, encoding="utf-8"))
hit = key in d
print(f"[verify] cache hit: {hit}    entries: {len(d.get(key, []))}    expected: {n+1}")
for i, v in enumerate(d.get(key, [])[:n+1]):
    label = "canon" if i == 0 else "rewr "
    print(f"         v{i} ({label}): {v[:100]}{'...' if len(v)>100 else ''}")
PY
}

# ── Set all bon_mcts env vars for the suite ──────────────────────────────
# args: $1 run_root   $2 n_prompts
a6000_setup_bon_mcts_env() {
    local run_root="$1" n_prompts="$2"
    export METHODS="${METHODS:-bon_mcts}"
    export START_INDEX=0
    export END_INDEX="${n_prompts}"
    export SEEDS="${SEED:-42}"
    export N_SIMS="${N_SIMS:-30}"
    export BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS:-8}"
    export BON_MCTS_TOPK="${BON_MCTS_TOPK:-2}"
    export BON_MCTS_MIN_SIMS="${BON_MCTS_MIN_SIMS:-8}"
    export BON_MCTS_SIM_ALLOC=split
    export BON_MCTS_REFINE_METHOD=ours_tree
    export LOOKAHEAD_METHOD_MODE=rollout_tree_prior_adaptive_cfg
    : "${N_VARIANTS:=1}"; export N_VARIANTS
    export USE_QWEN=0
    export PRECOMPUTE_REWRITES=0
    export CORRECTION_STRENGTHS="0.0"
    export UCB_C=1.0
    # SLIM_MODE=1 disables ALL image dumps -- keep only rank_*.jsonl + summary.tsv.
    # Useful for big multi-method / multi-prompt grids where image disk usage
    # would balloon (200 prompts x 11 methods x 5 images each = 11000 files).
    if [[ "${SLIM_MODE:-0}" == "1" ]]; then
        export SAVE_BEST_IMAGES=0 SAVE_IMAGES=0 SAVE_VARIANTS=0
    else
        export SAVE_BEST_IMAGES=1 SAVE_IMAGES=1 SAVE_VARIANTS=1
    fi
    export REWARD_BACKEND="${SEARCH_REWARD:-imagereward}"
    export REWARD_TYPE="${SEARCH_REWARD:-imagereward}"
    export REWARD_BACKENDS="${SEARCH_REWARD:-imagereward}"
    export EVAL_BACKENDS="${SEARCH_REWARD:-imagereward}"
    export EVAL_BEST_IMAGES=1 EVAL_ALLOW_MISSING_BACKENDS=1 EVAL_REWARD_DEVICE=cuda
    export NUM_GPUS=1
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export OFFLOAD_TEXT_ENCODER_AFTER_ENCODE="${OFFLOAD_TEXT_ENCODER_AFTER_ENCODE:-1}"
    export MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
    export OUT_ROOT="${run_root}"
    # Step images / all-attempt dumps: gated on SLIM_MODE.  In slim mode we
    # unset these so the lookahead module's save hooks silently skip.
    if [[ "${SLIM_MODE:-0}" == "1" ]]; then
        unset SAVE_BEST_STEP_IMAGES_DIR SAVE_ALL_ATTEMPTS_DIR SAVE_ALL_STEP_IMAGES_DIR
    else
        export SAVE_BEST_STEP_IMAGES_DIR="${run_root}/step_images_inline"
        export SAVE_ALL_ATTEMPTS_DIR="${run_root}/all_attempts"
        if [[ "${SAVE_ALL_STEPS:-0}" == "1" ]]; then
            export SAVE_ALL_STEP_IMAGES_DIR="${run_root}/all_step_attempts"
            mkdir -p "${SAVE_ALL_STEP_IMAGES_DIR}"
        fi
        mkdir -p "${SAVE_BEST_STEP_IMAGES_DIR}" "${SAVE_ALL_ATTEMPTS_DIR}"
    fi
}

# ── Run bon_mcts via the suite ───────────────────────────────────────────
a6000_run_bon_mcts() {
    local run_root="$1"
    mkdir -p "${run_root}"
    echo "[a6000] running bon_mcts -> ${run_root}"
    # tee so suite progress streams to terminal AND the per-method log file.
    # Set A6000_QUIET=1 to revert to file-only (silent on stdout).
    if [[ "${A6000_QUIET:-0}" == "1" ]]; then
        bash "${SUITE}" > "${run_root}/_run.log" 2>&1 || \
          echo "[a6000] WARN suite exited non-zero (see ${run_root}/_run.log)"
    else
        bash "${SUITE}" 2>&1 | tee "${run_root}/_run.log"
        local rc=${PIPESTATUS[0]}
        [[ "${rc}" -ne 0 ]] && echo "[a6000] WARN suite exited non-zero (rc=${rc})"
    fi
}

# ── Render decision trees + action logs + trajectory strips ─────────────
a6000_render_viz() {
    local run_root="$1" n_prompts="$2"
    local backend="${BACKEND:-sid}"
    local prompt_range="0:${n_prompts}"
    "${PYTHON_BIN:-python}" "${SCRIPT_DIR}/render_trees_batch.py" \
        --run_root "${run_root}" --method bon_mcts \
        --prompt_range "${prompt_range}" \
        --out_dir "${run_root}/${backend}" \
        --title_prefix "ActDiff (${backend})" \
        --workers 2 || true
    "${PYTHON_BIN:-python}" "${SCRIPT_DIR}/dump_winner_log.py" \
        --run_root "${run_root}" --method bon_mcts \
        --prompt_range "${prompt_range}" \
        --out_dir "${run_root}/${backend}_logs" \
        --combined "${run_root}/${backend}_logs/_all.txt" || true
    if [[ -d "${run_root}/step_images_inline" ]]; then
        "${PYTHON_BIN:-python}" "${SCRIPT_DIR}/compose_trajectory_strips.py" \
            --in_dir "${run_root}/step_images_inline" \
            --out_dir "${run_root}/trajectory_strips" \
            --prompts_file "${PROMPT_FILE}" \
            --panel_size 384 --build_grid || true
    fi
}

# ── Quick variant-usage diagnostic ───────────────────────────────────────
a6000_verify_variant_usage() {
    local run_root="$1" n_variants="$2"
    python3 - "${run_root}" "${n_variants}" <<'PY'
import glob, json, sys, re
run_root, n_req = sys.argv[1], int(sys.argv[2])
ranks = glob.glob(f"{run_root}/run_*/bon_mcts/logs/rank_*.jsonl")
if not ranks:
    print("[verify] FAIL no rank file"); sys.exit(0)
for ln in open(ranks[0]):
    if not ln.strip(): continue
    r = json.loads(ln)
    if r.get("mode") not in ("mcts","bon_mcts"): continue
    acts = r.get("actions", [])
    vs = [int(a[0]) for a in acts]; cfgs = [float(a[1]) for a in acts]
    print(f"[verify] chosen v: {vs}   cfg: {cfgs}")
    break
attempts = glob.glob(f"{run_root}/all_attempts/prompt_0000/*.png")
if attempts:
    counts = {}
    for fp in attempts:
        m = re.search(r"_v([0-9-]+)_", fp)
        if m:
            for t in m.group(1).split("-"):
                counts[t] = counts.get(t, 0) + 1
    print(f"[verify] all-attempts variant counts: {dict(sorted(counts.items()))}    total: {len(attempts)}")
PY
}
