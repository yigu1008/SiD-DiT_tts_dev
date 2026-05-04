#!/usr/bin/env bash
# Master bash: runs ALL MCTS-related ablations in ONE AMLT job.
#   1. MCTS hyperparam ablation   (8 cells × 2 backends × 4 seeds = 64)
#   2. Qwen rewrite ablation      (2 cells × 2 backends × 4 seeds = 16)
# Total ~80 sequential cells, one env build, one reward server boot.
#
# Caller env (from AMLT yaml):
#   RUN_ROOT_BASE      - parent dir; sub-ablations will write to RUN_ROOT_BASE/{mcts_param,qwen_rewrite}
#   REWARD_SERVER_URL  - shared reward server URL
#   PYTHON_BIN
#
# Optional:
#   ABLATIONS          - subset of {mcts_param, qwen_rewrite} (default both)
#   BACKENDS, SEEDS, N_PROMPTS, FAIL_FAST  — passed through to each sub-ablation

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_heartbeat.sh"
start_heartbeat "all-ablations"

: "${RUN_ROOT_BASE:?RUN_ROOT_BASE must be set}"
: "${REWARD_SERVER_URL:?REWARD_SERVER_URL must be set}"

ABLATIONS="${ABLATIONS:-mcts_param qwen_rewrite}"
FAIL_FAST="${FAIL_FAST:-0}"
mkdir -p "${RUN_ROOT_BASE}"

failed=()
for abl in ${ABLATIONS}; do
    case "${abl}" in
        mcts_param)
            echo
            echo "================================================================"
            echo "[all-ablations] === MCTS hyperparam ablation ==="
            echo "================================================================"
            RUN_ROOT="${RUN_ROOT_BASE}/mcts_param" \
            bash "${SCRIPT_DIR}/hpsv2_mcts_param_ablation.sh" || {
                rc=$?; echo "[all-ablations] FAIL mcts_param rc=${rc}" >&2
                failed+=("mcts_param")
                if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
            }
            ;;
        qwen_rewrite)
            echo
            echo "================================================================"
            echo "[all-ablations] === Qwen 4B vs 7B rewrite ablation ==="
            echo "================================================================"
            RUN_ROOT="${RUN_ROOT_BASE}/qwen_rewrite" \
            bash "${SCRIPT_DIR}/hpsv2_qwen_rewrite_ablation.sh" || {
                rc=$?; echo "[all-ablations] FAIL qwen_rewrite rc=${rc}" >&2
                failed+=("qwen_rewrite")
                if [[ "${FAIL_FAST}" == "1" ]]; then exit "${rc}"; fi
            }
            ;;
        *)
            echo "[all-ablations] ERROR unknown ablation '${abl}'" >&2
            failed+=("${abl}/unknown")
            ;;
    esac
done

echo
if (( ${#failed[@]} > 0 )); then
    echo "[all-ablations] DONE with failures: ${failed[*]}"
    exit 1
fi
echo "[all-ablations] DONE all ${ABLATIONS} OK"
echo "  mcts_param results: ${RUN_ROOT_BASE}/mcts_param"
echo "  qwen_rewrite:       ${RUN_ROOT_BASE}/qwen_rewrite"
echo "  Compare:"
echo "    python3 mcts_param_compare.py --root ${RUN_ROOT_BASE}/mcts_param"
echo "    python3 mcts_param_compare.py --root ${RUN_ROOT_BASE}/qwen_rewrite --cells qwen_4b qwen_8b"
