#!/usr/bin/env bash
# Progress check for the latest `my_actdiff_grid` run.
# Usage:
#   bash check_my_actdiff_grid.sh              # one-shot status
#   bash check_my_actdiff_grid.sh watch        # auto-refresh every 30s
#   bash check_my_actdiff_grid.sh /path/run    # explicit run dir
#   RUN=/path/run bash check_my_actdiff_grid.sh

set -uo pipefail

# Pick the run dir: arg > env > newest matching dir
if [[ -n "${1:-}" && "$1" != "watch" ]]; then
    RUN="$1"
elif [[ -n "${RUN:-}" ]]; then
    :
else
    RUN="$(ls -dt /data/ygu/runs/my_actdiff_grid_*/ 2>/dev/null | head -1)"
fi

if [[ -z "${RUN:-}" || ! -d "${RUN}" ]]; then
    echo "[FATAL] no my_actdiff_grid run dir found." >&2
    echo "        Looked in: /data/ygu/runs/my_actdiff_grid_*/" >&2
    exit 1
fi

# How many prompts were requested?  Pull from the run's _run.log if possible.
N_TOTAL="${N_TOTAL:-}"
if [[ -z "${N_TOTAL}" ]]; then
    N_TOTAL="$(grep -m1 -oE 'N_PROMPTS *= *[0-9]+' "${RUN%/}/_run.log" 2>/dev/null | grep -oE '[0-9]+' | head -1)"
    : "${N_TOTAL:=200}"
fi

METHODS=(baseline bon bon_actdiff_cfg bon_actdiff_full sop sop_actdiff_cfg sop_actdiff_full smc smc_actdiff_cfg smc_actdiff_full bon_mcts)

_print_status() {
    echo "=== $(date '+%Y-%m-%d %H:%M:%S')  run=${RUN%/}  expected=${N_TOTAL} ==="
    local done_total=0
    local methods_done=0
    for m in "${METHODS[@]}"; do
        local rf
        rf="$(ls "${RUN%/}/${m}"/run_*/bon_mcts/logs/rank_*.jsonl 2>/dev/null | head -1)"
        if [[ -f "${rf}" ]]; then
            local n
            n=$(wc -l < "${rf}" 2>/dev/null | tr -d ' ')
            printf "  %-22s %s/%s\n" "${m}:" "${n}" "${N_TOTAL}"
            done_total=$((done_total + n))
            if [[ "${n}" -ge "${N_TOTAL}" ]]; then
                methods_done=$((methods_done + 1))
            fi
        else
            printf "  %-22s pending\n" "${m}:"
        fi
    done
    local pct=$((done_total * 100 / (${N_TOTAL} * ${#METHODS[@]})))
    echo "------------------------------------------------------------"
    echo "  methods finished: ${methods_done}/${#METHODS[@]}"
    echo "  overall progress: ${done_total}/$((N_TOTAL * ${#METHODS[@]}))  (~${pct}%)"
    if [[ -f "${RUN%/}/summary.tsv" ]]; then
        echo "  summary.tsv:      ready"
    fi
}

if [[ "${1:-}" == "watch" ]]; then
    while true; do
        clear
        _print_status
        sleep 30
    done
else
    _print_status
fi
