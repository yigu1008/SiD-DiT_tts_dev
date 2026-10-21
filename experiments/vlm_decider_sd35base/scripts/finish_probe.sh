#!/usr/bin/env bash
# Finish the probe on ONE ryan-free GPU: basic stock mcts -> judge it -> regenerate PROBE.md.   bash finish_probe.sh <GPU>
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python; GPU=$1
IDS=/data/ygu/reward_orthogonality/followup_2026-09-24/prompt_ids_99.txt; log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/probe_driver.log; }
t0=$(date +%s); log "--- probe mcts (basic, no refine) start gpu=$GPU"
CUDA_VISIBLE_DEVICES=$GPU OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u $R/scripts/base_arm.py --arm mcts --ids $IDS --limit 1 --n_sims 40 --run_dir $R/probe/mcts --reward_server http://127.0.0.1:5141 --trace >> $R/logs/probe_mcts.log 2>&1; rc=$?; log "--- probe mcts end rc=$rc in $(( $(date +%s)-t0 ))s"
[ $rc -eq 0 ] && [ -s $R/probe/mcts/records.jsonl ] && bash $R/scripts/judge_base.sh $GPU $R/config/prompt_ids_probe.txt $R/probe/mcts
$PY $R/scripts/make_probe_md.py; log "FINISH PROBE COMPLETE"
