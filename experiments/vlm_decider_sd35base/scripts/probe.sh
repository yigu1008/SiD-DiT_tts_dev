#!/usr/bin/env bash
# Timing probe on base SD3.5: baseline then stock MCTS on the first N prompts of the 99-cohort.  bash probe.sh <GPU> <N> [N_SIMS]
set -uo pipefail
R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python; GPU=$1; N=${2:-1}; NS=${3:-40}
IDS=/data/ygu/reward_orthogonality/followup_2026-09-24/prompt_ids_99.txt; URL=http://127.0.0.1:5141
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/probe_driver.log; }
curl -s --max-time 5 $URL/health | grep -q '"status": "ok"' || { log "IR server not healthy"; exit 1; }
for ARM in baseline mcts; do t0=$(date +%s); log "--- probe $ARM start (n=$N sims=$NS gpu=$GPU)"
  CUDA_VISIBLE_DEVICES=$GPU OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u $R/scripts/base_arm.py --arm $ARM --ids $IDS --limit $N --n_sims $NS --run_dir $R/probe/$ARM --reward_server $URL --trace >> $R/logs/probe_${ARM}.log 2>&1
  log "--- probe $ARM end rc=$? in $(( $(date +%s)-t0 ))s"; done
log "PROBE COMPLETE"
