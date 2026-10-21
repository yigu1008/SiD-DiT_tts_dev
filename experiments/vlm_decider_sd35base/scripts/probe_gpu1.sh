#!/usr/bin/env bash
# Sequential probe on ONE GPU shared with the IR server: stock mcts then rand_prune (no VLM/Jev needed).  bash probe_gpu1.sh <GPU> [N_SIMS=40]
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python; GPU=$1; NS=${2:-40}
IDS=/data/ygu/reward_orthogonality/followup_2026-09-24/prompt_ids_99.txt; URL=http://127.0.0.1:5141
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/probe_driver.log; }
curl -s --max-time 5 $URL/health | grep -q '"status": "ok"' || { log "IR server not healthy"; exit 1; }
t0=$(date +%s); log "--- probe mcts start (gpu=$GPU sims=$NS, shared with IR server)"
CUDA_VISIBLE_DEVICES=$GPU OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u $R/scripts/base_arm.py --arm mcts --ids $IDS --limit 1 --n_sims $NS --run_dir $R/probe/mcts --reward_server $URL --trace >> $R/logs/probe_mcts.log 2>&1; log "--- probe mcts end rc=$? in $(( $(date +%s)-t0 ))s"
t0=$(date +%s); log "--- probe rand_prune start (gpu=$GPU sims=$NS)"
CUDA_VISIBLE_DEVICES=$GPU OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u $R/scripts/vlm_mcts.py --arm rand_prune --ids $IDS --limit 1 --n_sims $NS --run_dir $R/probe/rand_prune --reward_server $URL >> $R/logs/probe_rand_prune.log 2>&1; log "--- probe rand_prune end rc=$? in $(( $(date +%s)-t0 ))s"
log "PROBE GPU1 COMPLETE"
