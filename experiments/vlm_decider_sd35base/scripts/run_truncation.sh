#!/usr/bin/env bash
# bash run_truncation.sh <IDS_FILE> [N_TRAJ=8]   -- needs ONE free GPU (sampler + IR server share it)
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
IDS=$1; NT=${2:-8}; log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/trunc_driver.log; }
free_gpus(){ APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv,noheader,nounits | while IFS=, read g u m; do
    n=$(echo "$APPS" | grep -c "${u// /}"); [ "${m// /}" -lt 2000 ] && [ "$n" = 0 ] && echo $g; done; }
DEADLINE=$(( $(date +%s) + 18000 )); A=""
while [ $(date +%s) -lt $DEADLINE ]; do F=($(free_gpus)); [ ${#F[@]} -ge 1 ] && { sleep 20; G=($(free_gpus)); echo " ${G[*]} " | grep -q " ${F[0]} " && { A=${F[0]}; break; }; }; sleep 60; done
[ -n "$A" ] || { log "no free GPU"; exit 1; }
log "truncation probe on gpu$A"; bash $R/scripts/ir_server.sh $A 5141 || exit 1
CUDA_VISIBLE_DEVICES=$A OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u $R/scripts/truncation_probe.py \
  --ids $IDS --n_traj $NT --run_dir $R/main/truncation --reward_server http://127.0.0.1:5141 2>&1 | tee -a $R/logs/truncation.log
kill $(cat $R/state/ir_server.pid) 2>/dev/null; log "TRUNCATION PROBE COMPLETE"
