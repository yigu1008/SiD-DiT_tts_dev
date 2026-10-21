#!/usr/bin/env bash
# Configuration sweep of the online arm on a dev id-set, then judging + a comparison table.
#   bash run_sweep.sh <IDS_FILE> <N_SIMS> <tag=args> [tag=args ...]
# Each config runs sequentially on one sampler GPU; servers live on a second GPU. Resumable per prompt per config.
set -uo pipefail
R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
IDS=$1; NS=$2; shift 2; CFGS=("$@"); log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/sweep_driver.log; }
free_gpus(){ APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv,noheader,nounits | while IFS=, read g u m; do
    n=$(echo "$APPS" | grep -c "${u// /}"); [ "${m// /}" -lt 2000 ] && [ "$n" = 0 ] && echo $g; done; }
log "=== sweep: ids=$IDS sims=$NS configs=${#CFGS[@]} :: ${CFGS[*]}"
DEADLINE=$(( $(date +%s) + 36000 )); F=()
while [ $(date +%s) -lt $DEADLINE ]; do
  F=($(free_gpus)); if [ ${#F[@]} -ge 2 ]; then ok=1; for j in 1 2; do sleep 20; G=($(free_gpus)); [ "${G[*]}" = "${F[*]}" ] || { ok=0; break; }; done; [ $ok = 1 ] && break; fi
  sleep 60; done
F=($(free_gpus)); [ ${#F[@]} -ge 2 ] || { log "no 2 stable free GPUs; giving up"; exit 1; }
A=${F[0]}; B=${F[1]}; log "free GPUs ${F[*]} -> sampler gpu$A, servers gpu$B"
bash $R/scripts/ir_server.sh $A 5141 && bash $R/scripts/vlm_server.sh $B 5151 && bash $R/scripts/jev_server.sh $B 5161 || { log "server start failed"; exit 1; }
nohup bash $R/scripts/keepalive2.sh $A $B $B > /dev/null 2>&1 & echo $! > $R/state/keepalive.pid
DIRS=""
for entry in "${CFGS[@]}"; do
  TAG="${entry%%=*}"; ARGS="${entry#*=}"; DIR=$R/main/sw_$TAG; t=$(date +%s)
  log "--- sw_$TAG start gpu$A :: $ARGS"
  CUDA_VISIBLE_DEVICES=$A OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u $R/scripts/jev_online.py \
    --run_dir $DIR --ids $IDS --n_sims $NS --reward_server http://127.0.0.1:5141 --vlm_server http://127.0.0.1:5151 --jev_server http://127.0.0.1:5161 $ARGS \
    >> $R/logs/sweep_sw_$TAG.log 2>&1
  log "--- sw_$TAG end rc=$? in $(( $(date +%s)-t ))s records=$(wc -l < $DIR/records.jsonl 2>/dev/null)"
  [ -s $DIR/records.jsonl ] && DIRS="$DIRS $DIR"
done
kill $(cat $R/state/keepalive.pid) 2>/dev/null; kill $(cat $R/state/vlm_server.pid) $(cat $R/state/jev_server.pid) 2>/dev/null; sleep 3
[ -n "$DIRS" ] && bash $R/scripts/judge_base.sh $A $IDS $DIRS
kill $(cat $R/state/ir_server.pid) 2>/dev/null
$PY $R/scripts/sweep_table.py $IDS | tee -a $R/logs/sweep_driver.log
log "SWEEP COMPLETE"
