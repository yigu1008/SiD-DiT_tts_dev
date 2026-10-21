#!/usr/bin/env bash
# Jev as SELECTOR over the same 11 candidate noises the reward selectors used. No generation: images exist.
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/jevsel_driver.log; }
DEADLINE=$(( $(date +%s) + 21600 ))
WAIT_FOR="ts_k2m2 ts_k3m0 ts_k4m4 ts_k6m0 ts_k2m4 ts_k4m0"
log "waiting for the cheap band so a GPU is free for Jev (~24 GB)"
while :; do
  n=0; for A in $WAIT_FOR; do c=$(wc -l < $R/main/$A/records.jsonl 2>/dev/null || echo 0); n=$((n+c)); done
  [ "$n" -ge 54 ] && { log "cheap band done ($n/54)"; break; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { log "STALL at $n/54; not launching"; exit 1; }
  sleep 120
done
free_gpus(){ APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv,noheader,nounits | while IFS=, read g u m; do
    n=$(echo "$APPS" | grep -c "${u// /}"); [ "${m// /}" -lt 2000 ] && [ "$n" = 0 ] && echo $g; done; }
sleep 30; F=($(free_gpus)); log "free gpus: ${F[*]}"
[ ${#F[@]} -ge 1 ] || { log "no free GPU for Jev; not launching"; exit 1; }
bash $R/scripts/jev_server.sh ${F[0]} 5161 >> $R/logs/jevsel_driver.log 2>&1 || { log "jev server failed"; exit 1; }
$PY -u $R/scripts/jev_select.py --ids $R/config/prompt_ids_rsel9.txt --run_dir $R/main/sel_jev --k 11 \
  >> $R/logs/sel_jev.log 2>&1; log "sel_jev rc=$?"
kill $(cat $R/state/jev_server.pid) 2>/dev/null
log "JEV SELECTOR COMPLETE"
