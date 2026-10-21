#!/usr/bin/env bash
# Queue the CHEAP two-stage band, starting once the 576-616 NFE sweep frees its GPUs.
# Matched-NFE pairs so the allocation question is answered at low budget too, in one seed pool.
#   bash run_cheap.sh
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/cheap_driver.log; }
WAIT_FOR="ts_k11m0 ts_k8m4 ts_k6m8 ts_k4m12 ts_k3m14"
DEADLINE=$(( $(date +%s) + 14400 ))            # hard 4h stall guard (never a bare wait)
log "waiting for the sweep to finish: $WAIT_FOR"
while :; do
  n=0; for A in $WAIT_FOR; do c=$(wc -l < $R/main/$A/records.jsonl 2>/dev/null || echo 0); n=$((n+c)); done
  [ "$n" -ge 45 ] && { log "sweep complete ($n/45 records)"; break; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { log "STALL: deadline hit with $n/45 records; not launching"; exit 1; }
  sleep 60
done
# only use GPUs that are genuinely free AND not ryan's
free_gpus(){ APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv,noheader,nounits | while IFS=, read g u m; do
    n=$(echo "$APPS" | grep -c "${u// /}"); [ "${m// /}" -lt 2000 ] && [ "$n" = 0 ] && echo $g; done; }
sleep 30; F=($(free_gpus)); log "free gpus: ${F[*]}"
[ ${#F[@]} -ge 3 ] || { log "need 3 free GPUs, have ${#F[@]} - ryan may have taken them; not launching"; exit 1; }
curl -s --max-time 5 http://127.0.0.1:5142/health | grep -q '"status": "ok"' || { log "mr3 server down"; exit 1; }
# name k m   (three two-stage points + their pure-noise partners at the same NFE)
SPECS=("ts_k2m2 2 2" "ts_k3m0 3 0" "ts_k4m4 4 4" "ts_k6m0 6 0" "ts_k2m4 2 4" "ts_k4m0 4 0")
i=0; PIDS=()
for S in "${SPECS[@]}"; do
  set -- $S; NAME=$1; K=$2; M=$3; G=${F[$(( i % ${#F[@]} ))]}
  ( t=$(date +%s); log "--- $NAME (k=$K m=$M, $((K*56+M*32)) NFE) gpu$G"
    cd /home/ygu/project/SiD-DiT_tts_dev
    CUDA_VISIBLE_DEVICES=$G OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY -u $R/scripts/two_stage.py --ids $R/config/prompt_ids_rsel9.txt --run_dir $R/main/$NAME \
      --k $K --m $M --reward_server http://127.0.0.1:5142 >> $R/logs/$NAME.log 2>&1
    log "--- $NAME end rc=$? in $(( $(date +%s)-t ))s records=$(wc -l < $R/main/$NAME/records.jsonl 2>/dev/null)" ) & PIDS+=($!)
  i=$((i+1))
  [ $(( i % ${#F[@]} )) -eq 0 ] && { for p in "${PIDS[@]}"; do wait $p; done; PIDS=(); }
done
for p in "${PIDS[@]}"; do wait $p; done
log "CHEAP BAND COMPLETE"
