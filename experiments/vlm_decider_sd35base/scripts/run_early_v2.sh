#!/usr/bin/env bash
# THE decisive re-test of the action axis with a better proposer.  Re-runs the early/late control probe on the v2
# bank (7.33 distinct wordings / 49 actions, 0 artifacts) instead of the shipped one (4.00 / 28, 4 artifacts).
# Question: with a wider, cleaner bank, does EARLY control finally beat redrawing the noise?  On the shipped bank
# early-control best-of-6 lost to fresh best-of-6 by -1.384 [-2.215, -0.511].
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/earlyv2_driver.log; }
WAIT_FOR="ts_k2m2 ts_k3m0 ts_k4m4 ts_k6m0 ts_k2m4 ts_k4m0"
DEADLINE=$(( $(date +%s) + 25200 ))
log "waiting for the cheap band to finish ($WAIT_FOR)"
while :; do
  n=0; for A in $WAIT_FOR; do c=$(wc -l < $R/main/$A/records.jsonl 2>/dev/null || echo 0); n=$((n+c)); done
  [ "$n" -ge 54 ] && { log "cheap band complete ($n/54)"; break; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { log "STALL: deadline hit at $n/54; not launching"; exit 1; }
  sleep 120
done
free_gpus(){ APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv,noheader,nounits | while IFS=, read g u m; do
    n=$(echo "$APPS" | grep -c "${u// /}"); [ "${m// /}" -lt 2000 ] && [ "$n" = 0 ] && echo $g; done; }
sleep 30; F=($(free_gpus)); log "free gpus: ${F[*]}"
[ ${#F[@]} -ge 3 ] || { log "need 3 free GPUs, have ${#F[@]}; not launching"; exit 1; }
curl -s --max-time 5 http://127.0.0.1:5142/health | grep -q '"status": "ok"' || { log "mr3 down"; exit 1; }
DIR=$R/main/ec_v2; PIDS=()
for i in 0 1 2; do
  ( t=$(date +%s); log "--- ec_v2 shard $i gpu${F[$i]}"
    cd /home/ygu/project/SiD-DiT_tts_dev
    CUDA_VISIBLE_DEVICES=${F[$i]} OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY -u $R/scripts/early_control.py --ids $R/config/prompt_ids_rsel9.txt --run_dir $DIR \
      --rewrites_file $R/config/rewrites_v2_9.json --shard_index $i --num_shards 3 \
      --reward_server http://127.0.0.1:5142 >> $R/logs/ecv2_shard$i.log 2>&1
    log "--- shard $i end rc=$? rows=$(wc -l < $DIR/records_shard$i.jsonl 2>/dev/null) in $(( $(date +%s)-t ))s" ) & PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait $p; done
cat $DIR/records_shard*.jsonl > $DIR/records_all.jsonl
log "EC_V2 COMPLETE: $(wc -l < $DIR/records_all.jsonl) rows"
