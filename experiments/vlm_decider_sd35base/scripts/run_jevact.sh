#!/usr/bin/env bash
# Jev as the POLICY: it executes the action it picks. Three arms, matched 616 NFE, FIXED noise (seed 42) so this
# measures the action policy alone.  Queued behind the cheap band + ec_v2 + sel_jev.
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/jevact_driver.log; }
DEADLINE=$(( $(date +%s) + 36000 ))
log "waiting for sel_jev (last queued stage) to finish"
while :; do
  c=$(wc -l < $R/main/sel_jev/records.jsonl 2>/dev/null || echo 0)
  [ "$c" -ge 9 ] && { log "sel_jev done"; break; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { log "STALL: sel_jev at $c/9; launching anyway (independent experiment)"; break; }
  sleep 180
done
free_gpus(){ APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv,noheader,nounits | while IFS=, read g u m; do
    n=$(echo "$APPS" | grep -c "${u// /}"); [ "${m// /}" -lt 2000 ] && [ "$n" = 0 ] && echo $g; done; }
sleep 30; F=($(free_gpus)); log "free gpus: ${F[*]}"
[ ${#F[@]} -ge 3 ] || { log "need 3 free GPUs (1 for vlm+jev, 2 samplers), have ${#F[@]}; not launching"; exit 1; }
curl -s --max-time 5 http://127.0.0.1:5142/health | grep -q '"status": "ok"' || { log "mr3 down"; exit 1; }
SRV=${F[0]}
bash $R/scripts/vlm_server.sh $SRV 5151 >> $R/logs/jevact_driver.log 2>&1 || { log "vlm failed"; exit 1; }
bash $R/scripts/jev_server.sh $SRV 5161 >> $R/logs/jevact_driver.log 2>&1 || { log "jev failed"; exit 1; }
run(){ local G=$1 N=$2 P=$3 t=$(date +%s); log "--- $N (policy=$P) gpu$G"
  cd /home/ygu/project/SiD-DiT_tts_dev
  CUDA_VISIBLE_DEVICES=$G OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u $R/scripts/jev_act.py --ids $R/config/prompt_ids_rsel9.txt --run_dir $R/main/$N --policy $P \
    --reward_server http://127.0.0.1:5142 --vlm_server http://127.0.0.1:5151 --jev_server http://127.0.0.1:5161 \
    >> $R/logs/$N.log 2>&1
  log "--- $N end rc=$? in $(( $(date +%s)-t ))s records=$(wc -l < $R/main/$N/records.jsonl 2>/dev/null)"; }
run ${F[1]} ja_jevact jev & P1=$!
run ${F[2]} ja_randact random & P2=$!
wait $P1; wait $P2
F2=($(free_gpus)); [ ${#F2[@]} -ge 1 ] && run ${F2[0]} ja_anchoract anchor
kill $(cat $R/state/vlm_server.pid) $(cat $R/state/jev_server.pid) 2>/dev/null
log "JEV-AS-POLICY COMPLETE"
