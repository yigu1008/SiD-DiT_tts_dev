#!/usr/bin/env bash
# Jev as the POLICY (it executes its chosen action), crossed with bank width.
# The question: the action optimum is PROMPT-SPECIFIC (cross-prompt quality correlation -0.08 wording / -0.01 CFG),
# and Jev is a 70%-vs-33% within-prompt comparator -- so does widening what it chooses from raise the ceiling?
#   ja_jevact      Jev chooses, shipped bank   (4.00 wordings, 28 actions)
#   ja_jevact_v2   Jev chooses, v2 bank        (7.33 wordings, 49 actions, 0 artifacts)   <- the user's proposal
#   ja_randact_v2  random choice, v2 bank      (matched control: same rich bank, no information)
#   ja_anchoract   anchor every segment        (the no-action-search floor)
# All 616 NFE, noise FIXED at seed 42, so this isolates the action policy.
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/jevact2_driver.log; }
free_gpus(){ APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv,noheader,nounits | while IFS=, read g u m; do
    n=$(echo "$APPS" | grep -c "${u// /}"); [ "${m// /}" -lt 2000 ] && [ "$n" = 0 ] && echo $g; done; }
DEADLINE=$(( $(date +%s) + 28800 ))
# RETRY for GPUs instead of exiting -- the previous launcher bailed permanently when ec_v2 held 0/1/2.
log "waiting for >=4 free GPUs (1 for vlm+jev, 3 samplers)"
while :; do
  F=($(free_gpus))
  [ ${#F[@]} -ge 4 ] && { log "got ${#F[@]} free: ${F[*]}"; break; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { log "STALL: only ${#F[@]} free after 8h; giving up"; exit 1; }
  sleep 120
done
sleep 20; F=($(free_gpus)); [ ${#F[@]} -ge 4 ] || { log "GPUs taken during settle; aborting"; exit 1; }
curl -s --max-time 5 http://127.0.0.1:5142/health | grep -q '"status": "ok"' || { log "mr3 down"; exit 1; }
SRV=${F[0]}
bash $R/scripts/vlm_server.sh $SRV 5151 >> $R/logs/jevact2_driver.log 2>&1 || { log "vlm failed"; exit 1; }
bash $R/scripts/jev_server.sh $SRV 5161 >> $R/logs/jevact2_driver.log 2>&1 || { log "jev failed"; exit 1; }
V2=$R/config/rewrites_v2_9.json
run(){ local G=$1 N=$2 P=$3 BANK=$4 t=$(date +%s); log "--- $N (policy=$P bank=$(basename $BANK)) gpu$G"
  cd /home/ygu/project/SiD-DiT_tts_dev
  CUDA_VISIBLE_DEVICES=$G OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u $R/scripts/jev_act.py --ids $R/config/prompt_ids_rsel9.txt --run_dir $R/main/$N --policy $P \
    --rewrites_file $BANK --reward_server http://127.0.0.1:5142 \
    --vlm_server http://127.0.0.1:5151 --jev_server http://127.0.0.1:5161 >> $R/logs/$N.log 2>&1
  log "--- $N end rc=$? in $(( $(date +%s)-t ))s records=$(wc -l < $R/main/$N/records.jsonl 2>/dev/null)"; }
SHIPPED=/data/ygu/human_eval_genai40_v1/sid_vqascore_algorithm_sweep/genai200_v1/shared_rewrites_cache.json
run ${F[1]} ja_jevact_v2  jev    $V2      & P1=$!
run ${F[2]} ja_randact_v2 random $V2      & P2=$!
run ${F[3]} ja_jevact     jev    $SHIPPED & P3=$!
wait $P1; wait $P2; wait $P3
G2=($(free_gpus)); [ ${#G2[@]} -ge 1 ] && run ${G2[0]} ja_anchoract anchor $SHIPPED
kill $(cat $R/state/vlm_server.pid) $(cat $R/state/jev_server.pid) 2>/dev/null
log "JEV-AS-POLICY x BANK-WIDTH COMPLETE"
