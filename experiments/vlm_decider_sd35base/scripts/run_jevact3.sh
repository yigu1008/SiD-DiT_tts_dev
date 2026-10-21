#!/usr/bin/env bash
# Corrected rerun after the 2026-10-21 review: (1) bank width actually delivered (--n_variants sized from the bank),
# (2) sheet panel order permuted per call. Jev-as-policy x bank width, plus ec_v2 on the real 9-10-wording bank.
# All 616 NFE, noise fixed at seed 42. Refuses to launch unless both fixes are present in jev_act.py.
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/jevact3_driver.log; }
grep -q "sheet_perm" $R/scripts/jev_act.py && grep -q "_bank_n_variants" $R/scripts/jev_act.py || { log "GUARD: review fixes not present in jev_act.py; refusing to launch"; exit 1; }
grep -q "_bank_n_variants" $R/scripts/early_control.py || { log "GUARD: bank fix missing in early_control.py"; exit 1; }
free_gpus(){ APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv,noheader,nounits | while IFS=, read g u m; do
    n=$(echo "$APPS" | grep -c "${u// /}"); [ "${m// /}" -lt 2000 ] && [ "$n" = 0 ] && echo $g; done; }
DEADLINE=$(( $(date +%s) + 172800 ))   # ryan holds every GPU right now; wait up to 48h, checking every 5 min
log "waiting for >=4 free GPUs (vlm+jev server, reward server, 2 samplers)"
while :; do F=($(free_gpus)); [ ${#F[@]} -ge 4 ] && { log "free: ${F[*]}"; break; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { log "STALL: gave up after 48h"; exit 1; }; sleep 300; done
sleep 30; F=($(free_gpus)); [ ${#F[@]} -ge 4 ] || { log "GPUs taken during settle; aborting"; exit 1; }
bash $R/scripts/mr3_server.sh ${F[0]} 5142 >> $R/logs/jevact3_driver.log 2>&1 || { log "mr3 failed"; exit 1; }
bash $R/scripts/vlm_server.sh ${F[1]} 5151 >> $R/logs/jevact3_driver.log 2>&1 || { log "vlm failed"; exit 1; }
bash $R/scripts/jev_server.sh ${F[1]} 5161 >> $R/logs/jevact3_driver.log 2>&1 || { log "jev failed"; exit 1; }
V2=$R/config/rewrites_v2_9.json
run(){ local G=$1 N=$2 P=$3 t=$(date +%s); log "--- $N (policy=$P, v2 bank, full width) gpu$G"
  cd /home/ygu/project/SiD-DiT_tts_dev
  CUDA_VISIBLE_DEVICES=$G OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u $R/scripts/jev_act.py --ids $R/config/prompt_ids_rsel9.txt --run_dir $R/main/$N --policy $P --rewrites_file $V2 \
    --reward_server http://127.0.0.1:5142 --vlm_server http://127.0.0.1:5151 --jev_server http://127.0.0.1:5161 >> $R/logs/$N.log 2>&1
  log "--- $N end rc=$? in $(( $(date +%s)-t ))s records=$(wc -l < $R/main/$N/records.jsonl 2>/dev/null)"; }
run ${F[2]} ja_jevact_v2w  jev    & P1=$!
run ${F[3]} ja_randact_v2w random & P2=$!
wait $P1; wait $P2
kill $(cat $R/state/vlm_server.pid) $(cat $R/state/jev_server.pid) 2>/dev/null; sleep 10
# ec_v2 on the real bank: 3 shards on whatever is free now (reward server still up)
G=($(free_gpus)); [ ${#G[@]} -ge 1 ] || { log "no GPU for ec_v2w"; kill $(cat $R/state/mr3_server.pid); exit 1; }
NS=$(( ${#G[@]} < 3 ? ${#G[@]} : 3 )); DIR=$R/main/ec_v2w; PIDS=()
for i in $(seq 0 $((NS-1))); do
  ( t=$(date +%s); log "--- ec_v2w shard $i/$NS gpu${G[$i]}"; cd /home/ygu/project/SiD-DiT_tts_dev
    CUDA_VISIBLE_DEVICES=${G[$i]} OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY -u $R/scripts/early_control.py --ids $R/config/prompt_ids_rsel9.txt --run_dir $DIR --rewrites_file $V2 \
      --shard_index $i --num_shards $NS --reward_server http://127.0.0.1:5142 >> $R/logs/ecv2w_shard$i.log 2>&1
    log "--- ec_v2w shard $i end rc=$? in $(( $(date +%s)-t ))s" ) & PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait $p; done
cat $DIR/records_shard*.jsonl > $DIR/records_all.jsonl 2>/dev/null
kill $(cat $R/state/mr3_server.pid) 2>/dev/null
log "CORRECTED RERUN COMPLETE: $(wc -l < $DIR/records_all.jsonl 2>/dev/null) ec_v2w rows"
