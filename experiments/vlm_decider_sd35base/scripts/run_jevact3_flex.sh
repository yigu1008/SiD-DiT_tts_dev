#!/usr/bin/env bash
# Flexible-GPU, self-resuming launcher for the corrected jevact rerun (same arms/args as run_jevact3.sh).
#   Phase A  ja_jevact_v2w + ja_randact_v2w : needs 3 CLEAN GPUs  [mr3] [vlm+jev 41 GB] [sampler ~26 GB]; 4+ -> arms in parallel
#   Phase B  ec_v2w (3 fixed shards)        : needs 2 CLEAN GPUs  [mr3] [sampler(s)]; shards round-robin over sampler GPUs
# Only 2 clean GPUs -> phase B runs first; phase A waits for a third. A running mr3 (health on :5142) counts as its own GPU.
# "Clean" = no compute process from anyone and <2 GB used: never share a card another user is on (their launches SIGTERM us).
# Resumable: jev_act.py skips finished prompts; early_control.py skips finished (prompt,noise,mode,vi) per shard file,
# so NS is pinned at 3 regardless of how many GPUs are available. Completion judged by RECORDS only.
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python; SID=/home/ygu/project/SiD-DiT_tts_dev
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/jevact3_driver.log; }
grep -q "sheet_perm" $R/scripts/jev_act.py && grep -q "_bank_n_variants" $R/scripts/jev_act.py || { log "GUARD: review fixes not present in jev_act.py; refusing to launch"; exit 1; }
grep -q "_bank_n_variants" $R/scripts/early_control.py || { log "GUARD: bank fix missing in early_control.py"; exit 1; }
free_gpus(){ APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv,noheader,nounits | while IFS=, read g u m; do
    n=$(echo "$APPS" | grep -c "${u// /}"); [ "${m// /}" -lt 2000 ] && [ "$n" = 0 ] && echo $g; done; }
V2=$R/config/rewrites_v2_9.json; DIR=$R/main/ec_v2w; NS=3
nrec(){ wc -l < "$1" 2>/dev/null || echo 0; }
A_done(){ [ "$(nrec $R/main/ja_jevact_v2w/records.jsonl)" -ge 9 ] && [ "$(nrec $R/main/ja_randact_v2w/records.jsonl)" -ge 9 ]; }
B_done(){ [ "$(cat $DIR/records_shard*.jsonl 2>/dev/null | wc -l)" -ge 324 ]; }
up(){ curl -s --max-time 5 http://127.0.0.1:$1/health | grep -q '"status": "ok"'; }
status(){ echo "jevact $(nrec $R/main/ja_jevact_v2w/records.jsonl)/9 randact $(nrec $R/main/ja_randact_v2w/records.jsonl)/9 ec_v2w $(cat $DIR/records_shard*.jsonl 2>/dev/null | wc -l)/324"; }
kill_servers(){ for s in vlm jev mr3; do kill $(cat $R/state/${s}_server.pid 2>/dev/null) 2>/dev/null; done; sleep 10; }

run_arm(){ local G=$1 N=$2 P=$3 t=$(date +%s); log "--- $N (policy=$P, v2 bank, full width) gpu$G"
  cd $SID && CUDA_VISIBLE_DEVICES=$G OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u $R/scripts/jev_act.py --ids $R/config/prompt_ids_rsel9.txt --run_dir $R/main/$N --policy $P --rewrites_file $V2 \
    --reward_server http://127.0.0.1:5142 --vlm_server http://127.0.0.1:5151 --jev_server http://127.0.0.1:5161 >> $R/logs/$N.log 2>&1
  log "--- $N end rc=$? in $(( $(date +%s)-t ))s records=$(nrec $R/main/$N/records.jsonl)"; }
run_shard(){ local G=$1 i=$2 t=$(date +%s); log "--- ec_v2w shard $i/$NS gpu$G"
  cd $SID && CUDA_VISIBLE_DEVICES=$G OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u $R/scripts/early_control.py --ids $R/config/prompt_ids_rsel9.txt --run_dir $DIR --rewrites_file $V2 \
    --shard_index $i --num_shards $NS --reward_server http://127.0.0.1:5142 >> $R/logs/ecv2w_shard$i.log 2>&1
  log "--- ec_v2w shard $i end rc=$? in $(( $(date +%s)-t ))s shard_records=$(nrec $DIR/records_shard$i.jsonl)"; }

# take_servers <gpus...> : start mr3 (and, if $NEED_VLM=1, vlm+jev) on the first clean GPUs unless already healthy;
# prints the remaining sampler GPUs. Returns 1 on a server failure.
take_servers(){ local F=("$@"); up 5142 || { bash $R/scripts/mr3_server.sh ${F[0]} 5142 >> $R/logs/jevact3_driver.log 2>&1 || return 1; F=("${F[@]:1}"); }
  if [ "${NEED_VLM:-0}" = 1 ]; then
    if ! up 5151 || ! up 5161; then kill $(cat $R/state/vlm_server.pid $R/state/jev_server.pid 2>/dev/null) 2>/dev/null; sleep 5
      bash $R/scripts/vlm_server.sh ${F[0]} 5151 >> $R/logs/jevact3_driver.log 2>&1 || return 1
      bash $R/scripts/jev_server.sh ${F[0]} 5161 >> $R/logs/jevact3_driver.log 2>&1 || return 1; F=("${F[@]:1}"); fi; fi
  echo "${F[@]}"; }

phaseA(){ local S; S=($(NEED_VLM=1 take_servers "$@")) || { log "phase A: server start failed"; return 1; }
  [ ${#S[@]} -ge 1 ] || { log "phase A: no sampler GPU left"; return 1; }
  log "phase A on samplers: ${S[*]} ($( [ ${#S[@]} -ge 2 ] && echo parallel || echo sequential ))"
  if [ ${#S[@]} -ge 2 ]; then run_arm ${S[0]} ja_jevact_v2w jev & P1=$!; run_arm ${S[1]} ja_randact_v2w random & P2=$!; wait $P1; wait $P2
  else [ "$(nrec $R/main/ja_jevact_v2w/records.jsonl)" -ge 9 ] || run_arm ${S[0]} ja_jevact_v2w jev
       [ "$(nrec $R/main/ja_randact_v2w/records.jsonl)" -ge 9 ] || run_arm ${S[0]} ja_randact_v2w random; fi
  kill $(cat $R/state/vlm_server.pid $R/state/jev_server.pid 2>/dev/null) 2>/dev/null; sleep 10; log "phase A pass done: $(status)"; }

phaseB(){ local S; S=($(NEED_VLM=0 take_servers "$@")) || { log "phase B: mr3 start failed"; return 1; }
  [ ${#S[@]} -ge 1 ] || { log "phase B: no sampler GPU left"; return 1; }
  local K=$(( ${#S[@]} < NS ? ${#S[@]} : NS )); log "phase B: $NS shards over sampler GPUs ${S[*]:0:$K}"; local PIDS=()
  for g in $(seq 0 $((K-1))); do ( for i in $(seq 0 $((NS-1))); do [ $((i % K)) -eq $g ] && run_shard ${S[$g]} $i; done ) & PIDS+=($!); done
  for p in "${PIDS[@]}"; do wait $p; done
  cat $DIR/records_shard*.jsonl > $DIR/records_all.jsonl 2>/dev/null; log "phase B pass done: $(status)"; }

DEADLINE=$(( $(date +%s) + 172800 )); log "=== flex launcher start: $(status)"
while :; do
  if A_done && B_done; then kill_servers; log "CORRECTED RERUN COMPLETE: $(status)"; exit 0; fi
  [ "$(date +%s)" -ge "$DEADLINE" ] && { kill_servers; log "STALL: 48 h deadline, $(status)"; exit 1; }
  F=($(free_gpus)); n=${#F[@]}; m=0; up 5142 && m=1          # a healthy mr3 saves one GPU
  if ! A_done && [ $((n+m)) -ge 3 ]; then need=$((3-m)); ph=A
  elif ! B_done && [ $((n+m)) -ge 2 ]; then need=$((2-m)); ph=B
  else sleep 300; continue; fi
  sleep 30; F=($(free_gpus)); [ ${#F[@]} -ge $need ] || { log "GPUs taken during settle (wanted $need for phase $ph)"; sleep 120; continue; }
  log "clean GPUs: ${F[*]} -> phase $ph"
  if [ $ph = A ]; then phaseA "${F[@]}"; else phaseB "${F[@]}"; fi
  rc=$?; [ $rc -ne 0 ] && { log "phase $ph exited rc=$rc; retrying in 10 min"; sleep 600; }
done
