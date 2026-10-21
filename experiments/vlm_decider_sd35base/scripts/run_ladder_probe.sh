#!/usr/bin/env bash
# bash run_ladder_probe.sh <G0> <G1> <G2> [IDS] [N_SEEDS]   (mr3 server must already be up on 5142)
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
G=("$1" "$2" "$3"); IDS=${4:-$R/config/prompt_ids_rsel9.txt}; NS=${5:-32}; DIR=$R/main/ns_ladder
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/ladder_driver.log; }
curl -s --max-time 5 http://127.0.0.1:5142/health | grep -q '"status": "ok"' || { log "mr3 not healthy"; exit 1; }
log "=== ladder probe: gpus ${G[*]}, n_seeds=$NS, n=$(grep -c . $IDS)"
PIDS=()
for i in 0 1 2; do
  ( t=$(date +%s)
    CUDA_VISIBLE_DEVICES=${G[$i]} OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY -u $R/scripts/noise_ladder_probe.py --ids $IDS --run_dir $DIR --n_seeds $NS --shard_index $i --num_shards 3 \
      --save_previews --reward_server http://127.0.0.1:5142 >> $R/logs/ladder_shard$i.log 2>&1
    log "--- shard $i end rc=$? in $(( $(date +%s)-t ))s rows=$(wc -l < $DIR/records_shard$i.jsonl 2>/dev/null)" ) & PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait $p; done
cat $DIR/records_shard*.jsonl > $DIR/records_all.jsonl
log "ladder probe done: $(wc -l < $DIR/records_all.jsonl) trajectories"
