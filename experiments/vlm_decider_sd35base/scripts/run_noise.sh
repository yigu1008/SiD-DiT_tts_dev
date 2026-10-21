#!/usr/bin/env bash
# Initial-noise probe, sharded across sampler GPUs against an already-running mr3 server on 5142.
#   bash run_noise.sh <G0> <G1> <G2> [IDS] [N_SEEDS]
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
G=("$1" "$2" "$3"); IDS=${4:-$R/config/prompt_ids_rsel9.txt}; NS=${5:-16}; DIR=$R/main/ns_probe
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/noise_driver.log; }
curl -s --max-time 5 http://127.0.0.1:5142/health | grep -q '"status": "ok"' || { log "mr3 server not healthy"; exit 1; }
log "=== noise probe: ${#G[@]} shards on gpus ${G[*]}, n_seeds=$NS, n=$(grep -c . $IDS)"
PIDS=()
for i in 0 1 2; do
  ( t=$(date +%s)
    CUDA_VISIBLE_DEVICES=${G[$i]} OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY -u $R/scripts/noise_probe.py --ids $IDS --run_dir $DIR --n_seeds $NS --shard_index $i --num_shards 3 \
      --reward_server http://127.0.0.1:5142 >> $R/logs/noise_shard$i.log 2>&1
    log "--- shard $i end rc=$? in $(( $(date +%s)-t ))s rows=$(wc -l < $DIR/records_shard$i.jsonl 2>/dev/null)" ) & PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait $p; done
cat $DIR/records_shard*.jsonl > $DIR/records_all.jsonl
log "noise probe done: $(wc -l < $DIR/records_all.jsonl) images total"
