#!/usr/bin/env bash
# Fixed-schedule joint (noise x action) allocation grid at 616 NFE. mr3 server must be up on 5142.
#   bash run_joint.sh <G0> <G1> <G2> <G3> [IDS]
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
G=("$1" "$2" "$3" "$4"); IDS=${5:-$R/config/prompt_ids_rsel9.txt}
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/joint_driver.log; }
curl -s --max-time 5 http://127.0.0.1:5142/health | grep -q '"status": "ok"' || { log "mr3 not healthy"; exit 1; }
# name screen_n keep
SPECS=("jt_s4k1 4 1" "jt_s8k1 8 1" "jt_s8k3 8 3" "jt_s16k1 16 1")
log "=== joint grid on gpus ${G[*]}, n=$(grep -c . $IDS): ${SPECS[*]}"
PIDS=()
for i in 0 1 2 3; do
  set -- ${SPECS[$i]}; NAME=$1; SN=$2; KP=$3
  ( t=$(date +%s); log "--- $NAME start (screen_n=$SN keep=$KP) gpu${G[$i]}"
    CUDA_VISIBLE_DEVICES=${G[$i]} OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY -u $R/scripts/joint_arm.py --ids $IDS --run_dir $R/main/$NAME --screen_n $SN --keep $KP \
      --reward_server http://127.0.0.1:5142 >> $R/logs/joint_$NAME.log 2>&1
    log "--- $NAME end rc=$? in $(( $(date +%s)-t ))s records=$(wc -l < $R/main/$NAME/records.jsonl 2>/dev/null)" ) & PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait $p; done
log "joint grid done"
