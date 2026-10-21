#!/usr/bin/env bash
# Does WHERE you branch matter at the 10-sim operating point? Blind MCTS, same prompts/budget, different key-step
# schedules. If these all land together there is nothing for a decider to choose and the Jev version is not worth building.
#   bash run_keysteps.sh <IDS_FILE> [SIMS=10]
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
IDS=$1; NS=${2:-10}; log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/ks_driver.log; }
free_gpus(){ APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv,noheader,nounits | while IFS=, read g u m; do
    n=$(echo "$APPS" | grep -c "${u// /}"); [ "${m// /}" -lt 2000 ] && [ "$n" = 0 ] && echo $g; done; }
DEADLINE=$(( $(date +%s) + 36000 ))
while [ $(date +%s) -lt $DEADLINE ]; do F=($(free_gpus)); [ ${#F[@]} -ge 1 ] && { sleep 20; G=($(free_gpus)); echo " ${G[*]} " | grep -q " ${F[0]} " && { A=${F[0]}; break; }; }; sleep 60; done
[ -n "${A:-}" ] || { log "no free GPU"; exit 1; }
log "key-step sweep on gpu$A, sims=$NS"; bash $R/scripts/ir_server.sh $A 5141 || exit 1
DIRS=""
# name=schedule   (run_mcts prepends step 0 when absent; 'every' = branch at all 28 steps)
for E in "spread8=0,4,8,12,16,20,24,27" "early4=0,1,2,3" "late4=0,21,24,27" "spread4=0,9,18,27" "mid4=0,10,14,18" "two=0,14"; do
  TAG="${E%%=*}"; KS="${E#*=}"; NAME=ks_${TAG}; DIR=$R/main/$NAME; t=$(date +%s); log "--- $NAME start key_steps=[$KS]"
  CUDA_VISIBLE_DEVICES=$A OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u $R/scripts/base_arm.py \
    --arm mcts --ids $IDS --n_sims $NS --key_steps "$KS" --run_dir $DIR --reward_server http://127.0.0.1:5141 >> $R/logs/ks_$NAME.log 2>&1
  log "--- $NAME end rc=$? in $(( $(date +%s)-t ))s records=$(wc -l < $DIR/records.jsonl 2>/dev/null)"
  if [ -s $DIR/records.jsonl ]; then
    $PY - "$DIR" "$NAME" <<'PY'
import json, sys, pathlib
d, name = sys.argv[1], sys.argv[2]; f = pathlib.Path(d) / "records.jsonl"
recs = [json.loads(l) for l in open(f)]
for r in recs: r["arm"] = name
open(f, "w").write("".join(json.dumps(r) + "\n" for r in recs)); print("arm ->", name, len(recs))
PY
    DIRS="$DIRS $DIR"; fi
done
[ -n "$DIRS" ] && bash $R/scripts/judge_base.sh $A $IDS $DIRS
kill $(cat $R/state/ir_server.pid) 2>/dev/null; log "KEYSTEP SWEEP COMPLETE"
