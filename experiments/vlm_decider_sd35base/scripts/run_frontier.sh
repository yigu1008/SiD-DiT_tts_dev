#!/usr/bin/env bash
# Budget frontier: run one arm at several simulation counts on the same prompts, judge all.
#   bash run_frontier.sh <IDS_FILE> <blind|jev> <sims...>
# blind needs 1 GPU (sampler + IR server); jev needs 2 (servers on the second).
set -uo pipefail
R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
IDS=$1; KIND=$2; shift 2; SIMS=("$@"); log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/frontier_driver.log; }
free_gpus(){ APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv,noheader,nounits | while IFS=, read g u m; do
    n=$(echo "$APPS" | grep -c "${u// /}"); [ "${m// /}" -lt 2000 ] && [ "$n" = 0 ] && echo $g; done; }
NEED=1; [ "$KIND" = jev ] && NEED=2
log "=== frontier $KIND sims=[${SIMS[*]}] ids=$IDS; waiting for $NEED free GPU(s)"
DEADLINE=$(( $(date +%s) + 36000 )); F=()
while [ $(date +%s) -lt $DEADLINE ]; do
  F=($(free_gpus)); if [ ${#F[@]} -ge $NEED ]; then ok=1; for j in 1 2; do sleep 20; G=($(free_gpus)); [ "${G[*]:0:$NEED}" = "${F[*]:0:$NEED}" ] || { ok=0; break; }; done; [ $ok = 1 ] && break; fi
  sleep 60; done
F=($(free_gpus)); [ ${#F[@]} -ge $NEED ] || { log "no $NEED stable free GPU(s); giving up"; exit 1; }
A=${F[0]}; log "sampler+IR gpu$A"; bash $R/scripts/ir_server.sh $A 5141 || exit 1
if [ "$KIND" = jev ]; then B=${F[1]}; log "servers gpu$B"; bash $R/scripts/vlm_server.sh $B 5151 && bash $R/scripts/jev_server.sh $B 5161 || exit 1
  nohup bash $R/scripts/keepalive2.sh $A $B $B > /dev/null 2>&1 & echo $! > $R/state/keepalive.pid; fi
DIRS=""
for NS in "${SIMS[@]}"; do
  NAME=fr_${KIND}_s${NS}; DIR=$R/main/$NAME; t=$(date +%s); log "--- $NAME start gpu$A"
  if [ "$KIND" = blind ]; then
    CUDA_VISIBLE_DEVICES=$A OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u $R/scripts/base_arm.py \
      --arm mcts --ids $IDS --n_sims $NS --run_dir $DIR --reward_server http://127.0.0.1:5141 >> $R/logs/frontier_$NAME.log 2>&1
  else
    CUDA_VISIBLE_DEVICES=$A OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u $R/scripts/vlm_mcts.py \
      --arm jev_prune --ids $IDS --n_sims $NS --run_dir $DIR --reward_server http://127.0.0.1:5141 \
      --vlm_server http://127.0.0.1:5151 --jev_server http://127.0.0.1:5161 >> $R/logs/frontier_$NAME.log 2>&1
  fi
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
kill $(cat $R/state/keepalive.pid 2>/dev/null) 2>/dev/null; [ "$KIND" = jev ] && kill $(cat $R/state/vlm_server.pid) $(cat $R/state/jev_server.pid) 2>/dev/null
[ -n "$DIRS" ] && bash $R/scripts/judge_base.sh $A $IDS $DIRS
kill $(cat $R/state/ir_server.pid) 2>/dev/null; log "FRONTIER $KIND COMPLETE"
