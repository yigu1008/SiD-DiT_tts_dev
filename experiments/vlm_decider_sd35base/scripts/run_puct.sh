#!/usr/bin/env bash
# PUCT soft prior vs hard pruning vs blind, all at the same sims.  bash run_puct.sh <IDS_FILE> [SIMS=10]
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
IDS=$1; NS=${2:-10}; log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/puct_driver.log; }
free_gpus(){ APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv,noheader,nounits | while IFS=, read g u m; do
    n=$(echo "$APPS" | grep -c "${u// /}"); [ "${m// /}" -lt 2000 ] && [ "$n" = 0 ] && echo $g; done; }
DEADLINE=$(( $(date +%s) + 36000 ))
while [ $(date +%s) -lt $DEADLINE ]; do F=($(free_gpus)); [ ${#F[@]} -ge 2 ] && { sleep 20; G=($(free_gpus)); [ "${G[*]:0:2}" = "${F[*]:0:2}" ] && break; }; sleep 60; done
F=($(free_gpus)); [ ${#F[@]} -ge 2 ] || { log "no 2 free GPUs"; exit 1; }
A=${F[0]}; B=${F[1]}; log "sampler+IR gpu$A, servers gpu$B, sims=$NS"
bash $R/scripts/ir_server.sh $A 5141 && bash $R/scripts/vlm_server.sh $B 5151 && bash $R/scripts/jev_server.sh $B 5161 || exit 1
nohup bash $R/scripts/keepalive2.sh $A $B $B > /dev/null 2>&1 & echo $! > $R/state/keepalive.pid
DIRS=""; NAME=pu_puct_s${NS}; DIR=$R/main/$NAME; t=$(date +%s); log "--- $NAME start (joint prior, no pruning)"
CUDA_VISIBLE_DEVICES=$A OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u $R/scripts/vlm_mcts.py \
  --arm jev_prune --ids $IDS --n_sims $NS --run_dir $DIR --reward_server http://127.0.0.1:5141 \
  --vlm_server http://127.0.0.1:5151 --jev_server http://127.0.0.1:5161 --joint --puct >> $R/logs/puct_$NAME.log 2>&1
log "--- $NAME end rc=$? in $(( $(date +%s)-t ))s records=$(wc -l < $DIR/records.jsonl 2>/dev/null)"
if [ -s $DIR/records.jsonl ]; then
  $PY - "$DIR" "$NAME" <<'PY'
import json, sys, pathlib
d, name = sys.argv[1], sys.argv[2]; f = pathlib.Path(d) / "records.jsonl"
recs = [json.loads(l) for l in open(f)]
for r in recs: r["arm"] = name
open(f, "w").write("".join(json.dumps(r) + "\n" for r in recs)); print("arm ->", name, len(recs))
PY
  DIRS="$DIR"; fi
kill $(cat $R/state/keepalive.pid) 2>/dev/null; kill $(cat $R/state/vlm_server.pid) $(cat $R/state/jev_server.pid) 2>/dev/null
[ -n "$DIRS" ] && bash $R/scripts/judge_base.sh $A $IDS $DIRS
kill $(cat $R/state/ir_server.pid) 2>/dev/null; log "PUCT RUN COMPLETE"
