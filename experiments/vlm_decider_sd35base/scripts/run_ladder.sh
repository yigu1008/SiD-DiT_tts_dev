#!/usr/bin/env bash
# Blind-search ladder on an id-set: baseline + stock basic MCTS (+ optional mcts_refine). Needs ONE free GPU
# (the 2.3 GB ImageReward server shares it with the 26 GB sampler). No VLM/Jev servers involved.
#   bash run_ladder.sh <IDS_FILE> <PREFIX> [N_SIMS=40] [ARMS="baseline mcts"]
set -uo pipefail
R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
IDS=$1; PRE=$2; NS=${3:-40}; ARMS="${4:-baseline mcts}"; log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/ladder_driver.log; }
free_gpus(){ APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv,noheader,nounits | while IFS=, read g u m; do
    n=$(echo "$APPS" | grep -c "${u// /}"); [ "${m// /}" -lt 2000 ] && [ "$n" = 0 ] && echo $g; done; }
log "=== ladder: ids=$IDS arms=[$ARMS] sims=$NS prefix=$PRE; waiting for 1 free GPU"
DEADLINE=$(( $(date +%s) + 36000 )); A=""
while [ $(date +%s) -lt $DEADLINE ]; do
  F=($(free_gpus)); if [ ${#F[@]} -ge 1 ]; then ok=1; for j in 1 2; do sleep 20; G=($(free_gpus)); echo " ${G[*]} " | grep -q " ${F[0]} " || { ok=0; break; }; done
    [ $ok = 1 ] && { A=${F[0]}; break; }; fi
  sleep 60; done
[ -n "$A" ] || { log "no stable free GPU; giving up"; exit 1; }
log "using gpu$A for sampler + IR server"
bash $R/scripts/ir_server.sh $A 5141 || { log "IR server failed"; exit 1; }
DIRS=""
for ARM in $ARMS; do
  DIR=$R/main/${PRE}_$ARM; t=$(date +%s); log "--- ${PRE}_$ARM start gpu$A"
  CUDA_VISIBLE_DEVICES=$A OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u $R/scripts/base_arm.py \
    --arm $ARM --ids $IDS --n_sims $NS --run_dir $DIR --reward_server http://127.0.0.1:5141 >> $R/logs/ladder_${PRE}_$ARM.log 2>&1
  log "--- ${PRE}_$ARM end rc=$? in $(( $(date +%s)-t ))s records=$(wc -l < $DIR/records.jsonl 2>/dev/null)"
  if [ -s $DIR/records.jsonl ]; then
    $PY - "$DIR" "${PRE}_$ARM" <<'PY'
import json, sys, pathlib
d, name = sys.argv[1], sys.argv[2]; f = pathlib.Path(d) / "records.jsonl"
recs = [json.loads(l) for l in open(f)]
for r in recs: r["arm"] = name
open(f, "w").write("".join(json.dumps(r) + "\n" for r in recs)); print("arm ->", name, len(recs))
PY
    DIRS="$DIRS $DIR"; fi
done
[ -n "$DIRS" ] && bash $R/scripts/judge_base.sh $A $IDS $DIRS
kill $(cat $R/state/ir_server.pid) 2>/dev/null
$PY $R/scripts/sweep_table.py $IDS; log "LADDER COMPLETE"
