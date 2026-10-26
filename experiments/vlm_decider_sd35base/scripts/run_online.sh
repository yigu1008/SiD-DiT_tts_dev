#!/usr/bin/env bash
# Online wording-search arm + matched fixed-bank control on the same prompts, then judging.
#   bash run_online.sh <IDS_FILE> <SUBDIR_PREFIX> [N_SIMS=40] [MAXW_MIN=600]
# Waits for 2 GPUs with NO compute processes (ryan's workers come and go), puts sampler+IR on one and VLM+Jev on the other.
set -uo pipefail
R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
IDS=$1; PRE=$2; NS=${3:-40}; MAXW=${4:-600}; ONLINE_ARGS="${5:-}"; log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/online_driver.log; }
free_gpus(){ APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv,noheader,nounits | while IFS=, read g u m; do
    n=$(echo "$APPS" | grep -c "${u// /}"); [ "${m// /}" -lt 2000 ] && [ "$n" = 0 ] && echo $g; done; }
N=$(grep -c . $IDS); log "=== online run: ids=$IDS n=$N sims=$NS prefix=$PRE; waiting for 2 free GPUs (max $MAXW min)"
DEADLINE=$(( $(date +%s) + MAXW*60 )); F=()
while [ $(date +%s) -lt $DEADLINE ]; do
  F=($(free_gpus)); if [ ${#F[@]} -ge 2 ]; then ok=1; for j in 1 2; do sleep 20; G=($(free_gpus)); [ "${G[*]}" = "${F[*]}" ] || { ok=0; break; }; done; [ $ok = 1 ] && break; fi
  sleep 60; done
F=($(free_gpus)); [ ${#F[@]} -ge 2 ] || { log "no 2 stable free GPUs by the deadline; giving up"; exit 1; }
A=${F[0]}; B=${F[1]}; log "free GPUs: ${F[*]} -> sampler+IR gpu$A, VLM+Jev gpu$B"
bash $R/scripts/ir_server.sh $A 5141 && bash $R/scripts/vlm_server.sh $B 5151 && bash $R/scripts/jev_server.sh $B 5161 || { log "server start failed"; exit 1; }
nohup bash $R/scripts/keepalive2.sh $A $B $B > /dev/null 2>&1 & echo $! > $R/state/keepalive.pid; log "watchdog pid $(cat $R/state/keepalive.pid)"
COMMON="--ids $IDS --n_sims $NS --reward_server http://127.0.0.1:5141 --vlm_server http://127.0.0.1:5151 --jev_server http://127.0.0.1:5161"
t=$(date +%s); log "--- ${PRE}_jev_online start gpu$A"
CUDA_VISIBLE_DEVICES=$A OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u $R/scripts/jev_online.py --arm ${PRE}_jev_online --run_dir $R/main/${PRE}_jev_online $COMMON $ONLINE_ARGS >> $R/logs/online_${PRE}_jev_online.log 2>&1
log "--- ${PRE}_jev_online end rc=$? in $(( $(date +%s)-t ))s records=$(wc -l < $R/main/${PRE}_jev_online/records.jsonl 2>/dev/null)"
t=$(date +%s); log "--- ${PRE}_jev_prune (fixed-bank control) start gpu$A"
CUDA_VISIBLE_DEVICES=$A OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u $R/scripts/vlm_mcts.py --arm jev_prune --run_dir $R/main/${PRE}_jev_prune $COMMON >> $R/logs/online_${PRE}_jev_prune.log 2>&1
log "--- ${PRE}_jev_prune end rc=$? in $(( $(date +%s)-t ))s records=$(wc -l < $R/main/${PRE}_jev_prune/records.jsonl 2>/dev/null)"
kill $(cat $R/state/keepalive.pid) 2>/dev/null; kill $(cat $R/state/vlm_server.pid) $(cat $R/state/jev_server.pid) 2>/dev/null; sleep 3
$PY - "$R/main/${PRE}_jev_prune" <<'PY'
import json, sys, pathlib
d = sys.argv[1]; f = pathlib.Path(d) / "records.jsonl"
if f.exists():
    recs = [json.loads(l) for l in open(f)]
    for r in recs: r["arm"] = pathlib.Path(d).name
    open(f, "w").write("".join(json.dumps(r) + "\n" for r in recs)); print("control arm renamed:", pathlib.Path(d).name, len(recs))
PY
DIRS=""; for s in jev_online jev_prune; do [ -s $R/main/${PRE}_$s/records.jsonl ] && DIRS="$DIRS $R/main/${PRE}_$s"; done
[ -n "$DIRS" ] && bash $R/scripts/judge_base.sh $A $IDS $DIRS
kill $(cat $R/state/ir_server.pid) 2>/dev/null
D=/data/ygu/reward_orthogonality/followup_2026-09-24/scripts
log "=== paired contrasts: ${PRE}_jev_online minus ${PRE}_jev_prune ==="; FF=$R/main/contrast_${PRE}_online_minus_prune.txt; : > $FF
for M in ir hpsv3 pickscore hpsv2 vqascore qwen_alignment; do $PY $D/same_day_analysis.py paired --scores $R/scores.csv --ids $IDS --a ${PRE}_jev_online --b ${PRE}_jev_prune --metric $M >> $FF 2>&1; done
cat $FF | tee -a $R/logs/online_driver.log; log "ONLINE RUN COMPLETE"
