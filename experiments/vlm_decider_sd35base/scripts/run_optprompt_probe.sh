#!/usr/bin/env bash
# Wait for TWO ryan-free GPUs (checked at launch time), then: IR server + sampler on GPU A, VLM + Jev servers on GPU B (17+24 GB),
# run opt_prompt on the first N prompts.   bash run_optprompt_probe.sh <N> [MAX_WAIT_MIN=600]
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python; N=${1:-5}; MAXW=${2:-600}; SUB=${3:-opt_prompt}; IDSOVR=${4:-}; RUNDIR=${5:-$R/probe/$SUB}
IDS=${IDSOVR:-/data/ygu/reward_orthogonality/followup_2026-09-24/prompt_ids_99.txt}; log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/optprompt_driver.log; }
free_gpus(){ # a GPU is free only if < 2 GB used AND it has NO compute process at all (owners may be invisible to ps, e.g. containers)
  APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv,noheader,nounits | while IFS=, read g u m; do
    n=$(echo "$APPS" | grep -c "${u// /}"); [ "${m// /}" -lt 2000 ] && [ "$n" = 0 ] && echo $g; done; }
log "waiting for 2 ryan-free GPUs (max $MAXW min) for opt_prompt probe n=$N"
DEADLINE=$(( $(date +%s) + MAXW*60 )); F=()
while [ $(date +%s) -lt $DEADLINE ]; do
  F=($(free_gpus)); if [ ${#F[@]} -ge 2 ]; then   # require 3 consecutive stable readings 20 s apart (ryan's workers restart within seconds)
    ok=1; for j in 1 2; do sleep 20; G=($(free_gpus)); [ "${G[*]}" = "${F[*]}" ] || { ok=0; break; }; done
    [ $ok = 1 ] && break; log "transient free reading [${F[*]}] not stable; keep waiting"; fi
  sleep 60; done
F=($(free_gpus)); [ ${#F[@]} -ge 2 ] || { log "no 2 stable free GPUs by the deadline ($MAXW min); giving up"; exit 1; }
A=${F[0]}; B=${F[1]}; log "free GPUs: ${F[*]} -> sampler+IR on gpu$A, VLM+Jev on gpu$B"
bash $R/scripts/ir_server.sh $A 5141 && bash $R/scripts/vlm_server.sh $B 5151 && bash $R/scripts/jev_server.sh $B 5161 || { log "server start failed"; exit 1; }
t0=$(date +%s); log "--- opt_prompt start gpu=$A n=$N"
CUDA_VISIBLE_DEVICES=$A OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u $R/scripts/opt_prompt_arm.py --ids $IDS --limit $N --run_dir $RUNDIR --reward_server http://127.0.0.1:5141 --vlm_server http://127.0.0.1:5151 --jev_server http://127.0.0.1:5161 >> $R/logs/probe_$SUB.log 2>&1
log "--- opt_prompt end rc=$? in $(( $(date +%s)-t0 ))s records=$(wc -l < $RUNDIR/records.jsonl 2>/dev/null)"
kill $(cat $R/state/keepalive.pid 2>/dev/null) 2>/dev/null; kill $(cat $R/state/vlm_server.pid) $(cat $R/state/jev_server.pid) $(cat $R/state/ir_server.pid) 2>/dev/null; log "watchdog + servers stopped; OPTPROMPT PROBE COMPLETE"
