#!/usr/bin/env bash
# Per-node reward selection: Jev matches the VLM-observed deficiency to each reward's focused area.
#   bash run_rsel.sh [IDS=config/prompt_ids_rsel9.txt] [SIMS=10]
# Four arms at a matched 10-sim / 616-NFE operating point, identical machinery except the reward step:
#   rs_jev   Jev picks the reward at each node from the evidence
#   rs_ir    fixed ImageReward          (matched control: isolates the selection step)
#   rs_hps   fixed HPSv3                (matched control: what following appearance alone does)
#   rs_blind blind key-step MCTS on IR  (no Jev, no VLM -- the reference the whole study is measured against)
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
IDS=${1:-$R/config/prompt_ids_rsel9.txt}; NS=${2:-10}
mkdir -p $R/logs $R/main $R/state; log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/rsel_driver.log; }
free_gpus(){ APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv,noheader,nounits | while IFS=, read g u m; do
    n=$(echo "$APPS" | grep -c "${u// /}"); [ "${m// /}" -lt 2000 ] && [ "$n" = 0 ] && echo $g; done; }
DEADLINE=$(( $(date +%s) + 36000 ))
while [ $(date +%s) -lt $DEADLINE ]; do F=($(free_gpus)); [ ${#F[@]} -ge 6 ] && { sleep 20; G=($(free_gpus)); [ "${G[*]:0:6}" = "${F[*]:0:6}" ] && break; }; sleep 60; done
F=($(free_gpus)); [ ${#F[@]} -ge 6 ] || { log "need 6 free GPUs, have ${#F[@]} (${F[*]}); ryan is using the rest -- not launching"; exit 1; }
S1=${F[0]}; S2=${F[1]}; S3=${F[2]}; S4=${F[3]}; MR=${F[4]}; VJ=${F[5]}
log "=== rsel: samplers gpu$S1/$S2/$S3 (+blind gpu$S4), reward server gpu$MR, vlm+jev gpu$VJ, sims=$NS, n=$(grep -c . $IDS)"
bash $R/scripts/mr3_server.sh $MR 5142 && bash $R/scripts/vlm_server.sh $VJ 5151 && bash $R/scripts/jev_server.sh $VJ 5161 || { log "server startup failed"; exit 1; }
nohup bash $R/scripts/keepalive3.sh $MR $VJ $VJ > /dev/null 2>&1 & echo $! > $R/state/keepalive.pid
URL=http://127.0.0.1:5142
run_sel(){ # <gpu> <name> <policy>
  local G=$1 NAME=$2 POL=$3 t=$(date +%s); log "--- $NAME start (policy=$POL) gpu$G"
  CUDA_VISIBLE_DEVICES=$G OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u $R/scripts/jev_reward_select.py --ids $IDS --n_sims $NS --run_dir $R/main/$NAME --reward_policy $POL \
    --reward_server $URL --vlm_server http://127.0.0.1:5151 --jev_server http://127.0.0.1:5161 >> $R/logs/rsel_$NAME.log 2>&1
  log "--- $NAME end rc=$? in $(( $(date +%s)-t ))s records=$(wc -l < $R/main/$NAME/records.jsonl 2>/dev/null)"; }
run_blind(){ local G=$1 t=$(date +%s); log "--- rs_blind start gpu$G"
  CUDA_VISIBLE_DEVICES=$G OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u $R/scripts/base_arm.py --arm mcts --ids $IDS --n_sims $NS --run_dir $R/main/rs_blind --reward_server $URL >> $R/logs/rsel_rs_blind.log 2>&1
  log "--- rs_blind end rc=$? in $(( $(date +%s)-t ))s records=$(wc -l < $R/main/rs_blind/records.jsonl 2>/dev/null)"; }
run_sel $S1 rs_jev jev & P1=$!
run_sel $S2 rs_ir  imagereward & P2=$!
run_sel $S3 rs_hps hpsv3 & P3=$!
run_blind $S4 & P4=$!
for p in $P1 $P2 $P3 $P4; do wait $p; done
log "all arms done"
kill $(cat $R/state/keepalive.pid) 2>/dev/null
kill $(cat $R/state/vlm_server.pid) $(cat $R/state/jev_server.pid) $(cat $R/state/mr3_server.pid) 2>/dev/null; sleep 15
DIRS=""; for n in rs_jev rs_ir rs_hps rs_blind; do [ -s $R/main/$n/records.jsonl ] && DIRS="$DIRS $R/main/$n"; done
[ -n "$DIRS" ] && bash $R/scripts/judge_base.sh $S1 $IDS $DIRS
log "RSEL RUN COMPLETE"
