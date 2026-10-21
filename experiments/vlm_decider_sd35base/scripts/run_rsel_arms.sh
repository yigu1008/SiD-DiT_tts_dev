#!/usr/bin/env bash
# Reward-selection arms against ALREADY-RUNNING servers (mr3 on 5142, vlm 5151, jev 5161).
#   bash run_rsel_arms.sh <G_JEV> <G_IR> <G_HPS> <G_BLIND> [IDS] [SIMS]
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
S1=$1; S2=$2; S3=$3; S4=$4; IDS=${5:-$R/config/prompt_ids_rsel9.txt}; NS=${6:-10}
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/rsel_driver.log; }
for P in 5142 5151 5161; do curl -s --max-time 5 http://127.0.0.1:$P/health | grep -q '"status": "ok"' || { log "server on $P not healthy"; exit 1; }; done
URL=http://127.0.0.1:5142
log "=== rsel arms: jev=gpu$S1 ir=gpu$S2 hps=gpu$S3 blind=gpu$S4 sims=$NS n=$(grep -c . $IDS)"
run_sel(){ local G=$1 NAME=$2 POL=$3 t=$(date +%s); log "--- $NAME start (policy=$POL) gpu$G"
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
log "all rsel arms done"
