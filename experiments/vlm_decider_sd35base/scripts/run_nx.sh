#!/usr/bin/env bash
# Adaptive noise-budget arms, identical machinery, differing only in who decides.
#   bash run_nx.sh <G_JEV> <G_REWARD> <G_RANDOM> <G_NEVER> [IDS] [ABANDON_RATE] [EXPAND_RATE]
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
S1=$1; S2=$2; S3=$3; S4=$4; IDS=${5:-$R/config/prompt_ids_rsel9.txt}; AR=${6:-0.52}; ER=${7:-0.5}
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/nx_driver.log; }
for P in 5142 5151 5161; do curl -s --max-time 5 http://127.0.0.1:$P/health | grep -q '"status": "ok"' || { log "server $P not healthy"; exit 1; }; done
log "=== nx arms: jev=gpu$S1 reward=gpu$S2 random=gpu$S3 (ar=$AR er=$ER) never=gpu$S4  n=$(grep -c . $IDS)"
run(){ local G=$1 NAME=$2 POL=$3; shift 3; local t=$(date +%s); log "--- $NAME start (policy=$POL) gpu$G"
  CUDA_VISIBLE_DEVICES=$G OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u $R/scripts/jev_noise_budget.py --ids $IDS --run_dir $R/main/$NAME --policy $POL "$@" \
    --reward_server http://127.0.0.1:5142 --vlm_server http://127.0.0.1:5151 --jev_server http://127.0.0.1:5161 \
    >> $R/logs/nx_$NAME.log 2>&1
  log "--- $NAME end rc=$? in $(( $(date +%s)-t ))s records=$(wc -l < $R/main/$NAME/records.jsonl 2>/dev/null)"; }
run $S1 nx_jev    jev & P1=$!
run $S2 nx_reward reward & P2=$!
run $S3 nx_random random --abandon_rate $AR --expand_rate $ER & P3=$!
run $S4 nx_never  never & P4=$!
for p in $P1 $P2 $P3 $P4; do wait $p; done
log "all nx arms done"
