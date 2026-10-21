#!/usr/bin/env bash
# bash launch_cohort.sh <GPU> <arm: baseline|mcts|rand_prune|vlm_prune|jev_prune> <N_PROMPTS> [N_SIMS=40]
# Runs the arm over the first N ids of prompt_ids_99.txt into main/<arm> (resumable per prompt); log: logs/main_<arm>.log
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
GPU=$1; ARM=$2; N=$3; NS=${4:-40}; IDS=$R/config/prompt_ids_$N.txt; mkdir -p $R/config $R/main $R/logs $R/state
head -n $N /data/ygu/reward_orthogonality/followup_2026-09-24/prompt_ids_99.txt > $IDS
case $ARM in
  baseline|mcts|mcts_refine) S=$R/scripts/base_arm.py; EXTRA="" ;;
  opt_prompt) S=$R/scripts/opt_prompt_arm.py; EXTRA="--vlm_server http://127.0.0.1:5151 --jev_server http://127.0.0.1:5161" ;;
  rand_prune|vlm_prune|jev_prune) S=$R/scripts/vlm_mcts.py; EXTRA="--vlm_server http://127.0.0.1:5151 --jev_server http://127.0.0.1:5161" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
echo "[$(date '+%F %T')] --- main $ARM start gpu=$GPU n=$N sims=$NS ids=$(sha256sum $IDS | cut -c1-12)" | tee -a $R/logs/main_driver.log
CUDA_VISIBLE_DEVICES=$GPU OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup $PY -u $S --arm $ARM --ids $IDS --n_sims $NS --run_dir $R/main/$ARM --reward_server http://127.0.0.1:5141 $EXTRA >> $R/logs/main_$ARM.log 2>&1 &
echo $! > $R/state/main_$ARM.pid; echo "pid $(cat $R/state/main_$ARM.pid)"
