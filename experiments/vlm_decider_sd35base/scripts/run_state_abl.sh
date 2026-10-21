#!/usr/bin/env bash
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16
free_gpus(){ APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv,noheader,nounits | while IFS=, read g u m; do
    n=$(echo "$APPS" | grep -c "${u// /}"); [ "${m// /}" -lt 2000 ] && [ "$n" = 0 ] && echo $g; done; }
for i in $(seq 1 120); do F=($(free_gpus)); [ ${#F[@]} -ge 1 ] && { A=${F[0]}; break; }; sleep 30; done
[ -n "${A:-}" ] || { echo "no free GPU"; exit 1; }
echo "[$(date '+%F %T')] state ablation: Jev on gpu$A" | tee -a $R/logs/state_abl.log
bash $R/scripts/jev_server.sh $A 5161 >> $R/logs/state_abl.log 2>&1 || exit 1
/home/ygu/miniconda3/envs/sid_dit/bin/python $R/scripts/state_ablation.py http://127.0.0.1:5161 8 2>&1 | tee -a $R/logs/state_abl.log
kill $(cat $R/state/jev_server.pid) 2>/dev/null; echo "STATE ABLATION COMPLETE" | tee -a $R/logs/state_abl.log
