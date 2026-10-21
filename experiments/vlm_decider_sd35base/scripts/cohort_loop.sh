#!/usr/bin/env bash
# Relaunch the resumable opt_prompt cohort until all N records exist (max ATTEMPTS), then run the finish script.  bash cohort_loop.sh <N> [ATTEMPTS=4]
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; N=${1:-25}; ATT=${2:-4}; log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/optprompt_driver.log; }
for i in $(seq 1 $ATT); do
  have=$(wc -l < $R/main/opt_prompt/records.jsonl 2>/dev/null || echo 0); [ "$have" -ge "$N" ] && break
  log "cohort attempt $i: $have/$N records, relaunching"; bash $R/scripts/run_optprompt_probe.sh $N 600 opt_prompt $R/config/prompt_ids_$N.txt $R/main/opt_prompt
done
have=$(wc -l < $R/main/opt_prompt/records.jsonl 2>/dev/null || echo 0)
if [ "$have" -ge "$N" ]; then log "cohort complete ($have/$N); judging"; bash $R/scripts/finish_cohort_optprompt.sh 0; else log "cohort INCOMPLETE after $ATT attempts ($have/$N)"; fi
