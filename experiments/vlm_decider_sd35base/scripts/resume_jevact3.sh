#!/usr/bin/env bash
# Keep resuming run_jevact3.sh until it reports COMPLETE. Each pass waits for GPUs itself and skips finished
# prompts, so a SIGTERM from another user's launch (what happened 2026-10-21 19:53) only costs the in-flight prompt.
R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/jevact3_driver.log; }
DEADLINE=$(( $(date +%s) + 172800 )); pass=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  pass=$((pass+1)); log "=== resume pass $pass"
  bash $R/scripts/run_jevact3.sh; rc=$?
  [ $rc -eq 0 ] && { log "resume: complete after $pass pass(es)"; exit 0; }
  log "resume: pass $pass exited rc=$rc; sleeping 10 min before trying again"; sleep 600
done
log "resume: 48 h deadline reached without completion"; exit 1
