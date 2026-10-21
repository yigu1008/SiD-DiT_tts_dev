#!/usr/bin/env bash
# Watchdog with explicit GPUs: bash keepalive2.sh <IR_GPU> <VLM_GPU> <JEV_GPU>   (restarts a dead server on its GPU; checks every 30 s)
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; IRG=$1; VLG=$2; JVG=$3
log(){ echo "[$(date '+%F %T')] $*" >> $R/logs/keepalive.log; }; log "watchdog2 start pid $$ ir=gpu$IRG vlm=gpu$VLG jev=gpu$JVG"
while true; do
  curl -s --max-time 5 http://127.0.0.1:5141/health | grep -q '"status": "ok"' || { ps -o pid= -p $(cat $R/state/ir_server.pid 2>/dev/null || echo 0) >/dev/null 2>&1 || { log "IR down -> restart gpu$IRG"; bash $R/scripts/ir_server.sh $IRG 5141 >> $R/logs/keepalive.log 2>&1; }; }
  curl -s --max-time 5 http://127.0.0.1:5151/health | grep -q '"status": "ok"' || { ps -o pid= -p $(cat $R/state/vlm_server.pid 2>/dev/null || echo 0) >/dev/null 2>&1 || { log "VLM down -> restart gpu$VLG"; bash $R/scripts/vlm_server.sh $VLG 5151 >> $R/logs/keepalive.log 2>&1; }; }
  curl -s --max-time 5 http://127.0.0.1:5161/health | grep -q '"status": "ok"' || { ps -o pid= -p $(cat $R/state/jev_server.pid 2>/dev/null || echo 0) >/dev/null 2>&1 || { log "Jev down -> restart gpu$JVG"; bash $R/scripts/jev_server.sh $JVG 5161 >> $R/logs/keepalive.log 2>&1; }; }
  sleep 30; done
