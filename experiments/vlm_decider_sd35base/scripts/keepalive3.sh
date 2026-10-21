#!/usr/bin/env bash
# Watchdog for the reward-selection run: bash keepalive3.sh <MR3_GPU> <VLM_GPU> <JEV_GPU>
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; MRG=$1; VLG=$2; JVG=$3
log(){ echo "[$(date '+%F %T')] $*" >> $R/logs/keepalive.log; }; log "watchdog3 start pid $$ mr3=gpu$MRG vlm=gpu$VLG jev=gpu$JVG"
while true; do
  curl -s --max-time 5 http://127.0.0.1:5142/health | grep -q '"status": "ok"' || { ps -o pid= -p $(cat $R/state/mr3_server.pid 2>/dev/null || echo 0) >/dev/null 2>&1 || { log "mr3 down -> restart gpu$MRG"; bash $R/scripts/mr3_server.sh $MRG 5142 >> $R/logs/keepalive.log 2>&1; }; }
  curl -s --max-time 5 http://127.0.0.1:5151/health | grep -q '"status": "ok"' || { ps -o pid= -p $(cat $R/state/vlm_server.pid 2>/dev/null || echo 0) >/dev/null 2>&1 || { log "VLM down -> restart gpu$VLG"; bash $R/scripts/vlm_server.sh $VLG 5151 >> $R/logs/keepalive.log 2>&1; }; }
  curl -s --max-time 5 http://127.0.0.1:5161/health | grep -q '"status": "ok"' || { ps -o pid= -p $(cat $R/state/jev_server.pid 2>/dev/null || echo 0) >/dev/null 2>&1 || { log "Jev down -> restart gpu$JVG"; bash $R/scripts/jev_server.sh $JVG 5161 >> $R/logs/keepalive.log 2>&1; }; }
  sleep 30; done
