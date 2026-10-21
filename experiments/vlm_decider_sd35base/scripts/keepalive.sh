#!/usr/bin/env bash
# Watchdog: every 30 s check the IR / VLM / Jev servers; restart a dead one on its GPU.  bash keepalive.sh  (nohup)
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; SID=/home/ygu/project/SiD-DiT_tts_dev; RW=/home/ygu/miniconda3/envs/reward/bin/python
log(){ echo "[$(date '+%F %T')] $*" >> $R/logs/keepalive.log; }
log "watchdog start pid $$"
while true; do
  if ! curl -s --max-time 5 http://127.0.0.1:5141/health | grep -q '"status": "ok"'; then
    if ! ps -o pid= -p $(cat $R/state/ir_server.pid 2>/dev/null || echo 0) >/dev/null 2>&1; then
      log "IR server down; restarting on gpu1"
      CUDA_VISIBLE_DEVICES=1 HF_HUB_CACHE=/data/ygu/.cache/hub HF_HOME=/data/ygu/.cache/hub SID_FORCE_WANDB_STUB=1 WANDB_DISABLED=true PATH="$(dirname $RW):$PATH" nohup $RW -u $SID/reward_server.py --host 127.0.0.1 --port 5141 --device cuda:0 --backends imagereward --require_all_backends >> $R/logs/ir_server_gpu1_p5141_r2.log 2>&1 & echo $! > $R/state/ir_server.pid; sleep 60
    fi
  fi
  if ! curl -s --max-time 5 http://127.0.0.1:5151/health | grep -q '"status": "ok"'; then
    if ! ps -o pid= -p $(cat $R/state/vlm_server.pid 2>/dev/null || echo 0) >/dev/null 2>&1; then log "VLM server down; restarting on gpu3"; bash $R/scripts/vlm_server.sh 3 5151 >> $R/logs/keepalive.log 2>&1; fi
  fi
  if ! curl -s --max-time 5 http://127.0.0.1:5161/health | grep -q '"status": "ok"'; then
    if ! ps -o pid= -p $(cat $R/state/jev_server.pid 2>/dev/null || echo 0) >/dev/null 2>&1; then log "Jev server down; restarting on gpu6"; bash $R/scripts/jev_server.sh 6 5161 >> $R/logs/keepalive.log 2>&1; fi
  fi
  sleep 30
done
