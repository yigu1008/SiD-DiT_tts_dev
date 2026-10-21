#!/usr/bin/env bash
# bash ir_server.sh <GPU> [PORT=5141]  -- ImageReward-only reward server (reward env, ~2.3 GB); waits for /health
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; SID=/home/ygu/project/SiD-DiT_tts_dev; RW=/home/ygu/miniconda3/envs/reward/bin/python; GPU=$1; PORT=${2:-5141}
LOG=$R/logs/ir_server_gpu${GPU}_p${PORT}.log; echo "[$(date '+%F %T')] === start instance ===" >> $LOG
CUDA_VISIBLE_DEVICES=$GPU HF_HUB_CACHE=/data/ygu/.cache/hub HF_HOME=/data/ygu/.cache/hub SID_FORCE_WANDB_STUB=1 WANDB_DISABLED=true PATH="$(dirname $RW):$PATH" nohup bash -c "$RW -X faulthandler -u $SID/reward_server.py --host 127.0.0.1 --port $PORT --device cuda:0 --backends imagereward --require_all_backends; echo \"[\$(date '+%F %T')] EXIT code=\$?\"" >> $LOG 2>&1 &
sleep 0.5; echo $(pgrep -n -f "reward_server.py --host 127.0.0.1 --port $PORT") > $R/state/ir_server.pid; echo "[$(date '+%F %T')] IR server pid=$(cat $R/state/ir_server.pid) gpu=$GPU port=$PORT"
for i in $(seq 1 60); do sleep 5; curl -s --max-time 3 http://127.0.0.1:$PORT/health | grep -q '"status": "ok"' && { echo "healthy after $((i*5)) s"; exit 0; }; kill -0 $(cat $R/state/ir_server.pid) 2>/dev/null || { echo FATAL; tail -20 $LOG; exit 1; }; done; echo "not healthy after 300 s"; exit 1
