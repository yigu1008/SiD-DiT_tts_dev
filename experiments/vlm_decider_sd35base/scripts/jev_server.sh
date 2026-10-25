#!/usr/bin/env bash
# bash jev_server.sh <GPU> [PORT=5161]  -- Jev-Omni (~24 GB bf16) decision server in the jev env; waits for /health
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/jev/bin/python; GPU=$1; PORT=${2:-5161}
LOG=$R/logs/jev_server_gpu${GPU}_p${PORT}.log; echo "[$(date '+%F %T')] === start instance ===" >> $LOG
CUDA_VISIBLE_DEVICES=$GPU nohup bash -c "$PY -X faulthandler -u $R/scripts/jev_server.py --port $PORT; echo \"[\$(date '+%F %T')] EXIT code=\$? \"" >> $LOG 2>&1 &
sleep 0.5; echo $(pgrep -n -f "jev_server.py --port $PORT") > $R/state/jev_server.pid
echo "[$(date '+%F %T')] jev server pid=$(cat $R/state/jev_server.pid) gpu=$GPU port=$PORT"
for i in $(seq 1 180); do sleep 5; curl -s --max-time 3 http://127.0.0.1:$PORT/health | grep -q '"status": "ok"' && { echo "healthy after $((i*5)) s"; exit 0; }; kill -0 $(cat $R/state/jev_server.pid) 2>/dev/null || { echo FATAL; tail -25 $R/logs/jev_server_gpu${GPU}_p${PORT}.log; exit 1; }; done; echo "not healthy after 900 s"; exit 1
