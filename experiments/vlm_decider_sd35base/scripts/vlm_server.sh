#!/usr/bin/env bash
# bash vlm_server.sh <GPU> [PORT]   -- Qwen2.5-VL-7B decider on its own GPU (~17 GB), waits for /health
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python; GPU=$1; PORT=${2:-5151}
LOG=$R/logs/vlm_server_gpu${GPU}_p${PORT}.log; echo "[$(date '+%F %T')] === start instance ===" >> $LOG
CUDA_VISIBLE_DEVICES=$GPU nohup bash -c "$PY -X faulthandler -u $R/scripts/vlm_server.py --port $PORT; echo \"[\$(date '+%F %T')] EXIT code=\$? \"" >> $LOG 2>&1 &
sleep 0.5; echo $(pgrep -n -f "vlm_server.py --port $PORT") > $R/state/vlm_server.pid
echo "[$(date '+%F %T')] vlm server pid=$(cat $R/state/vlm_server.pid) gpu=$GPU port=$PORT"
for i in $(seq 1 60); do sleep 5; curl -s --max-time 3 http://127.0.0.1:$PORT/health | grep -q '"status": "ok"' && { echo "healthy after $((i*5)) s"; exit 0; }; kill -0 $(cat $R/state/vlm_server.pid) 2>/dev/null || { echo FATAL; tail -20 $R/logs/vlm_server_gpu${GPU}_p${PORT}.log; exit 1; }; done; echo "not healthy after 300 s"; exit 1
