#!/usr/bin/env bash
# bash mr3_server.sh <GPU> [PORT=5142]  -- 3-backend search reward server (imagereward+hpsv3+pickscore,
# ~23 GB, `reward` env, HPSv3 raw scale).  VQAScore is deliberately NOT loaded: it stays a held-out judge.
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; SID=/home/ygu/project/SiD-DiT_tts_dev
RW=/home/ygu/miniconda3/envs/reward/bin/python; GPU=$1; PORT=${2:-5142}
LOG=$R/logs/mr3_server_gpu${GPU}_p${PORT}.log; echo "[$(date '+%F %T')] === start instance ===" >> $LOG
CUDA_VISIBLE_DEVICES=$GPU HF_HUB_CACHE=/data/ygu/.cache/hub HF_HOME=/data/ygu/.cache/hub HPSV3_SCORE_STYLE=raw \
  SID_FORCE_WANDB_STUB=1 WANDB_DISABLED=true PATH="$(dirname $RW):$PATH" \
  nohup bash -c "$RW -X faulthandler -u $SID/reward_server.py --host 127.0.0.1 --port $PORT --device cuda:0 --backends imagereward hpsv3 pickscore --require_all_backends; echo \"[\$(date '+%F %T')] EXIT code=\$?\"" >> $LOG 2>&1 &
sleep 0.5; echo $(pgrep -n -f "reward_server.py --host 127.0.0.1 --port $PORT") > $R/state/mr3_server.pid
echo "[$(date '+%F %T')] mr3 server pid=$(cat $R/state/mr3_server.pid) gpu=$GPU port=$PORT"
for i in $(seq 1 72); do sleep 5
  curl -s --max-time 3 http://127.0.0.1:$PORT/health | grep -q '"status": "ok"' && { echo "healthy after $((i*5)) s: $(curl -s http://127.0.0.1:$PORT/health)"; exit 0; }
  kill -0 $(cat $R/state/mr3_server.pid) 2>/dev/null || { echo FATAL; tail -25 $LOG; exit 1; }
done; echo "not healthy after 360 s"; tail -25 $LOG; exit 1
