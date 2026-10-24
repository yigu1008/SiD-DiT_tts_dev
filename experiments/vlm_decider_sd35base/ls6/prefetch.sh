#!/usr/bin/env bash
# Run on a LOGIN node (survives compute jobs): bash prefetch.sh [a|b|c|all]   -> logs in $SCRATCH/jev_decider/logs/
D=/scratch/09398/yigu/jev_decider; mkdir -p $D/logs
G=${1:-all}
export HF_HOME=/scratch/09398/yigu/hf_cache HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
nohup /scratch/09398/yigu/jev_probe/venv_small/bin/python -u $D/setup/prefetch_models.py $G >> $D/setup/logs/prefetch_$G.log 2>&1 &
echo "started group $G pid $! on $(hostname); log $D/setup/logs/prefetch_$G.log"
