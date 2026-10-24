#!/usr/bin/env bash
#SBATCH -J jevdec
#SBATCH -p gpu-a100
#SBATCH -N 1 -n 1
#SBATCH -t 48:00:00
#SBATCH -A ASC25020
#SBATCH -o /data/ygu/runs/vlm_decider_sd35base_2026-10-16/logs/slurm_%j.out
# 48 h is the gpu-a100 ceiling on LS6; everything is resumable, so if it is cut, resubmit as is.
# Pass STAGES through:  sbatch --export=ALL,STAGES="actdiff judge" run_ls6.sbatch
source /data/ygu/runs/vlm_decider_sd35base_2026-10-16/env.sh
exec bash $RUN_DIR/scripts/run_ls6.sh
