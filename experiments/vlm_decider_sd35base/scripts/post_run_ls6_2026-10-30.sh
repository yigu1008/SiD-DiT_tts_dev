#!/usr/bin/env bash
# Runs automatically after run_ls6 (rerun2) exits: deploy the multi-image Jev server, fidelity judges on ec_v2w, Jev pairwise probe,
# VLM requirement-level half, final selector comparison, results into the repo. Logs: $RUN_DIR/logs/post_run.log
set -uo pipefail; source /scratch/09398/yigu/jev_decider/env.sh; R=$RUN_DIR; S=$REPO_DIR/experiments/vlm_decider_sd35base
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/post_run.log; }
PF=$R/state/run_ls6.pid; while kill -0 $(cat $PF) 2>/dev/null; do sleep 120; done; log "run_ls6 exited: $(tail -1 $R/logs/ls6_driver.log)"
sleep 20; for s in vlm jev mr3; do kill $(cat $R/state/${s}_server.pid 2>/dev/null) 2>/dev/null; done; sleep 10; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a $R/logs/post_run.log
log "1. sync repo -> run dir (multi-image jev_server, judge_ec_v2w)"; bash $R/sync_from_repo.sh >> $R/logs/post_run.log 2>&1
log "2. fidelity judges on ec_v2w (GPU0)"; bash $R/scripts/judge_ec_v2w.sh 0 >> $R/logs/post_run.log 2>&1; ls $R/scores_ec_v2w >> $R/logs/post_run.log 2>&1
log "3. Jev server (GPU1) + pairwise probe"; bash $R/scripts/jev_server.sh 1 5161 >> $R/logs/post_run.log 2>&1 && \
  $SIDDIT_PY_PREFIX/bin/python $S/scripts/jev_pairwise_probe.py $R/main/ec_v2w/records_all.jsonl --prompt_map $R/config/prompts_rsel9_index.json --out $S/results/jev_pairwise_ec_v2w_2026-10-30.json --pairs_per_prompt 40 >> $R/logs/post_run.log 2>&1
log "4. VLM server (GPU2) + requirement-level half + VLM pairwise-free grading already done"; bash $R/scripts/vlm_server.sh 2 5151 >> $R/logs/post_run.log 2>&1 && \
  $SIDDIT_PY_PREFIX/bin/python $S/scripts/requirements_grade.py $R/main/ec_v2w/records_all.jsonl --prompt_map $R/config/prompts_rsel9_index.json --reqs $R/config/requirements_rsel9.json --stage vlm --out $S/results/req_grade_vlm_ec_v2w_2026-10-30.json >> $R/logs/post_run.log 2>&1
log "5. results -> repo"; for a in ja_jevact_v2w ad_jevbank ad_randbank; do mkdir -p $S/results/main/$a; for f in records.jsonl decisions.jsonl decider_log.jsonl config.json aggregate_summary.json aggregate_summary.txt; do [ -f $R/main/$a/$f ] && cp $R/main/$a/$f $S/results/main/$a/; done; done
cp $R/scores.csv $S/results/scores.csv; mkdir -p $S/results/ls6_judge_2026-10-30 && cp -r $R/scores_ec_v2w $S/results/ls6_judge_2026-10-30/ 2>/dev/null; cp $R/logs/ls6_driver.log $S/results/ls6_logs/ls6_driver_rerun2.log 2>/dev/null
for s in vlm jev; do kill $(cat $R/state/${s}_server.pid 2>/dev/null) 2>/dev/null; done; log "POST-RUN CHAIN DONE"
