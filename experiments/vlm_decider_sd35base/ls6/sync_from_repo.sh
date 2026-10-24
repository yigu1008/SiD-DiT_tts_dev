#!/usr/bin/env bash
# Re-copy scripts/ config/ deps/ from the git clone into $RUN_DIR and rewrite the A6000 paths (node_remap.sh).
# Run after every `git pull`. Never touches main/ logs/ state/ scores.csv.
set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SRC=$REPO_DIR/experiments/vlm_decider_sd35base
rm -rf $RUN_DIR/scripts $RUN_DIR/config $RUN_DIR/deps
cp -r $SRC/scripts $SRC/config $SRC/deps $SRC/node_remap.sh $RUN_DIR/; rm -rf $RUN_DIR/scripts/__pycache__
cp $SRC/PLAN.md $SRC/RECOMMENDATION.md $SRC/RESUME.md $RUN_DIR/ 2>/dev/null || true
HF_CACHE=$HF_HOME bash $RUN_DIR/node_remap.sh
# shipped results that are INPUTS here: ns_ladder (early_control.py reads it); scores.csv seeded once so rsel_table can compare to old arms
[ -s $RUN_DIR/main/ns_ladder/records_all.jsonl ] || { mkdir -p $RUN_DIR/main; cp -r $SRC/results/main/ns_ladder $RUN_DIR/main/; }
[ -s $RUN_DIR/scores.csv ] || cp $SRC/results/scores.csv $RUN_DIR/scores.csv
echo "synced from $(git -C $REPO_DIR rev-parse --short HEAD) at $(date '+%F %T')" | tee $RUN_DIR/SYNCED_FROM
