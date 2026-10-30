#!/usr/bin/env bash
# Held-out grader test: 16 prompts NOT used so far x 12 noises at the anchor action; then VQAScore, Qwen, Jev score on the finals.
# PRE-SPECIFIED before seeing results (from the rsel9 ladder): baseline = bo-12 by HPSv3; primary = top-2 HPSv3 -> Jev E[score] on VQAScore;
# secondary = top-4 -> Jev; Jev-only over all 12. Generation has a hard timeout so judges always get time on this node.
set -uo pipefail; source /scratch/09398/yigu/jev_decider/env.sh; R=$RUN_DIR; S=$REPO_DIR/experiments/vlm_decider_sd35base; PY=$SIDDIT_PY_PREFIX/bin/python
A=ns_ladder_new16; DIR=$R/main/$A; IDS=$R/config/prompt_ids_new16.txt; NS=12; GEN_TIMEOUT=${GEN_TIMEOUT:-3700}; mkdir -p $DIR
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/$A.log; }
log "=== start on $(hostname), gen timeout ${GEN_TIMEOUT}s"
bash $R/scripts/mr3_server.sh 0 5142 >> $R/logs/$A.log 2>&1 || { log "FATAL mr3 failed"; exit 1; }
PIDS=(); for i in 0 1; do G=$((i+1)); ( cd $REPO_DIR && CUDA_VISIBLE_DEVICES=$G OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout $GEN_TIMEOUT \
    $PY -u $R/scripts/noise_ladder_probe.py --ids $IDS --run_dir $DIR --n_seeds $NS --shard_index $i --num_shards 2 --reward_server http://127.0.0.1:5142 >> $R/logs/${A}_shard$i.log 2>&1; log "--- shard $i end rc=$? rows=$(wc -l < $DIR/records_shard$i.jsonl 2>/dev/null)" ) & PIDS+=($!); done
for p in "${PIDS[@]}"; do wait $p; done; cat $DIR/records_shard*.jsonl > $DIR/records_all.jsonl; log "generated $(wc -l < $DIR/records_all.jsonl)/192"
kill $(cat $R/state/mr3_server.pid 2>/dev/null) 2>/dev/null; sleep 10
MAN=$R/manifests_$A; OUT=$R/scores_$A; mkdir -p $MAN $OUT
$PY - $DIR/records_all.jsonl $MAN/followup_$A.jsonl $A <<'PY'
import json, sys, hashlib, os
recs, out, A = sys.argv[1:4]; n = 0
with open(out, "w") as f:
    for r in map(json.loads, open(recs)):
        if os.path.exists(r["image"]):
            f.write(json.dumps({"backend": f"followup:{A}", "arm": A, "prompt_id": r["prompt_id"], "sample_index": int(r["noise_seed"]), "prompt": r["prompt"], "source": "genai_bench", "category": "", "skills": "", "seed": r["noise_seed"], "image": r["image"], "image_sha256": hashlib.sha256(open(r["image"], "rb").read()).hexdigest()}) + "\n"); n += 1
print("manifest", n)
PY
( CUDA_VISIBLE_DEVICES=0 PATH="$VQA_PY_PREFIX/bin:$PATH" $VQA_PY_PREFIX/bin/python $R/deps/score_vqa.py --manifest_dir $MAN --out $OUT/vqa_scores.csv --shard_index 0 --num_shards 1 > $R/logs/${A}_vqa.log 2>&1; log "vqascore exit=$?" ) & V=$!
( CUDA_VISIBLE_DEVICES=2 $PY $R/deps/score_vlm_judge.py --manifest_dirs $MAN --out $OUT/vlm_scores.csv --shard_index 0 --num_shards 1 > $R/logs/${A}_qwen.log 2>&1; log "qwen judge exit=$?" ) & Q=$!
bash $R/scripts/jev_server.sh 1 5161 >> $R/logs/$A.log 2>&1 && $PY $S/scripts/jev_grade_images.py $DIR/records_all.jsonl --out $S/results/jev_grade_${A}_2026-10-30.json >> $R/logs/$A.log 2>&1; log "jev grading rc=$?"
wait $V; wait $Q; kill $(cat $R/state/jev_server.pid 2>/dev/null) 2>/dev/null
$PY $S/scripts/merge_ec_judges.py $MAN/followup_$A.jsonl $OUT/vqa_scores.shard0.csv $OUT/vlm_scores.shard0.csv --out $S/results/judges_${A}_2026-10-30.json >> $R/logs/$A.log 2>&1; log "merge rc=$?"
mkdir -p $S/results/main/$A; cp $DIR/records_all.jsonl $DIR/config_shard*.json $S/results/main/$A/; log "CHAIN DONE"
