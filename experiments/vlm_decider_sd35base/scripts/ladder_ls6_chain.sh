#!/usr/bin/env bash
# Regenerate the noise ladder ON LS6 with images + step previews, then judge it (VQAScore, Qwen) and grade it with Jev, so the
# two-layer policy (BoN noise by HPSv3 -> re-rank top-k by Jev fidelity score) can be measured on 9 prompts x 32 noises.
#   GPU0 mr3 reward server; GPU1+GPU2 one ladder shard each (144 generations x 56 NFE, ~80 min); then GPU0 judges, GPU1 Jev grading.
# Output: $R/main/ns_ladder_ls6/{images,previews,records_*}, $R/scores_ns_ladder_ls6/, results/jev_grade_ns_ladder_ls6.json
set -uo pipefail; source /scratch/09398/yigu/jev_decider/env.sh; R=$RUN_DIR; S=$REPO_DIR/experiments/vlm_decider_sd35base; PY=$SIDDIT_PY_PREFIX/bin/python
DIR=$R/main/ns_ladder_ls6; IDS=$R/config/prompt_ids_rsel9.txt; NS=${NS:-32}; mkdir -p $DIR $R/logs
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/ladder_ls6.log; }
up(){ curl -s --max-time 5 http://127.0.0.1:$1/health | grep -q '"status": "ok"'; }
log "=== ladder regen on $(hostname): n_seeds=$NS"
up 5142 || bash $R/scripts/mr3_server.sh 0 5142 >> $R/logs/ladder_ls6.log 2>&1 || { log "mr3 failed"; exit 1; }
PIDS=(); for i in 0 1; do G=$((i+1))
  ( t=$(date +%s); cd $REPO_DIR && CUDA_VISIBLE_DEVICES=$G OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY -u $R/scripts/noise_ladder_probe.py --ids $IDS --run_dir $DIR --n_seeds $NS --shard_index $i --num_shards 2 --save_previews --reward_server http://127.0.0.1:5142 >> $R/logs/ladder_ls6_shard$i.log 2>&1
    log "--- shard $i (gpu$G) end rc=$? in $(( $(date +%s)-t ))s rows=$(wc -l < $DIR/records_shard$i.jsonl 2>/dev/null)" ) & PIDS+=($!); done
for p in "${PIDS[@]}"; do wait $p; done; cat $DIR/records_shard*.jsonl > $DIR/records_all.jsonl; log "ladder: $(wc -l < $DIR/records_all.jsonl) trajectories, images $(ls $DIR/images | wc -l), previews $(ls $DIR/previews 2>/dev/null | wc -l)"
kill $(cat $R/state/mr3_server.pid 2>/dev/null) 2>/dev/null; sleep 10
log "judges on ladder images (GPU0)"; MAN=$R/manifests_ns_ladder_ls6; OUT=$R/scores_ns_ladder_ls6; mkdir -p $MAN $OUT
$PY - $DIR/records_all.jsonl $MAN/followup_ns_ladder_ls6.jsonl <<'PY'
import json, sys, hashlib, os
recs, out = sys.argv[1:3]; n = 0
with open(out, "w") as f:
    for l in open(recs):
        r = json.loads(l)
        if not os.path.exists(r["image"]): continue
        f.write(json.dumps({"backend": "followup:ns_ladder_ls6", "arm": "ns_ladder_ls6", "prompt_id": r["prompt_id"], "sample_index": int(r["noise_seed"]), "prompt": r["prompt"], "source": "genai_bench", "category": "", "skills": "",
                            "seed": r["noise_seed"], "image": r["image"], "image_sha256": hashlib.sha256(open(r["image"], "rb").read()).hexdigest()}) + "\n"); n += 1
print("manifest", n)
PY
( CUDA_VISIBLE_DEVICES=0 PATH="$VQA_PY_PREFIX/bin:$PATH" $VQA_PY_PREFIX/bin/python $R/deps/score_vqa.py --manifest_dir $MAN --out $OUT/vqa_scores.csv --shard_index 0 --num_shards 1 > $R/logs/ladder_ls6_vqa.log 2>&1; log "vqascore exit=$?"
  CUDA_VISIBLE_DEVICES=0 $PY $R/deps/score_vlm_judge.py --manifest_dirs $MAN --out $OUT/vlm_scores.csv --shard_index 0 --num_shards 1 > $R/logs/ladder_ls6_qwen.log 2>&1; log "qwen judge exit=$?" ) & J=$!
log "Jev grading of ladder images (GPU1)"; bash $R/scripts/jev_server.sh 1 5161 >> $R/logs/ladder_ls6.log 2>&1 && \
  $PY $S/scripts/jev_grade_images.py $DIR/records_all.jsonl --out $S/results/jev_grade_ns_ladder_ls6_2026-10-30.json >> $R/logs/ladder_ls6.log 2>&1; log "jev grading rc=$?"
wait $J; kill $(cat $R/state/jev_server.pid 2>/dev/null) 2>/dev/null
$PY $S/scripts/merge_ec_judges.py $MAN/followup_ns_ladder_ls6.jsonl $OUT/vqa_scores.shard0.csv $OUT/vlm_scores.shard0.csv --out $S/results/judges_ns_ladder_ls6_2026-10-30.json >> $R/logs/ladder_ls6.log 2>&1
cp -r $DIR/records_all.jsonl $DIR/config_shard*.json $S/results/main/ns_ladder_ls6/ 2>/dev/null || { mkdir -p $S/results/main/ns_ladder_ls6; cp $DIR/records_all.jsonl $DIR/config_shard*.json $S/results/main/ns_ladder_ls6/; }
mkdir -p $S/results/ls6_judge_2026-10-30 && cp -r $OUT $S/results/ls6_judge_2026-10-30/; log "LADDER CHAIN DONE"
