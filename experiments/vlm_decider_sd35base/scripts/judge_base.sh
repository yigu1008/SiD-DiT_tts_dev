#!/usr/bin/env bash
# Held-out judging + export + paired contrasts for base-SD3.5 arms.
#   bash judge_base.sh <GPU> <IDS_FILE> <ARM_DIR>...        (arm dirs contain records.jsonl)
# Phases: manifests -> score_local (IR, PickScore, HPSv2, HPSv3; reward env) -> Qwen2-VL alignment judge (sid_dit env)
#         -> VQAScore (vqascore_reward env) -> scores.csv (prompt_id,seed,arm,metric,score) -> contrasts vs baseline & mcts.
set -uo pipefail
R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; RO=/data/ygu/reward_orthogonality; D=$RO/followup_2026-09-24
RW=/home/ygu/miniconda3/envs/reward/bin/python; SD=/home/ygu/miniconda3/envs/sid_dit/bin/python; VQ=/home/ygu/miniconda3/envs/vqascore_reward
# TAG carries seconds+pid: minute granularity let two judges launched in the same minute share the
# manifest and scores dirs and write them concurrently.
GPU=$1; IDS=$2; shift 2; DIRS="$*"; TAG=$(date +%m%d_%H%M%S)_$$; MAN=$R/manifests_$TAG; OUT=$R/scores_$TAG; mkdir -p $OUT $R/logs
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/judge_driver.log; }
log "=== judge gpu$GPU ids=$(grep -c . $IDS) dirs=[$DIRS] tag=$TAG"
$SD $D/scripts/build_followup_manifest.py --arm_dirs $DIRS --out_dir $MAN --ids $IDS || exit 1
t=$(date +%s); CUDA_VISIBLE_DEVICES=$GPU $RW $RO/score_local.py --manifest_dir $MAN --out $OUT/local_scores.csv --shard_index 0 --num_shards 1 > $R/logs/judge_${TAG}_local.log 2>&1; log "local judges exit=$? $(( $(date +%s)-t ))s"
t=$(date +%s); CUDA_VISIBLE_DEVICES=$GPU $SD $RO/score_vlm_judge.py --manifest_dirs $MAN --out $OUT/vlm_scores.csv --shard_index 0 --num_shards 1 > $R/logs/judge_${TAG}_vlm.log 2>&1; log "qwen2-vl judge exit=$? $(( $(date +%s)-t ))s"
t=$(date +%s); CUDA_VISIBLE_DEVICES=$GPU PATH="$VQ/bin:$PATH" $VQ/bin/python $RO/score_vqa.py --manifest_dir $MAN --out $OUT/vqa_scores.csv --shard_index 0 --num_shards 1 > $R/logs/judge_${TAG}_vqa.log 2>&1; log "vqascore exit=$? $(( $(date +%s)-t ))s"
[ -s $R/scores.csv ] || echo "prompt_id,seed,arm,metric,score" > $R/scores.csv
RECS=""; for d in $DIRS; do RECS="$RECS $d/records.jsonl"; done
$SD $D/scripts/export_scores.py --scores_csv $R/scores.csv --local_glob "$OUT/local_scores*.csv" --vlm_glob "$OUT/vlm_scores*.csv" --records $RECS --skip_existing | tee -a $R/logs/judge_driver.log
$SD - "$R/scores.csv" "$OUT" $RECS <<'PY'
import csv, glob, json, sys
scores, out, recs = sys.argv[1], sys.argv[2], sys.argv[3:]
have = {(r["prompt_id"], r["seed"], r["arm"], r["metric"]) for r in csv.DictReader(open(scores))}
vqa = {(r["backend"], r["prompt_id"]): float(r["vqascore"]) for f in glob.glob(f"{out}/vqa_scores*.csv") for r in csv.DictReader(open(f))}
new = []
for rp in recs:
    for r in map(json.loads, open(rp)):
        k = (f"followup:{r['arm']}", r["prompt_id"])
        if k in vqa and (r["prompt_id"], str(r["seed"]), r["arm"], "vqascore") not in have: new.append([r["prompt_id"], r["seed"], r["arm"], "vqascore", f"{vqa[k]:.6f}"])
csv.writer(open(scores, "a", newline="")).writerows(new); print(f"vqascore rows appended: {len(new)}")
PY
ARMS=$(for d in $DIRS; do basename $d; done)
log "=== paired contrasts (n=$(grep -c . $IDS)): each arm minus baseline, and minus mcts ==="
for A in $ARMS; do for B in baseline mcts; do [ "$A" = "$B" ] && continue; echo " $ARMS " | grep -q " $B " || continue
  F=$R/contrast_${A}_minus_${B}.txt; : > $F
  for M in ir hpsv3 pickscore hpsv2 qwen_alignment vqascore search_score; do $SD $D/scripts/same_day_analysis.py paired --scores $R/scores.csv --ids $IDS --a $A --b $B --metric $M >> $F 2>&1; done
  log "$A - $B:"; cat $F | tee -a $R/logs/judge_driver.log; done; done
log "JUDGE COMPLETE"
