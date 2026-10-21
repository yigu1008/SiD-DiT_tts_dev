#!/usr/bin/env bash
# After main/opt_prompt finishes: derive the unedited baseline (round-0 images) as arm opt_prompt_r0, judge both on the 25 ids, paired contrasts.  bash finish_cohort_optprompt.sh <GPU>
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python; D=/data/ygu/reward_orthogonality/followup_2026-09-24; GPU=${1:-0}
IDS=$R/config/prompt_ids_25.txt; log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/optprompt_driver.log; }
$PY - <<'PY'
import json, hashlib, os
R="/data/ygu/runs/vlm_decider_sd35base_2026-10-16"; src=f"{R}/main/opt_prompt/records.jsonl"; out=f"{R}/main/opt_prompt_r0"; os.makedirs(out, exist_ok=True)
n=0
with open(f"{out}/records.jsonl","w") as f:
    for l in open(src):
        r=json.loads(l); r0=r["rounds"][0]; img=r0["image"]
        f.write(json.dumps({"prompt_id":r["prompt_id"],"prompt":r["prompt"],"arm":"opt_prompt_r0","seed":r["seed"],"image":img,"image_sha256":hashlib.sha256(open(img,"rb").read()).hexdigest(),"search_score":r0["ir"],"note":"round-0 image = original wording, CFG 4.5, seed 42 (identical to the baseline arm by construction)"})+"\n"); n+=1
print("opt_prompt_r0 records:", n)
PY
bash $R/scripts/judge_base.sh $GPU $IDS $R/main/opt_prompt $R/main/opt_prompt_r0
log "=== paired contrasts n=25: opt_prompt minus opt_prompt_r0 ==="; F=$R/main/contrast_n25_opt_prompt_minus_r0.txt; : > $F
for M in ir hpsv3 pickscore hpsv2 vqascore qwen_alignment; do $PY $D/scripts/same_day_analysis.py paired --scores $R/scores.csv --ids $IDS --a opt_prompt --b opt_prompt_r0 --metric $M >> $F 2>&1; done; cat $F | tee -a $R/logs/optprompt_driver.log
$PY $R/scripts/make_optprompt_sheet.py $R/main/opt_prompt > /dev/null; log "COHORT OPTPROMPT COMPLETE"
