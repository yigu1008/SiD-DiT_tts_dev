#!/usr/bin/env bash
# Rename arm fields to the run-dir name in v3/v4/v5 records, purge their old rows, judge r0+v3+v4+v5 on the hard-5 prompts, regenerate the report.
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python; GPU=${1:-0}
$PY - <<'PY'
import json, csv
R="/data/ygu/runs/vlm_decider_sd35base_2026-10-16"
for v in ("opt_prompt_v3","opt_prompt_v4","opt_prompt_v5"):
    f=f"{R}/probe/{v}/records.jsonl"
    try: recs=[json.loads(l) for l in open(f)]
    except FileNotFoundError: continue
    for r in recs: r["arm"]=v
    open(f,"w").write("".join(json.dumps(r)+"\n" for r in recs)); print(v, "arm renamed in", len(recs), "records")
rows=list(csv.DictReader(open(f"{R}/scores.csv"))); keep=[r for r in rows if not r["arm"].startswith("opt_prompt")]
with open(f"{R}/scores.csv","w",newline="") as fh:
    w=csv.DictWriter(fh, fieldnames=["prompt_id","seed","arm","metric","score"]); w.writeheader(); w.writerows(keep)
print(f"purged {len(rows)-len(keep)} opt_prompt* rows; kept {len(keep)}")
PY
DIRS="$R/probe/opt_prompt_r0 $R/probe/opt_prompt_v3 $R/probe/opt_prompt_v4"; [ -s $R/probe/opt_prompt_v5/records.jsonl ] && DIRS="$DIRS $R/probe/opt_prompt_v5"
bash $R/scripts/judge_base.sh $GPU $R/config/prompt_ids_hard5.txt $DIRS
$PY $R/scripts/make_optprompt_md.py; echo "[$(date '+%F %T')] REJUDGE DONE" | tee -a $R/logs/optprompt_driver.log
