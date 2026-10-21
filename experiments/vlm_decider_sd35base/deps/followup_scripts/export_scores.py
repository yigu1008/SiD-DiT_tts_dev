#!/usr/bin/env python3
"""Append judged scores of the new arms to selected_scores.csv (prompt_id,seed,arm,metric,score).
Reads the judge CSVs produced for manifests with backend 'followup:<arm>'.
Metrics: ir, pickscore, hpsv2, hpsv3 (from score_local), qwen_alignment (score_vlm_judge),
and the arm's own search score as 'search_score' for the live-vs-offline consistency check.
Refuses to write a (prompt_id, seed, arm, metric) that already exists."""
import argparse, csv, glob, json
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument("--scores_csv", required=True); p.add_argument("--local_glob", required=True); p.add_argument("--vlm_glob", required=True)
p.add_argument("--records", nargs="+", required=True, help="records.jsonl files of the new arms")
p.add_argument("--skip_existing", action="store_true", help="skip (not refuse) rows already present; report max |rejudged - stored| for them"); a = p.parse_args()
existing = {(r["prompt_id"], r["seed"], r["arm"], r["metric"]): float(r["score"]) for r in csv.DictReader(open(a.scores_csv))}
loc = {}; [loc.__setitem__((r["backend"], r["prompt_id"]), r) for f in sorted(glob.glob(a.local_glob)) for r in csv.DictReader(open(f)) if int(r["sample_index"]) == 1]
vlm = {}; [vlm.__setitem__((r["backend"], r["prompt_id"]), float(r["vlm_yes"])) for f in sorted(glob.glob(a.vlm_glob)) for r in csv.DictReader(open(f))]
new = []; missing = []; skipped = 0; maxdiff = {}
for rp in a.records:
    for r in map(json.loads, open(rp)):
        arm, pid, seed = r["arm"], r["prompt_id"], str(r["seed"]); k = (f"followup:{arm}", pid)
        vals = {}
        if k in loc: vals.update(ir=float(loc[k]["imagereward"]), pickscore=float(loc[k]["pickscore"]), hpsv2=float(loc[k]["hpsv2"]), hpsv3=float(loc[k]["hpsv3"]))
        else: missing.append((arm, pid, "local"))
        if k in vlm: vals["qwen_alignment"] = vlm[k]
        else: missing.append((arm, pid, "vlm"))
        vals["search_score"] = float(r.get("selected_ir", r.get("selected_score", r.get("search_score"))))
        for m, v in vals.items():
            key = (pid, seed, arm, m)
            if key in existing:
                if not a.skip_existing: raise SystemExit(f"refusing to overwrite existing row {key}")
                skipped += 1; maxdiff[m] = max(maxdiff.get(m, 0.0), abs(existing[key] - float(v))); continue
            new.append({"prompt_id": pid, "seed": seed, "arm": arm, "metric": m, "score": f"{v:.6f}"})
with open(a.scores_csv, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["prompt_id","seed","arm","metric","score"]); w.writerows(new)
print(f"appended {len(new)} rows; skipped existing {skipped}; max |rejudged-stored| per metric: { {k: round(v, 5) for k, v in maxdiff.items()} }; missing judge scores: {len(missing)} {missing[:5]}")
