#!/usr/bin/env python3
"""Comparison table over the sweep arms on one id-set.  python sweep_table.py <ids_file> [--md OUT.md]"""
import json, csv, sys, os, glob, collections, statistics as st
R = "/data/ygu/runs/vlm_decider_sd35base_2026-10-16"
def _dedup(xs):
    seen = set(); return [x for x in xs if not (x in seen or seen.add(x))]
ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
out_md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
sc = collections.defaultdict(dict)
if os.path.exists(f"{R}/scores.csv"):
    for r in csv.DictReader(open(f"{R}/scores.csv")):
        if r["prompt_id"] in ids: sc[(r["arm"], r["prompt_id"])][r["metric"]] = float(r["score"])
dirs = _dedup(sorted(glob.glob(f"{R}/main/sw_*")) + [f"{R}/main/probe3_jev_online", f"{R}/main/probe3_jev_prune", f"{R}/main/dev3_baseline", f"{R}/main/dev3_mcts", f"{R}/main/dev3_mcts_refine", f"{R}/main/bm44_jev_online", f"{R}/main/bm44_jev_prune", f"{R}/main/sw_best", f"{R}/main/sw_beststate"])
mets = ["ir", "hpsv3", "pickscore", "hpsv2", "vqascore", "qwen_alignment"]
L = []; P = L.append
P("| arm | n | " + " | ".join(mets) + " | bank | +words | s/prompt | NFE |")
P("|---|---|" + "---|" * (len(mets) + 4))
rows = {}
for d in dirs:
    f = f"{d}/records.jsonl"
    if not os.path.exists(f) or os.path.getsize(f) == 0: continue
    recs = [json.loads(l) for l in open(f) if json.loads(l)["prompt_id"] in ids]
    if not recs: continue
    arm = recs[0]["arm"]; pts = [i for i in ids if (arm, i) in sc and all(m in sc[(arm, i)] for m in mets)]
    vals = {m: (st.fmean(sc[(arm, i)][m] for i in pts) if pts else float("nan")) for m in mets}
    rows[arm] = vals
    bank = st.fmean(r.get("bank_size", len(r.get("variants", []))) for r in recs)
    add = st.fmean(r.get("n_wordings_added", 0) for r in recs)
    P(f"| {arm} | {len(pts)} | " + " | ".join(f"{vals[m]:.3f}" for m in mets) +
      f" | {bank:.1f} | {add:.1f} | {st.fmean(r['seconds'] for r in recs):.0f} | {st.fmean(r['nfe_transformer'] for r in recs):.0f} |")
base = "probe3_jev_prune"
if base in rows:
    P(f"\nDelta vs the fixed-bank control ({base}); clean held-out metrics are hpsv3, pickscore, hpsv2, vqascore:\n")
    P("| arm | " + " | ".join(mets) + " |"); P("|---|" + "---|" * len(mets))
    for a, v in rows.items():
        if a != base: P(f"| {a} | " + " | ".join(f"{v[m]-rows[base][m]:+.3f}" for m in mets) + " |")
txt = "\n".join(L)
print(txt)
if out_md: open(out_md, "w").write(txt + "\n"); print("\nwrote", out_md)
