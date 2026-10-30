#!/usr/bin/env python3
"""Pre-specified contrasts on a noise ladder: pick by policy vs bo-N by HPSv3, paired over prompts, 95% prompt-bootstrap CI.
Usage: python heldout_grader_contrast.py <records_all.jsonl> <jev_grade.json> <judges.json> [--n_noises 12]"""
import argparse, json, random, statistics as st
ap = argparse.ArgumentParser(); ap.add_argument("records"); ap.add_argument("jev"); ap.add_argument("judges"); ap.add_argument("--n_noises", type=int, default=12); a = ap.parse_args()
rows = {}
for r in map(json.loads, open(a.records)): rows[r["image"]] = {"p": r["prompt_id"], "seed": r["noise_seed"], "hpsv3": r["final"]["hpsv3"]}
for f in (a.jev, a.judges):
    for r in json.load(open(f)):
        if r["image"] in rows: rows[r["image"]].update({k: r[k] for k in ("jev_score_E", "vqascore", "qwen_alignment") if k in r})
by = {}
for r in rows.values():
    if all(k in r for k in ("jev_score_E", "vqascore", "qwen_alignment")): by.setdefault(r["p"], []).append(r)
by = {p: sorted(v, key=lambda r: r["seed"])[:a.n_noises] for p, v in by.items() if len(v) >= a.n_noises}
print(f"n prompts = {len(by)}, noises per prompt = {a.n_noises}"); top = lambda v, k: sorted(v, key=lambda r: -r["hpsv3"])[:k]; base = lambda v: top(v, 1)[0]
T = ("vqascore", "qwen_alignment", "hpsv3"); rng = random.Random(0)
print("means: " + "  ".join(f"{n} " + "/".join(f"{st.mean(f(v)[t] for v in by.values()):.3f}" for t in T) for n, f in (("bo-N", base), ("top2->Jev", lambda v: max(top(v, 2), key=lambda r: r["jev_score_E"])), ("top4->Jev", lambda v: max(top(v, 4), key=lambda r: r["jev_score_E"])), ("Jev only", lambda v: max(v, key=lambda r: r["jev_score_E"])))))
print("random: " + "/".join(f"{st.mean(st.mean(r[t] for r in v) for v in by.values()):.3f}" for t in T))
for n, f in (("PRIMARY top-2->Jev", lambda v: max(top(v, 2), key=lambda r: r["jev_score_E"])), ("top-4->Jev", lambda v: max(top(v, 4), key=lambda r: r["jev_score_E"])), ("Jev only", lambda v: max(v, key=lambda r: r["jev_score_E"]))):
    print(n)
    for t in T:
        d = [f(v)[t] - base(v)[t] for v in by.values()]; b = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(10000))
        print(f"  {t:15} {st.mean(d):+.3f} [{b[250]:+.3f},{b[9749]:+.3f}] up {sum(x>0 for x in d)} down {sum(x<0 for x in d)} same {sum(x==0 for x in d)}")
