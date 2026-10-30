#!/usr/bin/env python3
"""Correlate Jev typed-grading readouts with any numeric fields (search rewards / judges) within prompt and pooled; pairwise ordering.
Usage: python jev_grade_analysis.py <jev_grade json> [--targets vec.hpsv3 vec.imagereward vec.pickscore] [--gap 0.5]"""
import json, sys, argparse, collections, itertools, statistics as st
ap = argparse.ArgumentParser(); ap.add_argument("json"); ap.add_argument("--targets", nargs="+", default=["vec.hpsv3", "vec.imagereward", "vec.pickscore"]); ap.add_argument("--gap", type=float, default=0.5)
a = ap.parse_args(); rows = json.load(open(a.json)); byp = collections.defaultdict(list); [byp[r["prompt_id"]].append(r) for r in rows]
def spearman(x, y):
    rk = lambda v: {i: s for s, i in enumerate(sorted(range(len(v)), key=lambda i: v[i]))}
    rx, ry = rk(x), rk(y); n = len(x); return 1 - 6 * sum((rx[i] - ry[i]) ** 2 for i in range(n)) / (n * (n * n - 1)) if n > 2 else float("nan")
reads = ["jev_yes", "jev_score_E", "jev_score_P4"] + [t for t in a.targets]
print(f"=== n={len(rows)} images, {len(byp)} prompts, {len(rows)//len(byp)} per prompt. Spearman pooled [within-prompt mean]")
print(f"{'readout':>14} " + " ".join(f"{t:>18}" for t in a.targets))
for k in reads:
    cells = []
    for t in a.targets:
        pooled = spearman([r[k] for r in rows], [r[t] for r in rows]); within = st.mean(spearman([r[k] for r in v], [r[t] for r in v]) for v in byp.values() if len(v) > 2)
        cells.append(f"{pooled:+.2f} [{within:+.2f}]")
    print(f"{k:>14} " + " ".join(f"{c:>18}" for c in cells))
t0 = a.targets[0]; print(f"\n=== within-prompt pairwise ordering accuracy vs {t0} (gap > {a.gap}):")
for k in ["jev_yes", "jev_score_E"] + a.targets[1:]:
    ok = n = 0
    for v in byp.values():
        for x, y in itertools.combinations(v, 2):
            if abs(x[t0] - y[t0]) <= a.gap or x[k] == y[k]: continue
            n += 1; ok += (x[k] > y[k]) == (x[t0] > y[t0])
    print(f"  {k:>14}: {100*ok/max(n,1):5.1f}%  (n={n})")
if "stratum" in rows[0]:
    print("\n=== by stratum/mode: mean Jev P(yes), E[s], and", t0)
    g = collections.defaultdict(list); [g[(r.get("stratum"), r.get("mode"))].append(r) for r in rows]
    for key in sorted(g): v = g[key]; print(f"  {str(key):>22}: P(yes) {st.mean(r['jev_yes'] for r in v):.2f}  E[s] {st.mean(r['jev_score_E'] for r in v):.2f}  {t0} {st.mean(r[t0] for r in v):.2f}  n={len(v)}")
    print("\n=== per prompt: does Jev P(yes) separate best-noise from worst-noise images? (mean P(yes) best vs worst; AUC-ish = P(best > worst))")
    for pid, v in sorted(byp.items()):
        b = [r["jev_yes"] for r in v if r.get("stratum") == "best"]; w = [r["jev_yes"] for r in v if r.get("stratum") == "worst"]
        if b and w: print(f"  {pid}: best {st.mean(b):.2f} worst {st.mean(w):.2f}  P(best>worst) {sum(x > y for x in b for y in w)/(len(b)*len(w)):.2f}   | hpsv3 best {st.mean(r['vec.hpsv3'] for r in v if r['stratum']=='best'):.2f} worst {st.mean(r['vec.hpsv3'] for r in v if r['stratum']=='worst'):.2f}")
