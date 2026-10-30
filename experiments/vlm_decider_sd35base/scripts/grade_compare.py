#!/usr/bin/env python3
"""Merge Jev and VLM grading jsons by image and compare every readout as a within-prompt SELECTOR against targets.
Usage: python grade_compare.py --jev jev.json --vlm vlm.json [--targets vec.hpsv3 vec.imagereward vec.pickscore] [--gap 0.5]"""
import json, argparse, collections, itertools, statistics as st
ap = argparse.ArgumentParser(); ap.add_argument("--jev"); ap.add_argument("--vlm"); ap.add_argument("--targets", nargs="+", default=["vec.hpsv3", "vec.imagereward", "vec.pickscore"]); ap.add_argument("--gap", type=float, default=0.5)
a = ap.parse_args(); rows = {}
for f, keys in ((a.jev, ["jev_yes", "jev_score_E", "jev_score_P4"]), (a.vlm, ["vlm_rating", "vlm_yes", "vlm_pass"])):
    if not f: continue
    for r in json.load(open(f)):
        d = rows.setdefault(r["image"], {k: v for k, v in r.items() if not isinstance(v, (list, dict))}); d.update({k: r.get(k) for k in keys})
rows = list(rows.values()); byp = collections.defaultdict(list); [byp[r["prompt_id"]].append(r) for r in rows]
def spearman(x, y):
    rk = lambda v: {i: s for s, i in enumerate(sorted(range(len(v)), key=lambda i: v[i]))}
    rx, ry = rk(x), rk(y); n = len(x); return 1 - 6 * sum((rx[i] - ry[i]) ** 2 for i in range(n)) / (n * (n * n - 1)) if n > 2 else float("nan")
reads = [k for k in ["jev_yes", "jev_score_E", "jev_score_P4", "vlm_rating", "vlm_yes", "vlm_pass"] if any(r.get(k) is not None for r in rows)] + a.targets
ok_rows = lambda k, t: [r for r in rows if r.get(k) is not None and r.get(t) is not None]
print(f"=== n={len(rows)} images, {len(byp)} prompts. Spearman pooled [within-prompt mean] vs targets")
print(f"{'readout':>14} " + " ".join(f"{t:>18}" for t in a.targets))
for k in reads:
    cells = []
    for t in a.targets:
        rr = ok_rows(k, t); bp = collections.defaultdict(list); [bp[r["prompt_id"]].append(r) for r in rr]
        pooled = spearman([r[k] for r in rr], [r[t] for r in rr]); within = st.mean(spearman([r[k] for r in v], [r[t] for r in v]) for v in bp.values() if len(v) > 2)
        cells.append(f"{pooled:+.2f} [{within:+.2f}]")
    print(f"{k:>14} " + " ".join(f"{c:>18}" for c in cells))
for t0 in a.targets[:2]:
    print(f"\n=== within-prompt pairwise ordering accuracy vs {t0} (|gap| > {a.gap if t0=='vec.hpsv3' else a.gap/5:.2f}); ties in the readout skipped")
    g = a.gap if t0 == "vec.hpsv3" else a.gap / 5
    for k in reads:
        if k == t0: continue
        ok = n = 0
        for v in byp.values():
            for x, y in itertools.combinations(v, 2):
                if x.get(k) is None or y.get(k) is None or abs(x[t0] - y[t0]) <= g or x[k] == y[k]: continue
                n += 1; ok += (x[k] > y[k]) == (x[t0] > y[t0])
        print(f"  {k:>14}: {100*ok/max(n,1):5.1f}%  (n={n})")
if rows and "stratum" in rows[0]:
    print("\n=== best-noise vs worst-noise separation per readout: mean P(readout_best > readout_worst) over prompts (0.5 = blind)")
    for k in reads:
        aucs = []
        for v in byp.values():
            b = [r[k] for r in v if r.get("stratum") == "best" and r.get(k) is not None]; w = [r[k] for r in v if r.get("stratum") == "worst" and r.get(k) is not None]
            if b and w: aucs.append(sum((x > y) + 0.5 * (x == y) for x in b for y in w) / (len(b) * len(w)))
        print(f"  {k:>14}: {st.mean(aucs):.2f}  (per prompt: {' '.join(f'{x:.2f}' for x in aucs)})")
