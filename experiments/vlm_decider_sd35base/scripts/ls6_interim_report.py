#!/usr/bin/env python3
"""Interim probe of the LS6 runs from RECORDS only (no held-out judges): jevact 2x2, Jev decision stats, ec_v2w, ad_* arms.
Usage: python ls6_interim_report.py [RUN_DIR]   (default: the results/ dir next to this script's parent)"""
import json, statistics as st, collections, csv, os, sys
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
L = lambda p: [json.loads(l) for l in open(p)]
J = lambda v: json.loads(v) if isinstance(v, str) else v
M = lambda p: os.path.join(R, "main", p)

A = L(M("ja_jevact_v2w/records.jsonl")); B = L(M("ja_randact_v2w/records.jsonl"))
pa = {r["prompt_id"]: r for r in A}; pb = {r["prompt_id"]: r for r in B}; P = sorted(set(pa) & set(pb))
print(f"=== jevact 2x2 (n={len(P)} prompts, 11 traj x 56 NFE each): HPSv3 of selected traj | oracle best-of-11 | pool mean")
print(f"{'pid':6} {'jev_sel':>8} {'rnd_sel':>8} | {'jev_orc':>8} {'rnd_orc':>8} | {'jev_pool':>8} {'rnd_pool':>8} | {'jev_s':>6} {'rnd_s':>6}")
agg = collections.defaultdict(list)
for p in P:
    ra, rb = pa[p], pb[p]; ha = [t["vec"]["hpsv3"] for t in ra["all_traj"]]; hb = [t["vec"]["hpsv3"] for t in rb["all_traj"]]
    row = dict(jev_sel=ra["search_score"], rnd_sel=rb["search_score"], jev_orc=max(ha), rnd_orc=max(hb), jev_pool=st.mean(ha), rnd_pool=st.mean(hb), jev_s=ra["seconds"], rnd_s=rb["seconds"])
    for k, v in row.items(): agg[k].append(v)
    print(f"{p:6} {row['jev_sel']:8.3f} {row['rnd_sel']:8.3f} | {row['jev_orc']:8.3f} {row['rnd_orc']:8.3f} | {row['jev_pool']:8.3f} {row['rnd_pool']:8.3f} | {row['jev_s']:6.0f} {row['rnd_s']:6.0f}")
d = lambda x, y: [a - b for a, b in zip(agg[x], agg[y])]
for lab, (x, y) in {"selected": ("jev_sel", "rnd_sel"), "oracle best-of-11": ("jev_orc", "rnd_orc"), "pool mean": ("jev_pool", "rnd_pool")}.items():
    dd = d(x, y); print(f"  paired jev-rnd {lab:>18}: {st.mean(dd):+.3f} (sd {st.pstdev(dd):.3f}, jev wins {sum(v>0 for v in dd)}/{len(dd)})")
print(f"  selection gap oracle-selected: jev {st.mean(d('jev_orc','jev_sel')):.3f}  rnd {st.mean(d('rnd_orc','rnd_sel')):.3f}")
def acts(recs):
    cw = collections.Counter(); cc = collections.Counter(); sw = []
    for r in recs:
        for t in r["all_traj"][1:]:
            a = J(t["actions"]); [ (cw.update([w]), cc.update([c])) for w, c, _ in a ]; sw.append(sum(1 for i in range(1, len(a)) if a[i][:2] != a[i-1][:2]))
    return cw, cc, st.mean(sw)
for name, recs in (("jev", A), ("rnd", B)):
    cw, cc, sw = acts(recs); tot = sum(cc.values())
    print(f"  {name} actions: CFG " + " ".join(f"{c}:{100*n/tot:.0f}%" for c, n in sorted(cc.items())) + " | wording " + " ".join(f"w{w}:{100*n/tot:.0f}%" for w, n in sorted(cw.items())) + f" | switches/traj {sw:.1f}")

dec = L(M("ja_jevact_v2w/decisions.jsonl")); n = len(dec)
pos = collections.Counter(int(x["chosen_panel"]) for x in dec); maxp = [max(J(x["jev_probs"])) for x in dec]
orig = collections.Counter(J(x["sheet_perm"])[int(x["chosen_panel"]) - 1] for x in dec)   # chosen_panel is 1-indexed
print(f"\n=== Jev decisions n={n}: chosen DISPLAY panel " + str({k: f'{100*v/n:.0f}%' for k, v in sorted(pos.items())}) + " | chosen ORIGINAL candidate " + str({k: f'{100*v/n:.0f}%' for k, v in sorted(orig.items())}))
print(f"  max prob mean {st.mean(maxp):.3f} median {st.median(maxp):.3f}, >0.5 in {100*sum(p>0.5 for p in maxp)/n:.0f}% (uniform=0.333)")
bystep = collections.defaultdict(list); [bystep[int(x["step"])].append(p) for x, p in zip(dec, maxp)]
print("  mean max-prob by key step:", {k: round(st.mean(v), 3) for k, v in sorted(bystep.items())})

ec = L(M("ec_v2w/records_all.jsonl"))
g = collections.defaultdict(list); [g[(r["mode"], r["stratum"])].append(r["vec"]["hpsv3"] - r["anchor_hpsv3"]) for r in ec]
print(f"\n=== ec_v2w n={len(ec)}: single-variant dHPSv3 vs same-noise anchor (win%)")
for k in sorted(g): print(f"  {k[0]:>6} {k[1]:>6}: {st.mean(g[k]):+.3f} ({100*sum(x>0 for x in g[k])/len(g[k]):.0f}%, n={len(g[k])})")
gc = collections.defaultdict(list); [gc[(r["mode"], r["action"][1])].append(r["vec"]["hpsv3"] - r["anchor_hpsv3"]) for r in ec]
cf = sorted({k[1] for k in gc}); print("  by action CFG:    " + " ".join(f"{c:>9}" for c in cf))
for m in ("early", "full", "late"): print(f"  {m:>6}            " + " ".join(f"{st.mean(gc[(m,c)]):+5.2f}({len(gc[(m,c)]):2})" if (m, c) in gc else f"{'-':>9}" for c in cf))
grp = collections.defaultdict(list); [grp[(r["prompt_id"], r["noise_seed"], r["stratum"], r["mode"])].append(r) for r in ec]
by = collections.defaultdict(lambda: [[], []])
for (_, _, strat, m), rs in grp.items():
    b = max(r["vec"]["hpsv3"] for r in rs); by[(strat, m)][0].append(b - rs[0]["anchor_hpsv3"]); by[(strat, m)][1].append(b - rs[0]["median_noise_hpsv3"])
print("  best-of-k variants per (prompt,noise,mode): vs anchor | vs fresh median noise (beats fresh)")
for (strat, m), (va, vf) in sorted(by.items()): print(f"  {strat:>6} {m:>6} k={len(grp[next(x for x in grp if x[2]==strat and x[3]==m)])}: {st.mean(va):+.3f} | {st.mean(vf):+.3f} ({100*sum(x>0 for x in vf)/len(vf):.0f}%)")

print("\n=== actdiff arms (terminal ImageReward search_score):")
for arm in ("ad_base", "ad_rank3", "ad_jevbank", "ad_randbank", "ad_jevprobe", "ad_randprobe"):
    p = M(f"{arm}/records.jsonl")
    if os.path.exists(p) and os.path.getsize(p): rs = L(p); print(f"  {arm:12} n={len(rs)} mean IR {st.mean(r['search_score'] for r in rs):.4f}  NFE/prompt {rs[0]['nfe_transformer']}")
sc = os.path.join(R, "scores.csv")
if os.path.exists(sc) and os.path.exists(M("ad_base/records.jsonl")):
    ids = {r["prompt_id"] for r in L(M("ad_base/records.jsonl"))}; ss = collections.defaultdict(list)
    for r in csv.DictReader(open(sc)):
        if r["metric"] == "search_score" and r["prompt_id"] in ids and r["arm"] in ("ns_bo11_ir", "rs_ir", "rs_blind"): ss[r["arm"]].append(float(r["score"]))
    print("  reference IR-searched arms on the same prompts (scores.csv): " + ", ".join(f"{a} {st.mean(v):.4f} (n={len(v)})" for a, v in sorted(ss.items())))
