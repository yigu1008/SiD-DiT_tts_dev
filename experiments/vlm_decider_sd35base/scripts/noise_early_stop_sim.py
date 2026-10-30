#!/usr/bin/env python3
"""Offline: can x0-preview rewards at early checkpoints screen initial noises, so a fixed NFE budget sees more z0?
Data: ns_ladder records (9 prompts x 32 noises, anchor action; partial reward vectors at steps 4..24, final at 28 = 56 NFE).
(1) predictability: rank correlation (across noises, within prompt) of preview reward at step t vs final;
(2) policy simulation at fixed budgets: uniform best-of-k vs screen N noises to step t, finish the top-m, pick final best.
Usage: python noise_early_stop_sim.py <records_all.jsonl> [--reward hpsv3] [--draws 400]"""
import json, sys, random, statistics as st, collections, argparse
ap = argparse.ArgumentParser(); ap.add_argument("records"); ap.add_argument("--reward", default="hpsv3"); ap.add_argument("--draws", type=int, default=400); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); RW = a.reward
rows = [json.loads(l) for l in open(a.records)]
P = collections.defaultdict(list)
for r in rows: P[r["prompt_id"]].append(r)
steps = sorted(int(s) for s in rows[0]["partial"]); STEPS = 28; nfe = lambda t: 2 * t
def spearman(x, y):
    rk = lambda v: {i: s for s, i in enumerate(sorted(range(len(v)), key=lambda i: v[i]))}
    rx, ry = rk(x), rk(y); n = len(x); return 1 - 6 * sum((rx[i] - ry[i]) ** 2 for i in range(n)) / (n * (n * n - 1))
print(f"=== (1) predictability of FINAL {RW} from the preview {RW} at step t (across {len(rows)//len(P)} noises, mean over {len(P)} prompts)")
print(f"{'step':>5} {'NFE':>4} {'spearman':>9} {'top1 hit':>9} {'best-of-top3':>13}  (top1 hit = preview argmax is the final argmax; best-of-top3 = final value of the best among preview top-3, minus final max)")
for t in steps:
    sp, hit, gap = [], [], []
    for pid, rs in P.items():
        x = [r["partial"][str(t)][RW] for r in rs]; y = [r["final"][RW] for r in rs]
        sp.append(spearman(x, y)); hit.append(max(range(len(x)), key=lambda i: x[i]) == max(range(len(y)), key=lambda i: y[i]))
        top3 = sorted(range(len(x)), key=lambda i: -x[i])[:3]; gap.append(max(y[i] for i in top3) - max(y))
    print(f"{t:5} {nfe(t):4} {st.mean(sp):9.3f} {100*st.mean(hit):8.0f}% {st.mean(gap):+13.3f}")
# cross-reward: does an EARLY cheap reward predict final hpsv3 better than early hpsv3 itself?
for t in (8, 12):
    print(f"    step {t:2}: " + "  ".join(f"{k}->{RW}: {st.mean(spearman([r['partial'][str(t)][k] for r in rs], [r['final'][RW] for r in rs]) for rs in P.values()):.3f}" for k in ("imagereward", "hpsv3", "pickscore")))
print(f"\n=== (2) policy simulation, mean final {RW} of the SELECTED noise (selection by its final reward), {a.draws} random noise orderings x {len(P)} prompts")
rng = random.Random(a.seed)
def uniform(rs, budget):
    k = budget // 56; pick = rng.sample(rs, min(k, len(rs))); return max(r["final"][RW] for r in pick) if pick else float("nan")
def screen(rs, budget, t, m):
    # start N noises to step t, finish the top-m by preview reward; N chosen to exhaust the budget: N*2t + m*(56-2t) <= budget
    N = (budget - m * (56 - nfe(t))) // nfe(t)
    if N < m or N < 1: return float("nan")
    cand = rng.sample(rs, min(N, len(rs))); top = sorted(cand, key=lambda r: -r["partial"][str(t)][RW])[:m]
    return max(r["final"][RW] for r in top)
budgets = [112, 224, 336, 448, 616, 896]
pols = [("uniform best-of-k", lambda rs, b: uniform(rs, b))]
for t in (4, 8, 12, 16): 
    for m in (1, 2, 3): pols.append((f"screen@{t:2} keep{m}", (lambda t, m: lambda rs, b: screen(rs, b, t, m))(t, m)))
ceiling = st.mean(max(r["final"][RW] for r in rs) for rs in P.values()); single = st.mean(st.mean(r["final"][RW] for r in rs) for rs in P.values())
print(f"single noise {single:.3f} | oracle best of all 32: {ceiling:.3f}")
print(f"{'policy':>22} " + " ".join(f"{b:>7}" for b in budgets) + "   (NFE budget; blank = policy infeasible)")
for name, f in pols:
    vals = []
    for b in budgets:
        v = [f(rs, b) for _ in range(a.draws) for rs in P.values()]; v = [x for x in v if x == x]
        vals.append(f"{st.mean(v):7.3f}" if v else f"{'':>7}")
    print(f"{name:>22} " + " ".join(vals))
