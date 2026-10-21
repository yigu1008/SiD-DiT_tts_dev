#!/usr/bin/env python3
"""Offline simulation of every adaptive noise-allocation policy at a fixed NFE budget.

Cost model (exact for this sampler): 2 NFE per Euler step, so taking a noise to step t costs 2t and finishing it costs
2(28-t). A policy that screens n noises to step t and finishes the best m spends n*2t + m*2*(28-t).
  uniform_k      : finish k noises outright (k = B/56).  This is ns_bo11_hps.
  screen(t,n,m)  : screen n to step t, rank by the PARTIAL score, finish top m.      <- the real policy
  oracle(t,n,m)  : same, but rank by the TRUE FINAL score.  Upper bound for ANY abandonment rule at step t,
                   Jev's included -- no decision model can exceed it.
  random(t,n,m)  : rank at random. Control: isolates how much is the screening INFORMATION vs just seeing more noises.
Batch screen-and-finish is the optimal form of "abandon when saturated, move to the next noise" under a budget
(successive-halving structure); an online threshold rule is a special case, so these oracles bound it too.
"""
import argparse, json, random, statistics as st
from collections import defaultdict
from pathlib import Path
R = Path("/data/ygu/runs/vlm_decider_sd35base_2026-10-16")
SR = ["imagereward", "hpsv3", "pickscore"]
LAB = {"imagereward": "ir", "hpsv3": "hpsv3", "pickscore": "pickscore"}

def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1): r[o[k]] = avg
            i = j + 1
        return r
    a, b = rk(x), rk(y); n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** .5
    return num / den if den else 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=616); p.add_argument("--steps", type=int, default=28)
    p.add_argument("--metric", default="hpsv3", choices=SR); p.add_argument("--draws", type=int, default=2000)
    a = p.parse_args()
    rows = [json.loads(l) for l in open(R / "main/ns_ladder/records_all.jsonl")]
    by = defaultdict(dict)
    for r in rows: by[r["prompt_id"]][r["noise_seed"]] = r
    pids = sorted(by); cps = rows[0]["checkpoints"]
    N = min(len(by[p_]) for p_ in pids)
    print(f"prompts={len(pids)}  noises/prompt={N}  checkpoints={cps}  budget={a.budget} NFE  "
          f"full generation={2*a.steps} NFE  selecting on {LAB[a.metric]}\n")

    print("=== 1. PREDICTABILITY: across-noise Spearman( x0-preview score at step t , FINAL score ) ===")
    print("    (if this is near 0 at a cheap step, no abandonment rule can work there -- Jev's included)")
    hdr = f"{'step':>6}{'NFE/noise':>11}" + "".join(f"{LAB[b]:>12}" for b in SR)
    print(hdr); print("-" * len(hdr))
    for t in cps:
        row = f"{t:>6}{2*t:>11}"
        for b in SR:
            rs = []
            for pid in pids:
                seeds = sorted(by[pid])
                xs = [by[pid][s]["partial"][str(t)][b] for s in seeds]
                ys = [by[pid][s]["final"][b] for s in seeds]
                rs.append(spearman(xs, ys))
            row += f"{sum(rs)/len(rs):>+12.3f}"
        print(row)
    print("    cross-metric: does the partial HPSv3 rank predict the FINAL HPSv3 better than partial IR does?")

    print(f"\n=== 2. BEST-OF-k over initial latents, finishing every one (k*{2*a.steps} NFE) ===")
    print(f"{'k':>4}{'NFE':>7}" + "".join(f"{LAB[b]:>12}" for b in SR))
    print("-" * 47)
    rnd = random.Random(0)
    for k in [1, 2, 4, 8, 11, 16, 24, N]:
        if k > N: continue
        row = f"{k:>4}{k*2*a.steps:>7}"
        for b in SR:
            tot = 0.0
            for pid in pids:
                seeds = sorted(by[pid])
                tot += sum(max(by[pid][s]["final"][b] for s in rnd.sample(seeds, k)) for _ in range(a.draws)) / a.draws
            row += f"{tot/len(pids):>12.4f}"
        print(row + ("   <-- 616-NFE budget" if k == 11 else ""))

    print(f"\n=== 3. SCREEN-AND-FINISH at {a.budget} NFE, selecting on {LAB[a.metric]} ===")
    print(f"{'policy':<34}{'t':>4}{'n':>5}{'m':>4}{'NFE':>7}{LAB[a.metric]:>11}{'vs uniform':>12}")
    print("-" * 77)
    full = 2 * a.steps
    k_uni = a.budget // full
    uni = 0.0
    for pid in pids:
        seeds = sorted(by[pid])
        uni += sum(max(by[pid][s]["final"][a.metric] for s in rnd.sample(seeds, k_uni)) for _ in range(a.draws)) / a.draws
    uni /= len(pids)
    print(f"{'uniform best-of-k (= ns_bo11)':<34}{a.steps:>4}{k_uni:>5}{k_uni:>4}{k_uni*full:>7}{uni:>11.4f}{0.0:>+12.4f}")
    best = {}
    for ranker in ("partial", "oracle", "random"):
        rows_out = []
        for t in cps:
            cs, cf = 2 * t, 2 * (a.steps - t)
            for n in range(2, N + 1):
                m_max = (a.budget - n * cs) // cf
                for m in range(1, min(n, m_max) + 1):
                    cost = n * cs + m * cf
                    if cost > a.budget: continue
                    tot = 0.0
                    rr = random.Random(hash((t, n, m, ranker)) & 0xffffffff)
                    for pid in pids:
                        seeds = sorted(by[pid]); acc = 0.0
                        for _ in range(a.draws):
                            pick = rr.sample(seeds, n)
                            if ranker == "partial": key = lambda s: by[pid][s]["partial"][str(t)][a.metric]
                            elif ranker == "oracle": key = lambda s: by[pid][s]["final"][a.metric]
                            else: key = lambda s: rr.random()
                            surv = sorted(pick, key=key, reverse=True)[:m]
                            acc += max(by[pid][s]["final"][a.metric] for s in surv)
                        tot += acc / a.draws
                    rows_out.append((tot / len(pids), t, n, m, cost))
        rows_out.sort(reverse=True)
        v, t, n, m, c = rows_out[0]; best[ranker] = v
        nm = {"partial": f"screen by PARTIAL {LAB[a.metric]}", "oracle": "screen by TRUE FINAL (oracle ceiling)",
              "random": "screen at RANDOM (control)"}[ranker]
        print(f"{nm:<34}{t:>4}{n:>5}{m:>4}{c:>7}{v:>11.4f}{v-uni:>+12.4f}")
    print("-" * 77)
    print(f"  oracle ceiling over uniform          : {best['oracle']-uni:+.4f}   <-- NO abandonment rule can beat this")
    print(f"  realisable by partial-score screening: {best['partial']-uni:+.4f}")
    print(f"  attributable to seeing more noises   : {best['random']-uni:+.4f}   (random ranker, same budget)")
    info = best['partial'] - best['random']
    print(f"  attributable to the INFORMATION      : {info:+.4f}")
    print(f"\n  => headroom for a Jev saturation decision is at most {best['oracle']-uni:+.4f} "
          f"and a free reward-based screen already gets {best['partial']-uni:+.4f} of it.")

if __name__ == "__main__":
    main()
