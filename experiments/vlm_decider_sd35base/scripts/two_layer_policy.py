#!/usr/bin/env python3
"""Two-layer selection on the noise ladder (one prompt = N noises at the anchor action, all finished):
  layer 1: best-of-k by HPSv3 (aesthetic search reward)      layer 2: re-rank the top-k by a fidelity grader (Jev score / VLM / Qwen)
Reports, per policy, the mean over prompts of the picked image's VQAScore, Qwen alignment and HPSv3, with the NFE spent (k x 56),
against random, bo-k, grader-only and oracle. Inputs: ladder records (HPSv3), Jev grading json, judges json (same images).
Usage: python two_layer_policy.py <records_all.jsonl> <jev_grade.json> <judges.json> [--ks 2 4 8 16 32] [--draws 300]"""
import json, sys, argparse, collections, statistics as st, random
ap = argparse.ArgumentParser(); ap.add_argument("records"); ap.add_argument("jev"); ap.add_argument("judges"); ap.add_argument("--ks", nargs="+", type=int, default=[2, 4, 8, 16, 32]); ap.add_argument("--draws", type=int, default=300)
a = ap.parse_args(); rows = {}
for r in map(json.loads, open(a.records)): rows[r["image"]] = {"image": r["image"], "prompt_id": r["prompt_id"], "hpsv3": r["final"]["hpsv3"], "ir": r["final"]["imagereward"], "pickscore": r["final"]["pickscore"]}
for f in (a.jev, a.judges):
    for r in json.load(open(f)):
        if r["image"] in rows: rows[r["image"]].update({k: v for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
rows = [r for r in rows.values() if "vqascore" in r and "jev_score_E" in r]; byp = collections.defaultdict(list); [byp[r["prompt_id"]].append(r) for r in rows]
N = min(len(v) for v in byp.values()); print(f"{len(rows)} images, {len(byp)} prompts, {N} noises/prompt usable"); rng = random.Random(0)
T = ["vqascore", "qwen_alignment", "hpsv3"]
def pick_stats(fn):
    acc = collections.defaultdict(list)
    for _ in range(a.draws):
        for v in byp.values():
            p = fn(v)
            for t in T: acc[t].append(p[t])
    return {t: st.mean(acc[t]) for t in T}
def argmax(v, k): m = max(r[k] for r in v); return rng.choice([r for r in v if r[k] == m])
def rankavg(v, k1, k2):
    rk = lambda key: {id(r): s for s, r in enumerate(sorted(v, key=lambda r: r[key]))}; r1, r2 = rk(k1), rk(k2); sc = {id(r): r1[id(r)] + r2[id(r)] for r in v}; m = max(sc.values()); return rng.choice([r for r in v if sc[id(r)] == m])
pols = [("random single noise", 56, lambda v: rng.choice(v))]
for k in a.ks:
    if k <= N: pols.append((f"bo-{k} by HPSv3", 56 * k, (lambda k: lambda v: argmax(rng.sample(v, k), "hpsv3"))(k)))
pols += [(f"Jev score only, all {N}", 56 * N, lambda v: argmax(v, "jev_score_E")), (f"Jev P(yes) only, all {N}", 56 * N, lambda v: argmax(v, "jev_yes"))]
for k in a.ks:
    if 2 <= k <= N:
        pols.append((f"two-layer: top-{k} HPSv3 -> Jev score", 56 * N, (lambda k: lambda v: argmax(sorted(v, key=lambda r: -r["hpsv3"])[:k], "jev_score_E"))(k)))
        pols.append((f"two-layer: top-{k} HPSv3 -> rank-avg(HPSv3,Jev)", 56 * N, (lambda k: lambda v: rankavg(sorted(v, key=lambda r: -r["hpsv3"])[:k], "hpsv3", "jev_score_E"))(k)))
pols += [("oracle VQAScore", 0, lambda v: argmax(v, "vqascore")), ("oracle Qwen", 0, lambda v: argmax(v, "qwen_alignment")), ("oracle HPSv3 (bo-all)", 0, lambda v: argmax(v, "hpsv3"))]
print(f"{'policy':>46} {'NFE':>5} " + " ".join(f"{t:>10}" for t in T))
for name, nfe, fn in pols:
    s = pick_stats(fn); print(f"{name:>46} {nfe:5} " + " ".join(f"{s[t]:10.3f}" for t in T))
