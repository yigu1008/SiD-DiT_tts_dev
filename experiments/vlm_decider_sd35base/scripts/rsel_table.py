#!/usr/bin/env python3
"""Frontier reading of the reward-selection arms. The question is whether adaptive switching moves the
ir-vs-hpsv3 frontier -- i.e. whether rs_jev DOMINATES the fixed-reward arms jointly -- not whether it
wins a single metric (an HPSv3-following arm beats an IR arm on HPSv3 by construction).
Usage: rsel_table.py [--arms rs_blind rs_ir rs_hps rs_jev] [--ids ...]"""
import argparse, csv, json, itertools
from collections import defaultdict
from pathlib import Path
R = Path("/data/ygu/runs/vlm_decider_sd35base_2026-10-16")
METRICS = ["ir", "hpsv3", "pickscore", "hpsv2", "qwen_alignment", "vqascore"]
HELD_OUT = {"vqascore", "qwen_alignment", "hpsv2"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=["rs_blind", "rs_ir", "rs_hps", "rs_jev"])
    p.add_argument("--ids", default=str(R / "config/prompt_ids_rsel9.txt"))
    p.add_argument("--boot", type=int, default=10000)
    a = p.parse_args()
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    v = defaultdict(dict)
    for r in csv.DictReader(open(R / "scores.csv")):
        if r["arm"] in a.arms and r["prompt_id"] in ids: v[(r["arm"], r["prompt_id"])][r["metric"]] = float(r["score"])
    print(f"cohort n={len(ids)}  arms={a.arms}\n")
    print("=== means over prompts with a score for every listed arm (held-out metrics marked *) ===")
    hdr = f"{'arm':10s}" + "".join(f"{m + ('*' if m in HELD_OUT else ''):>17s}" for m in METRICS)
    print(hdr); print("-" * len(hdr))
    common = {m: [i for i in ids if all(m in v[(ar, i)] for ar in a.arms)] for m in METRICS}
    means = {}
    for ar in a.arms:
        row = f"{ar:10s}"
        for m in METRICS:
            c = common[m]
            if not c: row += f"{'-':>17s}"; continue
            mu = sum(v[(ar, i)][m] for i in c) / len(c); means[(ar, m)] = mu
            row += f"{mu:>17.4f}"
        print(row)
    print("    n per metric: " + "  ".join(f"{m}={len(common[m])}" for m in METRICS))

    import random
    def paired(A, B, m):
        c = [i for i in common[m] if (A, i) in v and (B, i) in v]
        d = [v[(A, i)][m] - v[(B, i)][m] for i in c]
        if not d: return None
        mu = sum(d) / len(d); rnd = random.Random(0)
        bs = sorted(sum(rnd.choice(d) for _ in d) / len(d) for _ in range(a.boot))
        return mu, bs[int(.025 * a.boot)], bs[int(.975 * a.boot)], len(d)

    print("\n=== paired prompt-bootstrap, arm minus rs_blind (95% CI, 10k resamples; CI crossing 0 = inconclusive) ===")
    for ar in a.arms:
        if ar == "rs_blind": continue
        print(f"-- {ar} - rs_blind")
        for m in METRICS:
            r = paired(ar, "rs_blind", m)
            if r: print(f"   {m+('*' if m in HELD_OUT else ''):<16s} {r[0]:+.4f}  [{r[1]:+.4f}, {r[2]:+.4f}]  n={r[3]}"
                        + ("" if r[1] <= 0 <= r[2] else "   <-- excludes 0"))
    print("\n=== paired, rs_jev minus each fixed-reward arm (does switching beat committing?) ===")
    for ar in ("rs_ir", "rs_hps"):
        if ar not in a.arms: continue
        print(f"-- rs_jev - {ar}")
        for m in METRICS:
            r = paired("rs_jev", ar, m)
            if r: print(f"   {m+('*' if m in HELD_OUT else ''):<16s} {r[0]:+.4f}  [{r[1]:+.4f}, {r[2]:+.4f}]  n={r[3]}"
                        + ("" if r[1] <= 0 <= r[2] else "   <-- excludes 0"))

    print("\n=== FRONTIER: is any arm dominated on (ir, hpsv3)?  (and on the held-out pair) ===")
    for pair in [("ir", "hpsv3"), ("vqascore", "hpsv3"), ("qwen_alignment", "hpsv3")]:
        x, y = pair
        if not (common[x] and common[y]): continue
        print(f"-- ({x}, {y}):")
        pts = {ar: (means.get((ar, x)), means.get((ar, y))) for ar in a.arms if means.get((ar, x)) is not None}
        for ar, (px, py) in pts.items():
            dom = [o for o, (ox, oy) in pts.items() if o != ar and ox >= px and oy >= py and (ox > px or oy > py)]
            tag = "DOMINATED by " + ",".join(dom) if dom else "on the frontier"
            print(f"   {ar:10s} {x}={px:+.4f} {y}={py:+.4f}   {tag}")

    print("\n=== what rs_jev actually chose ===")
    rp = R / "main/rs_jev/records.jsonl"
    if rp.exists():
        tot = defaultdict(int); roots = defaultdict(int); nsw = 0
        for r in map(json.loads, open(rp)):
            roots[r["root_reward"]] += 1
            for k, n in (r.get("reward_votes") or {}).items(): tot[k] += n
            if len(set((r.get("node_rewards") or {}).values())) > 1: nsw += 1
            print(f"   {r['prompt_id']}  root={r['root_reward']:12s} votes={dict(r.get('reward_votes') or {})}"
                  f"  ir={r['search_vec']['imagereward']:.3f} hps={r['search_vec']['hpsv3']:.3f} ps={r['search_vec']['pickscore']:.3f}"
                  f"  nfe={r['nfe_transformer']}")
        print(f"   root-reward distribution: {dict(roots)}")
        print(f"   node-reward votes total:  {dict(tot)}")
        print(f"   prompts where nodes DISAGREED on the reward: {nsw}/{sum(roots.values())}")

if __name__ == "__main__":
    main()
