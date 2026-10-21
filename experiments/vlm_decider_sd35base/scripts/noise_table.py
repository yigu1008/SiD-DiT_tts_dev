#!/usr/bin/env python3
"""How much is the initial latent worth, and does best-of-k noise beat the search at matched NFE?

Builds budget-matched arm dirs (ns_bo11_ir, ns_bo11_hps) from the probe's FIRST 11 seeds -- 11 x 56 = 616 NFE,
exactly the operating point every search arm was matched to -- so they can be judged held-out alongside them.
"""
import argparse, csv, json, statistics as st
from collections import defaultdict
from pathlib import Path
R = Path("/data/ygu/runs/vlm_decider_sd35base_2026-10-16")
SR = ["imagereward", "hpsv3", "pickscore"]
LAB = {"imagereward": "ir", "hpsv3": "hpsv3", "pickscore": "pickscore"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=11, help="budget-matched best-of-k (616 NFE / 56 NFE per generation)")
    p.add_argument("--emit_arms", action="store_true", help="write ns_bo11_* arm dirs for held-out judging")
    a = p.parse_args()
    rows = [json.loads(l) for l in open(R / "main/ns_probe/records_all.jsonl")]
    by = defaultdict(dict)
    for r in rows: by[r["prompt_id"]][r["noise_seed"]] = r
    pids = sorted(by)
    seeds_order = [42] + sorted(s for s in set().union(*[set(v) for v in by.values()]) if s != 42)
    print(f"prompts={len(pids)} seeds/prompt={len(seeds_order)} (first is 42, the seed every other arm used)")
    print(f"NFE per generation = 56; budget-matched k = {a.k} -> {a.k*56} NFE\n")

    print("=== 1. SPREAD across initial latents at a FIXED action (wording A, CFG 4.5) ===")
    print(f"{'prompt':<8}{'metric':<11}{'seed42':>9}{'min':>9}{'max':>9}{'range':>9}{'sd':>8}{'rank of 42':>12}")
    print("-" * 75)
    agg = defaultdict(list)
    for pid in pids:
        for b in SR:
            vals = [by[pid][s]["vec"][b] for s in seeds_order if s in by[pid]]
            v42 = by[pid][42]["vec"][b]
            rank = 1 + sum(1 for x in vals if x > v42)
            print(f"{pid:<8}{LAB[b]:<11}{v42:>9.3f}{min(vals):>9.3f}{max(vals):>9.3f}{max(vals)-min(vals):>9.3f}"
                  f"{st.stdev(vals):>8.3f}{f'{rank}/{len(vals)}':>12}")
            agg[b].append((v42, min(vals), max(vals), st.stdev(vals), rank, len(vals)))
    print("-" * 75)
    for b in SR:
        d = agg[b]
        print(f"{'MEAN':<8}{LAB[b]:<11}{sum(x[0] for x in d)/len(d):>9.3f}{sum(x[1] for x in d)/len(d):>9.3f}"
              f"{sum(x[2] for x in d)/len(d):>9.3f}{sum(x[2]-x[1] for x in d)/len(d):>9.3f}"
              f"{sum(x[3] for x in d)/len(d):>8.3f}{sum(x[4] for x in d)/len(d):>11.1f}/{d[0][5]}")

    print("\n=== 2. BEST-OF-k over initial latents (first k seeds, selected by that same reward) ===")
    hdr = f"{'k':>4}{'NFE':>7}" + "".join(f"{LAB[b]:>12}" for b in SR)
    print(hdr); print("-" * len(hdr))
    for k in [1, 2, 4, 8, a.k, len(seeds_order)]:
        if k > len(seeds_order): continue
        row = f"{k:>4}{k*56:>7}"
        for b in SR:
            m = sum(max(by[pid][s]["vec"][b] for s in seeds_order[:k] if s in by[pid]) for pid in pids) / len(pids)
            row += f"{m:>12.4f}"
        print(row + ("   <-- budget-matched to the 616-NFE search arms" if k == a.k else ""))

    print("\n=== 3. Budget-matched: best-of-%d NOISE vs the 616-NFE SEARCH arms (search-reward scale) ===" % a.k)
    v = defaultdict(dict)
    for r in csv.DictReader(open(R / "scores.csv")):
        if r["prompt_id"] in pids: v[(r["arm"], r["prompt_id"])][r["metric"]] = float(r["score"])
    arms = ["rs_blind", "rs_ir", "rs_hps", "rs_jev"]
    print(f"{'arm':<22}{'NFE':>6}{'ir':>10}{'hpsv3':>10}{'pickscore':>11}")
    print("-" * 59)
    for b, nm in [("imagereward", f"best-of-{a.k} noise (ir)"), ("hpsv3", f"best-of-{a.k} noise (hpsv3)")]:
        sel = {pid: max((by[pid][s]["vec"][b], s) for s in seeds_order[:a.k] if s in by[pid])[1] for pid in pids}
        row = f"{nm:<22}{a.k*56:>6}"
        for m in SR: row += f"{sum(by[pid][sel[pid]]['vec'][m] for pid in pids)/len(pids):>{10 if m!='pickscore' else 11}.4f}"
        print(row)
    for ar in arms:
        if not all((ar, pid) in v for pid in pids): continue
        row = f"{ar:<22}{616:>6}"
        for m in ["ir", "hpsv3", "pickscore"]: row += f"{sum(v[(ar,pid)][m] for pid in pids)/len(pids):>{10 if m!='pickscore' else 11}.4f}"
        print(row)
    print("  (the noise columns are server scores, the arm columns are judge scores; both use the same models, and\n"
          "   seed-42 agreement is checked below. Held-out judging of the selected images is the real comparison.)")

    print("\n=== 4. Sanity: does probe seed 42 reproduce the existing baseline arm? ===")
    for ar in ("dev3_baseline", "baseline"):
        hits = [(pid, v[(ar, pid)]["ir"], by[pid][42]["vec"]["imagereward"]) for pid in pids if (ar, pid) in v and "ir" in v[(ar, pid)]]
        for pid, judged, probe in hits:
            print(f"  {ar} {pid}: judged ir={judged:.4f}  probe seed42 ir={probe:.4f}  diff={probe-judged:+.4f}")
        if hits: break
    else: print("  no baseline arm overlaps this cohort")

    if a.emit_arms:
        for b, nm in [("imagereward", "ns_bo11_ir"), ("hpsv3", "ns_bo11_hps")]:
            d = R / "main" / nm; d.mkdir(parents=True, exist_ok=True)
            with (d / "records.jsonl").open("w") as f:
                for pid in pids:
                    s = max((by[pid][ss]["vec"][b], ss) for ss in seeds_order[:a.k] if ss in by[pid])[1]
                    r = by[pid][s]
                    f.write(json.dumps({"prompt_id": pid, "prompt": r["prompt"], "arm": nm, "seed": 42,
                                        "noise_seed": s, "search_score": r["vec"][b], "search_vec": r["vec"],
                                        "actions": [[r["wording"], r["cfg"], 0.0]], "image": r["image"],
                                        "image_sha256": r["image_sha256"], "nfe_transformer": a.k * 56,
                                        "selected_by": b, "k": a.k}) + "\n")
            print(f"  wrote {d}/records.jsonl ({len(pids)} rows, selected by {b}, {a.k*56} NFE)")

if __name__ == "__main__":
    main()
