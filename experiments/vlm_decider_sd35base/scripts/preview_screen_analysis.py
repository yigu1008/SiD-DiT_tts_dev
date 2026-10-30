#!/usr/bin/env python3
"""Early-stop screening with Jev on step-t previews vs the free preview-HPSv3 screen. Inputs: preview grading json (jev_score_E/jev_yes
per preview + partial.hpsv3 + vec.hpsv3 final), finals judges json (vqascore, qwen_alignment keyed by final image).
Per prompt (N noises): rank-correlation of each preview readout with FINAL HPSv3 / VQAScore; then screen-and-finish at a budget:
screen all N to step t (2t NFE each), finish top-m by the readout (2(28-t) each), pick the final by HPSv3 -> report picked VQAScore/HPSv3.
Usage: python preview_screen_analysis.py <preview_grade.json> <judges.json> [--t 12] [--m 2 3 5]"""
import json, argparse, collections, statistics as st, random
ap = argparse.ArgumentParser(); ap.add_argument("prev"); ap.add_argument("judges"); ap.add_argument("--t", type=int, default=12); ap.add_argument("--m", nargs="+", type=int, default=[2, 3, 5]); a = ap.parse_args()
J = {r["image"]: r for r in json.load(open(a.judges))}
rows = []
import os
for r in json.load(open(a.prev)):   # jev_grade_images drops string fields except image: rebuild the final image path from the preview path
    fin = r.get("final_image") or os.path.join(os.path.dirname(os.path.dirname(r["image"])), "images", f"{r['prompt_id']}_s{int(r['noise_seed'])}.png")
    j = J.get(fin)
    if not j: continue
    rows.append({"prompt_id": r["prompt_id"], "jev_prev_E": r["jev_score_E"], "jev_prev_yes": r["jev_yes"], "prev_hpsv3": r.get("partial.hpsv3"), "prev_ir": r.get("partial.imagereward"), "final_hpsv3": r["vec.hpsv3"], "vqascore": j["vqascore"], "qwen": j.get("qwen_alignment")})
byp = collections.defaultdict(list); [byp[r["prompt_id"]].append(r) for r in rows]; N = min(len(v) for v in byp.values())
def spearman(x, y):
    rk = lambda v: {i: s for s, i in enumerate(sorted(range(len(v)), key=lambda i: v[i]))}
    rx, ry = rk(x), rk(y); n = len(x); return 1 - 6 * sum((rx[i] - ry[i]) ** 2 for i in range(n)) / (n * (n * n - 1))
print(f"=== {len(rows)} noises, {len(byp)} prompts; preview at step {a.t} ({2*a.t} NFE of 56). Within-prompt Spearman of preview readouts with the FINISHED image's judges:")
for k in ("jev_prev_E", "jev_prev_yes", "prev_hpsv3", "prev_ir"):
    if any(r[k] is None for r in rows): continue
    print(f"  {k:>13}: vs final HPSv3 {st.mean(spearman([r[k] for r in v], [r['final_hpsv3'] for r in v]) for v in byp.values()):+.3f} | vs VQAScore {st.mean(spearman([r[k] for r in v], [r['vqascore'] for r in v]) for v in byp.values()):+.3f} | vs Qwen {st.mean(spearman([r[k] for r in v], [r['qwen'] for r in v]) for v in byp.values()):+.3f}")
print(f"  (finished-image references) final HPSv3 vs VQAScore {st.mean(spearman([r['final_hpsv3'] for r in v], [r['vqascore'] for r in v]) for v in byp.values()):+.3f}")
rng = random.Random(0); T = ("vqascore", "qwen", "final_hpsv3")
print(f"\n=== screen all {N} noises to step {a.t}, finish the top-m by the readout, then pick by {{selector}} among the finished ones; mean picked values (300 draws)")
print(f"{'policy':>52} {'NFE':>5} " + " ".join(f"{t:>12}" for t in T))
def run(fn, nfe, name):
    acc = collections.defaultdict(list)
    for _ in range(300):
        for v in byp.values():
            p = fn(v); [acc[t].append(p[t]) for t in T]
    print(f"{name:>52} {nfe:5} " + " ".join(f"{st.mean(acc[t]):12.3f}" for t in T))
run(lambda v: rng.choice(v), 56, "single noise")
for k in (2, 5, 11):
    run((lambda k: lambda v: max(rng.sample(v, k), key=lambda r: r["final_hpsv3"]))(k), 56 * k, f"bo-{k} finished, pick HPSv3")
for m in a.m:
    nfe = N * 2 * a.t + m * 2 * (28 - a.t)
    run((lambda m: lambda v: max(sorted(v, key=lambda r: -r["prev_hpsv3"])[:m], key=lambda r: r["final_hpsv3"]))(m), nfe, f"screen by preview-HPSv3 -> top-{m} finished, pick HPSv3")
    run((lambda m: lambda v: max(sorted(v, key=lambda r: -r["jev_prev_E"])[:m], key=lambda r: r["final_hpsv3"]))(m), nfe, f"screen by Jev preview score -> top-{m}, pick HPSv3")
    run((lambda m: lambda v: max(sorted(v, key=lambda r: -r["jev_prev_E"])[:m], key=lambda r: r["jev_prev_E"]))(m), nfe, f"screen by Jev preview score -> top-{m}, pick Jev")
    run((lambda m: lambda v: max(sorted(v, key=lambda r: -r["vqascore"])[:m], key=lambda r: r["vqascore"]))(m), nfe, f"ORACLE fidelity screen -> top-{m}")
run(lambda v: max(v, key=lambda r: r["vqascore"]), 56 * N, "oracle VQAScore over all finished"); run(lambda v: max(v, key=lambda r: r["final_hpsv3"]), 56 * N, f"bo-{N} finished, pick HPSv3")
