#!/usr/bin/env python3
"""Decision value of each readout as a within-prompt SELECTOR: pick argmax readout among a prompt's images; report the mean fidelity
(VQAScore, Qwen) and HPSv3 of the picked image vs random pick (mean) and oracle (max). Also fair pairwise accuracy on COMMON untied pairs.
Usage: python selector_value.py <merged jsons...> [--k 36]"""
import json, sys, collections, itertools, statistics as st, random
files = [f for f in sys.argv[1:] if not f.startswith("--")]; rows = {}
for f in files:
    for r in json.load(open(f)): rows.setdefault(r["image"], {}).update({k: v for k, v in r.items() if not isinstance(v, (list, dict))})
rows = [r for r in rows.values() if "vqascore" in r and "jev_yes" in r]; byp = collections.defaultdict(list); [byp[r["prompt_id"]].append(r) for r in rows]
reads = ["jev_yes", "jev_score_E", "jev_req_pass", "vlm_yes", "vlm_rating", "vlm_pass", "vec.hpsv3", "vec.imagereward", "vec.pickscore", "qwen_alignment", "vqascore"]
tg = ["vqascore", "qwen_alignment", "vec.hpsv3"]
print(f"=== pick argmax(readout) among each prompt's {len(rows)//len(byp)} images -> mean over {len(byp)} prompts of the picked image's judge value (ties broken at random, 200 draws)")
print(f"{'selector':>16} " + " ".join(f"{t:>14}" for t in tg)); rng = random.Random(0)
base = {t: st.mean(st.mean(r[t] for r in v) for v in byp.values()) for t in tg}; orc = {t: st.mean(max(r[t] for r in v) for v in byp.values()) for t in tg}
print(f"{'random pick':>16} " + " ".join(f"{base[t]:14.3f}" for t in tg)); print(f"{'oracle':>16} " + " ".join(f"{orc[t]:14.3f}" for t in tg))
for k in reads:
    vals = {t: [] for t in tg}
    for _ in range(200):
        for v in byp.values():
            vv = [r for r in v if r.get(k) is not None]; m = max(r[k] for r in vv); pick = rng.choice([r for r in vv if r[k] == m])
            for t in tg: vals[t].append(pick[t])
    print(f"{k:>16} " + " ".join(f"{st.mean(vals[t]):14.3f}" for t in tg))
print("\n=== fair pairwise accuracy vs VQAScore (gap>0.05) on the COMMON set of pairs where every listed readout is untied")
sel = ["jev_yes", "vlm_yes", "vlm_pass", "vlm_rating", "vec.hpsv3", "qwen_alignment"]; pairs = []
for v in byp.values():
    for x, y in itertools.combinations(v, 2):
        if abs(x["vqascore"] - y["vqascore"]) > 0.05 and all(x.get(k) is not None and y.get(k) is not None and x[k] != y[k] for k in sel): pairs.append((x, y))
print(f"  n common pairs = {len(pairs)}"); [print(f"  {k:>16}: {100*st.mean((x[k]>y[k])==(x['vqascore']>y['vqascore']) for x,y in pairs):5.1f}%") for k in sel]
