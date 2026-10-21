#!/usr/bin/env python3
"""Summarize probe/ (or main/) arms: seconds, NFE, search score, tree shape, decision-model calls.  python summarize_probe.py [dir=probe]"""
import json, sys, glob, os, statistics as st
R = "/data/ygu/runs/vlm_decider_sd35base_2026-10-16"; d = sys.argv[1] if len(sys.argv) > 1 else "probe"
rows = []
for arm in ["baseline", "mcts", "rand_prune", "vlm_prune", "jev_prune"]:
    f = f"{R}/{d}/{arm}/records.jsonl"
    if not os.path.exists(f): continue
    recs = [json.loads(l) for l in open(f)]
    if not recs: continue
    m = lambda k: st.fmean(float(r.get(k, 0) or 0) for r in recs)
    depth = {}
    for r in recs:
        for k, v in (r.get("nodes_per_depth") or {}).items(): depth[int(k)] = depth.get(int(k), 0) + v
    rows.append((arm, len(recs), m("seconds"), m("nfe_transformer"), m("search_score"), m("peak_alloc_gib"), m("n_nodes"), m("n_vlm_calls"), m("vlm_seconds"),
                 " ".join(f"d{k}:{v}" for k, v in sorted(depth.items())), ",".join(sorted({r.get("selected_from", "-") for r in recs}))))
print(f"| arm | n | s/prompt | NFE | IR (search) | peak GiB | tree nodes | decider calls | decider s | nodes per depth | return |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for r in rows: print(f"| {r[0]} | {r[1]} | {r[2]:.0f} | {r[3]:.0f} | {r[4]:.4f} | {r[5]:.1f} | {r[6]:.0f} | {r[7]:.0f} | {r[8]:.0f} | {r[9]} | {r[10]} |")
for arm in ["vlm_prune", "jev_prune"]:
    f = f"{R}/{d}/{arm}/decisions.jsonl"
    if os.path.exists(f):
        ds = [json.loads(l) for l in open(f)]; parsed = sum(1 for x in ds if x.get("parsed")); print(f"\n{arm}: {len(ds)} decisions, parsed {parsed}; example comment (depth 1): " + next((x["comment"] for x in ds if x["depth"] == 1 and x.get("comment")), "-")[:300])
        if arm == "jev_prune":
            sp = [max(x["jev"]["jev_cfg_spread"]) for x in ds if x.get("jev")]; vs = [max(x["jev"]["jev_variant_probs"]) for x in ds if x.get("jev")]
            if sp: print(f"  Jev CFG order-spread max: mean {st.fmean(sp):.3f}; wording top-prob mean {st.fmean(vs):.3f}")
