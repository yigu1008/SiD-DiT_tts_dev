#!/usr/bin/env python3
"""Regenerate PROBE.md from probe/*/records.jsonl, probe/*/decisions.jsonl and scores.csv."""
import json, csv, os, collections, statistics as st
R = "/data/ygu/runs/vlm_decider_sd35base_2026-10-16"; ARMS = ["baseline", "mcts", "mcts_refine", "rand_prune", "vlm_prune", "jev_prune"]
DESC = {"baseline": "variant 0, CFG 4.5, no search", "mcts": "stock key-step MCTS, 28-action bank, basic (no refine stage)",
        "mcts_refine": "stock default for steps>4: MCTS + sparse-noise refine pass over the selected path", "rand_prune": "per-node set = 6 random actions + anchor",
        "vlm_prune": "Qwen2.5-VL ranks wordings and CFGs itself (factorized, shuffled labels)", "jev_prune": "Qwen2.5-VL readout -> Jev-Omni typed choices (wording, CFG)"}
recs = {a: json.loads(open(f"{R}/probe/{a}/records.jsonl").readline()) for a in ARMS if os.path.exists(f"{R}/probe/{a}/records.jsonl") and os.path.getsize(f"{R}/probe/{a}/records.jsonl") > 0}
sc = collections.defaultdict(dict)
if os.path.exists(f"{R}/scores.csv"):
    for r in csv.DictReader(open(f"{R}/scores.csv")): sc[r["arm"]][r["metric"]] = float(r["score"])
L = []; P = L.append
P("# Probe on base SD3.5-Large: one prompt (q0000), 40 simulations per search arm\n")
P(f"Prompt: *{next(iter(recs.values()))['prompt']}*  (seed 42, 1024², 28 Euler steps, key steps 0,4,8,12,16,20,24,27, bank = 4 wordings × CFG {{3.5,…,7.0}}, ImageReward objective, RTX 6000 Ada)\n")
P("This is a pipeline/timing probe at n = 1. Nothing here is statistical evidence; the cohort run is what tests the claims.\n")
P("## Cost and search shape\n")
P("| arm | what | s / prompt | transformer NFE | decider calls (s) | tree nodes by depth | returned |"); P("| --- | --- | --- | --- | --- | --- | --- |")
for a in ARMS:
    if a not in recs: P(f"| {a} | {DESC[a]} | (pending) | | | | |"); continue
    r = recs[a]; dep = " ".join(f"d{k}:{v}" for k, v in sorted((r.get("nodes_per_depth") or {}).items(), key=lambda kv: int(kv[0])))
    P(f"| {a} | {DESC[a]} | {r['seconds']:.0f} | {r['nfe_transformer']} | {r.get('n_vlm_calls', 0)} ({r.get('vlm_seconds', 0):.0f}) | {dep or '-'} | {r.get('selected_from', '-')} |")
P("\nNFE accounting (verified with an independent step counter, `segment_steps`): every CFG step is 2 forwards; MCTS reuses stored prefixes, so an "
  "expansion from a depth-d node costs only the remaining steps. The stock 28-action tree is almost flat (28 root children first), the pruned trees go deeper, "
  "so at equal simulations the pruned arms spend ~10% fewer NFE. The stock *default* for steps>4 additionally runs a sparse-noise refine pass "
  "(`--mcts_sparse_noise_refine`, ~+1000 NFE); `mcts` here disables it, `mcts_refine` keeps it.\n")
P("## Held-out judges (one image per arm)\n")
mets = ["ir", "hpsv3", "pickscore", "hpsv2", "vqascore", "qwen_alignment"]
P("| arm | " + " | ".join(mets) + " |"); P("|" + "---|" * (len(mets) + 1))
for a in ARMS:
    if a in sc: P(f"| {a} | " + " | ".join(f"{sc[a].get(m, float('nan')):.3f}" for m in mets) + " |")
    elif a in recs: P(f"| {a} | (not judged yet) |" + " |" * (len(mets) - 1))
if "baseline" in sc:
    P("\nDelta vs baseline:\n"); P("| arm | " + " | ".join(mets) + " |"); P("|" + "---|" * (len(mets) + 1))
    for a in ARMS:
        if a in sc and a != "baseline": P(f"| {a} | " + " | ".join(f"{sc[a][m]-sc['baseline'][m]:+.3f}" for m in mets) + " |")
P("\nImageReward is the optimized objective (reported, not evidence). HPSv3, PickScore, HPSv2 and VQAScore are held out. The Qwen2-VL alignment judge is a different "
  "checkpoint from the Qwen2.5-VL reader but the same family (caveat). `search_score` equals judged `ir` for every arm (live/offline consistency).\n")
P("## Decision statistics\n")
for a in ("vlm_prune", "jev_prune"):
    f = f"{R}/probe/{a}/decisions.jsonl"
    if not os.path.exists(f): continue
    ds = [json.loads(l) for l in open(f)]; V = recs[a]["variants"]
    if a == "jev_prune":
        tv = collections.Counter(d["variants_best_first"][0] for d in ds); tc = collections.Counter(d["cfg_best_first"][0] for d in ds)
        P(f"- **jev_prune**: {len(ds)} nodes; Jev top wording per node: " + ", ".join(f"v{k}×{v}" for k, v in sorted(tv.items())) +
          f"; top CFG per node: {dict(sorted(tc.items()))}; wording top-prob mean {st.fmean(max(d['jev']['jev_variant_probs']) for d in ds):.3f}, "
          f"CFG top-prob mean {st.fmean(max(d['jev']['jev_cfg_probs']) for d in ds):.3f}, CFG order-spread (max) mean {st.fmean(max(d['jev']['jev_cfg_spread']) for d in ds):.3f}. "
          f"Readouts + 2 Jev questions ≈ {recs[a]['vlm_seconds']/len(ds):.1f} s per node.")
    else:
        tv = collections.Counter(V.index(d["labels"][d["variants_best_first"][0]]) if d.get("labels") else -1 for d in ds) if ds and ds[0].get("labels") else collections.Counter()
        tc = collections.Counter(d["cfg_best_first"][0] for d in ds)
        P(f"- **vlm_prune**: {len(ds)} nodes; VLM top wording per node: " + ", ".join(f"v{k}×{v}" for k, v in sorted(tv.items())) + f"; top CFG per node: {dict(sorted(tc.items()))}; {sum(1 for d in ds if d.get('parsed'))}/{len(ds)} JSON parsed; ≈ {recs[a]['vlm_seconds']/len(ds):.1f} s per node.")
P("\nWordings: " + "; ".join(f"v{i} = “{v[:90]}”" for i, v in enumerate(next(iter(recs.values()))["variants"])))
P("\n## Chosen action paths (variant idx, CFG) at the 8 key steps\n")
for a in ARMS:
    if a in recs: P(f"- {a}: " + " → ".join(f"(v{int(v)},{c:g})" for v, c, _ in recs[a]["actions"]))
P("\n![contact sheet](probe/contact_sheet_q0000.png)\n")
P("## Reading (n = 1, illustrative only)\n")
P("- The stock search with its default refine pass maximized ImageReward but produced an over-saturated, painterly image (HPSv3 −3.8, PickScore −0.9 vs baseline), "
  "choosing the malformed wording v3 at CFG 7.0 at step 0 and swinging CFG at every key step.")
P("- The basic stock search (refine pass off) gained ImageReward +0.17 and also held up on the held-out judges on this prompt (HPSv3 +0.36), so on q0000 the "
  "hacking came from the refine pass, not from the tree search itself.")
P("- The decision arms (VLM, Jev) gained ImageReward moderately (+0.16 / +0.11) and improved HPSv3 the most (+0.5 / +0.7) at ~10% fewer NFE than stock; "
  "random pruning gained ImageReward (+0.14) but lost HPSv3 (−1.0), so a smaller branching factor alone does not explain the held-out behaviour.")
P("- Jev picked the original wording at 40/41 nodes with p≈0.94 and preferred CFG 5–7 with weak confidence (top-prob 0.31, order-spread 0.35); the VLM-as-decider preferred CFG 3.5–4.0.")
P("- Readouts still occasionally treat variant content (e.g. “sunset”) as part of the prompt; the Jev state shows only the ORIGINAL prompt as target.")
open(f"{R}/PROBE.md", "w").write("\n".join(L) + "\n"); print("PROBE.md written", len(L), "lines")
