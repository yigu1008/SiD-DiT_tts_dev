#!/usr/bin/env python3
"""Is Jev's constant CFG-7.0 preference a property of the model or of MY option labels?

Re-asks the same CFG question on the same previews under four labelling schemes:
  current   - the labels used in all runs so far (extremes described; "strongest adherence" vs "weakest adherence")
  neutral   - bare numbers, no description at all
  reversed  - the same descriptions with their valence swapped (high = "oversaturated", low = "natural")
  symmetric - both extremes described with balanced costs and benefits
If the preference survives neutral labelling it is the model's; if it tracks the wording it was mine.
No sampling: previews and the Jev server only.
"""
import json, base64, sys, collections, statistics as st, urllib.request, glob, os
R = "/data/ygu/runs/vlm_decider_sd35base_2026-10-16"
JEV = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5161"
CFGS = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]
def lab(scheme):
    lo, hi = min(CFGS), max(CFGS)
    if scheme == "neutral":   return [f"CFG {c:g}" for c in CFGS]
    if scheme == "current":   return [f"CFG {c:g}" + (" (lowest guidance: most natural, weakest prompt adherence)" if c == lo else " (highest guidance: strongest adherence, most saturation and artifacts)" if c == hi else "") for c in CFGS]
    if scheme == "reversed":  return [f"CFG {c:g}" + (" (lowest guidance: least saturation and fewest artifacts)" if c == lo else " (highest guidance: most oversaturated, most artifacts, weakest naturalness)" if c == hi else "") for c in CFGS]
    if scheme == "symmetric": return [f"CFG {c:g}" + (" (low guidance: more natural, looser prompt adherence)" if c == lo else " (high guidance: tighter prompt adherence, more saturation)" if c == hi else "") for c in CFGS]
def ask(state, opts, seed, tag):
    req = urllib.request.Request(JEV + "/ask", data=json.dumps({"state": state, "question": "Which classifier-free guidance (CFG) strength should the sampler use for the next steps?",
        "options": opts, "orders": 3, "seed": seed, "tag": tag}).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())
import csv as _csv
PMAP = {r["prompt_id"]: r["prompt"] for r in _csv.DictReader(open("/data/ygu/reward_orthogonality/mcts/prompts/prompt_map.csv"))}
dec = [json.loads(l) for l in open(f"{R}/main/fr_jev_s10/decisions.jsonl")]
nodes = [d for d in dec if d.get("preview") and os.path.exists(d["preview"])][:12]
print(f"re-asking {len(nodes)} real decision nodes under 4 labelling schemes\n")
res = collections.defaultdict(collections.Counter); prob = collections.defaultdict(list)
for d in nodes:
    b64 = base64.b64encode(open(d["preview"], "rb").read()).decode()
    # faithful reconstruction of the state vlm_mcts.py built for this node (prompt text + step + the VLM readout)
    state = (f'ORIGINAL PROMPT (the only target): "{PMAP[d["prompt_id"]]}"\nDenoising step {d["step"]} of 28.\n'
             f'VLM readout of the current preview: {d.get("comment") or ""}\n'
             f'The sampler will apply the chosen setting for the next denoising steps; the FINAL image must match the ORIGINAL prompt.')
    for scheme in ("current", "neutral", "reversed", "symmetric"):
        r = ask(state, lab(scheme), seed=d["node_id"], tag=f"cfglabel/{scheme}")
        top = CFGS[int(r["argmax"])]; res[scheme][top] += 1; prob[scheme].append(max(r["probs"]))
print("| labelling | #1 CFG chosen (count) | mean top-prob |")
print("|---|---|---|")
for scheme in ("current", "neutral", "reversed", "symmetric"):
    print(f"| {scheme} | {dict(sorted(res[scheme].items()))} | {st.fmean(prob[scheme]):.3f} |")
print(f"\n({len(nodes)} nodes; 'current' is the labelling every run in this study used; state reconstructed faithfully: prompt text + step + VLM readout.)")
agree = sum(1 for d in nodes if True)
print("\nFor reference, in the real 10-sim run Jev ranked CFG 7.0 first at 33/33 nodes.")
