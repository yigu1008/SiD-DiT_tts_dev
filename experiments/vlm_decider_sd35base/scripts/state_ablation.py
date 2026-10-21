#!/usr/bin/env python3
"""What in the state is Jev actually using?

Takes real decision nodes and re-asks the CFG question under ablations:
  full          - the state the runs actually used (prompt + step + VLM readout) + the preview image
  no_image      - same text, image withheld        -> does the preview contribute anything?
  swapped_read  - another node's readout pasted in -> does the readout contribute anything?
  no_readout    - readout line deleted             -> ditto, by removal
  plus_cfg      - full + "the preview was produced at CFG X" (the missing error signal)
Reports how often the top choice changes versus `full`, and the mean total-variation distance of the
probability vector. If no_image and swapped_read barely move it, the decision is driven by the prompt
and the question alone, which is what a near-constant output implies.
"""
import json, base64, sys, os, csv, random, statistics as st, urllib.request
import numpy as np
R = "/data/ygu/runs/vlm_decider_sd35base_2026-10-16"
JEV = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5161"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
CFGS = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]
COPTS = [f"CFG {c:g}" + (" (lowest guidance: most natural, weakest prompt adherence)" if c == min(CFGS)
         else " (highest guidance: strongest adherence, most saturation and artifacts)" if c == max(CFGS) else "") for c in CFGS]
PMAP = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open("/data/ygu/reward_orthogonality/mcts/prompts/prompt_map.csv"))}

def ask(state, b64, seed, tag):
    body = {"state": state, "question": "Which classifier-free guidance (CFG) strength should the sampler use for the next steps?",
            "options": COPTS, "orders": 3, "seed": seed, "tag": tag}
    if b64: body["image_b64"] = b64
    req = urllib.request.Request(JEV + "/ask", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())

def build(prompt, step, readout, cfg_now=None):
    s = f'ORIGINAL PROMPT (the only target): "{prompt}"\nDenoising step {step} of 28.\n'
    if cfg_now is not None: s += f"The attached preview was produced using CFG {cfg_now:g} so far.\n"
    if readout is not None: s += f"VLM readout of the current preview: {readout}\n"
    return s + "The sampler will apply the chosen setting for the next denoising steps; the FINAL image must match the ORIGINAL prompt."

dec = [json.loads(l) for l in open(f"{R}/main/fr_jev_s10/decisions.jsonl")]
nodes = [d for d in dec if d.get("preview") and os.path.exists(d["preview"]) and d.get("comment")][:N]
others = [d.get("comment") for d in dec if d.get("comment")]
rng = random.Random(0)
res = {k: {"flip": 0, "tv": []} for k in ("no_image", "swapped_read", "no_readout", "plus_cfg")}
print(f"state ablation on {len(nodes)} real nodes\n")
for d in nodes:
    pid = d["prompt_id"]; b64 = base64.b64encode(open(d["preview"], "rb").read()).decode()
    cfg_now = float(d["path_actions"][-1][1]) if d.get("path_actions") else 4.5
    base_state = build(PMAP[pid], d["step"], d["comment"])
    P0 = np.array(ask(base_state, b64, d["node_id"], "abl/full")["probs"])
    variants = {
        "no_image":     (base_state, None),
        "swapped_read": (build(PMAP[pid], d["step"], rng.choice([c for c in others if c != d["comment"]])), b64),
        "no_readout":   (build(PMAP[pid], d["step"], None), b64),
        "plus_cfg":     (build(PMAP[pid], d["step"], d["comment"], cfg_now=cfg_now), b64),
    }
    line = f"  node {d['node_id']:3d} {pid} (preview made at CFG {cfg_now:g}): full -> CFG {CFGS[int(P0.argmax())]:g}"
    for k, (stt, img) in variants.items():
        P = np.array(ask(stt, img, d["node_id"], f"abl/{k}")["probs"])
        res[k]["tv"].append(float(0.5 * np.abs(P - P0).sum()))
        if int(P.argmax()) != int(P0.argmax()): res[k]["flip"] += 1
        line += f" | {k} -> {CFGS[int(P.argmax())]:g}"
    print(line)
print(f"\n| ablation | top choice changed | mean total-variation distance |"); print("|---|---|---|")
for k in ("no_image", "swapped_read", "no_readout", "plus_cfg"):
    print(f"| {k} | {res[k]['flip']}/{len(nodes)} | {st.fmean(res[k]['tv']):.3f} |")
print("\n(TV distance 0 = the ablation changed nothing; 1 = completely different distribution.)")
