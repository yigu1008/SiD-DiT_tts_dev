#!/usr/bin/env python3
"""Does Jev's image conditioning work AT ALL in this setup?

The CFG decision ignores the image (0/8 flips when it is withheld). Two explanations:
  (a) the image path is broken/unused in my server calls, or
  (b) it works, and the CFG question is simply answered from a lexical prior instead.
This asks questions whose answer is *determined* by the image, on the same preview files:
  blurry   - an early x0 preview (step 4-8) vs a finished image: "Is this image sharp or blurry?"
  content  - the correct prompt vs a clearly wrong one: "Does this image show X or Y?"
If Jev answers these correctly the image path is live and (b) holds.
"""
import json, base64, sys, os, glob, collections
import urllib.request
R = "/data/ygu/runs/vlm_decider_sd35base_2026-10-16"
JEV = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5161"
def ask(state, q, opts, b64, seed, tag):
    body = {"state": state, "question": q, "options": opts, "orders": 2, "seed": seed, "tag": tag, "image_b64": b64}
    req = urllib.request.Request(JEV + "/ask", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())
def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()
# early previews (blurry by construction) and finished images (sharp)
early = sorted(glob.glob(f"{R}/main/fr_jev_s10/previews/*_d1_s4.png"))[:4]
final = sorted(glob.glob(f"{R}/main/fr_blind_s10/*_mcts.png"))[:4] or sorted(glob.glob(f"{R}/main/dev3_mcts/*.png"))[:4]
print(f"sanity: {len(early)} early previews (step 4), {len(final)} finished images\n")
print("=== Q1: sharpness — the answer is determined by the image ===")
ok = 0
for p in early:
    r = ask("An image produced by a diffusion sampler.", "Is this image sharp and fully resolved, or blurry and unresolved?",
            ["It is blurry and unresolved", "It is sharp and fully resolved"], b64(p), 1, "sanity/blurry-early")
    good = r["argmax"] == 0; ok += good
    print(f"  early preview  P(blurry)={r['probs'][0]:.3f} -> {'CORRECT' if good else 'WRONG'}  {os.path.basename(p)}")
for p in final:
    r = ask("An image produced by a diffusion sampler.", "Is this image sharp and fully resolved, or blurry and unresolved?",
            ["It is blurry and unresolved", "It is sharp and fully resolved"], b64(p), 1, "sanity/blurry-final")
    good = r["argmax"] == 1; ok += good
    print(f"  finished image P(sharp) ={r['probs'][1]:.3f} -> {'CORRECT' if good else 'WRONG'}  {os.path.basename(p)}")
print(f"  sharpness accuracy: {ok}/{len(early)+len(final)}")
print("\n=== Q2: content — the answer is determined by the image ===")
import csv
PMAP = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open("/data/ygu/reward_orthogonality/mcts/prompts/prompt_map.csv"))}
WRONG = "A bowl of ramen noodles on a wooden table, steam rising."
ok2 = 0
for p in final:
    pid = os.path.basename(p).split("_")[0]
    if pid not in PMAP: continue
    r = ask("An image produced by a diffusion sampler.", "Which description matches this image?",
            [PMAP[pid], WRONG], b64(p), 2, "sanity/content")
    good = r["argmax"] == 0; ok2 += good
    print(f"  {pid}: P(correct description)={r['probs'][0]:.3f} -> {'CORRECT' if good else 'WRONG'}")
print(f"  content accuracy: {ok2}/{len(final)}")
