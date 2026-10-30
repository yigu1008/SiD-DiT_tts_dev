#!/usr/bin/env python3
"""Jev pairwise `choice` with TWO separate images (needs jev_server.py with images_b64): which image matches the prompt better?
Pairs: within prompt, best-noise vs worst-noise ec_v2w images (both orders of presentation averaged by the server's option
permutation; image order is also swapped across pairs). Reports accuracy vs the target reward and, if given, vs judge csvs.
Usage: python jev_pairwise_probe.py <ec records_all.jsonl> --prompt_map idx.json --out pairs.json [--pairs_per_prompt 40] [--target vec.hpsv3]"""
import argparse, base64, io, json, os, random, time, urllib.request, collections, statistics as st
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("records"); ap.add_argument("--prompt_map", required=True); ap.add_argument("--out", required=True); ap.add_argument("--jev", default="http://127.0.0.1:5161")
ap.add_argument("--pairs_per_prompt", type=int, default=40); ap.add_argument("--target", default="hpsv3"); ap.add_argument("--max_px", type=int, default=640); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
def post(body):
    for i in range(30):
        try:
            req = urllib.request.Request(a.jev + "/ask", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}); return json.loads(urllib.request.urlopen(req, timeout=600).read())
        except Exception as e: print("retry", i, type(e).__name__, flush=True); time.sleep(5)
def b64(path):
    im = Image.open(path).convert("RGB"); im.thumbnail((a.max_px, a.max_px)); b = io.BytesIO(); im.save(b, "PNG"); return base64.b64encode(b.getvalue()).decode()
m = json.load(open(a.prompt_map)); lines = [l.rstrip("\n") for l in open(os.path.join(os.path.dirname(a.prompt_map), "prompts_rsel9.txt"))]; pmap = {pid: lines[int(i)] for i, pid in m.items()}
recs = [json.loads(l) for l in open(a.records)]; byp = collections.defaultdict(list); [byp[r["prompt_id"]].append(r) for r in recs if os.path.exists(r["image"])]
rng = random.Random(a.seed); out = []; t0 = time.time()
for pid, rs in sorted(byp.items()):
    best = [r for r in rs if r["stratum"] == "best"]; worst = [r for r in rs if r["stratum"] == "worst"]
    for k in range(a.pairs_per_prompt):
        x, y = rng.choice(best), rng.choice(worst); first_is_best = rng.random() < 0.5; A, B = (x, y) if first_is_best else (y, x)
        state = f'PROMPT: "{pmap[pid]}"\nTwo images are attached, in order: Image 1 then Image 2. Both are text-to-image generations for this prompt.'
        r = post({"state": state, "question": "Which image matches the prompt better?", "options": ["Image 1", "Image 2"], "images_b64": [b64(A["image"]), b64(B["image"])], "orders": 2, "seed": 1000 * k + hash(pid) % 1000, "tag": f"pair/{pid}/{k}"})
        p_first = r["probs"][0]; p_best = p_first if first_is_best else 1 - p_first
        out.append({"prompt_id": pid, "best": x["image"], "worst": y["image"], "first_is_best": first_is_best, "p_best": p_best, "spread": max(r["spread"]),
                    "best_" + a.target: x["vec"][a.target], "worst_" + a.target: y["vec"][a.target], "ms": r["ms"]})
    acc = st.mean(o["p_best"] > 0.5 for o in out if o["prompt_id"] == pid); print(f"  {pid}: P(pick best-noise image) {acc:.2f}  mean p_best {st.mean(o['p_best'] for o in out if o['prompt_id']==pid):.2f}  {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(a.out, "w"))
print(f"\noverall: picks the {a.target}-best image in {100*st.mean(o['p_best'] > 0.5 for o in out):.0f}% of {len(out)} pairs; first-position pick rate {100*st.mean((o['p_best']>0.5)==o['first_is_best'] for o in out):.0f}% (position bias check, 50 = none); mean spread {st.mean(o['spread'] for o in out):.2f}")
