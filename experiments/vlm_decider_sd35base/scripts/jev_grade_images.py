#!/usr/bin/env python3
"""Grade any set of generated images with Jev-Omni typed questions (single image per call): `score` 0-5 and `noul` yes/no.
Input: a records jsonl (needs prompt_id + image; prompt text taken from the record or from a prompt map), output: json rows with
jev_score_E, jev_score_P4, jev_yes (+spreads) and every numeric field of the record flattened (vec.* etc.) for later correlation.
Usage: python jev_grade_images.py <records.jsonl> --out <json> [--prompt_map prompts_rsel9_index.json] [--jev URL] [--limit N]"""
import argparse, base64, io, json, os, time, urllib.request
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("records"); ap.add_argument("--out", required=True); ap.add_argument("--prompt_map"); ap.add_argument("--jev", default="http://127.0.0.1:5161")
ap.add_argument("--max_px", type=int, default=768); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--image_root_fix", default="")
a = ap.parse_args()
LEVELS = ["0 - nothing from the prompt is present", "1 - only the general subject is present; key elements missing or wrong",
          "2 - several elements present but counts, attributes or layout are wrong", "3 - most elements present; one clear error",
          "4 - all elements present with at most a minor attribute slip", "5 - every element present and correct in count, attributes and layout"]
def post(body):
    for i in range(30):
        try:
            req = urllib.request.Request(a.jev + "/ask", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=600).read())
        except Exception as e: print("retry", i, type(e).__name__, flush=True); time.sleep(5)
def b64(path):
    im = Image.open(path).convert("RGB"); im.thumbnail((a.max_px, a.max_px)); b = io.BytesIO(); im.save(b, "PNG"); return base64.b64encode(b.getvalue()).decode()
pmap = {}
if a.prompt_map:   # either {prompt_id: text} or the rsel9 index {i: prompt_id} next to prompts_rsel9.txt (line i = text)
    m = json.load(open(a.prompt_map))
    if all(isinstance(v, str) and v.startswith("q") for v in m.values()):
        lines = [l.rstrip("\n") for l in open(os.path.join(os.path.dirname(a.prompt_map), "prompts_rsel9.txt"))]
        pmap = {pid: lines[int(i)] for i, pid in m.items()}
    else: pmap = dict(m)
def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flat(v, pre + k + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool): out[pre + k] = v
        elif isinstance(v, str) and k in ("prompt_id", "arm", "mode", "stratum", "image"): out[pre + k] = v
    return out
recs = [json.loads(l) for l in open(a.records)]; recs = recs[: a.limit] if a.limit else recs
done = {}
if os.path.exists(a.out): done = {r["image"]: r for r in json.load(open(a.out))}
out = list(done.values()); t0 = time.time()
for n, rec in enumerate(recs):
    p = rec["image"]; p = p if os.path.exists(p) or not a.image_root_fix else p.replace(*a.image_root_fix.split("=>"))
    if p in done or not os.path.exists(p): continue
    prompt = rec.get("prompt") or pmap.get(rec["prompt_id"]); assert prompt, rec["prompt_id"]
    img = b64(p); state = f'PROMPT: "{prompt}"\nOne image is attached: a text-to-image generation for this prompt.'
    s = post({"state": state, "question": "How well does this image satisfy the prompt?", "options": LEVELS, "image_b64": img, "orders": 3, "seed": n, "tag": f"grade2/{rec['prompt_id']}/{n}/score"})
    y = post({"state": state, "question": "Does the image show exactly what the prompt describes?", "options": ["Yes", "No"], "image_b64": img, "orders": 2, "seed": n, "tag": f"grade2/{rec['prompt_id']}/{n}/noul"})
    row = flat(rec); row.update({"image": p, "prompt": prompt, "jev_score_E": sum(i * q for i, q in enumerate(s["probs"])), "jev_score_P4": sum(s["probs"][4:]), "jev_score_probs": s["probs"],
                                 "jev_score_spread": max(s["spread"]), "jev_yes": y["probs"][0], "jev_yes_spread": max(y["spread"])}); out.append(row)
    if n % 25 == 0: print(f"  {n}/{len(recs)} {rec['prompt_id']} E[s]={row['jev_score_E']:.2f} P(yes)={row['jev_yes']:.2f} {time.time()-t0:.0f}s", flush=True); json.dump(out, open(a.out, "w"))
json.dump(out, open(a.out, "w")); print(f"done {len(out)} rows -> {a.out} in {time.time()-t0:.0f}s")
