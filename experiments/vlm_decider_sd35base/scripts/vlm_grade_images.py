#!/usr/bin/env python3
"""Grade generated images with the live Qwen2.5-VL server (/generate) as a candidate SELECTION metric, parallel to jev_grade_images.py:
  vlm_rating  : "rate 0-5" -> integer            vlm_yes : yes/no -> 1/0
  vlm_pass    : checklist of the prompt's requirements, each PASS/FAIL -> fraction passed (+ the failed items as reasons)
Usage: python vlm_grade_images.py <records.jsonl> --out <json> [--prompt_map prompts_rsel9_index.json] [--vlm URL] [--limit N]"""
import argparse, base64, io, json, os, re, time, urllib.request
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("records"); ap.add_argument("--out", required=True); ap.add_argument("--prompt_map"); ap.add_argument("--vlm", default="http://127.0.0.1:5151")
ap.add_argument("--max_px", type=int, default=768); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
def post(body):
    for i in range(30):
        try:
            req = urllib.request.Request(a.vlm + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=600).read())["text"]
        except Exception as e: print("retry", i, type(e).__name__, flush=True); time.sleep(5)
def b64(path):
    im = Image.open(path).convert("RGB"); im.thumbnail((a.max_px, a.max_px)); b = io.BytesIO(); im.save(b, "PNG"); return base64.b64encode(b.getvalue()).decode()
pmap = {}
if a.prompt_map:
    m = json.load(open(a.prompt_map))
    if all(isinstance(v, str) and v.startswith("q") for v in m.values()):
        lines = [l.rstrip("\n") for l in open(os.path.join(os.path.dirname(a.prompt_map), "prompts_rsel9.txt"))]; pmap = {pid: lines[int(i)] for i, pid in m.items()}
    else: pmap = dict(m)
def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flat(v, pre + k + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool): out[pre + k] = v
        elif isinstance(v, str) and k in ("prompt_id", "arm", "mode", "stratum", "image"): out[pre + k] = v
    return out
SYS = "You are a strict judge of text-to-image fidelity. Judge only whether the image matches the prompt's content: objects, counts, attributes, spatial relations. Ignore aesthetics."
recs = [json.loads(l) for l in open(a.records)]; recs = recs[: a.limit] if a.limit else recs
done = {r["image"]: r for r in json.load(open(a.out))} if os.path.exists(a.out) else {}; out = list(done.values()); t0 = time.time()
for n, rec in enumerate(recs):
    p = rec["image"]
    if p in done or not os.path.exists(p): continue
    prompt = rec.get("prompt") or pmap[rec["prompt_id"]]; img = [b64(p)]
    r1 = post({"system": SYS, "text": f'Prompt: "{prompt}"\nRate from 0 to 5 how well this image satisfies the prompt (5 = every element present and correct, 0 = nothing from the prompt). Reply with the single digit only.', "images_b64": img, "max_new_tokens": 4})
    r2 = post({"system": SYS, "text": f'Prompt: "{prompt}"\nDoes the image show exactly what the prompt describes? Reply Yes or No only.', "images_b64": img, "max_new_tokens": 3})
    r3 = post({"system": SYS, "text": f'Prompt: "{prompt}"\nList every checkable requirement in the prompt (objects, counts, attributes, positions, relations) as short items, one per line, in the form "<requirement> -- PASS" or "<requirement> -- FAIL" according to the image. No other text.', "images_b64": img, "max_new_tokens": 220})
    m = re.search(r"\d", r1); rating = int(m.group()) if m else None; yes = 1.0 if re.match(r"\s*yes", r2, re.I) else (0.0 if re.match(r"\s*no", r2, re.I) else None)
    items = [l.strip() for l in r3.splitlines() if re.search(r"--\s*(PASS|FAIL)", l, re.I)]; npass = sum(1 for l in items if re.search(r"--\s*PASS", l, re.I))
    row = flat(rec); row.update({"image": p, "prompt": prompt, "vlm_rating": rating, "vlm_yes": yes, "vlm_pass": (npass / len(items)) if items else None, "vlm_n_items": len(items),
                                 "vlm_failed": [l for l in items if re.search(r"--\s*FAIL", l, re.I)], "vlm_raw": [r1.strip(), r2.strip()]}); out.append(row)
    if n % 25 == 0: print(f"  {n}/{len(recs)} {rec['prompt_id']} rating={rating} yes={yes} pass={row['vlm_pass']} items={len(items)} {time.time()-t0:.0f}s", flush=True); json.dump(out, open(a.out, "w"))
json.dump(out, open(a.out, "w")); print(f"done {len(out)} rows -> {a.out} in {time.time()-t0:.0f}s")
