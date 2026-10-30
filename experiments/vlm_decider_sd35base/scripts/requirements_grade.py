#!/usr/bin/env python3
"""Requirement-level grading (Jev's native `noul` use): one canonical list of atomic requirements per prompt (VLM, text only),
then for every image x requirement a yes/no from Jev-Omni (typed noul, image as the only media) and from the VLM (/generate).
  --stage reqs : build <reqs_json> {prompt_id: [requirement, ...]}      (9 text-only VLM calls)
  --stage jev  : per image x requirement Jev P(yes)   -> rows with jev_req_pass (mean P(yes)), jev_req_min, jev_req_all (all P>0.5)
  --stage vlm  : per image x requirement VLM yes/no    -> rows with vlm_req_pass (fraction yes), vlm_req_all
Usage: python requirements_grade.py <records.jsonl> --reqs reqs.json --out out.json --stage {reqs,jev,vlm} [--prompt_map idx.json]"""
import argparse, base64, io, json, os, re, time, urllib.request
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("records"); ap.add_argument("--reqs", required=True); ap.add_argument("--out"); ap.add_argument("--stage", required=True, choices=["reqs", "jev", "vlm"])
ap.add_argument("--prompt_map"); ap.add_argument("--jev", default="http://127.0.0.1:5161"); ap.add_argument("--vlm", default="http://127.0.0.1:5151"); ap.add_argument("--max_px", type=int, default=768); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
def post(url, body, key):
    for i in range(30):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=600).read())[key]
        except Exception as e: print("retry", i, type(e).__name__, flush=True); time.sleep(5)
def b64(path):
    im = Image.open(path).convert("RGB"); im.thumbnail((a.max_px, a.max_px)); b = io.BytesIO(); im.save(b, "PNG"); return base64.b64encode(b.getvalue()).decode()
pmap = {}
if a.prompt_map:
    m = json.load(open(a.prompt_map)); lines = [l.rstrip("\n") for l in open(os.path.join(os.path.dirname(a.prompt_map), "prompts_rsel9.txt"))]; pmap = {pid: lines[int(i)] for i, pid in m.items()}
recs = [json.loads(l) for l in open(a.records)]; recs = recs[: a.limit] if a.limit else recs
prompts = {r["prompt_id"]: (r.get("prompt") or pmap[r["prompt_id"]]) for r in recs}
if a.stage == "reqs":
    reqs = {}
    for pid, prompt in sorted(prompts.items()):
        txt = post(a.vlm + "/generate", {"text": f'Text-to-image prompt: "{prompt}"\nDecompose it into its atomic, visually checkable requirements (objects and their counts, attributes, positions, relations). 3 to 6 items, each a short declarative sentence that is true of a correct image, one per line, no numbering, no other text.', "max_new_tokens": 160}, "text")
        items = [re.sub(r"^[\s\-\*\d\.\)]+", "", l).strip() for l in txt.splitlines()]; items = [x for x in items if 3 < len(x) < 140][:6]
        reqs[pid] = items; print(pid, "->", items, flush=True)
    json.dump(reqs, open(a.reqs, "w"), indent=1, ensure_ascii=False); raise SystemExit
reqs = json.load(open(a.reqs))
def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flat(v, pre + k + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool): out[pre + k] = v
        elif isinstance(v, str) and k in ("prompt_id", "arm", "mode", "stratum", "image"): out[pre + k] = v
    return out
done = {r["image"]: r for r in json.load(open(a.out))} if os.path.exists(a.out) else {}; out = list(done.values()); t0 = time.time()
SYS = "You are a strict judge of text-to-image fidelity. Answer only about what is visible in the image."
for n, rec in enumerate(recs):
    p = rec["image"]
    if p in done or not os.path.exists(p): continue
    prompt = prompts[rec["prompt_id"]]; img = b64(p); items = reqs[rec["prompt_id"]]; ps = []
    for j, q in enumerate(items):
        if a.stage == "jev":
            r = post(a.jev + "/ask", {"state": f'PROMPT: "{prompt}"\nOne image is attached: a text-to-image generation for this prompt.', "question": f"Is this true of the image: {q}", "options": ["Yes", "No"], "image_b64": img, "orders": 2, "seed": 100 * n + j, "tag": f"req/{rec['prompt_id']}/{n}/{j}"}, "probs"); ps.append(r[0])
        else:
            t = post(a.vlm + "/generate", {"system": SYS, "text": f'Is this true of the image: "{q}"? Reply Yes or No only.', "images_b64": [img], "max_new_tokens": 3}, "text"); ps.append(1.0 if re.match(r"\s*yes", t, re.I) else 0.0)
    row = flat(rec); row.update({"image": p, "prompt": prompt, "n_req": len(items)})
    k = a.stage; row.update({f"{k}_req_pass": sum(ps) / len(ps), f"{k}_req_min": min(ps), f"{k}_req_all": float(all(x > 0.5 for x in ps)), f"{k}_req_probs": ps}); out.append(row)
    if n % 25 == 0: print(f"  {n}/{len(recs)} {rec['prompt_id']} {k}_req_pass={row[f'{k}_req_pass']:.2f} {time.time()-t0:.0f}s", flush=True); json.dump(out, open(a.out, "w"))
json.dump(out, open(a.out, "w")); print(f"done {len(out)} rows -> {a.out} in {time.time()-t0:.0f}s")
