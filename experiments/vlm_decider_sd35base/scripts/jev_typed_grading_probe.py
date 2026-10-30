#!/usr/bin/env python3
"""Step 1+2 of the Jev iteration ladder, offline: typed grading of SINGLE images (no contact sheet).
For every judged image (arms with VQAScore rows in scores.csv) ask Jev-Omni two typed questions with the image as the only media:
  score : "How well does this image satisfy the prompt?"  6 ordered levels -> expected level E[s] and P(level>=4)
  noul  : "The image shows exactly what the prompt describes."  Yes/No -> P(yes)
Then Spearman of each Jev readout with the held-out judges (vqascore, qwen_alignment, hpsv2) and the search rewards
(hpsv3, ir, pickscore), pooled over images and within prompt (mean over prompts). Baseline rows: hpsv3 vs the same judges.
Usage: python jev_typed_grading_probe.py <RUN_DIR> [--jev http://127.0.0.1:5161] [--out results/jev_typed_grading.json]"""
import argparse, base64, csv, io, json, os, statistics as st, collections, time, urllib.request
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("--jev", default="http://127.0.0.1:5161"); ap.add_argument("--out", default="jev_typed_grading.json"); ap.add_argument("--max_px", type=int, default=768)
a = ap.parse_args(); R = a.run_dir
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
rows = list(csv.DictReader(open(os.path.join(R, "scores.csv"))))
J = collections.defaultdict(dict)
for r in rows: J[(r["arm"], r["prompt_id"])][r["metric"]] = float(r["score"])
arms = sorted({k[0] for k, v in J.items() if "vqascore" in v and os.path.isdir(os.path.join(R, "main", k[0]))})
items = []
for arm in arms:
    for l in open(os.path.join(R, "main", arm, "records.jsonl")):
        rec = json.loads(l); p = rec["image"]
        if not os.path.exists(p): p = p.replace(f"/main/{arm.replace('_dupbug','')}/", f"/main/{arm}/")
        if os.path.exists(p) and (arm, rec["prompt_id"]) in J: items.append((arm, rec["prompt_id"], rec["prompt"], p))
print(f"{len(items)} judged images from {len(arms)} arms", flush=True)
out = []
for n, (arm, pid, prompt, path) in enumerate(items):
    img = b64(path); state = f'PROMPT: "{prompt}"\nOne image is attached: a text-to-image generation for this prompt.'
    s = post({"state": state, "question": "How well does this image satisfy the prompt?", "options": LEVELS, "image_b64": img, "orders": 3, "seed": n, "tag": f"grade/{arm}/{pid}/score"})
    y = post({"state": state, "question": "Does the image show exactly what the prompt describes?", "options": ["Yes", "No"], "image_b64": img, "orders": 2, "seed": n, "tag": f"grade/{arm}/{pid}/noul"})
    es = sum(i * p for i, p in enumerate(s["probs"])); p4 = sum(s["probs"][4:])
    out.append({"arm": arm, "prompt_id": pid, "image": path, "jev_score_E": es, "jev_score_P4": p4, "jev_score_probs": s["probs"], "jev_score_spread": max(s["spread"]),
                "jev_yes": y["probs"][0], "jev_yes_spread": max(y["spread"]), **J[(arm, pid)]})
    if n % 12 == 0: print(f"  {n}/{len(items)} {arm} {pid} E[s]={es:.2f} P(yes)={y['probs'][0]:.2f} vqa={J[(arm,pid)]['vqascore']:.3f}", flush=True)
json.dump(out, open(a.out, "w"), indent=1)
def spearman(x, y):
    rk = lambda v: {i: s for s, i in enumerate(sorted(range(len(v)), key=lambda i: v[i]))}
    rx, ry = rk(x), rk(y); n = len(x); return 1 - 6 * sum((rx[i] - ry[i]) ** 2 for i in range(n)) / (n * (n * n - 1)) if n > 2 else float("nan")
judges = ["vqascore", "qwen_alignment", "hpsv2", "hpsv3", "ir", "pickscore"]
print(f"\n=== Spearman with judges, pooled over {len(out)} images (within-prompt mean over {len({o['prompt_id'] for o in out})} prompts in brackets)")
print(f"{'readout':>14} " + " ".join(f"{j:>16}" for j in judges))
for key in ["jev_score_E", "jev_score_P4", "jev_yes", "hpsv3", "ir", "pickscore"]:
    cells = []
    for j in judges:
        pooled = spearman([o[key] for o in out], [o[j] for o in out])
        byp = collections.defaultdict(list); [byp[o["prompt_id"]].append((o[key], o[j])) for o in out]
        within = st.mean(spearman([x for x, _ in v], [y for _, y in v]) for v in byp.values() if len(v) > 2)
        cells.append(f"{pooled:+.2f} [{within:+.2f}]")
    print(f"{key:>14} " + " ".join(f"{c:>16}" for c in cells))
print(f"\nJev readout stats: E[s] mean {st.mean(o['jev_score_E'] for o in out):.2f} sd {st.pstdev(o['jev_score_E'] for o in out):.2f}; P(yes) mean {st.mean(o['jev_yes'] for o in out):.2f} sd {st.pstdev(o['jev_yes'] for o in out):.2f}; "
      f"order-spread score {st.mean(o['jev_score_spread'] for o in out):.2f} yes {st.mean(o['jev_yes_spread'] for o in out):.2f}")
