#!/usr/bin/env python3
"""Score records.jsonl images through an already-running reward_server (/score_batch): no GPU allocation needed,
so finished arms can be judged on HPSv3/PickScore/IR while the sampler is still using the same server.
Writes scores.csv-schema rows (prompt_id,seed,arm,metric,score); resumable (skips rows already in --out)."""
import argparse, base64, csv, json, os, sys, urllib.request
p = argparse.ArgumentParser(); p.add_argument("--records", nargs="+", required=True); p.add_argument("--out", required=True)
p.add_argument("--server", default="http://127.0.0.1:5142"); p.add_argument("--backends", nargs="+", default=["hpsv3", "pickscore", "imagereward"])
p.add_argument("--batch", type=int, default=4); a = p.parse_args()
METRIC = {"imagereward": "ir"}
def post(path, body):
    req = urllib.request.Request(a.server + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())
health = json.loads(urllib.request.urlopen(a.server + "/health", timeout=10).read()); print("server:", health, flush=True)
missing = [b for b in a.backends if b not in health["backends"]]
if missing: sys.exit(f"backends not loaded on server: {missing}")
have = set()
if os.path.exists(a.out):
    have = {(r["prompt_id"], r["seed"], r["arm"], r["metric"]) for r in csv.DictReader(open(a.out))}
new = not os.path.exists(a.out); fo = open(a.out, "a", newline=""); w = csv.writer(fo)
if new: w.writerow(["prompt_id", "seed", "arm", "metric", "score"])
for rp in a.records:
    recs = [json.loads(l) for l in open(rp) if l.strip()]
    for r in recs:
        r["_img"] = next((r[k] for k in ("image", "image_path", "best_image") if isinstance(r.get(k), str)), None)
        if r["_img"] and not os.path.isabs(r["_img"]): r["_img"] = os.path.join(os.path.dirname(rp), r["_img"])
    recs = [r for r in recs if r["_img"] and os.path.exists(r["_img"])]
    for be in a.backends:
        m = METRIC.get(be, be)
        todo = [r for r in recs if (r["prompt_id"], str(r["seed"]), r["arm"], m) not in have]
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            items = [{"prompt": r["prompt"], "image_b64": base64.b64encode(open(r["_img"], "rb").read()).decode()} for r in chunk]
            res = post("/score_batch", {"items": items, "backend": be})
            for r, s in zip(chunk, res["scores"]): w.writerow([r["prompt_id"], r["seed"], r["arm"], m, f"{s['score']:.6f}"])
            fo.flush()
        print(f"{os.path.basename(os.path.dirname(rp))} {m}: scored {len(todo)} (had {len(recs)-len(todo)})", flush=True)
print("DONE", flush=True)
