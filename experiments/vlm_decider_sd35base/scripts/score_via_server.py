#!/usr/bin/env python3
"""Score records.jsonl images through an already-running reward_server (/score_batch): no GPU allocation needed,
so finished arms can be judged on HPSv3/PickScore/IR while the sampler is still using the same server.
Writes scores.csv-schema rows (prompt_id,seed,arm,metric,score); resumable (skips rows already in --out).
Exits non-zero if any record is dropped (missing image / empty prompt) so an unscored arm cannot pass silently."""
import argparse, base64, csv, json, os, sys, time, urllib.request, urllib.error
p = argparse.ArgumentParser(); p.add_argument("--records", nargs="+", required=True); p.add_argument("--out", required=True)
p.add_argument("--server", default="http://127.0.0.1:5142"); p.add_argument("--backends", nargs="+", default=["hpsv3", "pickscore", "imagereward"])
p.add_argument("--batch", type=int, default=4); p.add_argument("--tries", type=int, default=6); a = p.parse_args()
METRIC = {"imagereward": "ir"}
def post(path, body, timeout=900, tries=6, sleep=10):
    """Same retrying shape as jev_act.py/early_control.py: the server is single-threaded and shared with the sampler."""
    for t in range(tries):
        try:
            req = urllib.request.Request(a.server + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            res = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            if "error" in res: raise RuntimeError(res["error"])
            return res
        except Exception as e:
            if t == tries - 1: raise
            print(f"  [retry {t+1}/{tries}] {path}: {type(e).__name__}: {str(e)[:120]}", flush=True); time.sleep(sleep)
health = json.loads(urllib.request.urlopen(a.server + "/health", timeout=10).read()); print("server:", health, flush=True)
missing = [b for b in a.backends if b not in health["backends"]]
if missing: sys.exit(f"backends not loaded on server: {missing}")
have = set()
if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
    have = {(r["prompt_id"], r["seed"], r["arm"], r["metric"]) for r in csv.DictReader(open(a.out))}
new = not (os.path.exists(a.out) and os.path.getsize(a.out) > 0); fo = open(a.out, "a", newline=""); w = csv.writer(fo)
if new: w.writerow(["prompt_id", "seed", "arm", "metric", "score"])
dropped_total = 0
for rp in a.records:
    raw = [json.loads(l) for l in open(rp) if l.strip()]; recs, seen, dropped = [], set(), []
    for r in raw:
        img = next((r[k] for k in ("image", "image_path", "best_image") if isinstance(r.get(k), str)), None)
        if img and not os.path.isabs(img): img = os.path.join(os.path.dirname(rp), img)
        key = (r.get("prompt_id"), str(r.get("seed")), r.get("arm"))
        if not img or not os.path.exists(img): dropped.append((key, "missing image")); continue
        if not str(r.get("prompt", "")).strip(): dropped.append((key, "empty prompt")); continue
        if key in seen: dropped.append((key, "duplicate record")); continue
        seen.add(key); r["_b64"] = base64.b64encode(open(img, "rb").read()).decode(); recs.append(r)
    for key, why in dropped: print(f"  [DROPPED] {rp} {key}: {why}", flush=True)
    dropped_total += len(dropped)
    for be in a.backends:
        m = METRIC.get(be, be)
        todo = [r for r in recs if (r["prompt_id"], str(r["seed"]), r["arm"], m) not in have]
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            res = post("/score_batch", {"items": [{"prompt": r["prompt"], "image_b64": r["_b64"]} for r in chunk], "backend": be}, tries=a.tries)
            for r, s in zip(chunk, res["scores"]):
                w.writerow([r["prompt_id"], r["seed"], r["arm"], m, f"{s['score']:.6f}"]); have.add((r["prompt_id"], str(r["seed"]), r["arm"], m))
            fo.flush()
        print(f"{os.path.basename(os.path.dirname(rp))} {m}: scored {len(todo)} (had {len(recs)-len(todo)}, dropped {len(dropped)})", flush=True)
fo.close()
if dropped_total: sys.exit(f"FAILED: {dropped_total} record(s) dropped (see [DROPPED] lines); scores for the rest were written to {a.out}")
print("DONE", flush=True)
