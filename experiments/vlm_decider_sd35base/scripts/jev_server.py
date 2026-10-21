#!/usr/bin/env python3
"""Jev-Omni decision server (jev env).  Pinned local snapshot, order-averaged typed decisions, fully logged.

GET  /health -> {"status":"ok","model":"akhilaaa3/Jev-Omni","rev":...,"requests_served":N}
POST /ask    -> {"state": str, "question": str, "options": [str,...] (2..256, unique), "image_b64": str|null, "orders": int=3, "seed": int=0, "tag": str?}
             -> {"probs": [...canonical order...], "spread": [...], "argmax": i, "perms": [...], "ms": float}
Order-averaging follows jev-cub/src/omni.py: never the identity permutation alone; mean over shuffled option orders,
spread kept as the noise floor.  Every call is appended to <run_dir>/logs/jev_calls.jsonl with the full state text.
"""
from __future__ import annotations
import argparse, base64, io, itertools, json, math, os, random, sys, time, traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from PIL import Image

REV = "5addda86ddee081a68fb067477ea100c221b8917"
PINNED = Path(f"/data/ygu/hf_cache/hub/models--akhilaaa3--Jev-Omni/snapshots/{REV}")
p = argparse.ArgumentParser(); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=5161)
p.add_argument("--log", default="/data/ygu/runs/vlm_decider_sd35base_2026-10-16/logs/jev_calls.jsonl"); A = p.parse_args()
sys.path.insert(0, str(PINNED)); import jev_omni
from transformers import AutoConfig, AutoProcessor
import transformers
t0 = time.time(); cfg = AutoConfig.from_pretrained(PINNED)
model = getattr(transformers, cfg.architectures[0]).from_pretrained(PINNED, dtype=torch.bfloat16, device_map="cuda:0").eval()
decision = json.loads((PINNED / "decision_config.json").read_text())
head = jev_omni._Head256(decision["hidden_size"]).to("cuda:0").eval(); head.load_state_dict(torch.load(PINNED / "head.pt", map_location="cuda:0", weights_only=True))
_, decoder = jev_omni._find_backbone(model)
CLF = jev_omni.JevOmni(model, head, AutoProcessor.from_pretrained(PINNED), decoder, "cuda:0")
print(f"[jev_server] loaded Jev-Omni rev {REV[:8]} in {time.time()-t0:.0f}s, resident {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
_count = 0; TMP = Path(A.log).parent / "jev_tmp"; TMP.mkdir(parents=True, exist_ok=True)

def ask(state, question, options, image_b64=None, orders=3, seed=0, tag=None):
    n = len(options)
    if len(set(options)) != n: raise ValueError("options must be unique")
    if not 2 <= n <= 256: raise ValueError("need 2..256 options")
    rng = random.Random(seed); orders = max(1, int(orders))
    if math.factorial(n) <= orders: perms = [list(q) for q in itertools.permutations(range(n))]
    else:
        perms, seen = [], set()
        while len(perms) < orders:
            q = list(range(n)); rng.shuffle(q)
            if tuple(q) in seen: continue
            seen.add(tuple(q)); perms.append(q)
    media = None
    if image_b64:
        media = TMP / f"img_{os.getpid()}_{int(time.time()*1000)}.png"; Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB").save(media)
    t = time.time(); per_order = []
    try:
        for perm in perms:
            shown = [options[i] for i in perm]
            r = CLF.predict(state=state, question=question, options=shown, media=str(media) if media else None, modality="image" if media else "text")
            pr = r["probabilities"]; back = [0.0] * n
            for pos, orig in enumerate(perm): back[orig] = float(pr[shown[pos]])
            per_order.append(back)
    finally:
        if media: media.unlink(missing_ok=True)
    ms = (time.time() - t) * 1000
    mean = [sum(o[i] for o in per_order) / len(per_order) for i in range(n)]
    spread = [max(o[i] for o in per_order) - min(o[i] for o in per_order) for i in range(n)]
    out = {"probs": mean, "spread": spread, "argmax": max(range(n), key=lambda i: mean[i]), "perms": perms, "orders": per_order, "ms": round(ms, 1)}
    with open(A.log, "a") as f:
        f.write(json.dumps({"t": time.time(), "rev": REV, "tag": tag, "modality": "image" if image_b64 else "text", "question": question, "n_options": n,
                            "options": options, "state": state, "perms": perms, "probs": mean, "spread": spread, "orders": per_order, "ms": ms}) + "\n")
    return out

class H(BaseHTTPRequestHandler):
    def _json(self, data, status=200):
        b = json.dumps(data).encode(); self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health": self._json({"status": "ok", "model": "akhilaaa3/Jev-Omni", "rev": REV, "requests_served": _count})
        else: self._json({"error": "not found"}, 404)
    def do_POST(self):
        global _count
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path != "/ask": return self._json({"error": "not found"}, 404)
            r = ask(body["state"], body["question"], list(body["options"]), body.get("image_b64"), int(body.get("orders", 3)), int(body.get("seed", 0)), body.get("tag")); _count += 1; self._json(r)
        except Exception as exc:
            tb = traceback.format_exc(); print(f"[jev_server] ERROR: {exc}\n{tb}", flush=True); self._json({"error": str(exc), "traceback": tb}, 500)
    def log_message(self, *a): pass

print(f"[jev_server] serving on {A.host}:{A.port}", flush=True); HTTPServer((A.host, A.port), H).serve_forever()
