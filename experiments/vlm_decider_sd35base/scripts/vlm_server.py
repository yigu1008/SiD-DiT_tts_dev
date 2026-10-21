#!/usr/bin/env python3
"""VLM decider server (Qwen2.5-VL-7B-Instruct, bf16, sdpa).  Same JSON-over-HTTP style as reward_server.py.

GET  /health                 -> {"status":"ok","model":...,"requests_served":N}
POST /generate               -> generic multimodal chat: {"system": str?, "text": str, "images_b64": [..], "max_new_tokens": int?, "temperature": float?}
                                -> {"text": str, "seconds": float, "n_images": int}
POST /decide                 -> structured decision used by the VLM-MCTS arm:
   {"prompt": str, "step": int, "total_steps": int, "preview_b64": str|null, "candidates": [{"id": int, "variant": str, "cfg": float}], "k": int}
   -> {"comment": str, "ranking": [ids...], "keep": [ids (top-k)], "raw": str, "parsed": bool, "seconds": float}
The decider ALWAYS judges against the ORIGINAL prompt; candidate variant texts are shown only as the available actions.
Decoding is greedy (do_sample=False) so decisions are reproducible for a given input.
"""
from __future__ import annotations
import argparse, base64, io, json, os, re, time, traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
from PIL import Image

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=5151)
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--max_pixels", type=int, default=1024 * 28 * 28); p.add_argument("--min_pixels", type=int, default=128 * 28 * 28)
    return p.parse_args()

A = parse_args()
os.environ.setdefault("HF_HUB_CACHE", "/data/ygu/.cache/hub"); os.environ.setdefault("HF_HOME", "/data/ygu/.cache/hub"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
t0 = time.time()
MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(A.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}).eval()
PROC = AutoProcessor.from_pretrained(A.model, min_pixels=A.min_pixels, max_pixels=A.max_pixels)
print(f"[vlm_server] loaded {A.model} in {time.time()-t0:.0f}s, resident {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
_count = 0

def _img(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

@torch.no_grad()
def generate(text: str, images: list[Image.Image], system: str | None = None, max_new_tokens: int = 384, temperature: float = 0.0) -> str:
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": text}]
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    chat = PROC.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = PROC(text=[chat], images=image_inputs if image_inputs else None, videos=video_inputs if video_inputs else None, padding=True, return_tensors="pt").to(MODEL.device)
    gen_kwargs = dict(max_new_tokens=int(max_new_tokens), do_sample=temperature > 0, pad_token_id=PROC.tokenizer.eos_token_id)
    if temperature > 0: gen_kwargs["temperature"] = float(temperature)
    out = MODEL.generate(**inputs, **gen_kwargs)
    return PROC.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

DECIDE_SYSTEM = ("You are an expert art director guiding a text-to-image diffusion sampler. The ONLY target is the ORIGINAL prompt; "
                 "rewritten prompt variants are tools, not targets: content a variant adds beyond the original (e.g. a sunset, a mood word) is NOT required "
                 "and should only be chosen if it helps render what the original asks for. CFG (classifier-free guidance): higher = stronger prompt adherence "
                 "but more saturation, contrast and artifacts; lower = more natural and varied but weaker adherence. Variant letters are arbitrary labels. "
                 "Answer in the requested JSON only.")

def _letters(n): return [chr(65 + i) for i in range(n)]

def decide(body: dict) -> dict:
    """Factorized decision: rank prompt variants (shuffled letter labels) and CFG levels separately; keep = top-nv variants x top-nc CFGs."""
    import random, hashlib
    prompt = body["prompt"]; step = int(body.get("step", 0)); total = int(body.get("total_steps", 28)); k = int(body.get("k", 6))
    cands = body["candidates"]; prev = body.get("preview_b64")
    variants = sorted({c["variant"] for c in cands}, key=lambda v: min(int(c["id"]) for c in cands if c["variant"] == v))
    cfgs = sorted({float(c["cfg"]) for c in cands})
    nv = int(body.get("n_variants_keep", 2)); nc = int(body.get("n_cfg_keep", max(1, k // nv)))
    order = list(range(len(variants))); random.Random(int(hashlib.sha1(f"{prompt}|{step}|{body.get('salt','')}".encode()).hexdigest()[:8], 16)).shuffle(order)
    letters = _letters(len(variants)); label_of_variant = {variants[vi]: letters[pos] for pos, vi in enumerate(order)}; variant_of_label = {l: v for v, l in label_of_variant.items()}
    var_lines = "\n".join(f"  {letters[pos]}: {variants[vi]}" for pos, vi in enumerate(order))
    cfg_line = ", ".join(f"{c:g}" for c in cfgs)
    if prev:
        head = (f'ORIGINAL PROMPT: "{prompt}"\n\nThe attached image is the sampler\'s CURRENT x0 preview at denoising step {step}/{total} '
                f"(early steps are blurry; judge composition and content, not sharpness). In at most two sentences: what already matches the ORIGINAL prompt, "
                f"and what is missing, wrong, or at risk? ")
    else:
        head = (f'ORIGINAL PROMPT: "{prompt}"\n\nNo image yet (step 0/{total}). In at most two sentences: which parts of the ORIGINAL prompt are hardest '
                f"for a diffusion model and what should be prioritized? ")
    text = (head + f"Then decide the sampler settings for the next steps.\n\nPROMPT VARIANTS (labels are arbitrary):\n{var_lines}\n\nCFG LEVELS: {cfg_line}\n\n"
            f"Rank ALL variants best-first for rendering the ORIGINAL prompt, and rank ALL CFG levels best-first for this stage. "
            f'Respond with JSON exactly like: {{"comment": "...", "variants_best_first": ["{letters[0]}", ...], "cfg_best_first": [number, ...]}}')
    t = time.time(); raw = generate(text, [_img(prev)] if prev else [], system=DECIDE_SYSTEM, max_new_tokens=int(body.get("max_new_tokens", 320)))
    parsed = False; comment = ""; vr: list[str] = []; cr: list[float] = []
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            j = json.loads(m.group(0)); comment = str(j.get("comment", "")); parsed = True
            vr = [variant_of_label[str(x).strip().upper()[:1]] for x in j.get("variants_best_first", []) if str(x).strip().upper()[:1] in variant_of_label]
            cr = [float(x) for x in j.get("cfg_best_first", []) if isinstance(x, (int, float)) and float(x) in cfgs]
        except Exception:
            comment = raw[:400]
    def dedup(xs):
        seen = set(); return [x for x in xs if not (x in seen or seen.add(x))]
    vr = dedup(vr) + [v for v in variants if v not in set(vr)]; cr = dedup(cr) + [c for c in cfgs if c not in set(cr)]
    # action ranking = lexicographic over (variant rank, cfg rank); keep = top-nv variants x top-nc cfgs
    by_pair = {(c["variant"], float(c["cfg"])): int(c["id"]) for c in cands}
    ranking = [by_pair[(v, c)] for v in vr for c in cr if (v, c) in by_pair]
    keep = [by_pair[(v, c)] for v in vr[:nv] for c in cr[:nc] if (v, c) in by_pair]
    return {"comment": comment, "ranking": ranking, "keep": keep, "variants_best_first": [label_of_variant[v] for v in vr], "variant_texts_best_first": vr,
            "cfg_best_first": cr, "labels": {l: v for l, v in variant_of_label.items()}, "raw": raw, "parsed": parsed, "seconds": round(time.time() - t, 2)}

class H(BaseHTTPRequestHandler):
    def _json(self, data, status=200):
        b = json.dumps(data).encode(); self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self._json({"status": "ok", "model": A.model, "requests_served": _count} if self.path == "/health" else {"error": "not found"}, 200 if self.path == "/health" else 404)
    def do_POST(self):
        global _count
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path == "/generate":
                t = time.time(); imgs = [_img(b) for b in body.get("images_b64", [])]
                txt = generate(body["text"], imgs, system=body.get("system"), max_new_tokens=int(body.get("max_new_tokens", 384)), temperature=float(body.get("temperature", 0.0)))
                _count += 1; self._json({"text": txt, "seconds": round(time.time() - t, 2), "n_images": len(imgs)})
            elif self.path == "/decide":
                r = decide(body); _count += 1; self._json(r)
            else: self._json({"error": "not found"}, 404)
        except Exception as exc:
            tb = traceback.format_exc(); print(f"[vlm_server] ERROR {self.path}: {exc}\n{tb}", flush=True); self._json({"error": str(exc), "traceback": tb}, 500)
    def log_message(self, *a): pass

print(f"[vlm_server] serving on {A.host}:{A.port}", flush=True)
HTTPServer((A.host, A.port), H).serve_forever()
