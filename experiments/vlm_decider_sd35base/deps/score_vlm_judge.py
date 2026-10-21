#!/usr/bin/env python3
"""Held-out ALIGNMENT judge: zero-shot Qwen2-VL-7B-Instruct, P(yes) to
"does this image depict the description?".

Why: PickScore and HPSv2 are preference models, so the study's held-out set
could not fairly judge alignment. This judge is a different model family from
CLIP-FlanT5 VQAScore and is not preference-trained. (HPSv3 is fine-tuned from
Qwen2-VL-7B; this is the base instruct model used zero-shot -- related lineage,
different objective; stated as a caveat.)
"""
from __future__ import annotations
import argparse, csv, json, os, time
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_dirs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    a = p.parse_args()
    os.environ.setdefault("HF_HOME", "/data/ygu/.cache/hub"); os.environ.setdefault("HF_HUB_CACHE", "/data/ygu/.cache/hub"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from PIL import Image
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    rows = []
    for d in a.manifest_dirs:
        for mf in sorted(Path(d).glob("*.jsonl")):
            rows += [json.loads(l) for l in open(mf) if l.strip()]
    seen, uniq = set(), []
    for r in rows:
        k = (r["backend"], r["prompt_id"], r["sample_index"])
        if k not in seen: seen.add(k); uniq.append(r)
    uniq.sort(key=lambda r: (r["backend"], r["prompt_id"], r["sample_index"]))
    shard = [r for i, r in enumerate(uniq) if i % a.num_shards == a.shard_index]
    if a.limit: shard = shard[: a.limit]
    print(f"[vlm] shard {a.shard_index}/{a.num_shards}: {len(shard)} of {len(uniq)} images", flush=True)

    model = Qwen2VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    proc = AutoProcessor.from_pretrained(a.model, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
    tok = proc.tokenizer
    def first_ids(words): return sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})
    yes_ids, no_ids = first_ids(["Yes", " Yes", "yes", " yes"]), first_ids(["No", " No", "no", " no"])
    print(f"[vlm] yes ids {yes_ids} no ids {no_ids}", flush=True)

    out = Path(a.out).with_suffix(f".shard{a.shard_index}.csv"); out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out.open("w", newline="") as f, torch.inference_mode():
        w = csv.writer(f); w.writerow(["backend", "prompt_id", "sample_index", "vlm_yes"])
        for i, r in enumerate(shard):
            q = (f'Does this image accurately depict the following description?\nDescription: "{r["prompt"]}"\n'
                 "Consider every object, count, relation and negation in the description. Answer with Yes or No.")
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            img = Image.open(r["image"]).convert("RGB")
            inputs = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
            logits = model(**inputs).logits[0, -1].float()
            ly, ln = torch.logsumexp(logits[yes_ids], 0), torch.logsumexp(logits[no_ids], 0)
            p_yes = torch.softmax(torch.stack([ly, ln]), 0)[0].item()
            w.writerow([r["backend"], r["prompt_id"], r["sample_index"], f"{p_yes:.6f}"])
            if i % 100 == 0: f.flush(); print(f"  {i}/{len(shard)} {time.time()-t0:.0f}s", flush=True)
    print(f"[vlm] wrote {out} in {time.time()-t0:.0f}s ({len(shard)/max(time.time()-t0,1):.2f} img/s)", flush=True)

if __name__ == "__main__":
    main()
