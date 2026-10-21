#!/usr/bin/env python3
"""records.jsonl of each new arm -> one manifest JSONL per arm in the schema the
existing judge scorers (score_local.py, score_vqa.py, score_vlm_judge.py) read:
backend='followup:<arm>', prompt_id, sample_index=1, prompt, image (+ seed, arm)."""
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument("--arm_dirs", nargs="+", required=True); p.add_argument("--out_dir", required=True); p.add_argument("--ids", required=True)
a = p.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); ids = {l.strip() for l in open(a.ids) if l.strip()}
for d in a.arm_dirs:
    recs = [json.loads(l) for l in open(Path(d)/"records.jsonl")]; recs = [r for r in recs if r["prompt_id"] in ids]
    arm = recs[0]["arm"]
    with (out/f"followup_{arm}.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps({"backend": f"followup:{arm}", "arm": arm, "prompt_id": r["prompt_id"], "sample_index": 1, "prompt": r["prompt"],
                                "source": "genai_bench", "category": "", "skills": "", "seed": r["seed"], "image": r["image"], "image_sha256": r["image_sha256"]}) + "\n")
    print(f"{arm}: {len(recs)} rows -> {out/f'followup_{arm}.jsonl'}")
