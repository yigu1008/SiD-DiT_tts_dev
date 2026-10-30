#!/usr/bin/env python3
"""Jev as an EARLY-STOP noise screener: grade the step-t preview PNGs (ns_ladder_ls6/previews/{pid}_s{seed}_t{t}.png) with the typed
`score` question and compare, across noises within prompt, with the final HPSv3 / VQAScore and with the preview-HPSv3 screen.
Writes rows via jev_grade_images-style calls. Usage: python preview_screen_probe.py <records_all.jsonl> --t 12 --out json"""
import argparse, json, os, re, subprocess, sys
ap = argparse.ArgumentParser(); ap.add_argument("records"); ap.add_argument("--t", type=int, default=12); ap.add_argument("--out", required=True); a = ap.parse_args()
recs = [json.loads(l) for l in open(a.records)]; tmp = a.out + ".previews.jsonl"
with open(tmp, "w") as f:
    for r in recs:
        p = os.path.join(os.path.dirname(os.path.dirname(r["image"])), "previews", f"{r['prompt_id']}_s{r['noise_seed']}_t{a.t}.png")
        if os.path.exists(p): f.write(json.dumps({"prompt_id": r["prompt_id"], "prompt": r["prompt"], "image": p, "final_image": r["image"], "noise_seed": r["noise_seed"], "vec": r["final"], "partial": r["partial"].get(str(a.t), {})}) + "\n")
print("preview rows:", sum(1 for _ in open(tmp)))
subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "jev_grade_images.py"), tmp, "--out", a.out], check=False)
