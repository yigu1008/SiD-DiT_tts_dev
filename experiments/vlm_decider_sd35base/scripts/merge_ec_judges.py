#!/usr/bin/env python3
"""Attach the ec_v2w fidelity judges (VQAScore, Qwen2-VL alignment) to per-image rows via the manifest (backend,prompt_id,sample_index)->image.
Usage: python merge_ec_judges.py <manifest.jsonl> <vqa_scores.csv> <vlm_scores.csv> --out judges_ec_v2w.json"""
import json, csv, argparse
ap = argparse.ArgumentParser(); ap.add_argument("manifest"); ap.add_argument("vqa"); ap.add_argument("vlm"); ap.add_argument("--out", required=True); a = ap.parse_args()
man = {(r["backend"], r["prompt_id"], str(r["sample_index"])): r for r in map(json.loads, open(a.manifest))}
rows = {}
for r in man.values(): rows[r["image"]] = {"image": r["image"], "prompt_id": r["prompt_id"], "stratum": r.get("stratum"), "mode": r.get("mode"), "noise_seed": r.get("seed")}
for f, col, key in ((a.vqa, "vqascore", "vqascore"), (a.vlm, "vlm_yes", "qwen_alignment")):
    for r in csv.DictReader(open(f)):
        m = man.get((r["backend"], r["prompt_id"], str(r["sample_index"])))
        if m: rows[m["image"]][key] = float(r[col])
out = list(rows.values()); json.dump(out, open(a.out, "w")); print(f"{len(out)} images; with vqascore {sum('vqascore' in r for r in out)}, with qwen {sum('qwen_alignment' in r for r in out)}")
