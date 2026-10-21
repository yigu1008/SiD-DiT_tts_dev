#!/usr/bin/env python3
"""Convert an actdiff run (rank_*.jsonl + saved best images) into the records.jsonl schema judge_base.sh expects:
prompt_id, prompt, arm, seed, image, image_sha256, search_score, nfe_transformer (+ diagnostics kept)."""
import argparse, glob, hashlib, json, os
IMG_KEYS = ("image", "image_path", "best_image", "best_image_path", "out_path", "path", "png")

def main():
    p = argparse.ArgumentParser(); p.add_argument("--run_dir", required=True); p.add_argument("--index_map", required=True)
    p.add_argument("--arm", required=True); p.add_argument("--mode", default="mcts"); p.add_argument("--out", default=None)
    a = p.parse_args(); idx = json.load(open(a.index_map)); out = a.out or os.path.join(a.run_dir, "records.jsonl")
    rows = []
    for f in sorted(glob.glob(os.path.join(a.run_dir, "**", "rank_*.jsonl"), recursive=True)):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            if r.get("mode") != a.mode or "prompt_index" not in r: continue
            rows.append(r)
    seen = set(); n = 0
    with open(out, "w") as fo:
        for r in rows:
            pi = str(int(r["prompt_index"])); pid = idx.get(pi)
            if pid is None or pid in seen: continue
            # sd35_ddp_experiment saves the image as images/p{prompt_index:05d}_{mode}.png and stores NO path key
            img = os.path.join(a.run_dir, "images", f"p{int(pi):05d}_{a.mode}.png")
            if not os.path.exists(img):
                img = next((r[k] for k in IMG_KEYS if isinstance(r.get(k), str) and r[k].endswith(".png")), None)
                if img and not os.path.isabs(img): img = os.path.join(a.run_dir, img)
            if not img or not os.path.exists(img):
                # fall back: look for a saved best image for this prompt index
                cands = glob.glob(os.path.join(a.run_dir, "**", f"*{int(pi):04d}*{a.mode}*.png"), recursive=True) + \
                        glob.glob(os.path.join(a.run_dir, "**", f"*prompt_{int(pi):04d}*.png"), recursive=True)
                img = sorted(cands)[-1] if cands else None
            if not img: print(f"  [warn] no image for prompt_index {pi} ({pid})"); continue
            seen.add(pid)
            diag = r.get("search_diagnostics") or {}
            fo.write(json.dumps({"prompt_id": pid, "prompt": r.get("prompt", ""), "arm": a.arm, "seed": int(r.get("seed", 42)),
                                 "image": img, "image_sha256": hashlib.sha256(open(img, "rb").read()).hexdigest(),
                                 "search_score": float(r.get("score", float("nan"))), "nfe_transformer": int(r.get("nfe", -1)),
                                 "mode": a.mode, "prescreen": (diag.get("bon_mcts") or {}).get("prescreen"),
                                 "n_node_logs": len(((diag.get("bon_mcts") or {}).get("lookahead_node_logs") or []))}) + "\n"); n += 1
    print(f"wrote {out}: {n} records from {len(rows)} {a.mode} rows")

if __name__ == "__main__":
    main()
