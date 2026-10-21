#!/usr/bin/env python3
"""Score generated images with HPSv3, ImageReward, PickScore and HPSv2.

PickScore and HPSv2 are never optimized against in this study; they are the
held-out judges for the reward-hacking analysis.

Runs in the `reward` conda env, not `sid_dit`. hpsv3 pins transformers 4.45.2
and its modeling code reaches into Qwen2-VL internals (`model.embed_tokens`,
`visual.*` checkpoint keys) that transformers renamed in 4.52 -- so under
sid_dit's 4.52 the checkpoint fails to load and, once force-loaded, the forward
pass still breaks. The `reward` env has the versions hpsv3 was built against,
which is what reward_server.py was written to use. No compatibility shims here
by design: if this file starts needing them, it is running in the wrong env.

Both models load in one process via UnifiedRewardScorer's composite_hpsv3_ir
backend, then are queried individually so we keep the two raw scores rather
than their blend.
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time
from pathlib import Path

SID_REPO = "/home/ygu/project/SiD-DiT_tts_dev"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_dir", default="/data/ygu/reward_orthogonality/manifests")
    p.add_argument("--out", default="/data/ygu/reward_orthogonality/scores/local_scores.csv")
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    args = p.parse_args()

    os.environ.setdefault("HF_HUB_CACHE", "/data/ygu/.cache/hub")
    os.environ.setdefault("HF_HOME", "/data/ygu/.cache/hub")
    os.environ.setdefault("SID_FORCE_WANDB_STUB", "1")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("HPSV3_SCORE_STYLE", "raw")   # keep native scale
    os.environ.pop("REWARD_SERVER_URL", None)           # score in-process
    sys.path.insert(0, SID_REPO)
    os.chdir(SID_REPO)

    import transformers
    from PIL import Image
    from reward_unified import UnifiedRewardScorer

    tv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if tv >= (4, 52):
        raise RuntimeError(
            f"transformers {transformers.__version__} renamed the Qwen2-VL submodules; "
            "hpsv3 1.0.0 requires <=4.45.x. Run this with the `reward` env python."
        )

    scorer = UnifiedRewardScorer(device="cuda", backend="all")
    print("[scorer]", scorer.describe(), flush=True)
    for need in ("hpsv3", "imagereward", "pickscore", "hpsv2"):
        if need not in scorer.available:
            raise RuntimeError(f"{need} failed to load; available={scorer.available}")
    if getattr(scorer.state, "hpsv3_impl", None) != "official":
        raise RuntimeError(f"HPSv3 loaded via fallback impl={scorer.state.hpsv3_impl!r}, not the official model")

    rows = []
    for mf in sorted(Path(args.manifest_dir).glob("*.jsonl")):
        with mf.open(encoding="utf-8") as f:
            rows.extend(json.loads(l) for l in f if l.strip())
    seen, uniq = set(), []
    for r in rows:  # manifests append on resume; keep one row per image
        k = (r["backend"], r["prompt_id"], r["sample_index"])
        if k not in seen:
            seen.add(k); uniq.append(r)
    uniq.sort(key=lambda r: (r["backend"], r["prompt_id"], r["sample_index"]))
    shard = [r for i, r in enumerate(uniq) if i % args.num_shards == args.shard_index]
    print(f"[scorer] shard {args.shard_index}/{args.num_shards}: {len(shard)} of {len(uniq)} images", flush=True)

    out_path = Path(args.out).with_suffix(f".shard{args.shard_index}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["backend", "prompt_id", "sample_index", "source", "category",
                    "skills", "seed", "hpsv3", "imagereward", "pickscore", "hpsv2", "image"])
        for i, r in enumerate(shard):
            with Image.open(r["image"]) as im:
                im = im.convert("RGB")
                hps = float(scorer._score_hpsv3(r["prompt"], im))
                ir = float(scorer._score_imagereward(r["prompt"], im))
                ps = float(scorer._score_pickscore(r["prompt"], im))
                h2 = float(scorer._score_hpsv2(r["prompt"], im))
            w.writerow([r["backend"], r["prompt_id"], r["sample_index"], r.get("source", ""),
                        r.get("category", ""), r.get("skills", ""), r.get("seed", ""),
                        f"{hps:.6f}", f"{ir:.6f}", f"{ps:.6f}", f"{h2:.6f}", r["image"]])
            if i % 50 == 0:
                f.flush()
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                print(f"  {i}/{len(shard)}  {time.time()-t0:.0f}s  {rate:.2f} img/s", flush=True)
    print(f"[scorer] wrote {out_path} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
