#!/usr/bin/env python3
"""Score generated images with CLIP-FlanT5 VQAScore (vqascore_reward env).

Runs in its own env because t2v_metrics pins transformers 4.49 while the
generation env is on 4.52. Scores straight from file paths in batches.
"""
from __future__ import annotations

import argparse, csv, json, os, time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_dir", default="/data/ygu/reward_orthogonality/manifests")
    p.add_argument("--out", default="/data/ygu/reward_orthogonality/scores/vqa_scores.csv")
    p.add_argument("--model", default="clip-flant5-xxl")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    args = p.parse_args()

    # t2v_metrics hardcodes HF_CACHE_DIR = "./hf_cache/" -- a RELATIVE path that
    # ignores HF_HOME/HF_HUB_CACHE entirely. Left alone it creates ./hf_cache in
    # whatever the cwd happens to be and re-downloads all 22GB of clip-flant5-xxl
    # (this filled /data to zero once already). Pass cache_dir explicitly AND run
    # from the SiD repo, so both the explicit arg and the relative default land
    # on the copy that is already downloaded.
    CACHE = "/home/ygu/project/SiD-DiT_tts_dev/hf_cache"
    if not Path(CACHE, "models--zhiqiulin--clip-flant5-xxl").is_dir():
        raise RuntimeError(f"clip-flant5-xxl cache not found at {CACHE}")
    os.chdir("/home/ygu/project/SiD-DiT_tts_dev")
    os.environ.setdefault("HF_HOME", CACHE)
    os.environ.setdefault("HF_HUB_CACHE", CACHE)
    os.environ.setdefault("WANDB_DISABLED", "true")

    import t2v_metrics

    scorer = t2v_metrics.VQAScore(model=args.model, device="cuda:0", cache_dir=CACHE)
    print(f"[vqa] loaded {args.model}", flush=True)

    rows = []
    for mf in sorted(Path(args.manifest_dir).glob("*.jsonl")):
        with mf.open(encoding="utf-8") as f:
            rows.extend(json.loads(l) for l in f if l.strip())
    seen, uniq = set(), []
    for r in rows:
        k = (r["backend"], r["prompt_id"], r["sample_index"])
        if k not in seen:
            seen.add(k); uniq.append(r)
    uniq.sort(key=lambda r: (r["backend"], r["prompt_id"], r["sample_index"]))
    shard = [r for i, r in enumerate(uniq) if i % args.num_shards == args.shard_index]
    print(f"[vqa] shard {args.shard_index}/{args.num_shards}: {len(shard)} of {len(uniq)} images", flush=True)

    out_path = Path(args.out).with_suffix(f".shard{args.shard_index}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["backend", "prompt_id", "sample_index", "vqascore"])
        for start in range(0, len(shard), args.batch_size):
            chunk = shard[start : start + args.batch_size]
            # Pairwise diagonal: t2v_metrics returns an [n_img, n_txt] matrix.
            scores = scorer(images=[r["image"] for r in chunk],
                            texts=[r["prompt"] for r in chunk])
            for j, r in enumerate(chunk):
                w.writerow([r["backend"], r["prompt_id"], r["sample_index"],
                            f"{float(scores[j][j]):.6f}"])
            if start % (args.batch_size * 20) == 0:
                f.flush()
                print(f"  {start}/{len(shard)}  {time.time() - t0:.0f}s", flush=True)
    print(f"[vqa] wrote {out_path} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
