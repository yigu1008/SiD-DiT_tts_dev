#!/usr/bin/env python3
"""Prefetch the Jev-decider study's weights into $SCRATCH/hf_cache (idempotent: re-run until every line says DONE).
Usage: python prefetch_models.py <group>   groups: a | b | c | all
Gated repos (SD3.5-Large) need a token: `hf auth login` once (token lands in ~/.cache/huggingface/token) or export HF_TOKEN.
"""
import os, sys, time, json
os.environ.setdefault("HF_HOME", "/scratch/09398/yigu/hf_cache")
from huggingface_hub import snapshot_download, get_token

GROUPS = {
    "a": [("Qwen/Qwen2.5-VL-7B-Instruct", None), ("MizzenAI/HPSv3", None), ("yuvalkirstain/PickScore_v1", None)],
    "b": [("akhilaaa3/Jev-Omni", "5addda86ddee081a68fb067477ea100c221b8917"), ("zhiqiulin/clip-flant5-xxl", None)],
    "c": [("stabilityai/stable-diffusion-3.5-large", None)],
}
g = sys.argv[1] if len(sys.argv) > 1 else "all"
repos = sum(GROUPS.values(), []) if g == "all" else GROUPS[g]
state = os.path.join(os.environ["HF_HOME"], "prefetch_state.json")
done = json.load(open(state)) if os.path.exists(state) else {}
for repo, rev in repos:
    if done.get(repo): print(f"DONE (cached) {repo} -> {done[repo]}", flush=True); continue
    t = time.time(); print(f"[{time.strftime('%F %T')}] START {repo} rev={rev} token={'yes' if get_token() else 'no'}", flush=True)
    for attempt in range(1, 6):
        try:
            p = snapshot_download(repo, revision=rev, max_workers=4)
            done = json.load(open(state)) if os.path.exists(state) else {}
            done[repo] = p; json.dump(done, open(state, "w"), indent=1)
            print(f"[{time.strftime('%F %T')}] DONE {repo} -> {p} in {time.time()-t:.0f}s", flush=True); break
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:200]}"
            if "401" in msg or "403" in msg or "gated" in msg.lower() or "Unauthorized" in msg:
                print(f"[{time.strftime('%F %T')}] NEEDS TOKEN {repo}: {msg}", flush=True); break
            print(f"[{time.strftime('%F %T')}] retry {attempt}/5 {repo}: {msg}", flush=True); time.sleep(30)
    else:
        print(f"[{time.strftime('%F %T')}] FAILED {repo}", flush=True)
