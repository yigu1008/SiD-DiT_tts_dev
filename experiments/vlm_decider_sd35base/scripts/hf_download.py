#!/usr/bin/env python3
"""Download one HF repo into the /data hub cache. usage: hf_download.py <repo_id> [allow_pattern ...]"""
import sys, os, time
from huggingface_hub import snapshot_download
rid = sys.argv[1]; allow = sys.argv[2:] or None
tok = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
t0 = time.time()
p = snapshot_download(rid, cache_dir="/data/ygu/.cache/hub", token=tok, allow_patterns=allow,
                      ignore_patterns=["*.fp16*", "*fp8*", "*.ckpt", "*.msgpack", "*.h5"], max_workers=8)
print(f"DONE {rid} -> {p} in {time.time()-t0:.0f}s", flush=True)
