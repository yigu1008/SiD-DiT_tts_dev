#!/usr/bin/env python3
"""Smoke-test client for reward_server.py.

Hits GET /health and POST /score for each available backend with a dummy
PIL image. Writes a JSON report and exits non-zero if hpsv3 did not return
a numeric score.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.request

from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reward server smoke client")
    p.add_argument("--url", required=True, help="Base URL, e.g. http://localhost:5100")
    p.add_argument("--out_json", required=True)
    p.add_argument("--prompt", default="a photo of a blue sky")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--require_backend", default="hpsv3",
                    help="If this backend is listed in /health but does not return a numeric score, exit non-zero.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    with urllib.request.urlopen(args.url + "/health", timeout=30) as r:
        health = json.loads(r.read().decode())
    print("health:", health, flush=True)
    backends = health.get("backends", [])

    img = Image.new("RGB", (args.width, args.height), color=(120, 160, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    results = {"health": health, "scores": {}}
    for backend in backends:
        payload = json.dumps({
            "prompt": args.prompt,
            "image_b64": img_b64,
            "backend": backend,
        }).encode()
        req = urllib.request.Request(
            args.url + "/score",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read().decode())
            print(f"  {backend}: {resp}", flush=True)
            results["scores"][backend] = resp
        except Exception as e:
            print(f"  {backend}: ERROR {e}", flush=True)
            results["scores"][backend] = {"error": str(e)}

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out_json, flush=True)

    if args.require_backend in backends:
        s = results["scores"].get(args.require_backend, {})
        if not isinstance(s.get("score"), (int, float)):
            print(f"SMOKE FAIL: {args.require_backend} did not return a numeric score", flush=True)
            return 2
    print("SMOKE OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
