#!/usr/bin/env python3
"""Lightweight HTTP reward scoring server.

Runs in a separate conda env that has hpsv3/ImageReward-compatible deps
(transformers==4.45.2, trl==0.12.2, etc.) so the main generation env
does not need to satisfy those constraints.

Usage:
    # In the hpsv3-compatible env:
    python reward_server.py --port 5100 --device cuda:0 --backends hpsv3 imagereward

    # From the main env, set:
    export REWARD_SERVER_URL=http://localhost:5100
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional

from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reward scoring server")
    p.add_argument("--port", type=int, default=5100)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--backends", nargs="+", default=["hpsv3", "imagereward"],
                    help="Backends to load (hpsv3, imagereward, hpsv2, pickscore)")
    p.add_argument("--image_reward_model", default="ImageReward-v1.0")
    p.add_argument("--pickscore_model", default="yuvalkirstain/PickScore_v1")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Backend loaders — each returns (scorer_callable, name) or raises.
# scorer_callable(prompt: str, image: PIL.Image) -> float
# ---------------------------------------------------------------------------

def _load_hpsv3(device: str):
    """Load HPSv3 in this process."""
    # Wandb stub — trl imports wandb at module level
    _inject_wandb_stub()

    import huggingface_hub as _hfhub
    _hpsv3_ckpt = _hfhub.hf_hub_download(
        "MizzenAI/HPSv3", "HPSv3.safetensors", repo_type="model"
    )
    assert isinstance(_hpsv3_ckpt, str), f"Bad checkpoint path: {type(_hpsv3_ckpt)}"
    print(f"[reward_server] HPSv3 checkpoint: {_hpsv3_ckpt}")

    # Save and restore huggingface_hub after hpsv3 import
    saved_fns = {k: getattr(_hfhub, k) for k in ("hf_hub_download", "snapshot_download") if hasattr(_hfhub, k)}
    import hpsv3
    for k, v in saved_fns.items():
        if getattr(_hfhub, k, None) is not v:
            setattr(_hfhub, k, v)

    inferencer = hpsv3.HPSv3RewardInferencer(device=device, checkpoint_path=_hpsv3_ckpt)

    import torch

    def score_fn(prompt: str, image: Image.Image) -> float:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f, format="PNG")
            tmp_path = f.name
        try:
            with torch.no_grad():
                rewards = inferencer.reward([tmp_path], [prompt])
            return float(rewards[0][0].item())
        finally:
            os.unlink(tmp_path)

    return score_fn


def _load_imagereward(device: str, model_name: str = "ImageReward-v1.0"):
    """Load ImageReward in this process."""
    _inject_wandb_stub()
    import ImageReward as RM
    model = RM.load(model_name, device=device)
    model.eval()

    import torch

    def score_fn(prompt: str, image: Image.Image) -> float:
        with torch.no_grad():
            return float(model.score(prompt, image))

    return score_fn


def _load_hpsv2(device: str):
    """Load HPSv2."""
    import hpsv2
    # Just use the hpsv2.score API
    def score_fn(prompt: str, image: Image.Image) -> float:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f, format="PNG")
            tmp_path = f.name
        try:
            result = hpsv2.score([tmp_path], prompt, hps_version="v2.1")
            return float(result[0]) if isinstance(result, (list, tuple)) else float(result)
        finally:
            os.unlink(tmp_path)

    return score_fn


def _load_pickscore(device: str, model_name: str = "yuvalkirstain/PickScore_v1"):
    """Load PickScore."""
    from transformers import AutoModel, AutoProcessor
    import torch

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def score_fn(prompt: str, image: Image.Image) -> float:
        inputs = processor(
            images=[image], text=[prompt], return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits_per_image
        return float(logits[0][0].item())

    return score_fn


def _inject_wandb_stub():
    """Inject a minimal wandb stub to prevent import errors."""
    import importlib.machinery
    import types

    if sys.modules.get("wandb") is not None:
        return

    def _noop(*a, **kw):
        return None

    class _DummyRun:
        def log(self, *a, **kw): return None
        def finish(self, *a, **kw): return None
        def __getattr__(self, _n): return _noop

    stub = types.ModuleType("wandb")
    stub.__codex_stub__ = True
    stub.init = lambda *a, **kw: _DummyRun()
    stub.log = _noop; stub.finish = _noop; stub.login = _noop
    stub.watch = _noop; stub.define_metric = _noop
    stub.config = {}; stub.run = None
    stub.__package__ = "wandb"
    stub.__path__ = []
    stub.__file__ = ""
    stub.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None, is_package=True)

    def _make_sub(name, parent_pkg="wandb"):
        sub = types.ModuleType(name)
        sub.__package__ = parent_pkg
        sub.__path__ = []
        sub.__file__ = ""
        sub.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        return sub

    sdk_stub = _make_sub("wandb.sdk"); stub.sdk = sdk_stub
    proto_stub = _make_sub("wandb.proto")
    telem_stub = _make_sub("wandb.proto.wandb_telemetry_pb2", "wandb.proto")
    telem_stub.Imports = type("Imports", (), {"__getattr__": lambda s, n: _noop})()
    proto_stub.wandb_telemetry_pb2 = telem_stub; stub.proto = proto_stub

    sys.modules["wandb"] = stub
    sys.modules.setdefault("wandb.sdk", sdk_stub)
    sys.modules.setdefault("wandb.proto", proto_stub)
    sys.modules.setdefault("wandb.proto.wandb_telemetry_pb2", telem_stub)
    print("[reward_server] wandb stub injected.")


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

_scorers: Dict[str, Any] = {}
_request_count = 0


def _resolve_backend(backend: Optional[str]) -> tuple:
    """Resolve backend name to (scorer_fn, backend_name) or (None, error_msg)."""
    if backend and backend in _scorers:
        return _scorers[backend], backend
    elif backend:
        return None, f"backend {backend!r} not loaded"
    else:
        name = next(iter(_scorers.keys()))
        return _scorers[name], name


class RewardHandler(BaseHTTPRequestHandler):
    """Simple JSON API:

    POST /score
    {
        "prompt": "a cat sitting on a mat",
        "image_b64": "<base64-encoded PNG>",
        "backend": "hpsv3"          # optional, defaults to first available
    }
    → {"score": 1.234, "backend": "hpsv3"}

    POST /score_batch
    {
        "items": [
            {"prompt": "a cat", "image_b64": "..."},
            {"prompt": "a dog", "image_b64": "..."}
        ],
        "backend": "hpsv3"          # optional
    }
    → {"scores": [{"score": 1.2, "backend": "hpsv3"}, ...]}

    GET /health
    → {"status": "ok", "backends": ["hpsv3", "imagereward"], "requests_served": 42}
    """

    def do_GET(self):
        if self.path == "/health":
            self._json_response({
                "status": "ok",
                "backends": list(_scorers.keys()),
                "requests_served": _request_count,
            })
        else:
            self._json_response({"error": "not found"}, 404)

    def do_POST(self):
        global _request_count

        if self.path == "/score":
            self._handle_score()
        elif self.path == "/score_batch":
            self._handle_score_batch()
        else:
            self._json_response({"error": "not found"}, 404)

    def _handle_score(self):
        global _request_count
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            prompt = body["prompt"]
            image_b64 = body["image_b64"]

            image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
            fn, backend = _resolve_backend(body.get("backend"))
            if fn is None:
                self._json_response({"error": backend}, 400)
                return

            score = fn(prompt, image)
            _request_count += 1
            self._json_response({"score": float(score), "backend": backend})

        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[reward_server] ERROR in /score: {exc}\n{tb}")
            self._json_response({"error": str(exc), "traceback": tb}, 500)

    def _handle_score_batch(self):
        global _request_count
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            items = body["items"]
            fn, backend = _resolve_backend(body.get("backend"))
            if fn is None:
                self._json_response({"error": backend}, 400)
                return

            results = []
            for item in items:
                image = Image.open(
                    io.BytesIO(base64.b64decode(item["image_b64"]))
                ).convert("RGB")
                score = fn(item["prompt"], image)
                results.append({"score": float(score), "backend": backend})
            _request_count += len(items)
            self._json_response({"scores": results})

        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[reward_server] ERROR in /score_batch: {exc}\n{tb}")
            self._json_response({"error": str(exc), "traceback": tb}, 500)

    def _json_response(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Quieter logging
        pass


def main():
    args = parse_args()

    loaders = {
        "hpsv3": lambda: _load_hpsv3(args.device),
        "imagereward": lambda: _load_imagereward(args.device, args.image_reward_model),
        "hpsv2": lambda: _load_hpsv2(args.device),
        "pickscore": lambda: _load_pickscore(args.device, args.pickscore_model),
    }

    for backend in args.backends:
        if backend not in loaders:
            print(f"[reward_server] Unknown backend: {backend}, skipping.")
            continue
        try:
            print(f"[reward_server] Loading {backend} on {args.device} ...")
            _scorers[backend] = loaders[backend]()
            print(f"[reward_server] {backend} loaded OK.")
        except Exception as exc:
            print(f"[reward_server] {backend} FAILED: {exc}")
            traceback.print_exc()

    if not _scorers:
        print("[reward_server] ERROR: No backends loaded. Exiting.")
        sys.exit(1)

    print(f"[reward_server] Serving on {args.host}:{args.port} with backends: {list(_scorers.keys())}")
    server = HTTPServer((args.host, args.port), RewardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[reward_server] Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
