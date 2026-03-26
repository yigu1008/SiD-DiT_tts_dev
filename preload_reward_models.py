#!/usr/bin/env python3
"""
Pre-download all reward model weights to persistent cache before torchrun starts.

Rank-aware: LOCAL_RANK=0 downloads; other ranks wait up to WAIT_TIMEOUT seconds.
Idempotent: skips models already present on disk.

Usage (in YAML / shell setup, before torchrun):
    python preload_reward_models.py

Relevant env vars:
    LOCAL_RANK            - DDP local rank (default 0 = standalone)
    HF_HOME               - HuggingFace hub cache root
    IMAGEREWARD_CACHE     - Directory for ImageReward.pt (overrides ~/.cache/ImageReward)
    PICKSCORE_MODEL       - HF repo id (default yuvalkirstain/PickScore_v1)
    REWARD_BACKENDS       - Space-separated list of backends to preload
                            (default: imagereward pickscore hpsv2)
"""

import os
import sys
import time
from pathlib import Path

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WAIT_TIMEOUT = int(os.environ.get("PRELOAD_WAIT_TIMEOUT", "300"))

_BACKENDS_ENV = os.environ.get("REWARD_BACKENDS", "imagereward pickscore hpsv2")
BACKENDS = set(_BACKENDS_ENV.lower().split())


def _log(msg: str) -> None:
    print(f"[preload rank={LOCAL_RANK}] {msg}", flush=True)


def _wait_for_sentinel(sentinel: str, label: str) -> bool:
    """Non-rank-0 ranks wait until sentinel file appears."""
    for elapsed in range(WAIT_TIMEOUT):
        if os.path.exists(sentinel):
            return True
        if elapsed % 30 == 0:
            _log(f"waiting for {label} ... ({elapsed}s)")
        time.sleep(1)
    _log(f"WARNING: timed out waiting for {label} (sentinel={sentinel})")
    return False


def _sentinel_path(name: str) -> str:
    cache_base = os.environ.get(
        "HF_HOME", str(Path.home() / ".cache" / "huggingface")
    )
    return str(Path(cache_base) / f".preload_done_{name}")


def _mark_done(name: str) -> None:
    sentinel = _sentinel_path(name)
    Path(sentinel).parent.mkdir(parents=True, exist_ok=True)
    Path(sentinel).touch()


# ---------------------------------------------------------------------------
# ImageReward
# ---------------------------------------------------------------------------

def preload_imagereward() -> None:
    cache_dir = os.environ.get(
        "IMAGEREWARD_CACHE", str(Path.home() / ".cache" / "ImageReward")
    )
    pt_path = Path(cache_dir) / "ImageReward.pt"
    sentinel = _sentinel_path("imagereward")

    if pt_path.exists():
        _log(f"ImageReward.pt already cached at {cache_dir}")
        _mark_done("imagereward")
        return

    if LOCAL_RANK == 0:
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download("THUDM/ImageReward", "ImageReward.pt", local_dir=str(cache_dir))
            _log(f"ImageReward.pt downloaded to {cache_dir}")
            _mark_done("imagereward")
        except Exception as exc:
            _log(f"ERROR downloading ImageReward.pt: {exc}")
    else:
        _wait_for_sentinel(sentinel, "ImageReward.pt")


# ---------------------------------------------------------------------------
# PickScore
# ---------------------------------------------------------------------------

def preload_pickscore() -> None:
    model_id = os.environ.get("PICKSCORE_MODEL", "yuvalkirstain/PickScore_v1")
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    cache_dir = Path(hf_home) / "hub" / f"models--{model_id.replace('/', '--')}"
    sentinel = _sentinel_path("pickscore")

    if cache_dir.exists() and any(cache_dir.iterdir()):
        _log(f"PickScore already cached at {cache_dir}")
        _mark_done("pickscore")
        return

    if LOCAL_RANK == 0:
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(model_id)
            _log(f"PickScore downloaded ({model_id})")
            _mark_done("pickscore")
        except Exception as exc:
            _log(f"ERROR downloading PickScore: {exc}")
    else:
        _wait_for_sentinel(sentinel, "PickScore")


# ---------------------------------------------------------------------------
# HPSv2
# ---------------------------------------------------------------------------

def preload_hpsv2() -> None:
    # hpsv2 downloads its checkpoint from xswu/HPSv2 on HuggingFace Hub.
    # root_path = $HPS_ROOT or ~/.cache/hpsv2
    hps_root = os.environ.get("HPS_ROOT", str(Path.home() / ".cache" / "hpsv2"))
    checkpoint = Path(hps_root) / "HPS_v2_compressed.pt"
    sentinel = _sentinel_path("hpsv2")

    if checkpoint.exists():
        _log(f"HPSv2 checkpoint already cached at {hps_root}")
        _mark_done("hpsv2")
        return

    if LOCAL_RANK == 0:
        try:
            from huggingface_hub import hf_hub_download
            Path(hps_root).mkdir(parents=True, exist_ok=True)
            hf_hub_download("xswu/HPSv2", "HPS_v2_compressed.pt", local_dir=hps_root)
            _log(f"HPSv2 checkpoint downloaded to {hps_root}")
            _mark_done("hpsv2")
        except Exception as exc:
            _log(f"ERROR downloading HPSv2: {exc}")
    else:
        _wait_for_sentinel(sentinel, "HPSv2 checkpoint")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _log(f"starting (backends={sorted(BACKENDS)})")

    if "imagereward" in BACKENDS:
        preload_imagereward()
    if "pickscore" in BACKENDS:
        preload_pickscore()
    if "hpsv2" in BACKENDS:
        preload_hpsv2()

    _log("done")


if __name__ == "__main__":
    main()
