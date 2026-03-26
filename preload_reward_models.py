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
    HPS_ROOT              - Directory for HPSv2 checkpoint (overrides ~/.cache/hpsv2)
    REWARD_BACKENDS       - Space-separated list of backends to preload
                            (default: imagereward pickscore hpsv2)
"""

import os
import shutil
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


def _hf_download(repo_id: str, filename: str, local_dir: str) -> str:
    """
    Download a single file from HuggingFace Hub to local_dir/filename.

    Uses local_dir_use_symlinks=False to force a real file copy — Azure blob
    FUSE mounts do not support symlinks, so the default symlink behaviour of
    newer huggingface_hub versions silently leaves a broken link.

    Falls back to a manual copy from the hub cache if local_dir_use_symlinks
    is not supported by the installed version.
    """
    from huggingface_hub import hf_hub_download

    dest = Path(local_dir) / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        # huggingface_hub >= 0.17 supports local_dir_use_symlinks
        cached = hf_hub_download(
            repo_id,
            filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
    except TypeError:
        # Older huggingface_hub: download to hub cache, then copy manually
        cached = hf_hub_download(repo_id, filename)
        if str(Path(cached).resolve()) != str(dest.resolve()):
            shutil.copy2(cached, str(dest))
        cached = str(dest)

    # Verify the destination is a real file (not a broken symlink)
    if not dest.is_file():
        # Symlink exists but target unreachable — copy the hub-cached file
        hub_cached = hf_hub_download(repo_id, filename)
        shutil.copy2(hub_cached, str(dest))

    _log(f"  -> {dest} ({dest.stat().st_size // (1024*1024)} MB)")
    return str(dest)


# ---------------------------------------------------------------------------
# ImageReward
# ---------------------------------------------------------------------------

def preload_imagereward() -> None:
    cache_dir = os.environ.get(
        "IMAGEREWARD_CACHE", str(Path.home() / ".cache" / "ImageReward")
    )
    pt_path = Path(cache_dir) / "ImageReward.pt"
    sentinel = _sentinel_path("imagereward")

    if pt_path.is_file() and pt_path.stat().st_size > 0:
        _log(f"ImageReward.pt already cached at {cache_dir}")
        _mark_done("imagereward")
        return

    if LOCAL_RANK == 0:
        _log("downloading ImageReward.pt ...")
        try:
            _hf_download("THUDM/ImageReward", "ImageReward.pt", cache_dir)
            _log(f"ImageReward.pt ready at {cache_dir}")
            _mark_done("imagereward")
        except Exception as exc:
            _log(f"ERROR downloading ImageReward.pt: {exc}")
            raise
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
        _log(f"downloading PickScore ({model_id}) ...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(model_id)
            _log(f"PickScore ready")
            _mark_done("pickscore")
        except Exception as exc:
            _log(f"ERROR downloading PickScore: {exc}")
            raise
    else:
        _wait_for_sentinel(sentinel, "PickScore")


# ---------------------------------------------------------------------------
# HPSv2
# ---------------------------------------------------------------------------

def preload_hpsv2() -> None:
    hps_root = os.environ.get("HPS_ROOT", str(Path.home() / ".cache" / "hpsv2"))
    checkpoint = Path(hps_root) / "HPS_v2_compressed.pt"
    sentinel = _sentinel_path("hpsv2")

    if checkpoint.is_file() and checkpoint.stat().st_size > 0:
        _log(f"HPSv2 checkpoint already cached at {hps_root}")
        _mark_done("hpsv2")
        return

    if LOCAL_RANK == 0:
        _log("downloading HPSv2 checkpoint ...")
        try:
            _hf_download("xswu/HPSv2", "HPS_v2_compressed.pt", hps_root)
            _log(f"HPSv2 checkpoint ready at {hps_root}")
            _mark_done("hpsv2")
        except Exception as exc:
            _log(f"ERROR downloading HPSv2: {exc}")
            raise
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
