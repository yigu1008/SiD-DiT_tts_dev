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

def _preload_clip_weights() -> None:
    """Pre-download CLIP ViT-H-14 weights that ImageReward needs internally."""
    clip_cache = os.environ.get(
        "CLIP_CACHE_DIR",
        os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")) + "/clip",
    )
    Path(clip_cache).mkdir(parents=True, exist_ok=True)

    # ImageReward uses clip.load("ViT-H-14") which downloads from OpenAI
    # Pre-cache it so it works in offline mode
    clip_path = Path(clip_cache) / "ViT-H-14.pt"
    if clip_path.is_file() and clip_path.stat().st_size > 0:
        _log(f"CLIP ViT-H-14 already cached at {clip_path}")
        return

    _log("downloading CLIP ViT-H-14 for ImageReward ...")
    try:
        import clip
        clip.load("ViT-H-14", device="cpu", download_root=clip_cache)
        _log(f"CLIP ViT-H-14 cached at {clip_cache}")
    except Exception:
        # clip.load may not support ViT-H-14 directly — try open_clip fallback
        try:
            import urllib.request
            url = "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin"
            _log(f"  downloading from {url}")
            urllib.request.urlretrieve(url, str(clip_path))
            _log(f"  CLIP ViT-H-14 downloaded to {clip_path}")
        except Exception as exc2:
            _log(f"WARNING: could not pre-cache CLIP ViT-H-14: {exc2}")


def preload_imagereward() -> None:
    cache_dir = os.environ.get(
        "IMAGEREWARD_CACHE", str(Path.home() / ".cache" / "ImageReward")
    )
    pt_path = Path(cache_dir) / "ImageReward.pt"
    med_config_path = Path(cache_dir) / "med_config.json"
    sentinel = _sentinel_path("imagereward")

    if pt_path.is_file() and pt_path.stat().st_size > 0 and med_config_path.is_file():
        _log(f"ImageReward fully cached at {cache_dir}")
        # Also ensure CLIP weights are cached
        _preload_clip_weights()
        _mark_done("imagereward")
        return

    if LOCAL_RANK == 0:
        _log("downloading full THUDM/ImageReward repo ...")
        try:
            from huggingface_hub import snapshot_download
            # Download all files (ImageReward.pt, med_config.json, etc.)
            # so RM.load() can run fully offline without any network access.
            snapshot_download(
                "THUDM/ImageReward",
                local_dir=cache_dir,
                local_dir_use_symlinks=False,
            )
            _preload_clip_weights()
            _log(f"ImageReward fully ready at {cache_dir}")
            _mark_done("imagereward")
        except Exception as exc:
            _log(f"ERROR downloading THUDM/ImageReward: {exc}")
            raise
    else:
        _wait_for_sentinel(sentinel, "ImageReward")


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
# HPSv3
# ---------------------------------------------------------------------------

def _stage_hf_to_local_disk() -> None:
    """Copy HPSv3 + Qwen2-VL-7B-Instruct from HF_HOME (often blob FUSE) to
    a fast local disk (default /tmp/hf_cache_local).  Writes a sentinel file
    so the launching shell can redirect HF_HOME for the reward server
    subprocess.  Controlled by STAGE_REWARD_WEIGHTS_LOCAL=1.
    """
    if os.environ.get("STAGE_REWARD_WEIGHTS_LOCAL", "0") != "1":
        return
    import subprocess

    src_root = Path(
        os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    ) / "hub"
    dst_base = Path(os.environ.get("REWARD_LOCAL_HF_CACHE", "/tmp/hf_cache_local"))
    dst_root = dst_base / "hub"
    dst_root.mkdir(parents=True, exist_ok=True)

    ok = True
    for repo in ("models--Qwen--Qwen2-VL-7B-Instruct", "models--MizzenAI--HPSv3"):
        src = src_root / repo
        dst = dst_root / repo
        if not src.exists():
            _log(f"stage: source missing, skipping: {src}")
            ok = False
            continue
        if dst.exists() and any(dst.iterdir()):
            _log(f"stage: already present locally: {dst}")
            continue
        _log(f"stage: copying {src} -> {dst} ...")
        t0 = time.time()
        try:
            subprocess.run(
                ["rsync", "-a", "--copy-links", f"{src}/", f"{dst}/"], check=True
            )
        except Exception as exc:
            _log(f"stage: rsync failed ({exc}); falling back to shutil.copytree")
            try:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst, symlinks=False)
            except Exception as exc2:
                _log(f"stage: copy failed for {repo}: {exc2}")
                ok = False
                continue
        _log(f"stage: done {dst} in {time.time() - t0:.1f}s")

    if ok:
        (dst_base / ".stage_done").touch()
        _log(f"stage: sentinel written at {dst_base / '.stage_done'}")
    else:
        _log("stage: incomplete; NOT writing sentinel (server will use HF_HOME)")


def preload_hpsv3() -> None:
    """Pre-download HPSv3 checkpoint and base model weights."""
    sentinel = _sentinel_path("hpsv3")

    if LOCAL_RANK == 0:
        _log("pre-downloading HPSv3 model weights ...")

        # Step 1: always download the HPSv3 checkpoint via huggingface_hub
        # (works even if hpsv3 deps are partially broken)
        try:
            from huggingface_hub import hf_hub_download, snapshot_download

            hf_hub_download("MizzenAI/HPSv3", "HPSv3.safetensors", repo_type="model")
            _log("HPSv3 checkpoint downloaded")

            # HPSv3 uses Qwen2-VL-7B-Instruct as its base model
            snapshot_download("Qwen/Qwen2-VL-7B-Instruct", max_workers=1)
            _log("Qwen2-VL-7B-Instruct (HPSv3 base) cached")
        except Exception as exc:
            _log(f"WARNING: HPSv3 weight download error (non-fatal): {exc}")

        # Step 2 (optional): CPU init sanity check. Loads the full ~7B
        # Qwen2-VL-7B-Instruct into CPU RAM purely to validate imports —
        # easily adds 3–5 min and ~28 GB RAM. The reward server instantiates
        # the same model on cuda:0 anyway, so for server-mode preload we
        # skip this by default. Set PRELOAD_HPSV3_INIT_CHECK=1 to re-enable.
        if os.environ.get("PRELOAD_HPSV3_INIT_CHECK", "0") == "1":
            try:
                import hpsv3  # noqa: F811

                inferencer_cls = getattr(hpsv3, "HPSv3RewardInferencer", None)
                if inferencer_cls is not None:
                    inferencer = None
                    for kwargs in ({"device": "cpu"}, {}):
                        try:
                            inferencer = inferencer_cls(**kwargs)
                            break
                        except Exception:
                            pass
                    if inferencer is not None:
                        _log("HPSv3 full init OK on CPU")
                        del inferencer
                    else:
                        _log("WARNING: HPSv3RewardInferencer init failed on CPU; will retry on GPU at runtime")
            except Exception as exc:
                _log(f"WARNING: HPSv3 import/init check failed (non-fatal): {exc}")
        else:
            _log("HPSv3 weights cached; skipping CPU init check (set PRELOAD_HPSV3_INIT_CHECK=1 to enable)")

        # Optionally stage weights to local disk for faster server startup.
        try:
            _stage_hf_to_local_disk()
        except Exception as exc:
            _log(f"stage: unexpected error (non-fatal): {exc}")

        _mark_done("hpsv3")
    else:
        _wait_for_sentinel(sentinel, "HPSv3")


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
    if "hpsv3" in BACKENDS:
        preload_hpsv3()

    _log("done")


if __name__ == "__main__":
    main()
