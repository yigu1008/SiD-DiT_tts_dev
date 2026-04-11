#!/usr/bin/env python3
"""
Toy script to test loading UnifiedReward and HPSv3 on cluster.
Downloads models directly from HuggingFace (no Azure copy needed).

Usage:
  # Test both
  python test_reward_cluster.py --test all --device cuda

  # Test individually
  python test_reward_cluster.py --test unifiedreward --device cuda
  python test_reward_cluster.py --test hpsv3 --device cuda

  # CPU-only smoke test
  python test_reward_cluster.py --test all --device cpu

Prerequisites (install before running):
  # UnifiedReward deps (pinned versions for cluster compatibility)
  pip install --no-cache-dir "transformers>=4.52,<4.53" "qwen-vl-utils==0.0.14"
  pip install --no-cache-dir "accelerate>=0.30" "sentencepiece"

  # HPSv3 deps
  pip install --no-cache-dir "hpsv3" "open_clip_torch" "omegaconf" "hydra-core"

  # Shared
  pip install --no-cache-dir "Pillow>=10.0" "torch>=2.1"
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image


def make_test_image(width=1024, height=1024):
    """Create a simple test image (gradient + color blocks)."""
    import numpy as np
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Gradient background
    for y in range(height):
        for x in range(width):
            img[y, x] = [
                int(255 * x / width),
                int(255 * y / height),
                128,
            ]
    return Image.fromarray(img)


def test_unifiedreward(device="cuda", prompt="a cat sitting on a windowsill"):
    """Test UnifiedReward model loading and scoring."""
    print("\n" + "=" * 60)
    print("TEST: UnifiedReward")
    print("=" * 60)

    # Step 1: Check dependencies
    print("  Checking dependencies...")
    errors = []
    try:
        import transformers
        print(f"    transformers: {transformers.__version__}")
    except ImportError:
        errors.append("transformers not installed")
    try:
        from qwen_vl_utils import process_vision_info
        print(f"    qwen_vl_utils: OK")
    except ImportError:
        errors.append("qwen_vl_utils not installed (pip install qwen-vl-utils==0.0.14)")
    try:
        import accelerate
        print(f"    accelerate: {accelerate.__version__}")
    except ImportError:
        errors.append("accelerate not installed")

    if errors:
        print(f"\n  MISSING DEPS: {'; '.join(errors)}")
        print("  Install with:")
        print('    pip install --no-cache-dir "transformers>=4.52,<4.53" "qwen-vl-utils==0.0.14" "accelerate>=0.30" "sentencepiece"')
        return False

    # Step 2: Download model (direct from HF, no Azure copy)
    model_id = "CodeGoat24/UnifiedReward-qwen-7b"
    print(f"\n  Downloading/loading model: {model_id}")
    print(f"  HF_HOME: {os.environ.get('HF_HOME', '~/.cache/huggingface')}")

    t0 = time.time()
    try:
        from huggingface_hub import snapshot_download
        cache_path = snapshot_download(
            model_id,
            resume_download=True,
            max_workers=1,
        )
        print(f"  Model cached at: {cache_path}")
    except Exception as e:
        print(f"  WARNING: snapshot_download failed: {e}")
        print(f"  Will try loading directly (may download on first access)")

    # Step 3: Load via reward_unified
    print(f"\n  Loading via UnifiedRewardScorer(device='{device}', backend='unifiedreward')...")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from reward_unified import UnifiedRewardScorer

        t1 = time.time()
        scorer = UnifiedRewardScorer(device=device, backend="unifiedreward")
        t2 = time.time()
        print(f"  Loaded in {t2 - t1:.1f}s")
        print(f"  Available backends: {scorer.available}")
        print(f"  Description: {scorer.describe()}")

        if torch.cuda.is_available() and "cuda" in str(device):
            print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

        # Step 4: Score test image
        print(f"\n  Scoring test image...")
        img = make_test_image()
        t3 = time.time()
        score = float(scorer.score(prompt, img))
        t4 = time.time()
        print(f"  Score: {score:.4f} ({t4 - t3:.1f}s)")

        # Score a second time to check caching
        score2 = float(scorer.score("a beautiful sunset over mountains", img))
        print(f"  Score (different prompt): {score2:.4f}")

        print(f"\n  UnifiedReward: OK")
        del scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hpsv3(device="cuda", prompt="a cat sitting on a windowsill"):
    """Test HPSv3 model loading and scoring."""
    print("\n" + "=" * 60)
    print("TEST: HPSv3")
    print("=" * 60)

    # Step 1: Check dependencies
    print("  Checking dependencies...")
    errors = []
    try:
        import hpsv3
        print(f"    hpsv3: {getattr(hpsv3, '__version__', 'installed')}")
        inferencer_cls = getattr(hpsv3, "HPSv3RewardInferencer", None)
        if inferencer_cls is None:
            errors.append("hpsv3 module missing HPSv3RewardInferencer class")
        else:
            print(f"    HPSv3RewardInferencer: found")
    except ImportError:
        errors.append("hpsv3 not installed")
    try:
        import open_clip
        print(f"    open_clip: {open_clip.__version__}")
    except ImportError:
        errors.append("open_clip_torch not installed")
    try:
        import omegaconf
        print(f"    omegaconf: {omegaconf.__version__}")
    except ImportError:
        errors.append("omegaconf not installed")

    if errors:
        print(f"\n  MISSING DEPS: {'; '.join(errors)}")
        print("  Install with:")
        print('    pip install --no-cache-dir "hpsv3" "open_clip_torch" "omegaconf" "hydra-core"')
        return False

    # Step 2: Try loading via reward_unified
    print(f"\n  Loading via UnifiedRewardScorer(device='{device}', backend='hpsv3')...")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from reward_unified import UnifiedRewardScorer

        t1 = time.time()
        scorer = UnifiedRewardScorer(device=device, backend="hpsv3")
        t2 = time.time()
        print(f"  Loaded in {t2 - t1:.1f}s")
        print(f"  Available backends: {scorer.available}")
        print(f"  Description: {scorer.describe()}")

        if torch.cuda.is_available() and "cuda" in str(device):
            print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

        # Step 3: Score test image
        print(f"\n  Scoring test image...")
        img = make_test_image()
        t3 = time.time()
        score = float(scorer.score(prompt, img))
        t4 = time.time()
        print(f"  Score: {score:.4f} ({t4 - t3:.1f}s)")

        score2 = float(scorer.score("a beautiful sunset over mountains", img))
        print(f"  Score (different prompt): {score2:.4f}")

        print(f"\n  HPSv3: OK")
        del scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()

        # Try direct loading as fallback diagnostic
        print(f"\n  --- Direct HPSv3 load (bypassing reward_unified) ---")
        try:
            import hpsv3
            inferencer = hpsv3.HPSv3RewardInferencer(device=device)
            img = make_test_image()
            score = inferencer.score(prompt, img)
            print(f"  Direct HPSv3 score: {score}")
            print(f"  Direct load works — issue is in reward_unified integration")
        except Exception as e2:
            print(f"  Direct load also failed: {e2}")
            traceback.print_exc()
        return False


def test_both_together(device="cuda", prompt="a cat sitting on a windowsill"):
    """Test loading both models simultaneously (memory check)."""
    print("\n" + "=" * 60)
    print("TEST: Both models simultaneously")
    print("=" * 60)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from reward_unified import UnifiedRewardScorer

    if torch.cuda.is_available() and "cuda" in str(device):
        print(f"  GPU total: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        print(f"  GPU used before: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    img = make_test_image()
    results = {}

    for backend in ["unifiedreward", "hpsv3"]:
        try:
            print(f"\n  Loading {backend}...")
            t0 = time.time()
            scorer = UnifiedRewardScorer(device=device, backend=backend)
            t1 = time.time()
            score = float(scorer.score(prompt, img))
            t2 = time.time()
            results[backend] = {
                "score": score,
                "load_time": t1 - t0,
                "score_time": t2 - t1,
                "status": "OK",
            }
            if torch.cuda.is_available() and "cuda" in str(device):
                results[backend]["gpu_mb"] = torch.cuda.memory_allocated() / 1e6
            print(f"    {backend}: score={score:.4f} load={t1-t0:.1f}s score={t2-t1:.1f}s")
            del scorer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            results[backend] = {"status": f"FAILED: {e}"}
            print(f"    {backend}: FAILED — {e}")

    print(f"\n  Summary:")
    for k, v in results.items():
        print(f"    {k}: {v['status']}" + (f" score={v['score']:.4f}" if "score" in v else ""))

    return all(v["status"] == "OK" for v in results.values())


def main():
    parser = argparse.ArgumentParser(description="Test UnifiedReward + HPSv3 on cluster")
    parser.add_argument("--test", choices=["unifiedreward", "hpsv3", "both", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default="a cat sitting on a windowsill")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"HF_HOME: {os.environ.get('HF_HOME', '~/.cache/huggingface')}")
    print(f"Python: {sys.version}")

    ok = True
    if args.test in ("unifiedreward", "all"):
        ok &= test_unifiedreward(args.device, args.prompt)

    if args.test in ("hpsv3", "all"):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ok &= test_hpsv3(args.device, args.prompt)

    if args.test in ("both", "all"):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ok &= test_both_together(args.device, args.prompt)

    print("\n" + "=" * 60)
    print(f"RESULT: {'ALL PASSED' if ok else 'SOME FAILED'}")
    print("=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
