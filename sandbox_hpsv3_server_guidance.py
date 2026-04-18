#!/usr/bin/env python3
"""Minimal sandbox for official HPSv3 reward-server-guided sampling.

What it validates:
1) Reward server is reachable and exposes hpsv3.
2) UnifiedRewardScorer can score via REWARD_SERVER_URL with backend=hpsv3.
3) (Optional) Tiny FLUX MCTS run uses hpsv3 as search reward and produces output.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from reward_unified import UnifiedRewardScorer


def _http_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _make_probe_image(size: int) -> Image.Image:
    size = max(64, int(size))
    img = Image.new("RGB", (size, size), (20, 24, 30))
    draw = ImageDraw.Draw(img)
    for i in range(0, size, max(4, size // 20)):
        color = (40 + (i * 5) % 150, 70 + (i * 7) % 150, 100 + (i * 11) % 120)
        draw.rectangle([(0, i), (size - 1, min(size - 1, i + max(2, size // 32)))], fill=color)
    draw.ellipse(
        [size // 6, size // 6, size - size // 6, size - size // 6],
        outline=(235, 220, 130),
        width=max(2, size // 96),
    )
    return img


def _parse_summary(summary_path: Path) -> tuple[float | None, float | None, float | None]:
    if not summary_path.exists():
        return None, None, None
    try:
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None, None
    if not isinstance(rows, list) or len(rows) <= 0:
        return None, None, None
    samples = rows[0].get("samples", [])
    if not samples:
        return None, None, None
    sample = samples[0]
    base = sample.get("baseline_score")
    search = sample.get("search_score")
    delta = sample.get("delta_score")
    try:
        return (
            float(base) if base is not None else None,
            float(search) if search is not None else None,
            float(delta) if delta is not None else None,
        )
    except Exception:
        return None, None, None


def _run_tiny_sampling(
    args: argparse.Namespace,
    out_dir: Path,
    prompt: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = out_dir / "prompt.txt"
    prompt_file.write_text(prompt.strip() + "\n", encoding="utf-8")

    sampling_script = Path(args.sampling_script).expanduser().resolve()
    if not sampling_script.exists():
        raise FileNotFoundError(f"sampling script not found: {sampling_script}")

    cmd = [
        str(args.python_bin),
        str(sampling_script),
        "--search_method",
        "mcts",
        "--backend",
        str(args.sampling_backend),
        "--model_id",
        str(args.model_id),
        "--prompt_file",
        str(prompt_file),
        "--n_prompts",
        "1",
        "--n_samples",
        "1",
        "--steps",
        str(int(args.steps)),
        "--width",
        str(int(args.width)),
        "--height",
        str(int(args.height)),
        "--seed",
        str(int(args.seed)),
        "--n_variants",
        str(int(args.n_variants)),
        "--cfg_scales",
        *[str(x) for x in args.cfg_scales],
        "--n_sims",
        str(int(args.n_sims)),
        "--ucb_c",
        str(float(args.ucb_c)),
        "--reward_backend",
        "hpsv3",
        "--reward_device",
        str(args.reward_device),
        "--out_dir",
        str(out_dir),
        "--save_first_k",
        "1",
    ]
    if str(args.dtype).strip():
        cmd += ["--dtype", str(args.dtype)]
    if str(args.device).strip():
        cmd += ["--device", str(args.device)]
    if bool(args.auto_select_gpu):
        cmd += ["--auto_select_gpu"]

    env = os.environ.copy()
    env["REWARD_SERVER_URL"] = str(args.reward_server_url).rstrip("/")
    env.setdefault("SID_FORCE_WANDB_STUB", "1")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    elapsed = time.perf_counter() - t0

    summary_path = out_dir / "summary.json"
    base, search, delta = _parse_summary(summary_path)
    return {
        "ok": proc.returncode == 0 and summary_path.exists(),
        "returncode": int(proc.returncode),
        "elapsed_sec": round(float(elapsed), 3),
        "cmd": cmd,
        "summary_path": str(summary_path),
        "baseline_score": base,
        "search_score": search,
        "delta_score": delta,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-40:]),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal official HPSv3 reward-server guidance sandbox.")
    p.add_argument("--reward_server_url", default=os.environ.get("REWARD_SERVER_URL", "http://localhost:5100"))
    p.add_argument("--reward_device", default="cuda")
    p.add_argument("--prompt", default="a cinematic portrait of a woman in soft rim light, 85mm, ultra detailed")
    p.add_argument("--image_size", type=int, default=320)
    p.add_argument("--out_json", default="./hpsv3_server_guidance_sandbox_report.json")

    p.add_argument("--run_sampling", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--python_bin", default=sys.executable)
    p.add_argument("--sampling_script", default="./sampling_flux_unified.py")
    p.add_argument("--sampling_backend", choices=["flux", "senseflow_flux"], default="flux")
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-schnell")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_variants", type=int, default=2)
    p.add_argument("--cfg_scales", nargs="+", type=float, default=[1.0, 1.5])
    p.add_argument("--n_sims", type=int, default=6)
    p.add_argument("--ucb_c", type=float, default=1.41)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--auto_select_gpu", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sampling_out_dir", default="./sandbox_hpsv3_server_guidance_out")
    return p


def main() -> None:
    args = build_parser().parse_args()
    report: dict[str, Any] = {
        "ok": False,
        "timestamp_unix": time.time(),
        "reward_server_url": str(args.reward_server_url).rstrip("/"),
        "run_sampling": bool(args.run_sampling),
    }

    # Stage 1: reward server health
    health_url = f"{str(args.reward_server_url).rstrip('/')}/health"
    try:
        health = _http_json(health_url, timeout=10.0)
        report["health"] = health
        backends = [str(x) for x in health.get("backends", [])]
        if "hpsv3" not in backends:
            raise RuntimeError(f"reward server is up but hpsv3 not loaded. backends={backends}")
        print(f"[sandbox] reward server OK: {health_url} backends={backends}")
    except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
        report["error"] = f"reward server health failed: {exc}"
        Path(args.out_json).expanduser().resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise SystemExit(report["error"])

    # Stage 2: score through UnifiedRewardScorer via server
    try:
        os.environ["REWARD_SERVER_URL"] = str(args.reward_server_url).rstrip("/")
        os.environ.setdefault("SID_FORCE_WANDB_STUB", "1")
        os.environ.setdefault("WANDB_DISABLED", "true")
        probe = _make_probe_image(int(args.image_size))
        scorer = UnifiedRewardScorer(
            device=str(args.reward_device),
            backend="hpsv3",
            image_reward_model="ImageReward-v1.0",
            pickscore_model="yuvalkirstain/PickScore_v1",
            unifiedreward_model="CodeGoat24/UnifiedReward-qwen-7b",
        )
        score = float(scorer.score(str(args.prompt), probe))
        report["scorer_probe"] = {
            "ok": True,
            "available_backends": list(getattr(scorer, "available", [])),
            "describe": str(scorer.describe()),
            "score": float(score),
        }
        print(f"[sandbox] scorer probe OK: score={score:.6f}")
    except Exception as exc:  # noqa: BLE001
        report["scorer_probe"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        Path(args.out_json).expanduser().resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise SystemExit(f"scorer probe failed: {exc}")

    # Stage 3 (optional): tiny sampling run with hpsv3 guidance
    if args.run_sampling:
        try:
            run_out = Path(args.sampling_out_dir).expanduser().resolve()
            sampling_report = _run_tiny_sampling(args, run_out, str(args.prompt))
            report["sampling_probe"] = sampling_report
            if sampling_report.get("ok"):
                d = sampling_report.get("delta_score")
                print(f"[sandbox] sampling probe OK: delta={d}")
            else:
                print("[sandbox] sampling probe FAILED (see report JSON tails).")
        except Exception as exc:  # noqa: BLE001
            report["sampling_probe"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # Final
    report["ok"] = bool(report.get("scorer_probe", {}).get("ok")) and (
        (not args.run_sampling) or bool(report.get("sampling_probe", {}).get("ok"))
    )
    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[sandbox] report: {out_path}")
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
