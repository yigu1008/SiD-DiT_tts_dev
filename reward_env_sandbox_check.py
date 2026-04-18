#!/usr/bin/env python3
"""
Reward environment sandbox checker.

Purpose:
- Reproduce reward-backend environment issues in isolation.
- Run each backend case in a fresh subprocess so imports/state do not leak.
- Produce a single JSON report that can be compared across machines/runs.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


_JSON_PREFIX = "__REWARD_SANDBOX_JSON__"


@dataclass(frozen=True)
class CaseSpec:
    backend: str
    env: dict[str, str]
    note: str


_CASES: dict[str, CaseSpec] = {
    "imagereward": CaseSpec(
        backend="imagereward",
        env={"SID_FORCE_WANDB_STUB": "1", "WANDB_DISABLED": "true"},
        note="ImageReward only (local load path).",
    ),
    "imagereward_realwandb": CaseSpec(
        backend="imagereward",
        env={"SID_FORCE_WANDB_STUB": "0", "WANDB_DISABLED": "true"},
        note="ImageReward only with real wandb import path.",
    ),
    "hpsv3_imscore": CaseSpec(
        backend="hpsv3",
        env={"SID_HPSV3_IMPL": "imscore", "SID_FORCE_WANDB_STUB": "1", "WANDB_DISABLED": "true"},
        note="HPSv3 forced to imscore implementation.",
    ),
    "hpsv3_official": CaseSpec(
        backend="hpsv3",
        env={"SID_HPSV3_IMPL": "official", "SID_FORCE_WANDB_STUB": "1", "WANDB_DISABLED": "true"},
        note="HPSv3 forced to official implementation.",
    ),
    "hpsv3_official_realwandb": CaseSpec(
        backend="hpsv3",
        env={"SID_HPSV3_IMPL": "official", "SID_FORCE_WANDB_STUB": "0", "WANDB_DISABLED": "true"},
        note="HPSv3 official implementation with real wandb import path.",
    ),
    "hpsv3_auto": CaseSpec(
        backend="hpsv3",
        env={"SID_HPSV3_IMPL": "auto", "SID_FORCE_WANDB_STUB": "1", "WANDB_DISABLED": "true"},
        note="HPSv3 official-first with imscore fallback.",
    ),
    "blend_hpsv3_imscore": CaseSpec(
        backend="blend",
        env={"SID_HPSV3_IMPL": "imscore", "SID_FORCE_WANDB_STUB": "1", "WANDB_DISABLED": "true"},
        note="Blend backend with HPS branch from imscore HPSv3.",
    ),
}


def _parse_csv(raw: str) -> list[str]:
    out: list[str] = []
    for tok in str(raw).replace(" ", ",").split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _safe_version(dist_name: str) -> str:
    try:
        return md.version(dist_name)
    except Exception:
        return "missing"


def _tail(text: str, n_lines: int = 40) -> str:
    lines = str(text).splitlines()
    if len(lines) <= n_lines:
        return "\n".join(lines)
    return "\n".join(lines[-n_lines:])


def _make_test_image(size: int) -> Image.Image:
    size = max(64, int(size))
    img = Image.new("RGB", (size, size), (18, 23, 31))
    draw = ImageDraw.Draw(img)
    for i in range(0, size, max(4, size // 16)):
        color = (40 + (i * 3) % 180, 80 + (i * 5) % 140, 120 + (i * 7) % 120)
        draw.rectangle([(0, i), (size - 1, min(size - 1, i + max(2, size // 32)))], fill=color)
    draw.ellipse(
        [size // 5, size // 5, size - size // 5, size - size // 5],
        outline=(230, 220, 120),
        width=max(2, size // 96),
    )
    return img


def _extract_payload(stdout: str, stderr: str) -> dict[str, Any] | None:
    for blob in (stdout, stderr):
        for line in blob.splitlines():
            if line.startswith(_JSON_PREFIX):
                try:
                    return json.loads(line[len(_JSON_PREFIX) :].strip())
                except Exception:
                    continue
    return None


def _worker(args: argparse.Namespace) -> int:
    case_name = str(args.worker_case)
    spec = _CASES.get(case_name)
    if spec is None:
        payload = {
            "ok": False,
            "case": case_name,
            "error_type": "UnknownCase",
            "error": f"Unknown case: {case_name}. Available: {sorted(_CASES)}",
        }
        print(_JSON_PREFIX + json.dumps(payload, ensure_ascii=True))
        return 2

    for k, v in spec.env.items():
        os.environ[k] = v
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    if args.force_local_reward:
        os.environ["REWARD_SERVER_URL"] = ""

    started = time.perf_counter()
    payload: dict[str, Any] = {
        "ok": False,
        "case": case_name,
        "backend": spec.backend,
        "note": spec.note,
        "env_overrides": dict(spec.env),
        "versions": {
            "python": platform.python_version(),
            "torch": _safe_version("torch"),
            "transformers": _safe_version("transformers"),
            "trl": _safe_version("trl"),
            "huggingface_hub": _safe_version("huggingface-hub"),
            "hpsv3": _safe_version("hpsv3"),
            "imscore": _safe_version("imscore"),
            "image_reward": _safe_version("image-reward"),
            "qwen_vl_utils": _safe_version("qwen-vl-utils"),
        },
    }

    try:
        import torch
        from reward_unified import UnifiedRewardScorer

        requested_device = str(args.device)
        resolved_device = requested_device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            if args.strict_device:
                raise RuntimeError("CUDA requested but unavailable and strict_device=1.")
            resolved_device = "cpu"

        image = _make_test_image(int(args.image_size))
        scorer = UnifiedRewardScorer(
            device=resolved_device,
            backend=spec.backend,
            image_reward_model=str(args.image_reward_model),
            pickscore_model=str(args.pickscore_model),
            unifiedreward_model=str(args.unifiedreward_model),
        )
        score = float(scorer.score(str(args.prompt), image))

        state = getattr(scorer, "state", None)
        payload.update(
            {
                "ok": True,
                "requested_device": requested_device,
                "resolved_device": resolved_device,
                "cuda_available": bool(torch.cuda.is_available()),
                "available_backends": list(getattr(scorer, "available", [])),
                "describe": str(scorer.describe()),
                "score": float(score),
                "hpsv3_impl_loaded": str(getattr(state, "hpsv3_impl", None)) if state is not None else None,
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    payload["duration_sec"] = round(time.perf_counter() - started, 3)
    print(_JSON_PREFIX + json.dumps(payload, ensure_ascii=True))
    return 0 if payload.get("ok") else 3


def _run_case_subprocess(args: argparse.Namespace, case_name: str) -> dict[str, Any]:
    cmd = [
        str(args.python_bin),
        str(Path(__file__).resolve()),
        "--worker-case",
        case_name,
        "--device",
        str(args.device),
        "--prompt",
        str(args.prompt),
        "--image-size",
        str(args.image_size),
        "--image-reward-model",
        str(args.image_reward_model),
        "--pickscore-model",
        str(args.pickscore_model),
        "--unifiedreward-model",
        str(args.unifiedreward_model),
    ]
    if args.force_local_reward:
        cmd.append("--force-local-reward")
    if args.strict_device:
        cmd.append("--strict-device")

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=float(args.timeout_sec),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "case": case_name,
            "error_type": "TimeoutExpired",
            "error": f"Timed out after {args.timeout_sec}s",
            "duration_sec": round(time.perf_counter() - t0, 3),
            "stdout_tail": _tail(exc.stdout or ""),
            "stderr_tail": _tail(exc.stderr or ""),
            "returncode": 124,
        }

    payload = _extract_payload(proc.stdout, proc.stderr)
    if payload is None:
        payload = {
            "ok": False,
            "case": case_name,
            "error_type": "MissingWorkerPayload",
            "error": "Worker did not emit JSON payload marker.",
        }

    payload["duration_sec_subprocess"] = round(time.perf_counter() - t0, 3)
    payload["returncode"] = int(proc.returncode)
    payload["stdout_tail"] = _tail(proc.stdout)
    payload["stderr_tail"] = _tail(proc.stderr)
    if proc.returncode != 0 and payload.get("ok", False):
        payload["ok"] = False
        payload["error_type"] = "NonZeroExit"
        payload["error"] = f"Worker exited with non-zero status {proc.returncode}."
    return payload


def _coordinator(args: argparse.Namespace) -> int:
    requested_cases = _parse_csv(args.cases)
    if not requested_cases:
        raise RuntimeError("No cases selected. Use --cases or --list-cases.")
    unknown = [c for c in requested_cases if c not in _CASES]
    if unknown:
        raise RuntimeError(f"Unknown cases: {unknown}. Available: {sorted(_CASES)}")

    allow_fail_cases = set(_parse_csv(args.allow_fail_cases))
    started = time.perf_counter()

    results: list[dict[str, Any]] = []
    for idx, case_name in enumerate(requested_cases, start=1):
        print(f"[sandbox] case {idx}/{len(requested_cases)} -> {case_name}")
        result = _run_case_subprocess(args, case_name)
        results.append(result)
        status = "PASS" if result.get("ok") else "FAIL"
        impl = result.get("hpsv3_impl_loaded")
        score = result.get("score")
        suffix = f" impl={impl}" if impl not in {None, "None"} else ""
        suffix += f" score={score:.6f}" if isinstance(score, (float, int)) else ""
        print(f"[sandbox]   {status}{suffix}")

    required_failures = [r for r in results if (not r.get("ok")) and (r.get("case") not in allow_fail_cases)]
    optional_failures = [r for r in results if (not r.get("ok")) and (r.get("case") in allow_fail_cases)]
    passed = [r for r in results if r.get("ok")]

    report = {
        "timestamp_unix": time.time(),
        "host": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cases_requested": requested_cases,
        "allow_fail_cases": sorted(allow_fail_cases),
        "num_cases": len(results),
        "num_passed": len(passed),
        "num_failed_required": len(required_failures),
        "num_failed_optional": len(optional_failures),
        "duration_sec": round(time.perf_counter() - started, 3),
        "results": results,
    }

    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[sandbox] report: {out_path}")
    print(
        f"[sandbox] summary: pass={len(passed)} "
        f"required_fail={len(required_failures)} optional_fail={len(optional_failures)}"
    )
    return 0 if not required_failures else 2


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reward backend sandbox checker.")
    p.add_argument("--cases", default="imagereward,hpsv3_imscore,hpsv3_official")
    p.add_argument("--allow-fail-cases", default="hpsv3_official")
    p.add_argument("--device", default="cuda")
    p.add_argument("--prompt", default="a cinematic portrait of a woman in soft rim light, 85mm, ultra detailed")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--timeout-sec", type=float, default=1800.0)
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--out-json", default="./reward_env_sandbox_report.json")
    p.add_argument("--image-reward-model", default="ImageReward-v1.0")
    p.add_argument("--pickscore-model", default="yuvalkirstain/PickScore_v1")
    p.add_argument("--unifiedreward-model", default="CodeGoat24/UnifiedReward-qwen-7b")
    p.add_argument("--list-cases", action="store_true")
    p.add_argument("--force-local-reward", action="store_true", default=True)
    p.add_argument("--strict-device", action="store_true")
    p.add_argument("--worker-case", default=None, help=argparse.SUPPRESS)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.list_cases:
        print("Available cases:")
        for name in sorted(_CASES):
            spec = _CASES[name]
            print(f"  - {name:20s} backend={spec.backend:12s} note={spec.note}")
        return

    if args.worker_case:
        raise SystemExit(_worker(args))
    raise SystemExit(_coordinator(args))


if __name__ == "__main__":
    main()
