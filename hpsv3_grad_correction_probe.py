#!/usr/bin/env python3
"""
HPSv3 gradient-correction feasibility probe.

Purpose:
- Measure memory/time for HPSv3 judge-only scoring vs. gradient scoring.
- Capture OOM/failure evidence in JSON for slide/report use.

Notes:
- Gradient mode is implemented for imscore HPSv3 (tensor-input path).
- "official" hpsv3 inferencer is judge-only in this probe.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe HPSv3 gradient-correction memory behavior.")
    p.add_argument("--impl", choices=["imscore", "official"], default="imscore")
    p.add_argument("--model_id", default="RE-N-Y/hpsv3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--prompt", default="A photo of a cat sitting in a sink.")
    p.add_argument("--resolutions", default="224,512,1024", help="Comma-separated square resolutions.")
    p.add_argument("--modes", default="judge,grad", help="Comma-separated: judge,grad")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float32")
    p.add_argument("--reserve_gb", type=float, default=0.0, help="Optional VRAM pre-reserve to mimic SD model residency.")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", default="hpsv3_grad_probe.json")
    return p.parse_args()


def parse_list(raw: str) -> list[str]:
    vals = []
    for tok in raw.replace(",", " ").split():
        t = tok.strip()
        if t:
            vals.append(t)
    return vals


def parse_resolutions(raw: str) -> list[int]:
    out: list[int] = []
    for tok in parse_list(raw):
        try:
            out.append(int(tok))
        except ValueError:
            pass
    out = sorted(set([v for v in out if v > 0]))
    return out if out else [224, 512, 1024]


def as_dtype(name: str) -> torch.dtype:
    m = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return m[name]


def bytes_to_gib(x: int) -> float:
    return float(x) / float(1024**3)


def reset_cuda_peak(device: str) -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch.device(device))


def cuda_mem_snapshot(device: str) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "allocated_gib": 0.0,
            "reserved_gib": 0.0,
            "max_allocated_gib": 0.0,
            "max_reserved_gib": 0.0,
        }
    d = torch.device(device)
    return {
        "allocated_gib": bytes_to_gib(torch.cuda.memory_allocated(d)),
        "reserved_gib": bytes_to_gib(torch.cuda.memory_reserved(d)),
        "max_allocated_gib": bytes_to_gib(torch.cuda.max_memory_allocated(d)),
        "max_reserved_gib": bytes_to_gib(torch.cuda.max_memory_reserved(d)),
    }


def find_first_tensor(obj: Any) -> torch.Tensor | None:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            t = find_first_tensor(v)
            if t is not None:
                return t
    if isinstance(obj, (list, tuple)):
        for v in obj:
            t = find_first_tensor(v)
            if t is not None:
                return t
    return None


def extract_scalar_tensor(obj: Any, device: torch.device, require_grad_tensor: bool) -> torch.Tensor | None:
    t = find_first_tensor(obj)
    if t is not None:
        s = t.reshape(-1)[0]
        if require_grad_tensor and not s.requires_grad:
            return None
        return s
    if isinstance(obj, (float, int)):
        # Scalar fallback: useful for judge mode; not differentiable for grad mode.
        if require_grad_tensor:
            return None
        return torch.tensor(float(obj), device=device)
    return None


def load_model(impl: str, model_id: str, device: str) -> tuple[Any, str]:
    if impl == "imscore":
        from imscore.hpsv3.model import HPSv3 as IMSCoreHPSv3  # noqa: PLC0415

        model = IMSCoreHPSv3.from_pretrained(model_id)
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        # Gradient correction only needs dR/dx, not dR/dtheta.
        if isinstance(model, torch.nn.Module):
            for p in model.parameters():
                p.requires_grad_(False)
        return model, "imscore"

    # official hpsv3 inferencer
    import hpsv3  # noqa: PLC0415
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    ckpt = hf_hub_download("MizzenAI/HPSv3", "HPSv3.safetensors", repo_type="model")
    inferencer = hpsv3.HPSv3RewardInferencer(device=device, checkpoint_path=ckpt)
    return inferencer, "official"


def call_imscore(model: Any, pixels: torch.Tensor, prompt: str, require_grad_tensor: bool) -> tuple[torch.Tensor, str]:
    attempts: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = [
        ("score", (pixels, [prompt]), {}),
        ("score", (pixels, prompt), {}),
        ("reward", (pixels, [prompt]), {}),
        ("reward", (pixels, prompt), {}),
        ("__call__", (pixels, [prompt]), {}),
        ("__call__", (pixels, prompt), {}),
        ("score", (), {"pixels": pixels, "prompts": [prompt]}),
        ("reward", (), {"pixels": pixels, "prompts": [prompt]}),
        ("__call__", (), {"pixels": pixels, "prompts": [prompt]}),
    ]
    errs: list[str] = []
    for name, args, kwargs in attempts:
        fn = model if name == "__call__" else getattr(model, name, None)
        if fn is None:
            continue
        try:
            out = fn(*args, **kwargs)
            s = extract_scalar_tensor(out, pixels.device, require_grad_tensor=require_grad_tensor)
            if s is None:
                errs.append(f"{name}: non-diff or non-scalar output")
                continue
            return s, f"{name}(args={len(args)} kwargs={list(kwargs.keys())})"
        except Exception as exc:  # noqa: BLE001
            errs.append(f"{name}: {type(exc).__name__}: {str(exc)[:180]}")
    msg = " | ".join(errs[:6]) if errs else "no callable API found"
    raise RuntimeError(f"Unable to score via imscore HPSv3. {msg}")


def call_official_judge(inferencer: Any, pixels: torch.Tensor, prompt: str) -> tuple[torch.Tensor, str]:
    # official inferencer is path/PIL based; in this probe we convert tensor to PIL-like temp via CPU.
    # To keep this script lightweight and dependency-free, use tensor->numpy->PIL only when needed.
    from PIL import Image  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    p = pixels.detach().clamp(0, 1)
    arr = (p[0].permute(1, 2, 0).cpu().float().numpy() * 255.0).astype("uint8")
    img = Image.fromarray(arr)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    img.save(path)

    try:
        # Try common official API signatures.
        for fn_name in ("reward", "score", "__call__"):
            fn = inferencer if fn_name == "__call__" else getattr(inferencer, fn_name, None)
            if fn is None:
                continue
            for kwargs in (
                {"prompts": [prompt], "image_paths": [path]},
                {"image_paths": [path], "prompts": [prompt]},
            ):
                try:
                    out = fn(**kwargs)
                    s = extract_scalar_tensor(out, pixels.device, require_grad_tensor=False)
                    if s is not None:
                        return s, f"{fn_name}(official)"
                except Exception:
                    pass
        raise RuntimeError("No official inferencer callable succeeded.")
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


def reserve_vram_gb(device: str, reserve_gb: float, dtype: torch.dtype) -> tuple[torch.Tensor | None, str]:
    if reserve_gb <= 0:
        return None, "none"
    if not torch.cuda.is_available():
        return None, "cuda_unavailable"
    num_bytes = int(reserve_gb * (1024**3))
    elem = torch.tensor([], dtype=dtype).element_size()
    numel = max(1, num_bytes // elem)
    t = torch.empty(numel, dtype=dtype, device=device)
    # Materialize pages.
    t.zero_()
    return t, f"{bytes_to_gib(t.numel() * t.element_size()):.3f}GiB"


def run_once(
    model: Any,
    impl: str,
    device: str,
    prompt: str,
    resolution: int,
    batch_size: int,
    dtype: torch.dtype,
    mode: str,
    reserve_holder: torch.Tensor | None,
) -> dict[str, Any]:
    del reserve_holder  # kept alive by caller scope
    info: dict[str, Any] = {
        "mode": mode,
        "resolution": resolution,
        "batch_size": batch_size,
        "dtype": str(dtype).replace("torch.", ""),
        "status": "ok",
    }
    reset_cuda_peak(device)
    t0 = time.perf_counter()

    try:
        pixels = torch.rand(
            (batch_size, 3, resolution, resolution),
            device=device,
            dtype=dtype,
        )
        api = ""
        if mode == "judge":
            with torch.no_grad():
                if impl == "imscore":
                    score_t, api = call_imscore(model, pixels, prompt, require_grad_tensor=False)
                else:
                    score_t, api = call_official_judge(model, pixels, prompt)
            info["score"] = float(score_t.detach().float().item())
            info["grad_norm"] = None
        elif mode == "grad":
            if impl != "imscore":
                raise RuntimeError("Gradient probe is supported only for impl=imscore in this script.")
            pixels = pixels.clone().detach().requires_grad_(True)
            score_t, api = call_imscore(model, pixels, prompt, require_grad_tensor=True)
            score_t.float().backward()
            g = pixels.grad
            info["score"] = float(score_t.detach().float().item())
            info["grad_norm"] = float(g.detach().float().norm().item()) if g is not None else None
        else:
            raise RuntimeError(f"Unknown mode: {mode}")

        info["api"] = api
    except RuntimeError as exc:
        msg = str(exc)
        low = msg.lower()
        if "out of memory" in low or "cuda error: out of memory" in low:
            info["status"] = "oom"
            info["error"] = msg
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            info["status"] = "error"
            info["error"] = f"{type(exc).__name__}: {msg}"
            info["traceback"] = traceback.format_exc(limit=2)
    except Exception as exc:  # noqa: BLE001
        info["status"] = "error"
        info["error"] = f"{type(exc).__name__}: {exc}"
        info["traceback"] = traceback.format_exc(limit=2)
    finally:
        info["elapsed_sec"] = float(time.perf_counter() - t0)
        info["cuda_mem"] = cuda_mem_snapshot(device)
    return info


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = str(args.device)
    modes = parse_list(args.modes)
    resolutions = parse_resolutions(args.resolutions)
    dtype = as_dtype(args.dtype)

    report: dict[str, Any] = {
        "impl": args.impl,
        "model_id": args.model_id,
        "device": device,
        "prompt": args.prompt,
        "resolutions": resolutions,
        "modes": modes,
        "batch_size": int(args.batch_size),
        "dtype": str(dtype).replace("torch.", ""),
        "reserve_gb": float(args.reserve_gb),
        "repeats": int(args.repeats),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "results": [],
    }

    print(f"[probe] impl={args.impl} model_id={args.model_id} device={device}")
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    model, loaded_impl = load_model(args.impl, args.model_id, device)
    report["loaded_impl"] = loaded_impl
    print(f"[probe] loaded_impl={loaded_impl}")

    reserve_holder, reserve_msg = reserve_vram_gb(device, float(args.reserve_gb), dtype=torch.float16)
    report["reserve_actual"] = reserve_msg
    if reserve_msg != "none":
        print(f"[probe] reserve_vram={reserve_msg}")

    for rep in range(int(args.repeats)):
        for res in resolutions:
            for mode in modes:
                print(f"[probe] repeat={rep+1}/{args.repeats} mode={mode} res={res}")
                out = run_once(
                    model=model,
                    impl=loaded_impl,
                    device=device,
                    prompt=args.prompt,
                    resolution=res,
                    batch_size=int(args.batch_size),
                    dtype=dtype,
                    mode=mode,
                    reserve_holder=reserve_holder,
                )
                out["repeat"] = rep + 1
                report["results"].append(out)
                status = out.get("status", "?")
                mem = out.get("cuda_mem", {})
                peak = float(mem.get("max_allocated_gib", 0.0))
                print(f"  -> status={status} peak_alloc={peak:.3f}GiB elapsed={out.get('elapsed_sec', 0.0):.2f}s")
                if status != "ok":
                    print(f"     error={out.get('error', '')}")

    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[probe] wrote {out_path}")


if __name__ == "__main__":
    main()

