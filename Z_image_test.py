import argparse
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from reward_unified import UnifiedRewardScorer


def parse_bool_token(value: str) -> bool:
    text = value.strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool token: {value}. Use one of true/false.")


def parse_dynamic_preset_token(value: str) -> Tuple[float, float, str]:
    parts = [p.strip() for p in re.split(r"[,/]", value) if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Invalid dynamic preset: {value}. Expected base_shift,max_shift,time_shift_type"
        )
    base_shift = float(parts[0])
    max_shift = float(parts[1])
    time_shift_type = parts[2]
    return (base_shift, max_shift, time_shift_type)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Tongyi-MAI/Z-Image-Turbo",
        help="HF model id",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt to test",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Optional negative prompt. Mostly useful when cfg > 1.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="zimage_turbo_cfg_test",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=9,
        help="num_inference_steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="Fixed CFG/guidance scale (default Turbo setting).",
    )
    parser.add_argument(
        "--enable_cfg_sweep",
        action="store_true",
        help="Enable CFG sweep ablation. If unset, uses only --guidance_scale.",
    )
    parser.add_argument(
        "--guidance_scales",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="CFG values for ablation sweep when --enable_cfg_sweep is set.",
    )
    parser.add_argument(
        "--attention",
        type=str,
        default="",
        choices=["", "flash", "_flash_3"],
        help="Optional attention backend",
    )
    parser.add_argument(
        "--compile_transformer",
        action="store_true",
        help="Compile transformer for speed after first run",
    )
    parser.add_argument(
        "--scheduler_test_theme",
        type=str,
        choices=["none", "shift", "stochastic", "dynamic", "sigma", "all"],
        default="none",
        help="FlowMatchEuler scheduler test theme.",
    )
    parser.add_argument(
        "--scheduler_shifts",
        type=float,
        nargs="+",
        default=[0.8, 1.0, 1.2, 1.4],
        help="Shift sweep values when scheduler_test_theme includes shift.",
    )
    parser.add_argument(
        "--scheduler_stochastic_values",
        type=parse_bool_token,
        nargs="+",
        default=[False, True],
        help="stochastic_sampling values for flow-match sweep.",
    )
    parser.add_argument(
        "--scheduler_dynamic_presets",
        type=parse_dynamic_preset_token,
        nargs="+",
        default=[
            (0.5, 1.15, "exponential"),
            (0.7, 1.30, "exponential"),
            (0.5, 1.15, "linear"),
        ],
        help="Dynamic shift presets as base_shift,max_shift,time_shift_type",
    )
    parser.add_argument(
        "--scheduler_sigma_families",
        type=str,
        nargs="+",
        choices=["default", "karras", "exponential", "beta"],
        default=["default", "karras", "exponential", "beta"],
        help="Sigma schedule family sweep for FlowMatchEuler.",
    )
    parser.add_argument(
        "--log_intermediates",
        action="store_true",
        help="Try to log intermediate states via callback_on_step_end.",
    )
    parser.add_argument(
        "--save_intermediate_images",
        action="store_true",
        help="Save decoded intermediate images per cfg sweep.",
    )
    parser.add_argument(
        "--score_intermediates",
        action="store_true",
        help="Compute ImageReward for intermediate images (expensive).",
    )
    parser.add_argument(
        "--reward_model",
        type=str,
        default="ImageReward-v1.0",
        help="ImageReward checkpoint name.",
    )
    parser.add_argument(
        "--reward_backend",
        type=str,
        choices=["auto", "imagereward", "hpsv2", "unified"],
        default="auto",
        help="Reward backend selector.",
    )
    parser.add_argument(
        "--reward_weights",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        help="Unified backend weights: imagereward hpsv2",
    )
    parser.add_argument(
        "--no_image_reward",
        action="store_true",
        help="Disable ImageReward scoring entirely.",
    )
    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    return torch.float32


def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(np.float32) / 255.0


def add_label(img: Image.Image, text: str, pad: int = 40) -> Image.Image:
    w, h = img.size
    canvas = Image.new("RGB", (w, h + pad), color=(255, 255, 255))
    canvas.paste(img, (0, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), text, fill=(0, 0, 0))
    return canvas


def make_grid(images: List[Image.Image], cols: int = 3) -> Image.Image:
    if not images:
        raise ValueError("images must be non-empty")
    w, h = images[0].size
    rows = math.ceil(len(images) / cols)
    grid = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
    for i, img in enumerate(images):
        x = (i % cols) * w
        y = (i // cols) * h
        grid.paste(img, (x, y))
    return grid


def decode_latents_to_pil(pipe: Any, latents: torch.Tensor) -> Image.Image:
    with torch.inference_mode():
        vae_param = next(pipe.vae.parameters())
        vae_dtype = vae_param.dtype
        vae_device = vae_param.device
        latents = latents.to(device=vae_device, dtype=vae_dtype)
        scaling = getattr(pipe.vae.config, "scaling_factor", 1.0)
        shift = getattr(pipe.vae.config, "shift_factor", 0.0)
        decoded = pipe.vae.decode((latents / scaling) + shift, return_dict=False)[0]
        image = pipe.image_processor.postprocess(decoded, output_type="pil")
    return image[0]


def score_image(reward_scorer: Optional[UnifiedRewardScorer], prompt: str, image: Image.Image) -> Optional[float]:
    if reward_scorer is None:
        return None
    return float(reward_scorer.score(prompt, image))


def write_cfg_intermediate_stats(
    outdir: str,
    cfg_value: float,
    records: List[Dict[str, Any]],
) -> None:
    if not records:
        return
    path = os.path.join(outdir, f"cfg_{cfg_value:.2f}_intermediate_stats.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("step_idx\ttimestep\timagereward\n")
        for rec in records:
            ir = rec["imagereward"]
            ir_text = f"{ir:.8f}" if ir is not None else "nan"
            f.write(f"{rec['step_idx']}\t{rec['timestep']}\t{ir_text}\n")
    print(f"Saved intermediate stats: {path}")


def save_intermediate_grid(
    outdir: str,
    cfg_value: float,
    images: List[Image.Image],
    cols: int = 3,
) -> None:
    if not images:
        return
    grid = make_grid(images, cols=min(cols, len(images)))
    path = os.path.join(outdir, f"cfg_{cfg_value:.2f}_intermediate_grid.png")
    grid.save(path)
    print(f"Saved intermediate grid: {path}")


def scheduler_config_to_dict(scheduler: Any) -> Dict[str, Any]:
    cfg = scheduler.config
    if hasattr(cfg, "to_dict"):
        return dict(cfg.to_dict())
    return dict(cfg)


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "variant"


def build_scheduler_variants(args: argparse.Namespace, scheduler: Any) -> List[Dict[str, Any]]:
    base_config = scheduler_config_to_dict(scheduler)
    supported_keys = set(base_config.keys())
    scheduler_name = scheduler.__class__.__name__
    if scheduler_name != "FlowMatchEulerDiscreteScheduler":
        print(
            f"Warning: current scheduler is {scheduler_name}. "
            "Scheduler tests target FlowMatchEulerDiscreteScheduler fields."
        )

    theme = args.scheduler_test_theme
    themes = ["shift", "stochastic", "dynamic", "sigma"] if theme == "all" else ([] if theme == "none" else [theme])

    variants: List[Dict[str, Any]] = []
    seen_payloads = set()

    def add_variant(name: str, overrides: Dict[str, Any]) -> None:
        filtered: Dict[str, Any] = {}
        dropped: List[str] = []
        for key, value in overrides.items():
            if key in supported_keys:
                filtered[key] = value
            else:
                dropped.append(key)
        if dropped:
            print(f"Skipping unsupported scheduler keys for {name}: {', '.join(dropped)}")
        payload = json.dumps(filtered, sort_keys=True, default=str)
        if payload in seen_payloads:
            return
        seen_payloads.add(payload)
        variants.append(
            {
                "name": name,
                "slug": slugify(name),
                "overrides": filtered,
            }
        )

    add_variant("default", {})

    if "shift" in themes:
        for shift in args.scheduler_shifts:
            add_variant(f"shift_{shift:g}", {"shift": float(shift)})

    if "stochastic" in themes:
        for stochastic in args.scheduler_stochastic_values:
            tag = "on" if stochastic else "off"
            add_variant(f"stochastic_{tag}", {"stochastic_sampling": bool(stochastic)})

    if "dynamic" in themes:
        add_variant("dynamic_off", {"use_dynamic_shifting": False})
        for base_shift, max_shift, time_shift_type in args.scheduler_dynamic_presets:
            add_variant(
                f"dynamic_on_b{base_shift:g}_m{max_shift:g}_{time_shift_type}",
                {
                    "use_dynamic_shifting": True,
                    "base_shift": float(base_shift),
                    "max_shift": float(max_shift),
                    "time_shift_type": str(time_shift_type),
                },
            )

    if "sigma" in themes:
        for family in args.scheduler_sigma_families:
            if family == "default":
                overrides = {
                    "use_karras_sigmas": False,
                    "use_exponential_sigmas": False,
                    "use_beta_sigmas": False,
                }
            elif family == "karras":
                overrides = {
                    "use_karras_sigmas": True,
                    "use_exponential_sigmas": False,
                    "use_beta_sigmas": False,
                }
            elif family == "exponential":
                overrides = {
                    "use_karras_sigmas": False,
                    "use_exponential_sigmas": True,
                    "use_beta_sigmas": False,
                }
            else:
                overrides = {
                    "use_karras_sigmas": False,
                    "use_exponential_sigmas": False,
                    "use_beta_sigmas": True,
                }
            add_variant(f"sigma_{family}", overrides)

    return variants


def apply_scheduler_variant(pipe: Any, variant: Dict[str, Any]) -> Dict[str, Any]:
    scheduler_cls = pipe.scheduler.__class__
    config = scheduler_config_to_dict(pipe.scheduler)
    config.update(variant["overrides"])

    sigma_keys = ("use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas")
    enabled_sigma = [key for key in sigma_keys if bool(config.get(key, False))]
    if len(enabled_sigma) > 1:
        preferred = None
        for key in sigma_keys:
            if bool(variant["overrides"].get(key, False)):
                preferred = key
                break
        if preferred is None:
            preferred = enabled_sigma[0]
        for key in sigma_keys:
            if key in config:
                config[key] = key == preferred

    pipe.scheduler = scheduler_cls.from_config(config)
    applied = {key: config[key] for key in variant["overrides"].keys() if key in config}
    return applied


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from diffusers import ZImagePipeline

    device = "cuda"
    dtype = get_dtype(args.dtype)
    reward_scorer = None
    if not args.no_image_reward:
        try:
            reward_scorer = UnifiedRewardScorer(
                device=device,
                backend=args.reward_backend,
                image_reward_model=args.reward_model,
                unified_weights=(float(args.reward_weights[0]), float(args.reward_weights[1])),
            )
            print(f"Reward: {reward_scorer.describe()}")
        except Exception as exc:
            print(f"Reward model unavailable ({exc}); scoring disabled.")
            reward_scorer = None

    print(f"Loading pipeline: {args.model}")
    pipe = ZImagePipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    ).to(device)

    if args.attention:
        print(f"Setting attention backend: {args.attention}")
        pipe.transformer.set_attention_backend(args.attention)

    if args.compile_transformer:
        print("Compiling transformer...")
        pipe.transformer.compile()

    scheduler_variants = build_scheduler_variants(args, pipe.scheduler)
    print(f"Scheduler test theme: {args.scheduler_test_theme} | variants={len(scheduler_variants)}")
    for variant in scheduler_variants:
        print(f"  {variant['slug']}: {variant['overrides'] if variant['overrides'] else '(default)'}")

    cfg_values = list(args.guidance_scales) if args.enable_cfg_sweep else [float(args.guidance_scale)]
    print(
        f"Guidance mode: {'sweep' if args.enable_cfg_sweep else 'fixed'} | "
        f"values={', '.join(f'{x:.2f}' for x in cfg_values)}"
    )

    multiple_variants = len(scheduler_variants) > 1
    all_summary_stats: List[Dict[str, Any]] = []

    for variant_idx, variant in enumerate(scheduler_variants, start=1):
        variant_outdir = args.outdir if not multiple_variants else os.path.join(args.outdir, variant["slug"])
        os.makedirs(variant_outdir, exist_ok=True)

        applied_overrides = apply_scheduler_variant(pipe, variant)
        print(
            f"\n[{variant_idx}/{len(scheduler_variants)}] Scheduler variant: {variant['slug']} "
            f"applied={applied_overrides if applied_overrides else '(default)'}"
        )

        cfg_ref = float(cfg_values[0])
        saved_images: List[Image.Image] = []
        summary_stats: List[Dict[str, Any]] = []
        base_img_np: Optional[np.ndarray] = None

        for gs in cfg_values:
            print(f"\nRunning guidance_scale={gs}")
            generator = torch.Generator(device).manual_seed(args.seed)

            kwargs: Dict[str, Any] = dict(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=float(gs),
                generator=generator,
            )
            if args.negative_prompt:
                kwargs["negative_prompt"] = args.negative_prompt

            per_cfg_step_records: List[Dict[str, Any]] = []
            per_cfg_step_images: List[Image.Image] = []
            callback_supported = False

            if args.log_intermediates:
                cfg_dir = os.path.join(variant_outdir, f"cfg_{gs:.2f}_steps")
                if args.save_intermediate_images:
                    os.makedirs(cfg_dir, exist_ok=True)

                def _on_step_end(_pipe, step_idx: int, timestep, callback_kwargs):
                    latents = callback_kwargs.get("latents", None)
                    if latents is None:
                        return callback_kwargs

                    step_image = decode_latents_to_pil(pipe, latents)
                    step_ir = None
                    if args.score_intermediates:
                        step_ir = score_image(reward_scorer, args.prompt, step_image)

                    record = {
                        "step_idx": int(step_idx),
                        "timestep": float(timestep) if hasattr(timestep, "__float__") else str(timestep),
                        "imagereward": step_ir,
                    }
                    per_cfg_step_records.append(record)

                    label_ir = "n/a" if step_ir is None else f"{step_ir:.4f}"
                    labeled = add_label(
                        step_image,
                        f"{variant['slug']} cfg={gs:.2f} step={step_idx} t={record['timestep']} IR={label_ir}",
                    )
                    per_cfg_step_images.append(labeled)

                    if args.save_intermediate_images:
                        fname = os.path.join(cfg_dir, f"step_{int(step_idx):03d}.png")
                        step_image.save(fname)
                    return callback_kwargs

                kwargs["callback_on_step_end"] = _on_step_end
                kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

            with torch.inference_mode():
                try:
                    result = pipe(**kwargs)
                    callback_supported = args.log_intermediates
                except TypeError as exc:
                    # Some pipeline versions may not expose callback_on_step_end.
                    if args.log_intermediates and "callback_on_step_end" in str(exc):
                        print("This ZImagePipeline version does not support callback_on_step_end; running without intermediates.")
                        kwargs.pop("callback_on_step_end", None)
                        kwargs.pop("callback_on_step_end_tensor_inputs", None)
                        result = pipe(**kwargs)
                        callback_supported = False
                    else:
                        raise

            image = result.images[0]
            final_ir = score_image(reward_scorer, args.prompt, image)

            fname = os.path.join(variant_outdir, f"cfg_{gs:.2f}.png")
            image.save(fname)
            print(f"Saved: {fname}")
            if final_ir is not None:
                print(f"Final ImageReward: {final_ir:.4f}")

            if callback_supported:
                write_cfg_intermediate_stats(variant_outdir, gs, per_cfg_step_records)
                save_intermediate_grid(variant_outdir, gs, per_cfg_step_images)

            img_np = pil_to_np(image)
            if base_img_np is None:
                base_img_np = img_np
                delta_l1 = 0.0
                delta_l2 = 0.0
            else:
                diff = img_np - base_img_np
                delta_l1 = float(np.mean(np.abs(diff)))
                delta_l2 = float(np.sqrt(np.mean(diff ** 2)))

            summary_row = {
                "scheduler_variant": variant["slug"],
                "scheduler_overrides": applied_overrides,
                "guidance_scale": float(gs),
                "cfg_reference": cfg_ref,
                "l1_vs_cfg_ref": delta_l1,
                "rmse_vs_cfg_ref": delta_l2,
                "final_imagereward": final_ir,
                "n_intermediate_logged": len(per_cfg_step_records),
            }
            summary_stats.append(summary_row)
            all_summary_stats.append(summary_row)

            ir_text = "n/a" if final_ir is None else f"{final_ir:.4f}"
            labeled = add_label(
                image,
                f"{variant['slug']} cfg={gs:.2f} | L1_vs_cfg{cfg_ref:.2f}={delta_l1:.5f} | "
                f"RMSE_vs_cfg{cfg_ref:.2f}={delta_l2:.5f} | FinalIR={ir_text}",
            )
            saved_images.append(labeled)

        grid = make_grid(saved_images, cols=min(3, len(saved_images)))
        grid_path = os.path.join(variant_outdir, "cfg_grid.png")
        grid.save(grid_path)
        print(f"\nSaved grid: {grid_path}")

        txt_path = os.path.join(variant_outdir, "cfg_stats.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("scheduler_variant\tguidance_scale\tcfg_reference\tL1_vs_cfg_ref\tRMSE_vs_cfg_ref\tFinalImageReward\tN_IntermediateLogged\n")
            for row in summary_stats:
                ir = row["final_imagereward"]
                ir_text = f"{ir:.8f}" if ir is not None else "nan"
                f.write(
                    f"{row['scheduler_variant']}\t"
                    f"{row['guidance_scale']:.2f}\t"
                    f"{row['cfg_reference']:.2f}\t"
                    f"{row['l1_vs_cfg_ref']:.8f}\t"
                    f"{row['rmse_vs_cfg_ref']:.8f}\t"
                    f"{ir_text}\t"
                    f"{row['n_intermediate_logged']}\n"
                )
        print(f"Saved stats: {txt_path}")

        json_path = os.path.join(variant_outdir, "cfg_stats.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_stats, f, indent=2)
        print(f"Saved stats json: {json_path}")

        print("\nSummary:")
        for row in summary_stats:
            ir = row["final_imagereward"]
            ir_text = "n/a" if ir is None else f"{ir:.4f}"
            print(
                f"  scheduler={row['scheduler_variant']} | cfg={row['guidance_scale']:.2f} | "
                f"L1_vs_cfg{row['cfg_reference']:.2f}={row['l1_vs_cfg_ref']:.6f} | "
                f"RMSE_vs_cfg{row['cfg_reference']:.2f}={row['rmse_vs_cfg_ref']:.6f} | "
                f"FinalIR={ir_text} | IntermediateLogged={row['n_intermediate_logged']}"
            )

    if multiple_variants:
        combined_txt_path = os.path.join(args.outdir, "scheduler_cfg_stats.txt")
        with open(combined_txt_path, "w", encoding="utf-8") as f:
            f.write(
                "scheduler_variant\tguidance_scale\tcfg_reference\tL1_vs_cfg_ref\t"
                "RMSE_vs_cfg_ref\tFinalImageReward\tN_IntermediateLogged\n"
            )
            for row in all_summary_stats:
                ir = row["final_imagereward"]
                ir_text = f"{ir:.8f}" if ir is not None else "nan"
                f.write(
                    f"{row['scheduler_variant']}\t"
                    f"{row['guidance_scale']:.2f}\t"
                    f"{row['cfg_reference']:.2f}\t"
                    f"{row['l1_vs_cfg_ref']:.8f}\t"
                    f"{row['rmse_vs_cfg_ref']:.8f}\t"
                    f"{ir_text}\t"
                    f"{row['n_intermediate_logged']}\n"
                )
        print(f"\nSaved combined stats: {combined_txt_path}")

        combined_json_path = os.path.join(args.outdir, "scheduler_cfg_stats.json")
        with open(combined_json_path, "w", encoding="utf-8") as f:
            json.dump(all_summary_stats, f, indent=2)
        print(f"Saved combined stats json: {combined_json_path}")


if __name__ == "__main__":
    main()
