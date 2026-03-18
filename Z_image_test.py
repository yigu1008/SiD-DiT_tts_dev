import argparse
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

from reward_unified import UnifiedRewardScorer


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
        "--guidance_scales",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5],
        help="CFG values to sweep",
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

    saved_images: List[Image.Image] = []
    summary_stats: List[Dict[str, Any]] = []
    base_img_np = None

    for gs in args.guidance_scales:
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
            cfg_dir = os.path.join(args.outdir, f"cfg_{gs:.2f}_steps")
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
                labeled = add_label(step_image, f"cfg={gs:.2f} step={step_idx} t={record['timestep']} IR={label_ir}")
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

        fname = os.path.join(args.outdir, f"cfg_{gs:.2f}.png")
        image.save(fname)
        print(f"Saved: {fname}")
        if final_ir is not None:
            print(f"Final ImageReward: {final_ir:.4f}")

        if callback_supported:
            write_cfg_intermediate_stats(args.outdir, gs, per_cfg_step_records)
            save_intermediate_grid(args.outdir, gs, per_cfg_step_images)

        img_np = pil_to_np(image)
        if gs == 0.0:
            base_img_np = img_np
            delta_l1 = 0.0
            delta_l2 = 0.0
        else:
            if base_img_np is None:
                raise RuntimeError("Please include 0.0 in guidance_scales for comparison.")
            diff = img_np - base_img_np
            delta_l1 = float(np.mean(np.abs(diff)))
            delta_l2 = float(np.sqrt(np.mean(diff ** 2)))

        summary_stats.append(
            {
                "guidance_scale": float(gs),
                "l1_vs_cfg0": delta_l1,
                "rmse_vs_cfg0": delta_l2,
                "final_imagereward": final_ir,
                "n_intermediate_logged": len(per_cfg_step_records),
            }
        )

        ir_text = "n/a" if final_ir is None else f"{final_ir:.4f}"
        labeled = add_label(
            image,
            f"cfg={gs:.2f} | L1_vs_cfg0={delta_l1:.5f} | RMSE_vs_cfg0={delta_l2:.5f} | FinalIR={ir_text}",
        )
        saved_images.append(labeled)

    grid = make_grid(saved_images, cols=min(3, len(saved_images)))
    grid_path = os.path.join(args.outdir, "cfg_grid.png")
    grid.save(grid_path)
    print(f"\nSaved grid: {grid_path}")

    txt_path = os.path.join(args.outdir, "cfg_stats.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("guidance_scale\tL1_vs_cfg0\tRMSE_vs_cfg0\tFinalImageReward\tN_IntermediateLogged\n")
        for row in summary_stats:
            ir = row["final_imagereward"]
            ir_text = f"{ir:.8f}" if ir is not None else "nan"
            f.write(
                f"{row['guidance_scale']:.2f}\t"
                f"{row['l1_vs_cfg0']:.8f}\t"
                f"{row['rmse_vs_cfg0']:.8f}\t"
                f"{ir_text}\t"
                f"{row['n_intermediate_logged']}\n"
            )
    print(f"Saved stats: {txt_path}")

    json_path = os.path.join(args.outdir, "cfg_stats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"Saved stats json: {json_path}")

    print("\nSummary:")
    for row in summary_stats:
        ir = row["final_imagereward"]
        ir_text = "n/a" if ir is None else f"{ir:.4f}"
        print(
            f"  cfg={row['guidance_scale']:.2f} | "
            f"L1_vs_cfg0={row['l1_vs_cfg0']:.6f} | "
            f"RMSE_vs_cfg0={row['rmse_vs_cfg0']:.6f} | "
            f"FinalIR={ir_text} | "
            f"IntermediateLogged={row['n_intermediate_logged']}"
        )


if __name__ == "__main__":
    main()
