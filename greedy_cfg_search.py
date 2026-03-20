"""
greedy_cfg_search.py
====================
Greedy per-step search over (cfg_scale, use_neg_embed) for SiD SANA.

Key memory fixes:
- Reward model defaults to CPU.
- If reward runs on CUDA, move reward to CPU before each decode and back before scoring.
- Offload text encoders to CPU after prompt encoding.
- Empty CUDA cache before decode.

Usage:
  python greedy_cfg_search.py --prompt "..."
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow")
    parser.add_argument(
        "--prompt",
        default="a studio portrait of an elderly woman smiling, soft window light, 85mm lens, photorealistic",
    )
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument("--neg_embed", default=None)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--cfg_scales",
        nargs="+",
        type=float,
        default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--time_scale", type=float, default=1000.0)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument(
        "--reward_device",
        default="cpu",
        help="cpu (recommended) | same | auto | cuda | cuda:N",
    )
    parser.add_argument(
        "--offload_text_encoder_after_encode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--empty_cache_before_decode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--out_dir", default="./greedy_out")
    return parser.parse_args()


def to_torch_dtype(name: str) -> torch.dtype:
    k = str(name).lower()
    if k == "bf16":
        return torch.bfloat16
    if k == "fp32":
        return torch.float32
    return torch.float16


def resolve_reward_device(req: str, model_device: str) -> str:
    r = str(req).strip().lower()
    if r in {"cpu"}:
        return "cpu"
    if r in {"same", "model", "auto"}:
        return model_device if model_device.startswith("cuda") else "cpu"
    if r == "cuda":
        return model_device if model_device.startswith("cuda") else "cuda:0"
    if r.startswith("cuda:"):
        return r
    raise RuntimeError(f"Unsupported --reward_device={req}")


def module_device(module: Any) -> str | None:
    try:
        return str(next(module.parameters()).device)
    except Exception:
        return None


class RewardRuntime:
    def __init__(self, model: Any, score_device: str, model_device: str):
        self.model = model
        self.score_device = score_device
        self.model_device = model_device
        self.runtime_device = score_device

    def _move(self, dst: str) -> None:
        dst = str(dst)
        if self.runtime_device == dst:
            return
        self.model.to(dst)
        if hasattr(self.model, "device"):
            try:
                self.model.device = dst
            except Exception:
                pass
        self.runtime_device = dst
        if self.model_device.startswith("cuda"):
            torch.cuda.empty_cache()

    def before_decode(self) -> None:
        if self.runtime_device.startswith("cuda"):
            self._move("cpu")

    def score(self, prompt: str, image: Image.Image) -> float:
        if self.score_device.startswith("cuda") and self.runtime_device != self.score_device:
            self._move(self.score_device)
        return float(self.model.score(prompt, image))


def move_text_encoders(pipe: Any, dst: str) -> None:
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        module = getattr(pipe, name, None)
        if module is None:
            continue
        cur = module_device(module)
        if cur != dst:
            module.to(dst)


def infer_latent_hw(pipe: Any, height: int, width: int) -> tuple[int, int, int]:
    scale = int(getattr(pipe, "vae_scale_factor", 0) or 0)
    if scale <= 1:
        try:
            enc_ch = getattr(pipe.vae.config, "encoder_block_out_channels", None)
            if enc_ch is not None and len(enc_ch) > 1:
                scale = int(2 ** (len(enc_ch) - 1))
        except Exception:
            pass
    if scale <= 1:
        scale = 32
    return max(1, int(height) // scale), max(1, int(width) // scale), scale


def make_latents(pipe: Any, device: str, latent_c: int, seed: int, h: int, w: int, dtype: torch.dtype) -> torch.Tensor:
    exp_h, exp_w, scale = infer_latent_hw(pipe, h, w)
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = pipe.prepare_latents(1, latent_c, h, w, dtype, device, generator)
    got_h, got_w = int(latents.shape[-2]), int(latents.shape[-1])
    if (got_h, got_w) == (int(h), int(w)) and (exp_h, exp_w) != (int(h), int(w)):
        print(
            "Warning: prepare_latents returned pixel-space latent "
            f"{got_h}x{got_w}; forcing latent-space {exp_h}x{exp_w} (scale={scale})."
        )
        latents = torch.randn((1, latent_c, exp_h, exp_w), device=device, dtype=dtype, generator=generator)
    return latents


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from sid import SiDSanaPipeline
    except ImportError as e:
        sys.exit(f"Cannot import SiDSanaPipeline: {e}")

    from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
        ASPECT_RATIO_512_BIN,
        ASPECT_RATIO_1024_BIN,
    )
    try:
        from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
    except Exception:
        ASPECT_RATIO_2048_BIN = ASPECT_RATIO_1024_BIN

    aspect_bins = {16: ASPECT_RATIO_512_BIN, 32: ASPECT_RATIO_1024_BIN, 64: ASPECT_RATIO_2048_BIN}

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(args.dtype)

    print("Loading pipeline ...")
    pipe = SiDSanaPipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    pipe.transformer.eval()
    latent_c = int(pipe.transformer.config.in_channels)
    out_c = int(getattr(pipe.transformer.config, "out_channels", latent_c))
    variance_split = (out_c // 2 == latent_c)
    print(f"  device={device} dtype={args.dtype} variance_split={variance_split}")

    print("Loading ImageReward ...")
    try:
        import transformers

        bert_cls = getattr(transformers, "BertModel", None)
        if bert_cls is not None and not hasattr(bert_cls, "all_tied_weights_keys"):
            bert_cls.all_tied_weights_keys = property(lambda self: getattr(self, "_tied_weights_keys", None))
    except Exception:
        pass

    import ImageReward as RM

    reward_device = resolve_reward_device(args.reward_device, device)
    reward_model = RM.load("ImageReward-v1.0", device=reward_device)
    reward_model.eval()
    reward_rt = RewardRuntime(reward_model, reward_device, device)
    print(f"  reward_device={reward_device}")

    pretrained_neg_embeds = None
    pretrained_neg_mask = None
    if args.neg_embed:
        ckpt = torch.load(args.neg_embed, map_location="cpu")
        if isinstance(ckpt, dict) and "neg_embeds" in ckpt:
            pretrained_neg_embeds = ckpt["neg_embeds"].to(device=device, dtype=dtype)
            pretrained_neg_mask = ckpt["neg_mask"].to(device=device)
        else:
            pretrained_neg_embeds = ckpt.to(device=device, dtype=dtype)
            pretrained_neg_mask = torch.ones(ckpt.shape[:2], device=device, dtype=torch.long)
        print(f"Loaded neg embedding: {tuple(pretrained_neg_embeds.shape)}")
    else:
        print("No --neg_embed provided.")

    if args.prompt_file:
        prompts = [line.strip() for line in open(args.prompt_file, encoding="utf-8") if line.strip()]
    else:
        prompts = [args.prompt]

    actions: list[tuple[float, bool]] = []
    for cfg in args.cfg_scales:
        cfg = float(cfg)
        actions.append((cfg, False))
        if pretrained_neg_embeds is not None and cfg != 1.0:
            actions.append((cfg, True))
    print(f"Action space: {len(actions)} x {args.steps}")

    @torch.no_grad()
    def decode_to_pil(dx: torch.Tensor, orig_h: int, orig_w: int) -> Image.Image:
        reward_rt.before_decode()
        if args.empty_cache_before_decode and device.startswith("cuda"):
            torch.cuda.empty_cache()
        image = pipe.vae.decode(dx / pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = pipe.image_processor.resize_and_crop_tensor(image, orig_h, orig_w)
        return pipe.image_processor.postprocess(image, output_type="pil")[0]

    @torch.no_grad()
    def prepare_embeds(prompt: str, max_seq_len: int = 256):
        if args.offload_text_encoder_after_encode and device.startswith("cuda"):
            move_text_encoders(pipe, device)
            torch.cuda.empty_cache()

        pe, pm, ne, nm = pipe.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=True,
            negative_prompt="",
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_seq_len,
        )

        pne = None
        pnm = None
        if pretrained_neg_embeds is not None and pretrained_neg_mask is not None:
            bsz, cond_len = pe.shape[0], pe.shape[1]
            pne = pretrained_neg_embeds.expand(bsz, -1, -1).to(dtype=pe.dtype, device=device)
            pnm = pretrained_neg_mask.expand(bsz, -1).to(device=device)
            neg_len = pne.shape[1]
            if neg_len < cond_len:
                pne = torch.cat(
                    [pne, torch.zeros(bsz, cond_len - neg_len, pne.shape[2], device=device, dtype=pne.dtype)],
                    dim=1,
                )
                pnm = torch.cat(
                    [pnm, torch.zeros(bsz, cond_len - neg_len, device=device, dtype=pnm.dtype)],
                    dim=1,
                )
            elif neg_len > cond_len:
                pne = pne[:, :cond_len]
                pnm = pnm[:, :cond_len]

        if args.offload_text_encoder_after_encode and device.startswith("cuda"):
            move_text_encoders(pipe, "cpu")
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

        return pe, pm, ne, nm, pne, pnm

    @torch.no_grad()
    def transformer_step(
        latents: torch.Tensor,
        pe: torch.Tensor,
        pm: torch.Tensor,
        ne: torch.Tensor,
        nm: torch.Tensor,
        pne: torch.Tensor | None,
        pnm: torch.Tensor | None,
        t_flat: torch.Tensor,
        guidance_scale: float,
        use_pretrained_neg: bool,
    ) -> torch.Tensor:
        if guidance_scale == 1.0:
            flow = pipe.transformer(
                hidden_states=latents,
                encoder_hidden_states=pe,
                encoder_attention_mask=pm,
                timestep=args.time_scale * t_flat,
                return_dict=False,
            )[0]
            if variance_split:
                flow = flow.chunk(2, dim=1)[0]
            return flow

        ue, um = (pne, pnm) if (use_pretrained_neg and pne is not None and pnm is not None) else (ne, nm)
        flow_both = pipe.transformer(
            hidden_states=torch.cat([latents, latents]),
            encoder_hidden_states=torch.cat([ue, pe]),
            encoder_attention_mask=torch.cat([um, pm]),
            timestep=args.time_scale * torch.cat([t_flat, t_flat]),
            return_dict=False,
        )[0]
        if variance_split:
            flow_both = flow_both.chunk(2, dim=1)[0]
        flow_uncond, flow_cond = flow_both.chunk(2, dim=0)
        return flow_uncond + guidance_scale * (flow_cond - flow_uncond)

    @torch.no_grad()
    def run_baseline(prompt: str, seed: int) -> tuple[Image.Image, float]:
        pe, pm, ne, nm, pne, pnm = prepare_embeds(prompt)
        del ne, nm, pne, pnm
        orig_h, orig_w = args.height, args.width
        h, w = orig_h, orig_w
        sample_size = int(pipe.transformer.config.sample_size)
        if sample_size in aspect_bins:
            h, w = pipe.image_processor.classify_height_width_bin(h, w, ratios=aspect_bins[sample_size])

        noise_g = torch.Generator(device=device).manual_seed(seed + 1001)
        latents = make_latents(pipe, device, latent_c, seed, h, w, pe.dtype)
        dx = torch.zeros_like(latents)

        for i in range(args.steps):
            scalar_t = 999.0 * (1.0 - float(i) / float(args.steps))
            t_flat = torch.full((latents.shape[0],), scalar_t / 999.0, device=device, dtype=latents.dtype)
            t_4d = t_flat.view(-1, 1, 1, 1)
            noise = latents if i == 0 else torch.randn(latents.shape, device=device, dtype=latents.dtype, generator=noise_g)
            latents = (1.0 - t_4d) * dx + t_4d * noise
            flow = pipe.transformer(
                hidden_states=latents,
                encoder_hidden_states=pe,
                encoder_attention_mask=pm,
                timestep=args.time_scale * t_flat,
                return_dict=False,
            )[0]
            if variance_split:
                flow = flow.chunk(2, dim=1)[0]
            dx = latents - t_4d * flow

        image = decode_to_pil(dx, orig_h, orig_w)
        score = reward_rt.score(prompt, image)
        return image, score

    @torch.no_grad()
    def run_greedy(prompt: str, seed: int) -> tuple[Image.Image, float, list[tuple[tuple[float, bool], float]]]:
        pe, pm, ne, nm, pne, pnm = prepare_embeds(prompt)
        orig_h, orig_w = args.height, args.width
        h, w = orig_h, orig_w
        sample_size = int(pipe.transformer.config.sample_size)
        if sample_size in aspect_bins:
            h, w = pipe.image_processor.classify_height_width_bin(h, w, ratios=aspect_bins[sample_size])

        noise_g = torch.Generator(device=device).manual_seed(seed + 2001)
        latents = make_latents(pipe, device, latent_c, seed, h, w, pe.dtype)
        dx = torch.zeros_like(latents)
        chosen: list[tuple[tuple[float, bool], float]] = []

        for i in range(args.steps):
            scalar_t = 999.0 * (1.0 - float(i) / float(args.steps))
            t_flat = torch.full((latents.shape[0],), scalar_t / 999.0, device=device, dtype=latents.dtype)
            t_4d = t_flat.view(-1, 1, 1, 1)
            noise = latents if i == 0 else torch.randn(latents.shape, device=device, dtype=latents.dtype, generator=noise_g)
            latents = (1.0 - t_4d) * dx + t_4d * noise

            best_score = -float("inf")
            best_action = actions[0]
            best_dx = None
            print(f"  step {i + 1}/{args.steps} evaluating {len(actions)} actions")

            for cfg, use_neg in actions:
                flow = transformer_step(latents, pe, pm, ne, nm, pne, pnm, t_flat, cfg, use_neg)
                cand_dx = latents - t_4d * flow
                cand_img = decode_to_pil(cand_dx, orig_h, orig_w)
                score = reward_rt.score(prompt, cand_img)
                print(f"    cfg={cfg:.2f} neg={'Y' if use_neg else 'N'} IR={score:.4f}")
                if score > best_score:
                    best_score = score
                    best_action = (cfg, use_neg)
                    best_dx = cand_dx.clone()

            assert best_dx is not None
            dx = best_dx
            chosen.append((best_action, float(best_score)))
            print(f"  -> choose cfg={best_action[0]:.2f} neg={'Y' if best_action[1] else 'N'} IR={best_score:.4f}")

        final_img = decode_to_pil(dx, orig_h, orig_w)
        final_score = reward_rt.score(prompt, final_img)
        return final_img, final_score, chosen

    def load_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except Exception:
            return ImageFont.load_default()

    summary_rows: list[dict[str, Any]] = []
    for prompt_idx, prompt in enumerate(prompts):
        slug = f"p{prompt_idx:02d}"
        print(f"\n{'=' * 72}\n[{slug}] {prompt}\n{'=' * 72}")

        base_img, base_score = run_baseline(prompt, args.seed)
        base_path = os.path.join(args.out_dir, f"{slug}_baseline.png")
        base_img.save(base_path)
        print(f"baseline IR={base_score:.4f} -> {base_path}")

        greedy_img, greedy_score, chosen = run_greedy(prompt, args.seed)
        greedy_path = os.path.join(args.out_dir, f"{slug}_greedy.png")
        greedy_img.save(greedy_path)
        print(f"greedy IR={greedy_score:.4f} delta={greedy_score - base_score:+.4f} -> {greedy_path}")

        w, h = base_img.size
        label_h = 44
        comp = Image.new("RGB", (w * 2, h + label_h), (18, 18, 18))
        draw = ImageDraw.Draw(comp)
        comp.paste(base_img, (0, label_h))
        comp.paste(greedy_img, (w, label_h))
        draw.text((4, 4), f"baseline IR={base_score:.3f}", fill=(200, 200, 200), font=load_font(15))
        draw.text((w + 4, 4), f"greedy IR={greedy_score:.3f}", fill=(100, 255, 100), font=load_font(15))
        action_text = " -> ".join(
            f"s{i + 1}:cfg{a[0]:.2f}{' neg' if a[1] else ''}" for i, (a, _) in enumerate(chosen)
        )
        draw.text((w + 4, 24), action_text[:100], fill=(255, 220, 50), font=load_font(11))
        comp_path = os.path.join(args.out_dir, f"{slug}_comparison.png")
        comp.save(comp_path)

        summary_rows.append(
            {
                "idx": prompt_idx,
                "prompt": prompt[:80],
                "baseline": float(base_score),
                "greedy": float(greedy_score),
                "delta": float(greedy_score - base_score),
                "actions": [(float(a[0]), bool(a[1]), float(s)) for a, s in chosen],
            }
        )

    log_path = os.path.join(args.out_dir, "greedy_summary.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"{'idx':<5} {'baseline':>10} {'greedy':>10} {'delta':>8}  actions\n")
        f.write("-" * 84 + "\n")
        for row in summary_rows:
            action_str = " ".join(
                f"cfg{cfg:.2f}{'N' if use_neg else '.'}" for cfg, use_neg, _score in row["actions"]
            )
            f.write(
                f"{row['idx']:<5} {row['baseline']:>10.4f} {row['greedy']:>10.4f} {row['delta']:>+8.4f}  {action_str}\n"
            )
            f.write(f"      {row['prompt']}\n\n")

    print(f"\nsummary -> {log_path}")
    print(f"outputs -> {os.path.abspath(args.out_dir)}/")


if __name__ == "__main__":
    main()
