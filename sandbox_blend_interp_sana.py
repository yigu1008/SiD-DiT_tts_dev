"""
Fast prompt-blend interpolation ablation (SiD-SANA).

No search (no GA/CEM/MCTS). For each prompt:
  - Baseline (original prompt)
  - NLERP interpolation at t in {0, 0.25, 0.5, 0.75, 1.0}
  - SLERP interpolation at t in {0, 0.25, 0.5, 0.75, 1.0}

Outputs per-image reward entries and summary aggregates.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sampling_unified as su
from eval_and_log import ensure_dir, save_json
from prompt_basis import build_prompt_basis, encode_prompt_basis, load_basis_cache, save_basis_cache
from rollout_runner import resolve_resolution, run_dynamic_rollout, run_single_prompt_rollout
from weight_policy import WeightParams


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast NLERP/SLERP interpolation sweep (no search).")

    # I/O
    p.add_argument("--prompt", type=str, default="a studio portrait of an elderly woman smiling, soft window light, 85mm lens, photorealistic")
    p.add_argument("--prompt_file", type=str, default=None)
    p.add_argument("--max_prompts", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="./sandbox_blend_interp_out")
    p.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_first_k", type=int, default=10)
    p.add_argument("--basis_cache_json", type=str, default=None)

    # Model/pipeline
    p.add_argument("--model_id", type=str, default="YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--neg_embed", type=str, default=None)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument("--gpu_id", type=int, default=-1)
    p.add_argument("--auto_select_gpu", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min_free_gb", type=float, default=12.0)

    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--resolution_binning", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--time_scale", type=float, default=1000.0)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # Runtime/memory
    p.add_argument("--cuda_alloc_conf", type=str, default="expandable_segments:True")
    p.add_argument("--vae_slicing", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--vae_tiling", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--empty_cache_after_decode", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--offload_text_encoder_after_encode", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sana_no_fp32_attn", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--decode_device", type=str, default="auto")
    p.add_argument("--decode_cpu_dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
    p.add_argument("--decode_cpu_if_free_below_gb", type=float, default=20.0)

    # Reward
    p.add_argument("--reward_type", choices=["imagereward", "auto", "unifiedreward", "unified", "pickscore", "hpsv2", "blend"], default="imagereward")
    p.add_argument("--reward_device", type=str, default="cpu")
    p.add_argument("--reward_model", type=str, default="CodeGoat24/UnifiedReward-qwen-7b")
    p.add_argument("--unifiedreward_model", type=str, default=None)
    p.add_argument("--image_reward_model", type=str, default="ImageReward-v1.0")
    p.add_argument("--reward_weights", nargs=2, type=float, default=[1.0, 1.0])
    p.add_argument("--reward_api_base", type=str, default=None)
    p.add_argument("--reward_api_key", type=str, default="unifiedreward")
    p.add_argument("--reward_api_model", type=str, default="UnifiedReward-7b-v1.5")
    p.add_argument("--reward_max_new_tokens", type=int, default=512)
    p.add_argument("--reward_prompt_mode", choices=["standard", "strict"], default="standard")

    # Qwen prompt basis
    p.add_argument("--no_qwen", action="store_true")
    p.add_argument("--qwen_id", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--qwen_python", type=str, default="python3")
    p.add_argument("--qwen_dtype", choices=["float16", "bfloat16"], default="bfloat16")

    # Interp setup
    p.add_argument("--interp_labels", nargs=2, default=["balanced", "subject"], help="Two basis labels to interpolate.")
    p.add_argument("--interp_values", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--families", nargs="+", default=["nlerp", "slerp"], help="Blend families to evaluate.")
    p.add_argument("--preview_every", type=int, default=-1, help="Set -1 to disable preview decodes for speed.")
    return p.parse_args(argv)


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [str(args.prompt).strip()]
    if int(args.max_prompts) > 0:
        prompts = prompts[: int(args.max_prompts)]
    if len(prompts) == 0:
        raise RuntimeError("No prompts found.")
    return prompts


def _pick_pair_indices(labels: list[str], want_a: str, want_b: str) -> tuple[int, int]:
    key_to_idx = {str(lbl).strip().lower(): i for i, lbl in enumerate(labels)}
    a = key_to_idx.get(str(want_a).strip().lower(), -1)
    b = key_to_idx.get(str(want_b).strip().lower(), -1)
    if a < 0:
        a = 0
    if b < 0:
        b = 1 if len(labels) > 1 else 0
    if a == b and len(labels) > 1:
        b = 1 if a == 0 else 0
    return int(a), int(b)


def _theta_for_t(t: float) -> WeightParams:
    tt = float(np.clip(t, 0.0, 1.0))
    if tt <= 0.0:
        b = np.asarray([20.0, -20.0], dtype=np.float64)
    elif tt >= 1.0:
        b = np.asarray([-20.0, 20.0], dtype=np.float64)
    else:
        b = np.log(np.asarray([1.0 - tt, tt], dtype=np.float64))
    a = np.zeros((2,), dtype=np.float64)
    return WeightParams(a=a, b=b)


def _family_list(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    for fam in args.families:
        f = str(fam).strip().lower()
        if f in {"nlerp", "slerp"} and f not in out:
            out.append(f)
    if len(out) == 0:
        out = ["nlerp", "slerp"]
    return out


def _t_slug(t: float) -> str:
    return str(f"{float(t):.2f}").replace(".", "p")


def _font(size: int = 15) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _build_interp_panel(
    baseline_img: Image.Image,
    baseline_score: float,
    families: list[str],
    t_values: list[float],
    panel_data: dict[str, dict[float, tuple[Image.Image, float]]],
    out_path: str,
) -> None:
    if len(families) == 0:
        return
    w, h = baseline_img.size
    cols = 1 + len(t_values)
    rows = len(families)
    label_h = 46
    panel = Image.new("RGB", (cols * w, rows * (h + label_h)), (18, 18, 18))
    draw = ImageDraw.Draw(panel)
    f_main = _font(15)
    f_sub = _font(13)

    for r, fam in enumerate(families):
        y0 = r * (h + label_h)
        # Baseline column for each row.
        panel.paste(baseline_img, (0, y0 + label_h))
        draw.text((8, y0 + 6), f"{fam} baseline", fill=(220, 220, 220), font=f_main)
        draw.text((8, y0 + 24), f"s={baseline_score:.4f}", fill=(200, 200, 200), font=f_sub)

        fam_map = panel_data.get(fam, {})
        for c, tval in enumerate(t_values, start=1):
            x0 = c * w
            item = fam_map.get(float(tval))
            if item is None:
                continue
            img, score = item
            delta = float(score - baseline_score)
            panel.paste(img, (x0, y0 + label_h))
            draw.text((x0 + 8, y0 + 6), f"{fam} t={float(tval):.2f}", fill=(230, 230, 230), font=f_main)
            col = (110, 255, 110) if delta >= 0.0 else (255, 120, 120)
            draw.text((x0 + 8, y0 + 24), f"s={score:.4f} d={delta:+.4f}", fill=col, font=f_sub)

    panel.save(out_path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_dir(args.out_dir)

    if args.cuda_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(args.cuda_alloc_conf)
        print(f"Set PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

    prompts = load_prompts(args)
    families = _family_list(args)
    t_values = [float(np.clip(v, 0.0, 1.0)) for v in args.interp_values]
    basis_cache = load_basis_cache(args.basis_cache_json)

    t0 = time.perf_counter()
    ctx = su.load_pipeline(args)
    reward_ctx = su.load_reward(args, ctx)
    if callable(getattr(reward_ctx, "before_decode", None)):
        setattr(ctx, "pre_decode_hook", reward_ctx.before_decode)
    neg_embeds, neg_mask = su.load_neg_embed(args, ctx)

    all_rows: list[dict[str, Any]] = []
    tsv_path = os.path.join(args.out_dir, "blend_interp_scores.tsv")
    with open(tsv_path, "w", encoding="utf-8", newline="") as ftsv:
        writer = csv.writer(ftsv, delimiter="\t")
        writer.writerow(["slug", "prompt", "family", "t", "score", "delta_vs_baseline", "image_file"])

        for i, prompt in enumerate(prompts):
            slug = f"p{i:04d}"
            prompt_dir = os.path.join(args.out_dir, slug)
            ensure_dir(prompt_dir)
            save_this = int(args.save_first_k) < 0 or i < int(args.save_first_k)

            print("\n" + "=" * 72)
            print(f"[{slug}] {prompt}")
            print("=" * 72)

            basis = build_prompt_basis(args, prompt, basis_cache)
            basis_emb = encode_prompt_basis(args, ctx, basis, neg_embeds, neg_mask, max_seq=256)
            orig_h, orig_w, h, w = resolve_resolution(args, ctx)
            basis_texts = {str(c.label): str(c.text) for c in basis.candidates}
            save_json(
                os.path.join(prompt_dir, "prompt_basis.json"),
                {
                    "original": str(prompt),
                    "basis": basis_texts,
                },
            )

            base_img_path = os.path.join(prompt_dir, "baseline_original.png") if save_this and args.save_images else None
            baseline = run_single_prompt_rollout(
                args=args,
                ctx=ctx,
                reward_ctx=reward_ctx,
                prompt=prompt,
                metadata=None,
                seed=int(args.seed),
                h=h,
                w=w,
                orig_h=orig_h,
                orig_w=orig_w,
                pe=basis_emb.orig_pe,
                pm=basis_emb.orig_pm,
                ue=basis_emb.orig_ue,
                um=basis_emb.orig_um,
                preview_every=int(args.preview_every),
                save_path=base_img_path,
                tag=f"{slug}_baseline_orig",
            )
            print(f"  baseline={baseline.final_score:.4f}")

            li0, li1 = _pick_pair_indices(
                basis_emb.labels,
                str(args.interp_labels[0]),
                str(args.interp_labels[1]),
            )
            pair_labels = [basis_emb.labels[li0], basis_emb.labels[li1]]
            pair_indices = [int(li0), int(li1)]
            print(f"  interp pair: {pair_labels[0]}->{pair_labels[1]} (indices={pair_indices})")

            entries: list[dict[str, Any]] = []
            panel_data: dict[str, dict[float, tuple[Image.Image, float]]] = {}
            writer.writerow([slug, prompt, "baseline", "baseline", f"{baseline.final_score:.6f}", "+0.000000", base_img_path or ""])

            for fam in families:
                for tval in t_values:
                    theta = _theta_for_t(tval)
                    img_path = (
                        os.path.join(prompt_dir, f"{fam}_t{_t_slug(tval)}.png")
                        if save_this and args.save_images
                        else None
                    )
                    trace = run_dynamic_rollout(
                        args=args,
                        ctx=ctx,
                        reward_ctx=reward_ctx,
                        prompt=prompt,
                        metadata=None,
                        seed=int(args.seed),
                        h=h,
                        w=w,
                        orig_h=orig_h,
                        orig_w=orig_w,
                        basis_emb=basis_emb,
                        basis_indices=pair_indices,
                        blend_family=fam,
                        weight_params=theta,
                        preview_every=int(args.preview_every),
                        save_path=img_path,
                        tag=f"{slug}_{fam}_t{_t_slug(tval)}",
                    )
                    score = float(trace.final_score)
                    delta = float(score - baseline.final_score)
                    entry = {
                        "family": fam,
                        "t": float(tval),
                        "score": score,
                        "delta_vs_baseline": delta,
                        "image_file": img_path,
                    }
                    entries.append(entry)
                    if save_this and args.save_images:
                        panel_data.setdefault(str(fam), {})[float(tval)] = (trace.final_image, float(score))
                    writer.writerow([slug, prompt, fam, f"{float(tval):.2f}", f"{score:.6f}", f"{delta:+.6f}", img_path or ""])
                    print(f"  {fam:>5} t={float(tval):.2f} score={score:.4f} delta={delta:+.4f}")

            # Endpoint sanity: nlerp/slerp should match when one side weight is 1.
            by_key = {(str(e["family"]), round(float(e["t"]), 6)): float(e["score"]) for e in entries}
            for t_edge in (0.0, 1.0):
                key_n = ("nlerp", round(t_edge, 6))
                key_s = ("slerp", round(t_edge, 6))
                if key_n in by_key and key_s in by_key:
                    gap = by_key[key_s] - by_key[key_n]
                    if abs(gap) > 1e-3:
                        print(
                            f"  [warn] endpoint mismatch at t={t_edge:.2f}: "
                            f"slerp-nlerp={gap:+.4f} (expected near 0)"
                        )

            row = {
                "slug": slug,
                "prompt": prompt,
                "basis_labels": list(basis_emb.labels),
                "basis_texts": basis_texts,
                "interp_pair_labels": pair_labels,
                "interp_pair_indices": pair_indices,
                "baseline_score": float(baseline.final_score),
                "results": entries,
            }
            if save_this and args.save_images:
                # One stacked panel for all families.
                all_panel = os.path.join(prompt_dir, "interp_comparison_panel.png")
                _build_interp_panel(
                    baseline_img=baseline.final_image,
                    baseline_score=float(baseline.final_score),
                    families=families,
                    t_values=t_values,
                    panel_data=panel_data,
                    out_path=all_panel,
                )
                # Per-method panels.
                method_panels: dict[str, str] = {}
                for fam in families:
                    fam_panel = os.path.join(prompt_dir, f"{fam}_interp_panel.png")
                    _build_interp_panel(
                        baseline_img=baseline.final_image,
                        baseline_score=float(baseline.final_score),
                        families=[fam],
                        t_values=t_values,
                        panel_data=panel_data,
                        out_path=fam_panel,
                    )
                    method_panels[str(fam)] = fam_panel
                row["panel_files"] = {
                    "stacked": all_panel,
                    "by_family": method_panels,
                }
            save_json(os.path.join(prompt_dir, "interp_result.json"), row)
            all_rows.append(row)

    # Aggregate by (family, t)
    agg: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        base = float(row["baseline_score"])
        for ent in row["results"]:
            key = f"{ent['family']}@{float(ent['t']):.2f}"
            cur = agg.setdefault(key, {"family": ent["family"], "t": float(ent["t"]), "scores": [], "deltas": []})
            cur["scores"].append(float(ent["score"]))
            cur["deltas"].append(float(ent["delta_vs_baseline"]))

    agg_rows: list[dict[str, Any]] = []
    for key in sorted(agg.keys()):
        cur = agg[key]
        scores = np.asarray(cur["scores"], dtype=np.float64)
        deltas = np.asarray(cur["deltas"], dtype=np.float64)
        agg_rows.append(
            {
                "family": str(cur["family"]),
                "t": float(cur["t"]),
                "count": int(scores.size),
                "mean_score": float(scores.mean()) if scores.size else 0.0,
                "mean_delta_vs_baseline": float(deltas.mean()) if deltas.size else 0.0,
                "std_delta_vs_baseline": float(deltas.std()) if deltas.size else 0.0,
            }
        )

    summary = {
        "num_prompts": len(all_rows),
        "families": families,
        "interp_values": [float(x) for x in t_values],
        "rows": all_rows,
        "aggregate": agg_rows,
        "scores_tsv": tsv_path,
    }
    summary_path = os.path.join(args.out_dir, "blend_interp_summary.json")
    save_json(summary_path, summary)
    save_basis_cache(args.basis_cache_json, basis_cache)

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 72)
    print(f"Done. prompts={len(all_rows)} elapsed_sec={elapsed:.2f}")
    print(f"Scores TSV: {tsv_path}")
    print(f"Summary  : {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
