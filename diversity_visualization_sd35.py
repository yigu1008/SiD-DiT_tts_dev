#!/usr/bin/env python3
"""Diversity visualization for SD3.5-SiD vs SD3.5-large.

Generates samples across three diversity axes for each prompt:
  - seed-only            : same prompt, same CFG, vary seed
  - cfg-varied           : same prompt, vary CFG, same seed
  - prompt-varied        : vary prompt rewrite, same CFG, same seed
  - all-varied           : vary all three (upper bound)

Outputs:
  out_dir/
    images/p<P>/v<V>_cfg<CFG>_s<SEED>.png        -- raw samples
    grids/p<P>_grid.png                          -- per-prompt visual grid
    metrics/diversity_<backend>.csv              -- pairwise LPIPS / CLIP per axis
    metrics/diversity_bar_<backend>.png          -- bar chart by axis
    metrics/diversity_tsne_<backend>.png         -- CLIP t-SNE colored by axis

Usage:
  python diversity_visualization_sd35.py \
      --backend sid --model_id YGu1998/SiD-DiT-SD3.5-large \
      --prompt_file hpsv2_subset.txt --rewrites_file rewrites_cache.json \
      --n_prompts 4 --n_seeds 4 --cfg_scales 1.0 3.0 5.0 7.0 \
      --out_dir ./diversity_out_sid

Run twice (sid + sd35_base), then `--compare_dirs` overlays both on one bar chart.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sampling_unified_sd35 import (  # noqa: E402
    EmbeddingContext,
    PipelineContext,
    encode_variants,
    load_pipeline,
    run_baseline,
)


# ── Lightweight reward stub (we don't need rewards here) ────────────────────
class _NoRewardScorer:
    def score(self, prompt, image):
        return 0.0


def _build_args_for_pipeline(cli) -> argparse.Namespace:
    """Minimal arg namespace expected by load_pipeline / run_baseline."""
    return argparse.Namespace(
        backend=cli.backend,
        model_id=cli.model_id,
        transformer_id=cli.transformer_id,
        transformer_subfolder=cli.transformer_subfolder,
        dtype=cli.dtype,
        steps=cli.steps,
        width=cli.width,
        height=cli.height,
        time_scale=cli.time_scale,
        max_sequence_length=cli.max_sequence_length,
        sigmas=None,
        x0_sampler="default",
        euler_sampler=False,
    )


def _read_prompts(path: str, n: int) -> list[str]:
    lines = [l.strip() for l in open(path) if l.strip()]
    return lines[:n] if n > 0 else lines


def _read_rewrites(path: str | None, prompt: str, n_variants: int) -> list[str]:
    """Return [prompt, rewrite_1, ..., rewrite_{n-1}] if cache available, else replicate."""
    if not path or not os.path.exists(path):
        return [prompt] * n_variants
    try:
        cache = json.load(open(path))
    except Exception:
        return [prompt] * n_variants
    entry = cache.get(prompt)
    if isinstance(entry, dict):
        rewrites = list(entry.get("rewrites") or entry.get("variants") or [])
    elif isinstance(entry, list):
        rewrites = list(entry)
    else:
        rewrites = []
    out = [prompt] + [r for r in rewrites if isinstance(r, str) and r != prompt]
    out = out[:n_variants]
    while len(out) < n_variants:
        out.append(prompt)
    return out


# ── Diversity metrics ──────────────────────────────────────────────────────
def _to_tensor(img: Image.Image, size: int = 256) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W] in [0,1]


@torch.no_grad()
def _clip_embed(images: list[Image.Image], device: str) -> np.ndarray:
    """OpenCLIP ViT-B/32 image embeddings, L2-normalized."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    model.eval()
    batch = torch.stack([preprocess(im) for im in images]).to(device)
    feats = model.encode_image(batch)
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return feats.cpu().float().numpy()


@torch.no_grad()
def _lpips_pairwise_mean(images: list[Image.Image], device: str) -> float:
    """Mean pairwise LPIPS distance among a small set of images."""
    if len(images) < 2:
        return 0.0
    import lpips
    net = lpips.LPIPS(net="alex").to(device)
    net.eval()
    tensors = torch.stack([_to_tensor(im) for im in images]).to(device)
    tensors = tensors * 2.0 - 1.0  # LPIPS expects [-1, 1]
    dists: list[float] = []
    for i, j in combinations(range(len(images)), 2):
        d = net(tensors[i : i + 1], tensors[j : j + 1]).item()
        dists.append(d)
    return float(np.mean(dists))


def _clip_pairwise_mean(feats: np.ndarray) -> float:
    if feats.shape[0] < 2:
        return 0.0
    sim = feats @ feats.T
    n = feats.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(1.0 - sim[iu].mean())  # cosine *distance*


# ── Main pipeline ──────────────────────────────────────────────────────────
def generate_grid(cli) -> dict:
    args = _build_args_for_pipeline(cli)
    ctx = load_pipeline(args)
    scorer = _NoRewardScorer()

    out_dir = Path(cli.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "grids").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    prompts = _read_prompts(cli.prompt_file, cli.n_prompts)
    cfgs = list(cli.cfg_scales)
    seeds = [cli.seed_base + k for k in range(cli.n_seeds)]
    n_variants = max(1, int(cli.n_variants))

    manifest: list[dict] = []
    for pi, prompt in enumerate(prompts):
        variants = _read_rewrites(cli.rewrites_file, prompt, n_variants)
        emb = encode_variants(ctx, variants, max_sequence_length=cli.max_sequence_length)
        # Per-prompt subdir
        pdir = out_dir / "images" / f"p{pi:02d}"
        pdir.mkdir(parents=True, exist_ok=True)
        with open(pdir / "variants.txt", "w") as f:
            for vi, t in enumerate(variants):
                f.write(f"v{vi}: {t}\n")

        for vi, variant_text in enumerate(variants):
            # build a per-variant emb where variant `vi` is the only conditional
            v_emb = EmbeddingContext(
                cond_text=[emb.cond_text[vi]],
                cond_pooled=[emb.cond_pooled[vi]],
                uncond_text=emb.uncond_text,
                uncond_pooled=emb.uncond_pooled,
            )
            for ci, cfg in enumerate(cfgs):
                for si, seed in enumerate(seeds):
                    fname = f"v{vi}_cfg{cfg:g}_s{seed}.png"
                    fpath = pdir / fname
                    if fpath.exists() and not cli.overwrite:
                        continue
                    img, _ = run_baseline(
                        args, ctx, v_emb, scorer, variant_text, seed, cfg_scale=cfg
                    )
                    img.save(fpath)
                    manifest.append({
                        "prompt_idx": pi, "variant_idx": vi, "cfg": cfg,
                        "seed": seed, "path": str(fpath),
                    })
        print(f"[diversity] prompt {pi+1}/{len(prompts)} done")

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return {"prompts": prompts, "cfgs": cfgs, "seeds": seeds, "n_variants": n_variants,
            "manifest": manifest, "out_dir": str(out_dir)}


def render_grids(meta: dict, backend_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(meta["out_dir"])
    cfgs, seeds = meta["cfgs"], meta["seeds"]
    n_variants = meta["n_variants"]
    rows = n_variants
    cols = len(cfgs)
    for pi, prompt in enumerate(meta["prompts"]):
        fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows))
        if rows == 1: axes = np.array([axes])
        if cols == 1: axes = axes.reshape(-1, 1)
        for vi in range(rows):
            for ci, cfg in enumerate(cfgs):
                ax = axes[vi, ci]
                fpath = out_dir / "images" / f"p{pi:02d}" / f"v{vi}_cfg{cfg:g}_s{seeds[0]}.png"
                if fpath.exists():
                    ax.imshow(Image.open(fpath))
                ax.set_xticks([]); ax.set_yticks([])
                if vi == 0:
                    ax.set_title(f"cfg={cfg:g}", fontsize=9)
                if ci == 0:
                    ax.set_ylabel(f"v{vi}", fontsize=9)
        fig.suptitle(f"[{backend_label}] {prompt[:70]}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "grids" / f"p{pi:02d}_grid.png", dpi=120)
        plt.close(fig)


def compute_diversity(meta: dict, backend_label: str, device: str) -> str:
    """For each prompt, measure pairwise distances under each axis configuration."""
    out_dir = Path(meta["out_dir"])
    cfgs, seeds = meta["cfgs"], meta["seeds"]
    n_variants = meta["n_variants"]
    cfg_default = cfgs[len(cfgs) // 2]

    rows: list[dict] = []
    for pi, prompt in enumerate(meta["prompts"]):
        pdir = out_dir / "images" / f"p{pi:02d}"

        def imgs_for(filter_fn) -> list[Image.Image]:
            paths = sorted(pdir.glob("*.png"))
            sel = []
            for p in paths:
                stem = p.stem  # vV_cfgC_sS
                parts = stem.split("_")
                v = int(parts[0][1:]); c = float(parts[1][3:]); s = int(parts[2][1:])
                if filter_fn(v, c, s):
                    sel.append(Image.open(p).convert("RGB"))
            return sel

        groups = {
            "seed_only":     imgs_for(lambda v, c, s: v == 0 and c == cfg_default),
            "cfg_varied":    imgs_for(lambda v, c, s: v == 0 and s == seeds[0]),
            "prompt_varied": imgs_for(lambda v, c, s: c == cfg_default and s == seeds[0]),
            "all_varied":    imgs_for(lambda v, c, s: True),
        }

        for axis_name, imgs in groups.items():
            if not imgs:
                continue
            lpips_d = _lpips_pairwise_mean(imgs, device)
            try:
                clip_d = _clip_pairwise_mean(_clip_embed(imgs, device))
            except Exception as exc:
                print(f"[diversity] clip skipped ({exc})")
                clip_d = float("nan")
            rows.append({
                "backend": backend_label, "prompt_idx": pi, "axis": axis_name,
                "n_images": len(imgs), "mean_lpips": lpips_d, "mean_clip_dist": clip_d,
            })
            print(f"[diversity] p{pi:02d} {axis_name:14s} n={len(imgs)} "
                  f"lpips={lpips_d:.4f} clip_d={clip_d:.4f}")

    csv_path = out_dir / "metrics" / f"diversity_{backend_label}.csv"
    with open(csv_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return str(csv_path)


def plot_bar_summary(csv_paths: list[str], out_path: str) -> None:
    """Bar chart: mean pairwise LPIPS by axis, grouped by backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_backend: dict[str, dict[str, list[float]]] = {}
    for path in csv_paths:
        for row in csv.DictReader(open(path)):
            be = row["backend"]; ax = row["axis"]
            try: v = float(row["mean_lpips"])
            except: continue
            by_backend.setdefault(be, {}).setdefault(ax, []).append(v)

    axes_order = ["seed_only", "cfg_varied", "prompt_varied", "all_varied"]
    backends = sorted(by_backend.keys())
    x = np.arange(len(axes_order))
    width = 0.8 / max(1, len(backends))

    fig, ax = plt.subplots(figsize=(7, 4))
    for bi, be in enumerate(backends):
        means = [np.mean(by_backend[be].get(a, [0.0])) for a in axes_order]
        stds = [np.std(by_backend[be].get(a, [0.0])) for a in axes_order]
        ax.bar(x + (bi - (len(backends) - 1) / 2) * width, means, width,
               yerr=stds, capsize=3, label=be)
    ax.set_xticks(x); ax.set_xticklabels(axes_order, rotation=15)
    ax.set_ylabel("mean pairwise LPIPS")
    ax.set_title("Diversity by source axis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[diversity] bar chart -> {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="sid", choices=["sid", "sd35_base"])
    p.add_argument("--model_id", required=True)
    p.add_argument("--transformer_id", default=None)
    p.add_argument("--transformer_subfolder", default=None)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--prompt_file", required=True)
    p.add_argument("--rewrites_file", default=None,
                   help="JSON cache of {prompt: {rewrites: [...]}}; if absent, prompt-axis collapses.")
    p.add_argument("--n_prompts", type=int, default=4)
    p.add_argument("--n_seeds", type=int, default=4)
    p.add_argument("--n_variants", type=int, default=4)
    p.add_argument("--cfg_scales", type=float, nargs="+", default=[1.0, 3.0, 5.0, 7.0])
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--time_scale", type=float, default=1000.0)
    p.add_argument("--max_sequence_length", type=int, default=256)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default=None, help="Backend label for plot legends.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_generation", action="store_true",
                   help="Reuse existing images in out_dir; only recompute metrics.")
    p.add_argument("--compare_dirs", nargs="+", default=None,
                   help="If set, just merges diversity_*.csv from each dir into a bar chart.")
    return p.parse_args()


def main():
    cli = parse_args()
    label = cli.label or cli.backend

    if cli.compare_dirs:
        csv_paths: list[str] = []
        for d in cli.compare_dirs:
            for p in Path(d).glob("metrics/diversity_*.csv"):
                csv_paths.append(str(p))
        out = Path(cli.out_dir) / "diversity_compare.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plot_bar_summary(csv_paths, str(out))
        return

    if cli.skip_generation:
        out_dir = Path(cli.out_dir)
        manifest = json.load(open(out_dir / "manifest.json"))
        meta = {"prompts": _read_prompts(cli.prompt_file, cli.n_prompts),
                "cfgs": list(cli.cfg_scales),
                "seeds": [cli.seed_base + k for k in range(cli.n_seeds)],
                "n_variants": cli.n_variants,
                "manifest": manifest, "out_dir": str(out_dir)}
    else:
        meta = generate_grid(cli)

    render_grids(meta, label)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    csv_path = compute_diversity(meta, label, device)
    plot_bar_summary([csv_path],
                     str(Path(meta["out_dir"]) / "metrics" / f"diversity_bar_{label}.png"))


if __name__ == "__main__":
    main()
