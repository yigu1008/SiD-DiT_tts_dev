#!/usr/bin/env python3
"""Full cfg x variant image surface for ONE prompt (deep-dive on a single example).

Reads a run_cfg_prompt_grid.py run and lays out, for a single prompt, the entire
rectangle of generated images: rows = cfg (ascending), columns = prompt variant
(v0 = original, v1..vN = rewrites). Each cell shows its reward. The global-best
cell is boxed gold and the baseline cell (v0, lowest cfg) is boxed grey, so you
can see the whole cfg x prompt landscape and where it peaks -- the elaborated
view behind a single row of the 4-corner synergy montage.

Usage:
  python plot_prompt_surface.py --run_root /data/ygu/cfg_prompt_grid/sd35_portraits \
    --prompt_index 1 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import os

from PIL import Image, ImageDraw, ImageFont


def _font(sz: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_root", required=True)
    p.add_argument("--prompt_index", type=int, default=1)
    p.add_argument("--seed", default="42")
    p.add_argument("--thumb", type=int, default=300)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    csv_path = os.path.join(a.run_root, "cfg_prompt_grid.csv")
    img_dir = os.path.join(a.run_root, "images")
    rew: dict[tuple[int, float], float] = {}   # (variant, cfg) -> reward
    vtext: dict[int, str] = {}
    orig = ""
    cfgs, variants = set(), set()
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        if int(r["prompt_index"]) != a.prompt_index or str(int(r["seed"])) != str(a.seed):
            continue
        vi, c = int(r["variant"]), float(r["cfg"])
        rew[(vi, c)] = float(r["reward"]); cfgs.add(c); variants.add(vi)
        vtext[vi] = r.get("variant_text", "")
        if vi == 0:
            orig = r.get("variant_text", "")
    if not rew:
        raise SystemExit(f"no rows for prompt {a.prompt_index} seed {a.seed} in {csv_path}")

    cfgs = sorted(cfgs)                 # low -> high (top -> bottom)
    variants = sorted(variants)        # v0 original, then rewrites
    best_cell = max(rew, key=rew.get)
    base_cell = (0, cfgs[0])

    T, lh, hh, pad = int(a.thumb), 74, 132, 8
    W = lh + len(variants) * (T + pad) + pad
    H = hh + len(cfgs) * (T + pad) + pad
    canvas = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(canvas)
    fh, fr, fs = _font(19), _font(13), _font(15)
    d.text((8, 8), f"p{a.prompt_index:05d}  “{orig[:70]}”   (seed {a.seed}) — full cfg x variant surface",
           fill=(235, 235, 235), font=fh)
    d.text((8, 34), "rows = cfg (low→high)   cols = prompt variant (v0 = original, v1+ = aesthetic rewrites)   "
                    "gold = global best   grey = baseline", fill=(170, 170, 170), font=fr)
    # column headers (variant label + short rewrite text)
    for ci, vi in enumerate(variants):
        x = lh + ci * (T + pad) + pad
        lab = "v0 (original)" if vi == 0 else f"v{vi}"
        d.text((x + 3, 62), lab, fill=(200, 220, 255), font=fs)
        d.text((x + 3, 82), (vtext.get(vi, "")[:34]), fill=(160, 160, 160), font=fr)
        d.text((x + 3, 100), (vtext.get(vi, "")[34:68]), fill=(160, 160, 160), font=fr)

    for ri, c in enumerate(cfgs):
        y = hh + ri * (T + pad) + pad
        d.text((6, y + T // 2 - 8), f"cfg\n{c:g}", fill=(210, 210, 140), font=fs)
        for ci, vi in enumerate(variants):
            x = lh + ci * (T + pad) + pad
            score = rew.get((vi, c))
            path = os.path.join(img_dir, f"p{a.prompt_index:05d}_s{a.seed}_v{vi}_cfg{c:.2f}.png")
            if os.path.exists(path):
                canvas.paste(Image.open(path).convert("RGB").resize((T, T)), (x, y))
            else:
                d.rectangle([x, y, x + T, y + T], outline=(70, 70, 70))
            if (vi, c) == best_cell:
                d.rectangle([x, y, x + T - 1, y + T - 1], outline=(240, 200, 70), width=5)
            elif (vi, c) == base_cell:
                d.rectangle([x, y, x + T - 1, y + T - 1], outline=(130, 130, 130), width=3)
            if score is not None:
                d.rectangle([x, y + T - 20, x + 70, y + T], fill=(0, 0, 0))
                d.text((x + 3, y + T - 18), f"{score:.3f}", fill=(255, 235, 120), font=fr)

    out = a.out or os.path.join(a.run_root, f"surface_p{a.prompt_index:05d}.png")
    canvas.save(out)
    bv, bc = best_cell
    print(f"[surface] wrote {out}  best = v{bv} cfg {bc:g} ({rew[best_cell]:.3f}), "
          f"baseline {rew.get(base_cell, float('nan')):.3f}")


if __name__ == "__main__":
    main()
