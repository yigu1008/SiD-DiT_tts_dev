#!/usr/bin/env python3
"""Image montage for the per-step CFG grid (run_cfg_step_grid.py with GRID_SAVE_IMAGES=1).

For a chosen denoising step, lay out: rows = prompts, columns = cfg applied AT
that step (with the all-baseline image as the first column), so you can SEE how
bumping cfg at step k changes the image across prompts and cfg values. Each cell
is annotated with its reward (from cfg_step_grid.csv); the best cfg per prompt is
boxed green.

Usage:
  python plot_cfg_step_montage.py --run_root /data/ygu/cfg_prompt_grid/step_exp \
    --step 3 --seed 42 --top 10 --out step3_montage.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

from PIL import Image, ImageDraw, ImageFont


def _font(sz: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def _parse_prompts(spec: str) -> list[int]:
    out: list[int] = []
    for tok in spec.replace(",", " ").split():
        if "-" in tok:
            x, y = tok.split("-"); out.extend(range(int(x), int(y) + 1))
        elif tok.isdigit():
            out.append(int(tok))
    return sorted(set(out))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_root", required=True, help="dir with images/ and cfg_step_grid.csv")
    p.add_argument("--step", type=int, default=0, help="denoising step to montage (0-indexed)")
    p.add_argument("--joint", action="store_true", help="montage the JOINT intervention (GRID_STEP_JOINT run)")
    p.add_argument("--seed", default="42")
    p.add_argument("--prompts", default=None, help="subset '0-9'/'0,3,7' (default: all with data)")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--rank_by_gain", action="store_true", help="order prompts by best gain at this step")
    p.add_argument("--thumb", type=int, default=224)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    img_dir = os.path.join(a.run_root, "images")
    csv_path = os.path.join(a.run_root, "cfg_step_grid.csv")
    rew: dict[tuple[int, float], float] = {}
    base: dict[int, float] = {}
    cfgs = set()
    target_step = -2 if a.joint else a.step
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        pi, rw = int(r["prompt_index"]), float(r["reward"])
        if int(r["is_baseline"]) == 1:
            base[pi] = rw
        elif int(r["step"]) == target_step:
            c = float(r["cfg"]); rew[(pi, c)] = rw; cfgs.add(c)
    cfgs = sorted(cfgs)
    if not cfgs:
        raise SystemExit(f"no rows for {'joint' if a.joint else f'step {a.step}'} in {csv_path}")

    prompts_txt: list[str] = []
    pf = (glob.glob(f"{a.run_root}/_prompts/backend_*.txt") or [None])[0]
    if pf and os.path.exists(pf):
        prompts_txt = [ln.rstrip("\n") for ln in open(pf, encoding="utf-8") if ln.strip()]

    pis = sorted(base)
    if a.prompts:
        want = set(_parse_prompts(a.prompts)); pis = [i for i in pis if i in want]
    if a.rank_by_gain:
        pis.sort(key=lambda i: max((rew.get((i, c), -1e9) for c in cfgs), default=-1e9) - base.get(i, 0.0), reverse=True)
    pis = pis[: a.top]

    cols = [("__base__", "baseline")] + [(c, f"cfg {c:g}") for c in cfgs]
    T, lw, hh, pad = int(a.thumb), 250, 26, 6
    W = lw + len(cols) * (T + pad) + pad
    H = hh + len(pis) * (T + pad) + pad
    canvas = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(canvas)
    fh, fr = _font(16), _font(12)
    d.text((8, 6), f"cfg @ {'joint steps' if a.joint else f'step {a.step}'} (seed {a.seed})", fill=(235, 235, 235), font=fh)
    for c, (_v, lab) in enumerate(cols):
        d.text((lw + c * (T + pad) + pad + 4, 6), lab, fill=(220, 220, 220), font=fr)

    for row, pi in enumerate(pis):
        y = hh + row * (T + pad) + pad
        best_c = max(cfgs, key=lambda c: rew.get((pi, c), -1e9)) if cfgs else None
        d.text((6, y + 4), f"p{pi:05d}", fill=(180, 200, 255), font=fr)
        txt = (prompts_txt[pi][:34] if pi < len(prompts_txt) else "")
        d.text((6, y + 22), txt, fill=(190, 190, 190), font=fr)
        d.text((6, y + 40), f"base {base.get(pi, float('nan')):.2f}", fill=(200, 200, 120), font=fr)
        for c, (v, _lab) in enumerate(cols):
            x = lw + c * (T + pad) + pad
            if v == "__base__":
                path = os.path.join(img_dir, f"p{pi:05d}_s{a.seed}_baseline.png"); score = base.get(pi)
            else:
                fn = (f"p{pi:05d}_s{a.seed}_joint_cfg{float(v):.2f}.png" if a.joint
                      else f"p{pi:05d}_s{a.seed}_step{a.step}_cfg{float(v):.2f}.png")
                path = os.path.join(img_dir, fn)
                score = rew.get((pi, float(v)))
            if os.path.exists(path):
                canvas.paste(Image.open(path).convert("RGB").resize((T, T)), (x, y))
            else:
                d.rectangle([x, y, x + T, y + T], outline=(70, 70, 70))
                d.text((x + 8, y + T // 2), "(missing)", fill=(140, 140, 140), font=fr)
            if v != "__base__" and best_c is not None and abs(float(v) - best_c) < 1e-9:
                d.rectangle([x, y, x + T - 1, y + T - 1], outline=(90, 220, 120), width=3)  # best cfg
            if score is not None:
                d.rectangle([x, y + T - 18, x + 78, y + T], fill=(0, 0, 0))
                d.text((x + 3, y + T - 16), f"{score:.3f}", fill=(255, 235, 120), font=fr)

    out = a.out or os.path.join(a.run_root, ("cfg_joint_montage.png" if a.joint else f"cfg_step{a.step}_montage.png"))
    canvas.save(out)
    print(f"[step-montage] wrote {out}  ({len(pis)} prompts x {len(cols)} cols @ step {a.step})")


if __name__ == "__main__":
    main()
