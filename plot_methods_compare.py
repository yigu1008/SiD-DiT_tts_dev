#!/usr/bin/env python3
"""Qualitative method-comparison montage for SD3.5 (or any backend suite run).

Lays out, per prompt, one image column per search method:
    base (baseline) | DAS (smc) | FK-Steering (fksteering) | MCTS/ours (bon_mcts)
Reads the per-method image dirs and the per-prompt search reward from each
method's logs/rank_*.jsonl. Rows are ranked by the ours-vs-base delta so the
prompts where MCTS wins most surface first; the MCTS (ours) cell is highlighted
and the row's best reward is shown in green.

Usage:
  python plot_methods_compare.py --run_dir <.../seedNN/run_TS> --seed 42 --top 8

--run_dir is the directory that CONTAINS the per-method subdirs (baseline/, smc/,
fksteering/, bon_mcts/). If you pass a parent that holds a single run_*/ dir, it
descends automatically.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from PIL import Image, ImageDraw, ImageFont

# (method subdir, image-filename suffix, column label)
DEFAULT_METHODS = [
    ("baseline", "base", "base"),
    ("smc", "smc", "DAS"),
    ("fksteering", "smc", "FK-Steering"),
    ("bon_mcts", "mcts", "MCTS (ours)"),
]


def _font(sz: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def _parse_prompts(spec):
    out = []
    for tok in spec.replace(",", " ").split():
        if "-" in tok:
            x, y = tok.split("-"); out.extend(range(int(x), int(y) + 1))
        elif tok.isdigit():
            out.append(int(tok))
    return sorted(set(out))


def _rewards(method_dir, seed):
    """prompt_index -> search score, from logs/rank_*.jsonl (filtered by seed)."""
    out = {}
    for lf in sorted(glob.glob(os.path.join(method_dir, "logs", "rank_*.jsonl"))):
        for line in open(lf, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "prompt_index" not in r or "score" not in r:
                continue
            if seed is not None and "seed" in r and str(r["seed"]) != str(seed):
                continue
            out[int(r["prompt_index"])] = float(r["score"])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="dir containing per-method subdirs (or a parent with one run_*/)")
    p.add_argument("--seed", default="42")
    p.add_argument("--prompts", default=None, help="subset '0-7'/'0,3,5' (default: all present)")
    p.add_argument("--top", type=int, default=8, help="show top-N prompts by (ours - base) delta")
    p.add_argument("--rank", choices=["delta", "ours", "index"], default="delta")
    p.add_argument("--thumb", type=int, default=300)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    run_dir = a.run_dir
    if not os.path.isdir(os.path.join(run_dir, "baseline")):
        sub = sorted(glob.glob(os.path.join(run_dir, "run_*")), reverse=True)
        if sub and os.path.isdir(os.path.join(sub[0], "baseline")):
            run_dir = sub[0]
            print(f"[methods] descended into {run_dir}")

    methods = [m for m in DEFAULT_METHODS if os.path.isdir(os.path.join(run_dir, m[0]))]
    if not methods:
        raise SystemExit(f"no method subdirs found under {run_dir} (looked for {[m[0] for m in DEFAULT_METHODS]})")
    rew = {sub: _rewards(os.path.join(run_dir, sub), a.seed) for sub, _s, _l in methods}
    base_r = rew.get("baseline", {})
    ours_r = rew.get("bon_mcts", {})

    def img_path(sub, suf, pi):
        return os.path.join(run_dir, sub, "images", f"p{pi:05d}_{suf}.png")

    # prompts present = those with a baseline image
    base_sub, base_suf = "baseline", "base"
    pis = sorted(int(os.path.basename(f)[1:6])
                 for f in glob.glob(os.path.join(run_dir, base_sub, "images", "p*_%s.png" % base_suf)))
    if a.prompts:
        want = set(_parse_prompts(a.prompts)); pis = [i for i in pis if i in want]
    if not pis:
        raise SystemExit(f"no baseline images under {run_dir}/baseline/images")

    def rank_key(pi):
        if a.rank == "index":
            return -pi
        if a.rank == "ours":
            return ours_r.get(pi, -1e9)
        return ours_r.get(pi, -1e9) - base_r.get(pi, -1e9)   # delta
    pis.sort(key=rank_key, reverse=True)
    pis = pis[: a.top]

    T, lw, hh, pad = int(a.thumb), 60, 40, 8
    cols = methods
    W = lw + len(cols) * (T + pad) + pad
    H = hh + len(pis) * (T + pad) + pad
    canvas = Image.new("RGB", (W, H), (245, 245, 247))
    d = ImageDraw.Draw(canvas)
    fh, fr, fb = _font(20), _font(13), _font(16)
    d.text((10, 10), f"SD3.5 qualitative comparison (seed {a.seed})", fill=(20, 20, 20), font=fh)
    for ci, (_sub, _suf, lab) in enumerate(cols):
        x = lw + ci * (T + pad) + pad
        ours = lab.startswith("MCTS")
        d.text((x + 4, 16), lab, fill=((30, 130, 70) if ours else (40, 40, 40)), font=fb)

    for row, pi in enumerate(pis):
        y = hh + row * (T + pad) + pad
        d.text((6, y + T // 2 - 8), f"p{pi:05d}", fill=(90, 90, 90), font=fr)
        row_scores = {sub: rew.get(sub, {}).get(pi) for sub, _s, _l in cols}
        vals = [v for v in row_scores.values() if v is not None]
        best = max(vals) if vals else None
        for ci, (sub, suf, lab) in enumerate(cols):
            x = lw + ci * (T + pad) + pad
            pth = img_path(sub, suf, pi)
            if os.path.exists(pth):
                canvas.paste(Image.open(pth).convert("RGB").resize((T, T)), (x, y))
            else:
                d.rectangle([x, y, x + T, y + T], outline=(180, 180, 180))
                d.text((x + 8, y + T // 2), "(missing)", fill=(150, 150, 150), font=fr)
            ours = lab.startswith("MCTS")
            if ours:
                d.rectangle([x, y, x + T - 1, y + T - 1], outline=(46, 158, 87), width=4)
            s = row_scores.get(sub)
            if s is not None:
                is_best = best is not None and abs(s - best) < 1e-9
                bg = (46, 158, 87) if is_best else (0, 0, 0)
                d.rectangle([x, y + T - 20, x + 74, y + T], fill=bg)
                d.text((x + 3, y + T - 18), f"{s:.3f}", fill=(255, 255, 255), font=fr)

    out = a.out or os.path.join(run_dir, "methods_compare.png")
    canvas.save(out)
    won = sum(1 for pi in pis if ours_r.get(pi, -1e9) >= (max((rew.get(s, {}).get(pi, -1e9)
             for s, _su, _l in cols), default=-1e9) - 1e-9))
    print(f"[methods] wrote {out}  ({len(pis)} prompts, MCTS best in {won})")


if __name__ == "__main__":
    main()
