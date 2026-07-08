#!/usr/bin/env python3
"""Qualitative method-comparison montage for SD3.5 (or any backend suite run).

Lays out, per prompt, one image column per method dir, reading each method's
images/ and per-prompt search reward from logs/rank_*.jsonl. Image suffix is
auto-detected from each method's own images/ (so smc / fksteering / smc_actdiff
all resolve correctly even though they share the `_smc` suffix in separate dirs).

Rows are ranked by (highlight - baseline) reward delta; the highlighted method
column is boxed, and each row's best reward is badged green.

Usage:
  # default 5-method comparison
  python plot_methods_compare.py --run_dir <.../seedNN/run_TS> --seed 42 --top 8

  # focused: does ActDiff help SMC?
  python plot_methods_compare.py --run_dir <run_dir> --seed 42 \
    --methods "baseline smc smc_actdiff_cfg" --highlight smc_actdiff_cfg

--run_dir contains the per-method subdirs; pass a parent with one run_*/ and it
descends automatically.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

from PIL import Image, ImageDraw, ImageFont

# pretty labels; any method dir not listed falls back to its own name
LABELS = {
    "baseline": "base",
    "das": "DAS",
    "smc": "SMC",
    "smc_actdiff_cfg": "SMC+ActDiff",
    "smc_actdiff_full": "SMC+ActDiff",
    "fksteering": "FK-Steering",
    "fksteering_actdiff_cfg": "FK+ActDiff",
    "bon_mcts": "MCTS (ours)",
}
DEFAULT_ORDER = ["baseline", "das", "smc", "fksteering", "bon_mcts"]


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


def _detect_suffix(method_dir):
    """Infer the image suffix (e.g. 'base', 'smc', 'mcts', 'bon') from filenames
    like p00000_<suffix>.png, ignoring *_comp.png composites."""
    for f in sorted(glob.glob(os.path.join(method_dir, "images", "p*_*.png"))):
        b = os.path.basename(f)
        if b.endswith("_comp.png"):
            continue
        m = re.match(r"p\d{5}_(.+)\.png$", b)
        if m:
            return m.group(1)
    return None


def _rewards(method_dir, seed):
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
    p.add_argument("--run_dir", required=True, help="dir with per-method subdirs (or a parent with one run_*/)")
    p.add_argument("--seed", default="42")
    p.add_argument("--methods", default=None,
                   help="space/comma list of method dir names, in column order (default: baseline das smc fksteering bon_mcts)")
    p.add_argument("--highlight", default=None,
                   help="method dir to box (default: last method in --methods)")
    p.add_argument("--prompts", default=None, help="subset '0-7'/'0,3,5' (default: all present)")
    p.add_argument("--top", type=int, default=8, help="top-N prompts by (highlight - baseline) delta")
    p.add_argument("--rank", choices=["delta", "highlight", "index"], default="delta")
    p.add_argument("--thumb", type=int, default=300)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    run_dir = a.run_dir
    if not os.path.isdir(os.path.join(run_dir, "baseline")):
        sub = sorted(glob.glob(os.path.join(run_dir, "run_*")), reverse=True)
        if sub and os.path.isdir(os.path.join(sub[0], "baseline")):
            run_dir = sub[0]
            print(f"[methods] descended into {run_dir}")

    order = (a.methods.replace(",", " ").split() if a.methods else DEFAULT_ORDER)
    methods = []
    for d in order:
        mdir = os.path.join(run_dir, d)
        if not os.path.isdir(mdir):
            print(f"[methods] skip '{d}': no dir {mdir}")
            continue
        suf = _detect_suffix(mdir)
        if suf is None:
            print(f"[methods] skip '{d}': no images in {mdir}/images")
            continue
        methods.append((d, suf, LABELS.get(d, d)))
    if not methods:
        raise SystemExit(f"no usable method dirs under {run_dir} (tried {order})")

    highlight = a.highlight or methods[-1][0]
    rew = {d: _rewards(os.path.join(run_dir, d), a.seed) for d, _s, _l in methods}
    ref = methods[0][0]                       # first column = delta reference (baseline or, e.g., smc)
    base_r = rew.get(ref, {})
    hi_r = rew.get(highlight, {})

    def img_path(d, suf, pi):
        return os.path.join(run_dir, d, "images", f"p{pi:05d}_{suf}.png")

    b0 = methods[0]  # first column = reference for "present" prompts
    pis = sorted(int(os.path.basename(f)[1:6])
                 for f in glob.glob(os.path.join(run_dir, b0[0], "images", f"p*_{b0[1]}.png")))
    if a.prompts:
        want = set(_parse_prompts(a.prompts)); pis = [i for i in pis if i in want]
    if not pis:
        raise SystemExit(f"no images under {run_dir}/{b0[0]}/images")

    def rank_key(pi):
        if a.rank == "index":
            return -pi
        if a.rank == "highlight":
            return hi_r.get(pi, -1e9)
        return hi_r.get(pi, -1e9) - base_r.get(pi, -1e9)
    pis.sort(key=rank_key, reverse=True)
    pis = pis[: a.top]

    T, lw, hh, pad = int(a.thumb), 60, 40, 8
    W = lw + len(methods) * (T + pad) + pad
    H = hh + len(pis) * (T + pad) + pad
    canvas = Image.new("RGB", (W, H), (245, 245, 247))
    d = ImageDraw.Draw(canvas)
    fh, fr, fb = _font(20), _font(13), _font(16)
    d.text((10, 10), f"SD3.5 qualitative comparison (seed {a.seed})", fill=(20, 20, 20), font=fh)
    for ci, (mdir, _suf, lab) in enumerate(methods):
        x = lw + ci * (T + pad) + pad
        hl = (mdir == highlight)
        d.text((x + 4, 16), lab, fill=((30, 130, 70) if hl else (40, 40, 40)), font=fb)

    for row, pi in enumerate(pis):
        y = hh + row * (T + pad) + pad
        d.text((6, y + T // 2 - 8), f"p{pi:05d}", fill=(90, 90, 90), font=fr)
        row_scores = {mdir: rew.get(mdir, {}).get(pi) for mdir, _s, _l in methods}
        vals = [v for v in row_scores.values() if v is not None]
        best = max(vals) if vals else None
        for ci, (mdir, suf, lab) in enumerate(methods):
            x = lw + ci * (T + pad) + pad
            pth = img_path(mdir, suf, pi)
            if os.path.exists(pth):
                canvas.paste(Image.open(pth).convert("RGB").resize((T, T)), (x, y))
            else:
                d.rectangle([x, y, x + T, y + T], outline=(180, 180, 180))
                d.text((x + 8, y + T // 2), "(missing)", fill=(150, 150, 150), font=fr)
            if mdir == highlight:
                d.rectangle([x, y, x + T - 1, y + T - 1], outline=(46, 158, 87), width=4)
            s = row_scores.get(mdir)
            if s is not None:
                is_best = best is not None and abs(s - best) < 1e-9
                bg = (46, 158, 87) if is_best else (0, 0, 0)
                d.rectangle([x, y + T - 20, x + 74, y + T], fill=bg)
                d.text((x + 3, y + T - 18), f"{s:.3f}", fill=(255, 255, 255), font=fr)

    out = a.out or os.path.join(run_dir, "methods_compare.png")
    canvas.save(out)
    won = sum(1 for pi in pis if hi_r.get(pi, -1e9) >= base_r.get(pi, -1e9))
    print(f"[methods] wrote {out}  ({len(pis)} prompts; highlight='{highlight}' >= base in {won})")


if __name__ == "__main__":
    main()
