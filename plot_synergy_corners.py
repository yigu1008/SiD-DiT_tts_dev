#!/usr/bin/env python3
"""Qualitative 1+1>2 montage from the fixed cfg x prompt grid (no search).

Reads a run_cfg_prompt_grid.py run (cfg_prompt_grid.csv + images/) and lays out,
per prompt, the FOUR corners of the cfg x prompt rectangle:

    baseline  = (original prompt, low cfg)
    +cfg      = (original prompt, high cfg)
    +prompt   = (rewritten prompt, low cfg)
    +both     = (rewritten prompt, high cfg)   <- should visibly pop

Synergy = reward(+both) - (reward(+cfg) + reward(+prompt) - reward(baseline)).
The rewrite shown for +prompt / +both is the SAME variant (the one that wins at
high cfg), so the only thing toggling across the 2x2 is cfg and prompt -- a clean
"1+1>2" read. Rows are ranked by synergy so the strongest examples surface first.

Usage:
  python plot_synergy_corners.py --run_root /data/ygu/cfg_prompt_grid/sd35_corners \
    --seed 42 --top 8
Defaults: cfg_lo / cfg_hi = min / max cfg present in the CSV.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

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
    p.add_argument("--run_root", required=True, help="dir with images/ and cfg_prompt_grid.csv")
    p.add_argument("--seed", default="42")
    p.add_argument("--cfg_lo", type=float, default=None, help="baseline cfg (default: min cfg in CSV)")
    p.add_argument("--cfg_hi", type=float, default=None, help="high cfg (default: max cfg in CSV)")
    p.add_argument("--prompts", default=None, help="subset '0-9'/'0,3,7' (default: all)")
    p.add_argument("--top", type=int, default=8, help="show the top-N prompts by synergy")
    p.add_argument("--thumb", type=int, default=320)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    csv_path = os.path.join(a.run_root, "cfg_prompt_grid.csv")
    img_dir = os.path.join(a.run_root, "images")
    # reward[(pi, variant, cfg)] = reward ; text[pi][variant] = variant_text
    rew: dict[tuple[int, int, float], float] = {}
    text: dict[int, dict[int, str]] = defaultdict(dict)
    cfgs, variants = set(), set()
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        if str(int(r["seed"])) != str(a.seed):
            continue
        pi, vi, c, rw = int(r["prompt_index"]), int(r["variant"]), float(r["cfg"]), float(r["reward"])
        rew[(pi, vi, c)] = rw; cfgs.add(c); variants.add(vi)
        text[pi][vi] = r.get("variant_text", "")
    if not rew:
        raise SystemExit(f"no rows for seed {a.seed} in {csv_path}")

    cfgs = sorted(cfgs)
    cfg_lo = a.cfg_lo if a.cfg_lo is not None else cfgs[0]
    cfg_hi = a.cfg_hi if a.cfg_hi is not None else cfgs[-1]
    rewrites = sorted(v for v in variants if v != 0) or [0]  # variant 0 = original

    def corners(pi):
        base = rew.get((pi, 0, cfg_lo))
        plus_cfg = rew.get((pi, 0, cfg_hi))
        if base is None or plus_cfg is None:
            return None
        # pick the rewrite variant that wins at HIGH cfg -> the "+both" cell
        cand = [(rew.get((pi, v, cfg_hi), -1e9), v) for v in rewrites]
        best_both, v_star = max(cand)
        plus_prompt = rew.get((pi, v_star, cfg_lo))
        if plus_prompt is None or best_both < -1e8:
            return None
        syn = best_both - (plus_cfg + plus_prompt - base)
        return {"v": v_star, "base": base, "cfg": plus_cfg, "prompt": plus_prompt,
                "both": best_both, "syn": syn}

    pis = sorted({pi for (pi, _v, _c) in rew})
    if a.prompts:
        want = set(_parse_prompts(a.prompts)); pis = [i for i in pis if i in want]
    scored = [(pi, c) for pi in pis if (c := corners(pi))]
    scored.sort(key=lambda t: t[1]["syn"], reverse=True)
    scored = scored[: a.top]
    if not scored:
        raise SystemExit("no prompt had all four corners present")

    cols = [("base", 0, cfg_lo, "baseline"),
            ("cfg", 0, cfg_hi, f"+cfg ({cfg_hi:g})"),
            ("prompt", None, cfg_lo, "+prompt"),
            ("both", None, cfg_hi, f"+both ({cfg_hi:g})")]
    T, lw, hh, pad = int(a.thumb), 210, 30, 8
    W = lw + len(cols) * (T + pad) + pad
    H = hh + len(scored) * (T + pad) + pad
    canvas = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(canvas)
    fh, fr, fs = _font(18), _font(13), _font(15)
    d.text((8, 7), f"cfg x prompt synergy — 1+1>2  (seed {a.seed}, cfg {cfg_lo:g}->{cfg_hi:g})",
           fill=(235, 235, 235), font=fh)
    for ci, (_k, _v, _c, lab) in enumerate(cols):
        d.text((lw + ci * (T + pad) + pad + 4, 9), lab, fill=(220, 220, 220), font=fs)

    for row, (pi, c) in enumerate(scored):
        y = hh + row * (T + pad) + pad
        d.text((6, y + 4), f"p{pi:05d}", fill=(180, 200, 255), font=fr)
        d.text((6, y + 22), (text[pi].get(0, "")[:30]), fill=(185, 185, 185), font=fr)
        syn_col = (120, 235, 140) if c["syn"] > 0 else (235, 140, 120)
        d.text((6, y + 44), f"synergy\n{c['syn']:+.3f}", fill=syn_col, font=fs)
        for ci, (key, vi, cfv, _lab) in enumerate(cols):
            x = lw + ci * (T + pad) + pad
            v_use = c["v"] if vi is None else vi
            fn = f"p{pi:05d}_s{a.seed}_v{v_use}_cfg{cfv:.2f}.png"
            path = os.path.join(img_dir, fn)
            if os.path.exists(path):
                canvas.paste(Image.open(path).convert("RGB").resize((T, T)), (x, y))
            else:
                d.rectangle([x, y, x + T, y + T], outline=(70, 70, 70))
                d.text((x + 8, y + T // 2), "(missing)", fill=(140, 140, 140), font=fr)
            if key == "both":
                d.rectangle([x, y, x + T - 1, y + T - 1], outline=(90, 220, 120), width=4)
            d.rectangle([x, y + T - 20, x + 84, y + T], fill=(0, 0, 0))
            d.text((x + 3, y + T - 18), f"{c[key]:.3f}", fill=(255, 235, 120), font=fr)

    out = a.out or os.path.join(a.run_root, "synergy_corners.png")
    canvas.save(out)
    n_pos = sum(1 for _pi, c in scored if c["syn"] > 0)
    print(f"[corners] wrote {out}  ({len(scored)} prompts, {n_pos} with synergy>0)")


if __name__ == "__main__":
    main()
