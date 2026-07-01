#!/usr/bin/env python3
"""Per-prompt synergy montage: rows = prompts, columns = the four cfg x prompt
cells (base / +cfg / +prompt / both) searched images, so you can SEE what each
axis adds and what is lost when an axis is removed.

Reads the images the runner saved as <cell>/images/<slug>_mcts.png (the cell's
searched result) and <slug>_base.png (the shared no-search baseline).

Usage:
  python make_synergy_montage.py \
    --run_root /data/ygu/synergy_local/run_<ts>/sid \
    --prompts 0-11 --seed 42 --include_baseline --out synergy_montage.png
"""
from __future__ import annotations

import argparse
import glob
import os

from PIL import Image, ImageDraw, ImageFont

# (cell method dir, display label). base = the (0,0) reference cell.
CELLS = [
    ("bon_mcts_static_cfg", "base (no axis)"),
    ("bon_mcts_adaptive_cfg", "+cfg"),
    ("bon_mcts_rewrite_only", "+prompt"),
    ("bon_mcts_full", "both"),
]


def _font(sz: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def _find(run_root: str, cell: str, slug: str, seed: str | None, suffix: str) -> str | None:
    pats = []
    if seed:
        pats.append(f"{run_root}/**/seed{seed}/**/{cell}/images/{slug}_{suffix}.png")
    pats.append(f"{run_root}/**/{cell}/images/{slug}_{suffix}.png")
    for pat in pats:
        m = sorted(glob.glob(pat, recursive=True))
        if m:
            return m[0]
    return None


def _parse_prompts(spec: str) -> list[int]:
    out: list[int] = []
    for tok in spec.replace(",", " ").split():
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif tok.isdigit():
            out.append(int(tok))
    return sorted(set(out))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_root", required=True, help="RUN_ROOT/<backend> (contains the per-cell dirs)")
    p.add_argument("--prompts", default="0-11", help="indices, e.g. '0-11' or '0,3,7'")
    p.add_argument("--seed", default=None, help="restrict to one seed (e.g. 42); default: any")
    p.add_argument("--prompt_file", default=None,
                   help="prompt text for row labels (default: <run_root>/../_prompts/backend_*.txt)")
    p.add_argument("--include_baseline", action="store_true", help="prepend the shared no-search baseline column")
    p.add_argument("--thumb", type=int, default=256)
    p.add_argument("--out", default="synergy_montage.png")
    a = p.parse_args()

    run_root = a.run_root.rstrip("/")
    prompts_txt: list[str] = []
    pf = a.prompt_file
    if pf is None:
        cand = glob.glob(f"{os.path.dirname(run_root)}/_prompts/backend_*.txt")
        pf = cand[0] if cand else None
    if pf and os.path.exists(pf):
        prompts_txt = [ln.rstrip("\n") for ln in open(pf, encoding="utf-8") if ln.strip()]

    idxs = _parse_prompts(a.prompts)
    cols = ([("__base__", "no-search")] if a.include_baseline else []) + CELLS
    T, label_w, head_h, pad = int(a.thumb), 300, 34, 8
    rows = len(idxs)
    W = label_w + len(cols) * (T + pad) + pad
    H = head_h + rows * (T + pad) + pad
    canvas = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(canvas)
    fh, fr = _font(18), _font(13)

    for c, (_cell, lab) in enumerate(cols):
        x = label_w + c * (T + pad) + pad
        d.text((x + 6, 8), lab, fill=(235, 235, 235), font=fh)

    n_found = 0
    for r, pi in enumerate(idxs):
        slug = f"p{pi:05d}"
        y = head_h + r * (T + pad) + pad
        label = prompts_txt[pi] if pi < len(prompts_txt) else ""
        d.text((8, y + 4), slug, fill=(180, 200, 255), font=fr)
        # wrap the prompt text into the label column
        words, line, ly = label.split(), "", y + 24
        for w in words:
            if len(line) + len(w) + 1 > 40:
                d.text((8, ly), line, fill=(200, 200, 200), font=fr); ly += 16; line = w
            else:
                line = (line + " " + w).strip()
            if ly > y + T - 16:
                break
        if line and ly <= y + T - 16:
            d.text((8, ly), line, fill=(200, 200, 200), font=fr)
        for c, (cell, _lab) in enumerate(cols):
            x = label_w + c * (T + pad) + pad
            if cell == "__base__":
                path = None
                for mcell, _ in CELLS:
                    path = _find(run_root, mcell, slug, a.seed, "base")
                    if path:
                        break
            else:
                path = _find(run_root, cell, slug, a.seed, "mcts")
            if path and os.path.exists(path):
                canvas.paste(Image.open(path).convert("RGB").resize((T, T)), (x, y))
                n_found += 1
            else:
                d.rectangle([x, y, x + T, y + T], outline=(70, 70, 70))
                d.text((x + 8, y + T // 2 - 6), "(missing)", fill=(140, 140, 140), font=fr)

    canvas.save(a.out)
    print(f"[montage] wrote {a.out}  ({rows} prompts x {len(cols)} cols, {n_found} images found)")


if __name__ == "__main__":
    main()
