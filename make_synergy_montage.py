#!/usr/bin/env python3
"""Per-prompt synergy montage built to SHOW cfg + prompt => 1+1>2.

Columns (per prompt): standard (no-search single seed) | +cfg | +prompt | both.
Each searched cell is annotated with its reward (the SEARCH_REWARD, e.g.
composite_3), read from the runner's rank_*.jsonl logs. A per-row synergy value

    synergy = both - ( (+cfg) + (+prompt) - standard )      ( > 0  =>  1+1>2 )

is printed on the left. With --rank_by_synergy the prompts with the largest
synergy are shown first, so the montage showcases the super-additive cases.

Usage:
  python make_synergy_montage.py --run_root /data/ygu/synergy_local/run_<ts>/sid \
    --seed 42 --rank_by_synergy --top 12 --out synergy_montage.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from PIL import Image, ImageDraw, ImageFont

# Searched cells: (method dir, display label, image suffix).
CFG_CELL = ("bon_mcts_adaptive_cfg", "+cfg")
PROMPT_CELL = ("bon_mcts_rewrite_only", "+prompt")
BOTH_CELL = ("bon_mcts_full", "both")
SEARCH_CELLS = [CFG_CELL, PROMPT_CELL, BOTH_CELL]


def _font(sz: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def _find_img(run_root: str, cell: str, slug: str, seed: str | None, suffix: str) -> str | None:
    pats = []
    if seed:
        pats.append(f"{run_root}/**/seed{seed}/**/{cell}/images/{slug}_{suffix}.png")
    pats.append(f"{run_root}/**/{cell}/images/{slug}_{suffix}.png")
    for pat in pats:
        m = sorted(glob.glob(pat, recursive=True))
        if m:
            return m[0]
    return None


def _read_scores(run_root: str, cell: str, seed: str | None) -> tuple[dict[int, float], dict[int, float]]:
    """Scan a cell's rank logs -> ({prompt_index: search_score}, {prompt_index: base_score}).
    Filtered to `seed` when given; otherwise keeps the best (max) search score per prompt."""
    search: dict[int, float] = {}
    base: dict[int, float] = {}
    for log in glob.glob(f"{run_root}/**/{cell}/logs/rank_*.jsonl", recursive=True):
        if log.endswith("_rewrite_examples.jsonl"):
            continue
        try:
            with open(log, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if seed and str(r.get("seed")) != str(seed):
                        continue
                    if "prompt_index" not in r or "score" not in r:
                        continue
                    pid = int(r["prompt_index"])
                    sc = float(r["score"])
                    if str(r.get("mode")) == "base":
                        base[pid] = sc
                    else:
                        search[pid] = max(search.get(pid, float("-inf")), sc)
                        if "baseline_score" in r and pid not in base:
                            base[pid] = float(r["baseline_score"])
        except Exception:
            continue
    return search, base


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
    p.add_argument("--seed", default=None, help="pin one seed (recommended, e.g. 42) for consistent img+score")
    p.add_argument("--prompts", default=None, help="explicit indices '0-11'/'0,3,7' (default: all with data)")
    p.add_argument("--rank_by_synergy", action="store_true", help="order prompts by synergy (1+1>2) descending")
    p.add_argument("--top", type=int, default=12, help="max prompts to show")
    p.add_argument("--prompt_file", default=None, help="prompt text for labels (default: <run_root>/../_prompts/backend_*.txt)")
    p.add_argument("--thumb", type=int, default=256)
    p.add_argument("--out", default="synergy_montage.png")
    a = p.parse_args()

    run_root = a.run_root.rstrip("/")
    prompts_txt: list[str] = []
    pf = a.prompt_file or (glob.glob(f"{os.path.dirname(run_root)}/_prompts/backend_*.txt") or [None])[0]
    if pf and os.path.exists(pf):
        prompts_txt = [ln.rstrip("\n") for ln in open(pf, encoding="utf-8") if ln.strip()]

    cfg_s, base_a = _read_scores(run_root, CFG_CELL[0], a.seed)
    prm_s, base_b = _read_scores(run_root, PROMPT_CELL[0], a.seed)
    both_s, base_c = _read_scores(run_root, BOTH_CELL[0], a.seed)
    base_s = {**base_a, **base_b, **base_c}  # standard (no-search) score per prompt

    # prompts that have all four numbers
    have = sorted(set(cfg_s) & set(prm_s) & set(both_s) & set(base_s))
    if a.prompts:
        want = set(_parse_prompts(a.prompts))
        have = [i for i in have if i in want]

    def synergy(pi: int) -> float:
        return both_s[pi] - (cfg_s[pi] + prm_s[pi] - base_s[pi])

    if a.rank_by_synergy:
        have.sort(key=synergy, reverse=True)
    idxs = have[: a.top]
    if not idxs:
        raise SystemExit(f"No prompts with scores in all cells under {run_root} (seed={a.seed}).")

    cols = [("__std__", "standard")] + SEARCH_CELLS
    T, label_w, head_h, pad = int(a.thumb), 320, 34, 8
    rows = len(idxs)
    W = label_w + len(cols) * (T + pad) + pad
    H = head_h + rows * (T + pad) + pad
    canvas = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(canvas)
    fh, fr, fsc = _font(18), _font(13), _font(15)

    for c, (_cell, lab) in enumerate(cols):
        x = label_w + c * (T + pad) + pad
        d.text((x + 6, 8), lab, fill=(235, 235, 235), font=fh)

    for r, pi in enumerate(idxs):
        slug = f"p{pi:05d}"
        y = head_h + r * (T + pad) + pad
        syn = synergy(pi)
        syn_col = (120, 230, 140) if syn > 0 else (230, 140, 140)
        d.text((8, y + 4), slug, fill=(180, 200, 255), font=fr)
        d.text((8, y + 22), f"synergy {syn:+.3f}", fill=syn_col, font=fsc)
        d.text((8, y + 42), "1+1>2" if syn > 0 else "additive", fill=syn_col, font=fr)
        # wrap prompt text below
        words, line, ly = (prompts_txt[pi].split() if pi < len(prompts_txt) else []), "", y + 66
        for w in words:
            if len(line) + len(w) + 1 > 42:
                d.text((8, ly), line, fill=(190, 190, 190), font=fr); ly += 15; line = w
            else:
                line = (line + " " + w).strip()
            if ly > y + T - 15:
                break
        if line and ly <= y + T - 15:
            d.text((8, ly), line, fill=(190, 190, 190), font=fr)

        for c, (cell, _lab) in enumerate(cols):
            x = label_w + c * (T + pad) + pad
            if cell == "__std__":
                path, score = _find_img(run_root, BOTH_CELL[0], slug, a.seed, "base"), base_s[pi]
                if path is None:
                    for mc, _, in SEARCH_CELLS:
                        path = _find_img(run_root, mc, slug, a.seed, "base")
                        if path:
                            break
            else:
                path = _find_img(run_root, cell, slug, a.seed, "mcts")
                score = {CFG_CELL[0]: cfg_s, PROMPT_CELL[0]: prm_s, BOTH_CELL[0]: both_s}[cell][pi]
            if path and os.path.exists(path):
                canvas.paste(Image.open(path).convert("RGB").resize((T, T)), (x, y))
            else:
                d.rectangle([x, y, x + T, y + T], outline=(70, 70, 70))
                d.text((x + 8, y + T // 2 - 6), "(missing)", fill=(140, 140, 140), font=fr)
            # reward chip
            d.rectangle([x, y + T - 20, x + 92, y + T], fill=(0, 0, 0))
            d.text((x + 4, y + T - 18), f"{score:.3f}", fill=(255, 235, 120), font=fsc)

    canvas.save(a.out)
    n_pos = sum(1 for pi in idxs if synergy(pi) > 0)
    print(f"[montage] wrote {a.out}  ({rows} prompts, {n_pos} with synergy>0)")


if __name__ == "__main__":
    main()
