#!/usr/bin/env python3
"""Paper-grade ActDiff (bon_mcts) pipeline visualization.

Renders the tree of choices made at each step of the bon_mcts pipeline:

    1. Initial noise z₀
    2. Prescreen: N_SEEDS candidates, score, pick top-K
    3. Refine: at each key step, MCTS expands the action axis (CFG / variant /
       correction) per surviving seed; UCB selects the best child
    4. Final image: highest-Q terminal node

Two modes:

    SCHEMATIC (default — fixed N_SEEDS=8, TOPK=2, KEY_STEPS=2, CFG_BANK=4):
        python plot_actdiff_tree.py --out actdiff_pipeline.png

    REAL DATA (read a prompt's diagnostics):
        python plot_actdiff_tree.py --run_root <RUN_ROOT> --method bon_mcts \
            --prompt_index 7 --out actdiff_prompt_7.png

Real-data mode reads `aggregate_ddp.json` and per-prompt MCTS traces.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ── Colors ─────────────────────────────────────────────────────────────────
C_NOISE       = "#888888"
C_PRESCREEN   = "#3D77BE"
C_TOPK        = "#3D77BE"
C_REFINE_NODE = "#E08540"
C_BEST_PATH   = "#B73B3B"
C_EDGE        = "#999999"
C_BEST_EDGE   = "#B73B3B"
C_SCORE_HIGH  = "#2E7D32"
C_SCORE_LOW   = "#C62828"


@dataclass
class TreeNode:
    label: str
    score: float | None = None
    x: float = 0.0
    y: float = 0.0
    color: str = C_REFINE_NODE
    size: float = 1.0
    on_best_path: bool = False
    children: list["TreeNode"] = field(default_factory=list)
    parent: "TreeNode | None" = None
    edge_label: str = ""


def _set_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })


def _score_to_color(score: float | None, lo: float = -0.5, hi: float = 2.0) -> str:
    if score is None:
        return C_NOISE
    t = max(0.0, min(1.0, (score - lo) / max(1e-6, hi - lo)))
    # Linear blend low (red) → high (green)
    def _mix(a, b, t):
        return tuple(int(round(a[i] * (1 - t) + b[i] * t)) for i in range(3))
    lo_rgb = (198, 40, 40)   # C_SCORE_LOW
    hi_rgb = (46, 125, 50)   # C_SCORE_HIGH
    r, g, b = _mix(lo_rgb, hi_rgb, t)
    return f"#{r:02x}{g:02x}{b:02x}"


def build_schematic_tree(
    n_seeds: int = 8,
    topk: int = 2,
    key_steps: int = 2,
    cfg_bank: int = 4,
    rng_seed: int = 42,
) -> TreeNode:
    import random
    rng = random.Random(rng_seed)

    # Root = initial noise z₀
    root = TreeNode(label="$z_0$\nnoise", color=C_NOISE)

    # Level 1: prescreen — N_SEEDS candidates
    prescreen_scores = sorted([rng.uniform(0.2, 1.4) for _ in range(n_seeds)], reverse=True)
    prescreen: list[TreeNode] = []
    for i, s in enumerate(prescreen_scores):
        n = TreeNode(
            label=f"$s_{i}$",
            score=s,
            color=C_PRESCREEN,
            edge_label="" if i > 0 else "prescreen\nN seeds",
        )
        n.parent = root
        root.children.append(n)
        prescreen.append(n)

    # Mark top-K as surviving
    top_seeds = prescreen[:topk]
    for n in top_seeds:
        n.color = C_TOPK
        n.size = 1.5

    # Levels 2..(key_steps+1): refine — each surviving seed branches CFG_BANK ways per key step
    rng2 = random.Random(rng_seed + 1)
    current_level = top_seeds[:]
    for ks in range(key_steps):
        next_level: list[TreeNode] = []
        for parent in current_level:
            # children: CFG_BANK candidate actions
            cfgs = [round(1.0 + 0.5 * j, 2) for j in range(cfg_bank)]
            child_nodes = []
            for cfg in cfgs:
                # Reward ≈ parent + small bump, plus per-cfg noise
                if parent.score is None:
                    base = 1.0
                else:
                    base = parent.score
                ch_score = base + rng2.uniform(-0.15, 0.3)
                edge_lbl = f"cfg={cfg}"
                ch = TreeNode(label=f"$r$={ch_score:.2f}", score=ch_score,
                              color=C_REFINE_NODE, edge_label=edge_lbl)
                ch.parent = parent
                parent.children.append(ch)
                child_nodes.append(ch)
            # UCB-like selection: keep the best
            best_child = max(child_nodes, key=lambda c: c.score or 0.0)
            best_child.color = "#D55B1A"
            best_child.size = 1.4
            next_level.append(best_child)
        current_level = next_level

    # Mark the best path
    final_best = max(current_level, key=lambda c: c.score or 0.0)
    final_best.color = C_BEST_PATH
    final_best.on_best_path = True
    final_best.size = 1.6
    cur = final_best
    while cur is not None:
        cur.on_best_path = True
        cur = cur.parent

    return root


def _assign_positions(root: TreeNode, x_pad: float = 0.6) -> tuple[float, float]:
    """Assign (x, y) coordinates to every node — simple left-to-right BFS layout."""
    # Group by depth.
    levels: dict[int, list[TreeNode]] = {}
    def walk(n: TreeNode, d: int) -> None:
        levels.setdefault(d, []).append(n)
        for ch in n.children:
            walk(ch, d + 1)
    walk(root, 0)
    max_w = 0.0
    for d, nodes in levels.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            node.x = (i - (n - 1) / 2.0) * x_pad
            node.y = -d
        max_w = max(max_w, n * x_pad)
    return max_w, len(levels)


def _draw_tree(ax: plt.Axes, root: TreeNode, show_edge_labels: bool = True) -> None:
    # Edges first.
    def walk(n: TreeNode) -> None:
        for ch in n.children:
            color = C_BEST_EDGE if (n.on_best_path and ch.on_best_path) else C_EDGE
            lw = 2.5 if (n.on_best_path and ch.on_best_path) else 0.9
            ax.plot([n.x, ch.x], [n.y, ch.y], "-", color=color, lw=lw, alpha=0.85, zorder=1)
            if show_edge_labels and ch.edge_label:
                mx, my = (n.x + ch.x) / 2.0, (n.y + ch.y) / 2.0
                ax.text(mx + 0.04, my, ch.edge_label, fontsize=8,
                        color="#444", style="italic",
                        ha="left", va="center", zorder=2)
            walk(ch)
    walk(root)

    # Nodes
    def walk_node(n: TreeNode) -> None:
        # Choose fill color: best-path gets red, else score-based.
        if n.on_best_path:
            fill = C_BEST_PATH if n.parent is not None or n.color == C_BEST_PATH else n.color
        else:
            fill = _score_to_color(n.score) if n.score is not None else n.color
        edge_color = "black" if n.on_best_path else "#444"
        edge_w = 1.6 if n.on_best_path else 0.8
        r = 0.08 * n.size
        circ = mpatches.Circle((n.x, n.y), r,
                                facecolor=fill, edgecolor=edge_color,
                                linewidth=edge_w, zorder=3)
        ax.add_patch(circ)
        ax.text(n.x, n.y - r - 0.06, n.label, ha="center", va="top", fontsize=8,
                fontweight="bold" if n.on_best_path else "normal",
                color="#222", zorder=4)
        for ch in n.children:
            walk_node(ch)
    walk_node(root)


def _draw_level_annotations(ax: plt.Axes,
                            level_labels: list[str],
                            x_left: float, x_right: float,
                            n_levels: int) -> None:
    for d, lab in enumerate(level_labels):
        ax.text(x_left - 0.3, -d, lab, fontsize=10, ha="right", va="center",
                color="#444", style="italic")


def _draw_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_NOISE,
               markersize=10, markeredgecolor="black", label="initial noise $z_0$"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_PRESCREEN,
               markersize=10, markeredgecolor="#444", label="prescreen candidate"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_REFINE_NODE,
               markersize=10, markeredgecolor="#444", label="refine action (CFG / variant)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_BEST_PATH,
               markersize=12, markeredgecolor="black", label="UCB-selected best path"),
        Line2D([0], [0], color=C_BEST_EDGE, lw=2.5, label="best path edges"),
        Line2D([0], [0], color=C_EDGE, lw=0.9, label="other actions"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=False)


def _load_real_data(run_root: Path, method: str, prompt_index: int) -> TreeNode | None:
    """Try to reconstruct a tree from per-run diagnostics. Returns None on failure."""
    # Search candidates for per-prompt MCTS diagnostics.  Different runners
    # name the file slightly differently; try a few.
    candidates: list[Path] = []
    for pat in (
        f"**/{method}/**/diagnostics_prompt_{prompt_index:04d}.json",
        f"**/{method}/**/diagnostics_prompt_{prompt_index:05d}.json",
        f"**/{method}/**/diagnostics_{prompt_index}.json",
        f"**/{method}/**/per_prompt_diagnostics.json",
        f"**/{method}/**/aggregate_ddp.json",
    ):
        candidates.extend(sorted(run_root.glob(pat)))
    diag = None
    for path in candidates:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        # Pull the per-prompt entry if this is an aggregate
        if isinstance(data, dict) and "per_prompt" in data:
            entries = data["per_prompt"]
            if isinstance(entries, list) and prompt_index < len(entries):
                diag = entries[prompt_index]
                break
        elif isinstance(data, dict) and "bon_mcts" in data:
            diag = data["bon_mcts"]
            break
        elif isinstance(data, dict) and "prescreen_ranked" in data:
            diag = data
            break
    if diag is None:
        print(f"[actdiff-tree] no per-prompt diagnostics found for prompt {prompt_index}; "
              f"falling back to schematic.")
        return None

    # Build a tree using diagnostics (best-effort — may be missing per-action data).
    root = TreeNode(label="$z_0$\nnoise", color=C_NOISE)
    prescreen = diag.get("prescreen_ranked", [])
    for i, row in enumerate(prescreen):
        s = float(row.get("prescreen_score", 0.0))
        n = TreeNode(label=f"seed {row.get('seed', i)}\n$r$={s:.2f}",
                     score=s, color=C_PRESCREEN,
                     edge_label="" if i > 0 else "prescreen")
        n.parent = root
        root.children.append(n)
    refine = diag.get("tree_refine", [])
    top_seeds = {int(row.get("seed", -1)): row for row in refine}
    for n in root.children:
        sd_str = n.label.split(" ")[1].split("\n")[0]
        try:
            sd = int(sd_str)
        except ValueError:
            continue
        if sd in top_seeds:
            n.color = C_TOPK
            n.size = 1.5
            refine_row = top_seeds[sd]
            r_score = float(refine_row.get("tree_search_score", 0.0))
            leaf = TreeNode(label=f"refined\n$r$={r_score:.2f}",
                            score=r_score, color=C_REFINE_NODE,
                            edge_label=f"{int(refine_row.get('n_sims_used', 0))} sims")
            leaf.parent = n
            n.children.append(leaf)
    # Mark best path
    winner_seed = diag.get("winner_seed", None)
    if winner_seed is not None:
        for n in root.children:
            if f"seed {winner_seed}" in n.label:
                n.on_best_path = True
                for ch in n.children:
                    ch.on_best_path = True
                    ch.color = C_BEST_PATH
                break
        root.on_best_path = True
    return root


def _tree_to_dict(root: TreeNode) -> dict:
    """Serialize a TreeNode tree as JSON-safe dict for re-rendering later."""
    def walk(n: TreeNode) -> dict:
        return {
            "label": n.label,
            "score": n.score,
            "x": n.x,
            "y": n.y,
            "color": n.color,
            "size": n.size,
            "on_best_path": n.on_best_path,
            "edge_label": n.edge_label,
            "children": [walk(c) for c in n.children],
        }
    return walk(root)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Output: --out_dir wins when --out is not given.  Default is `figures/raw/`
    # so every invocation drops a timestamped (png + json) pair alongside prior
    # ones — no overwrites, easy to browse.
    p.add_argument("--out", default=None, type=Path,
                   help="Output PNG path (explicit).  If omitted, writes to "
                        "<out_dir>/actdiff_<timestamp>_<mode>.png.")
    p.add_argument("--out_dir", default=Path("figures/raw"), type=Path,
                   help="Output folder when --out is not given.  Default: figures/raw/.")
    p.add_argument("--save_json", action=argparse.BooleanOptionalAction, default=True,
                   help="Save the tree structure as a sibling .json (for re-rendering).")
    p.add_argument("--mode", choices=["schematic", "real"], default="schematic")
    p.add_argument("--run_root", type=Path, default=None,
                   help="Required if --mode=real.  RUN_ROOT containing per-prompt diagnostics.")
    p.add_argument("--method", default="bon_mcts")
    p.add_argument("--prompt_index", type=int, default=0)
    # Schematic knobs
    p.add_argument("--n_seeds", type=int, default=8)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--key_steps", type=int, default=2)
    p.add_argument("--cfg_bank", type=int, default=4)
    p.add_argument("--rng_seed", type=int, default=42)
    p.add_argument("--title", default="ActDiff (bon_mcts) pipeline")
    p.add_argument("--no_edge_labels", action="store_true")
    p.add_argument("--prompt", default=None,
                   help="Optional prompt text to display as a subtitle / caption (useful for paper figures).")
    args = p.parse_args()

    root: TreeNode | None = None
    if args.mode == "real":
        if args.run_root is None:
            raise SystemExit("--mode=real requires --run_root")
        root = _load_real_data(args.run_root, args.method, args.prompt_index)
    if root is None:
        print("[actdiff-tree] using schematic mode")
        root = build_schematic_tree(
            n_seeds=args.n_seeds, topk=args.topk,
            key_steps=args.key_steps, cfg_bank=args.cfg_bank,
            rng_seed=args.rng_seed,
        )

    _set_paper_style()
    max_w, n_levels = _assign_positions(root, x_pad=0.6)
    fig_w = max(8.0, max_w + 4.0)
    fig_h = max(5.0, n_levels * 1.6 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _draw_tree(ax, root, show_edge_labels=not args.no_edge_labels)

    # Level labels on the left
    if args.mode == "schematic":
        level_labels = ["initial noise"] + ["prescreen"] + [
            f"key step {k+1}" for k in range(args.key_steps)
        ]
    else:
        level_labels = ["initial noise", "prescreen", "refine (MCTS)"]
    _draw_level_annotations(ax, level_labels, -max_w / 2, max_w / 2, n_levels)

    _draw_legend(ax)

    ax.set_xlim(-max_w / 2 - 0.8, max_w / 2 + 0.8)
    ax.set_ylim(-n_levels - 0.5, 0.8)
    ax.set_xticks([]); ax.set_yticks([])
    title_str = args.title
    if args.prompt:
        title_str = f"{args.title}\nprompt: “{args.prompt}”"
    ax.set_title(title_str, fontsize=13, fontweight="bold", pad=10)

    fig.tight_layout()

    # Resolve output path: explicit --out wins; else <out_dir>/actdiff_<ts>_<mode>.png
    if args.out is not None:
        out_png = Path(args.out)
    else:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = args.mode
        if args.mode == "real":
            suffix = f"real_p{args.prompt_index}_{args.method}"
        out_png = args.out_dir / f"actdiff_{ts}_{suffix}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"[actdiff-tree] saved → {out_png}")

    if args.save_json:
        out_json = out_png.with_suffix(".json")
        out_json.write_text(json.dumps({
            "mode": args.mode,
            "title": args.title,
            "config": {
                "n_seeds": args.n_seeds, "topk": args.topk,
                "key_steps": args.key_steps, "cfg_bank": args.cfg_bank,
                "rng_seed": args.rng_seed, "method": args.method,
                "prompt_index": args.prompt_index,
            },
            "tree": _tree_to_dict(root),
        }, indent=2))
        print(f"[actdiff-tree] tree json → {out_json}")


if __name__ == "__main__":
    main()
