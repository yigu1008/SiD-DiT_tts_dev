#!/usr/bin/env python3
"""Visualize an MCTS search trail collected by collect_mcts_trail.py.

Produces two figures:
  (1) tree.png   -- tree topology: nodes colored by mean Q (best rollout),
                    sized by visits, edges = parent→child labelled by action.
  (2) ladder.png -- time-axis ladder: x = simulation index, y = depth,
                    each sim plotted as a polyline from root to leaf,
                    color = rollout score, marker = expansion point.

Inputs:
  trail.jsonl   -- emitted by collect_mcts_trail.py
  run_meta.json -- emitted by collect_mcts_trail.py

Usage:
  python plot_mcts_trail.py --trail_dir /tmp/mcts_trail --out_dir /tmp/mcts_trail
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np


def load_trail(path: Path) -> list[dict]:
    events: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def build_tree(events: list[dict]) -> dict:
    """From expand events, reconstruct (parent_id, child_id, action, depth) edges.

    Also accumulate per-node visits and per-node (sum_score, count) from rollout
    backups (we attribute each rollout to all nodes on its select path).
    """
    edges: dict[int, list[tuple[int, tuple]]] = defaultdict(list)  # parent_id -> [(child_id, action), ...]
    node_depth: dict[int, int] = {}
    visits: dict[int, int] = defaultdict(int)
    score_sum: dict[int, float] = defaultdict(float)
    score_count: dict[int, int] = defaultdict(int)

    root_id = None
    for ev in events:
        if ev["event"] == "search_start":
            root_id = int(ev.get("root_id", 0))
            node_depth[root_id] = 0
        elif ev["event"] == "expand":
            pid, cid = int(ev["parent_id"]), int(ev["child_id"])
            edges[pid].append((cid, tuple(ev["action"])))
            node_depth[cid] = int(ev["child_depth"])
            if pid not in node_depth:
                node_depth[pid] = int(ev["parent_depth"])
        elif ev["event"] == "backup":
            score = float(ev["score"])
            for nid in ev["path_node_ids"]:
                visits[int(nid)] += 1
                score_sum[int(nid)] += score
                score_count[int(nid)] += 1

    return {
        "root_id": root_id if root_id is not None else 0,
        "edges": edges,
        "node_depth": node_depth,
        "visits": visits,
        "score_sum": score_sum,
        "score_count": score_count,
    }


def _layout_tree(root_id: int, edges: dict[int, list[tuple[int, tuple]]],
                 node_depth: dict[int, int]) -> dict[int, tuple[float, float]]:
    """Walk the tree DFS and assign x = leaf-order, y = -depth."""
    pos: dict[int, tuple[float, float]] = {}
    leaf_counter = [0]

    def walk(nid: int) -> tuple[float, float]:
        children = edges.get(nid, [])
        if not children:
            x = float(leaf_counter[0])
            leaf_counter[0] += 1
            y = -float(node_depth.get(nid, 0))
            pos[nid] = (x, y)
            return x, y
        xs = []
        for child_id, _ in children:
            cx, _cy = walk(int(child_id))
            xs.append(cx)
        x = sum(xs) / len(xs)
        y = -float(node_depth.get(nid, 0))
        pos[nid] = (x, y)
        return x, y

    walk(int(root_id))
    return pos


def _action_label(action: tuple) -> str:
    if len(action) >= 5:
        v, c, cs, gamma, eps = action[:5]
        if abs(float(gamma)) > 1e-12:
            return f"v{int(v)}/cfg{float(c):.1f}/g{float(gamma):.2f}"
    if len(action) >= 3:
        v, c, cs = action[:3]
        if abs(float(cs)) > 1e-12:
            return f"v{int(v)}/cfg{float(c):.1f}/cs{float(cs):.2f}"
        return f"v{int(v)}/cfg{float(c):.1f}"
    return str(action)


def plot_tree(events: list[dict], meta: dict, out_path: Path) -> None:
    tree = build_tree(events)
    edges = tree["edges"]
    node_depth = tree["node_depth"]
    visits = tree["visits"]
    score_sum = tree["score_sum"]
    score_count = tree["score_count"]
    root_id = tree["root_id"]

    pos = _layout_tree(root_id, edges, node_depth)
    if not pos:
        print("[plot] tree empty; skipping tree.png")
        return

    all_q = [score_sum[nid] / max(1, score_count[nid])
             for nid in pos.keys() if score_count.get(nid, 0) > 0]
    q_min, q_max = (min(all_q), max(all_q)) if all_q else (0.0, 1.0)
    if q_max - q_min < 1e-9:
        q_max = q_min + 1.0

    cmap = plt.get_cmap("viridis")

    def _q_color(nid: int) -> tuple:
        if score_count.get(nid, 0) == 0:
            return (0.6, 0.6, 0.6, 0.7)
        q = score_sum[nid] / score_count[nid]
        return cmap((q - q_min) / (q_max - q_min))

    fig, ax = plt.subplots(figsize=(12, 7))

    # Edges first (under nodes).
    edge_segs: list[tuple] = []
    for pid, kids in edges.items():
        if pid not in pos:
            continue
        px, py = pos[pid]
        for cid, action in kids:
            if cid not in pos:
                continue
            cx, cy = pos[cid]
            edge_segs.append(((px, py), (cx, cy)))
            mid_x, mid_y = (px + cx) / 2, (py + cy) / 2
            ax.text(mid_x, mid_y, _action_label(action), fontsize=6,
                    color="dimgray", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    if edge_segs:
        lc = LineCollection(edge_segs, colors="gray", linewidths=0.8, alpha=0.7, zorder=1)
        ax.add_collection(lc)

    # Nodes.
    xs, ys, sizes, colors = [], [], [], []
    for nid, (x, y) in pos.items():
        xs.append(x); ys.append(y)
        v = max(1, visits.get(nid, 0))
        sizes.append(40 + 25 * math.sqrt(v))
        colors.append(_q_color(nid))
    ax.scatter(xs, ys, s=sizes, c=colors, edgecolor="black", linewidth=0.5, zorder=3)

    # Mark root.
    if root_id in pos:
        rx, ry = pos[root_id]
        ax.scatter([rx], [ry], s=200, marker="*", c="red", edgecolor="black",
                   linewidth=0.7, zorder=4, label="root")

    title = (f"MCTS tree  prompt={meta.get('prompt','?')[:60]!r}  "
             f"sims={meta.get('n_sims','?')}  ucb_c={meta.get('ucb_c','?')}  "
             f"backend={meta.get('backend','?')}  selected_Q={meta.get('selected_score', 0):.3f}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("leaf-order layout (DFS)")
    ax.set_ylabel("depth (-key step idx)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)

    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=q_min, vmax=q_max), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("mean rollout score (Q)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[plot] tree → {out_path}")


def plot_ladder(events: list[dict], meta: dict, out_path: Path) -> None:
    """Per-simulation ladder: x = sim index, y = depth visited.

    For each sim:
      - draw a polyline from (sim_idx, 0) descending through each select_ucb /
        select_untried event's depth, ending at the leaf's depth.
      - the last point (leaf depth) is colored by the rollout score.
      - the expansion point gets a star marker.
    """
    sims_by_idx: dict[int, dict] = defaultdict(lambda: {"select_depths": [], "expand_depth": None,
                                                         "leaf_depth": None, "score": None})
    for ev in events:
        evt = ev["event"]
        if evt == "sim_start":
            sims_by_idx[int(ev["sim"])] = {"select_depths": [0], "expand_depth": None,
                                            "leaf_depth": None, "score": None}
        elif evt in ("select_ucb", "select_untried"):
            sims_by_idx[int(ev["sim"])]["select_depths"].append(int(ev["depth"]))
        elif evt == "expand":
            sims_by_idx[int(ev["sim"])]["expand_depth"] = int(ev["child_depth"])
        elif evt == "rollout":
            sims_by_idx[int(ev["sim"])]["leaf_depth"] = int(ev["leaf_depth"])
            sims_by_idx[int(ev["sim"])]["score"] = float(ev["score"])

    if not sims_by_idx:
        print("[plot] no sims found; skipping ladder.png")
        return

    sim_indices = sorted(sims_by_idx.keys())
    scores = [sims_by_idx[i]["score"] for i in sim_indices if sims_by_idx[i]["score"] is not None]
    s_min, s_max = (min(scores), max(scores)) if scores else (0.0, 1.0)
    if s_max - s_min < 1e-9:
        s_max = s_min + 1.0
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(11, 5))

    for i in sim_indices:
        rec = sims_by_idx[i]
        depths = list(rec["select_depths"])
        if rec["expand_depth"] is not None:
            depths.append(int(rec["expand_depth"]))
        if rec["leaf_depth"] is not None:
            depths.append(int(rec["leaf_depth"]))
        depths = depths or [0]
        score = rec["score"]
        if score is None:
            color = (0.7, 0.7, 0.7, 0.5)
        else:
            color = cmap((score - s_min) / (s_max - s_min))
        xs = [i] * len(depths)
        ys = [-d for d in depths]
        ax.plot(xs, ys, "-", color=color, alpha=0.75, linewidth=1.2)
        ax.scatter([i], [ys[0]], s=12, c=[color], edgecolor="none", zorder=3)
        if rec["expand_depth"] is not None:
            ax.scatter([i], [-rec["expand_depth"]], s=60, marker="*",
                       c=[color], edgecolor="black", linewidth=0.4, zorder=4)
        if rec["leaf_depth"] is not None:
            ax.scatter([i], [-rec["leaf_depth"]], s=20, marker="o",
                       c=[color], edgecolor="black", linewidth=0.3, zorder=4)

    ax.set_xlabel("simulation index")
    ax.set_ylabel("depth (-key step idx)")
    ax.set_title(f"MCTS rollouts ladder  prompt={meta.get('prompt','?')[:60]!r}  "
                 f"sims={meta.get('n_sims','?')}  ucb_c={meta.get('ucb_c','?')}",
                 fontsize=10)
    ax.grid(True, alpha=0.25)

    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=s_min, vmax=s_max), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("rollout score", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    legend = [
        mpatches.Patch(color="lightgray", label="root → leaf path"),
        plt.Line2D([0], [0], marker="*", color="black", markerfacecolor="white",
                   markersize=10, linestyle="None", label="expansion point"),
        plt.Line2D([0], [0], marker="o", color="black", markerfacecolor="white",
                   markersize=6, linestyle="None", label="rollout terminal"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=7, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[plot] ladder → {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot MCTS search trail.")
    parser.add_argument("--trail_dir", required=True, type=str,
                        help="Dir containing trail.jsonl + run_meta.json (from collect_mcts_trail.py).")
    parser.add_argument("--out_dir", default=None,
                        help="Where to write tree.png + ladder.png. Defaults to --trail_dir.")
    cli = parser.parse_args()

    trail_dir = Path(cli.trail_dir)
    # If user passed the bon-style parent dir, auto-resolve to best/.
    if not (trail_dir / "trail.jsonl").exists() and (trail_dir / "best" / "trail.jsonl").exists():
        print(f"[plot] auto-resolved {trail_dir} → {trail_dir / 'best'}")
        trail_dir = trail_dir / "best"
    out_dir = Path(cli.out_dir) if cli.out_dir else trail_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    trail = load_trail(trail_dir / "trail.jsonl")
    print(f"[plot] loaded {len(trail)} events from {trail_dir / 'trail.jsonl'}")

    meta_path = trail_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    plot_tree(trail, meta, out_dir / "tree.png")
    plot_ladder(trail, meta, out_dir / "ladder.png")

    print("[plot] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
