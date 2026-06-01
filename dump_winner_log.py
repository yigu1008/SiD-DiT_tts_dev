#!/usr/bin/env python3
"""Per-prompt log of MCTS winning action sequence.

For each prompt in a bon_mcts run, reads the per-step decisions from
`lookahead_node_logs` (added recently for both SD3.5 and FLUX), picks the
most-visited action at each step (= the exploit path MCTS committed to),
and writes a human-readable .txt log.

Usage:
    python dump_winner_log.py \
        --run_root /mnt/data/v-yigu/all_in_one/flux-newcfg/composite/flux_schnell/seed42 \
        --method bon_mcts \
        --prompts 0 5 10 17 \
        --out_dir figures/raw/flux_logs
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load_diag(run_root: Path, method: str, prompt_idx: int) -> tuple[dict | None, str | None]:
    """Return (diagnostics_dict, prompt_text) for the given prompt_idx, or (None, None)."""
    for jp in sorted(run_root.glob(f"**/{method}/**/rank_*.jsonl")):
        try:
            with open(jp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if int(row.get("prompt_index", -1)) != int(prompt_idx):
                        continue
                    d = row.get("diagnostics") or {}
                    if isinstance(d.get("bon_mcts"), dict):
                        return d["bon_mcts"], row.get("prompt", "")
                    return d, row.get("prompt", "")
        except Exception:
            continue
    return None, None


def _winning_path(diag: dict) -> list[dict]:
    """For each step_idx, pick the most-visited (variant, cfg) action across sims."""
    logs = diag.get("lookahead_node_logs") or diag.get("node_logs") or []
    if not isinstance(logs, list) or not logs:
        return []
    # Count visits per (step_idx, action_key) across all sims+phases.
    visit: Counter = Counter()
    reward_sum: dict = {}
    reward_n: dict = {}
    for r in logs:
        act = r.get("chosen_action") or {}
        step = int(r.get("step_idx", -1))
        if step < 0:
            continue
        key = (step,
               int(act.get("variant_idx", 0)),
               round(float(act.get("cfg", 0.0)), 4),
               round(float(act.get("cs", 0.0)), 4))
        visit[key] += 1
        pr = r.get("preview_reward")
        if pr is not None:
            reward_sum[key] = reward_sum.get(key, 0.0) + float(pr)
            reward_n[key] = reward_n.get(key, 0) + 1

    # For each step, pick the most-visited action.
    by_step: dict[int, list[tuple]] = {}
    for (step, v, cfg, cs), n in visit.items():
        by_step.setdefault(step, []).append((n, v, cfg, cs))
    out = []
    for step in sorted(by_step.keys()):
        cands = sorted(by_step[step], key=lambda t: t[0], reverse=True)
        n, v, cfg, cs = cands[0]
        key = (step, v, cfg, cs)
        mean_r = reward_sum.get(key, 0.0) / max(1, reward_n.get(key, 0)) if reward_n.get(key) else None
        out.append({
            "step": step, "variant_idx": v, "cfg": cfg, "cs": cs,
            "visits": n, "mean_reward": mean_r,
            "n_alternatives": len(cands),
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_root", required=True, type=Path)
    p.add_argument("--method", default="bon_mcts")
    p.add_argument("--prompts", nargs="+", type=int, default=None,
                   help="Explicit prompt indices (default: 0..19).")
    p.add_argument("--prompt_range", default=None, help="Inclusive-exclusive, e.g. 0:50.")
    p.add_argument("--out_dir", default=Path("figures/raw/winner_logs"), type=Path)
    p.add_argument("--combined", default=None, type=Path,
                   help="Also write a single combined log to this path.")
    args = p.parse_args()

    if args.prompts is None and args.prompt_range:
        a, b = args.prompt_range.split(":")
        args.prompts = list(range(int(a), int(b)))
    if args.prompts is None:
        args.prompts = list(range(0, 20))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined_lines: list[str] = []

    for pi in args.prompts:
        diag, prompt_text = _load_diag(args.run_root, args.method, pi)
        if diag is None:
            print(f"  prompt {pi:4d}  [no diagnostics found]")
            continue
        winners = diag.get("prescreen_ranked", [])
        winner_seed = diag.get("winner_seed", "?")
        path = _winning_path(diag)

        lines = [f"prompt #{pi}",
                 f'  text: "{prompt_text}"',
                 f"  winner seed: {winner_seed}",
                 ""]
        if not path:
            lines.append("  [no per-step decisions logged -- falling back to prescreen summary]")
            for row in (winners or [])[:5]:
                lines.append(f"    seed={row.get('seed')}  prescreen_score={row.get('prescreen_score'):.4f}")
        else:
            lines.append("  per-step MCTS decisions (most-visited action per step):")
            for r in path:
                mr = f"  r_bar={r['mean_reward']:.3f}" if r["mean_reward"] is not None else ""
                lines.append(
                    f"    step {r['step']}:  cfg={r['cfg']:<5.2f}  variant_idx={r['variant_idx']}  "
                    f"visits={r['visits']:3d}  n_alternatives={r['n_alternatives']:2d}{mr}"
                )

            # === Full per-step rollup of EVERY action MCTS evaluated ===
            logs = diag.get("lookahead_node_logs") or diag.get("node_logs") or []
            if logs:
                per_step: dict[int, list[dict]] = {}
                for r in logs:
                    step = int(r.get("step_idx", -1))
                    if step < 0:
                        continue
                    act = r.get("chosen_action") or {}
                    key = (int(act.get("variant_idx", 0)),
                           round(float(act.get("cfg", 0.0)), 4),
                           round(float(act.get("cs", 0.0)), 4))
                    bucket = per_step.setdefault(step, [])
                    entry = next((b for b in bucket if b["_key"] == key), None)
                    if entry is None:
                        entry = {"_key": key, "variant_idx": key[0], "cfg": key[1],
                                 "cs": key[2], "visits": 0, "preview_rewards": [],
                                 "phases": []}
                        bucket.append(entry)
                    entry["visits"] += 1
                    pr = r.get("preview_reward")
                    if pr is not None:
                        entry["preview_rewards"].append(float(pr))
                    ph = r.get("phase") or r.get("kind")
                    if ph and ph not in entry["phases"]:
                        entry["phases"].append(str(ph))

                lines.append("")
                lines.append("  ALL actions evaluated by MCTS, per step:")
                for step in sorted(per_step.keys()):
                    rows_step = sorted(per_step[step],
                                       key=lambda b: b["visits"], reverse=True)
                    lines.append(f"    --- step {step} ({len(rows_step)} unique actions) ---")
                    for b in rows_step:
                        rs = b["preview_rewards"]
                        rmean = sum(rs) / len(rs) if rs else None
                        rmax = max(rs) if rs else None
                        rmin = min(rs) if rs else None
                        phase = ",".join(b["phases"]) if b["phases"] else "-"
                        r_str = (f"  r_mean={rmean:+.4f}  r_min={rmin:+.4f}  r_max={rmax:+.4f}"
                                 if rmean is not None else "  r=<no preview>")
                        lines.append(
                            f"      v={b['variant_idx']}  cfg={b['cfg']:<5.2f}  "
                            f"cs={b['cs']:<5.2f}  visits={b['visits']:3d}  "
                            f"phase={phase}{r_str}"
                        )
        lines.append("")  # blank line

        out_txt = args.out_dir / f"prompt_{pi:04d}.txt"
        out_txt.write_text("\n".join(lines))
        combined_lines.extend(lines)
        print(f"  prompt {pi:4d}  -> {out_txt}  ({len(path)} steps)")

    if args.combined:
        args.combined.parent.mkdir(parents=True, exist_ok=True)
        args.combined.write_text("\n".join(combined_lines))
        print(f"\n  combined log -> {args.combined}")


if __name__ == "__main__":
    main()
