#!/usr/bin/env python3
"""Collect MCTS search trails across N seeds (bon_mcts-style), keep the winner.

Loads SD3.5 pipeline + reward model ONCE, then runs `run_mcts` per seed with
MCTS_TRAIL_LOGGER capturing per-sim events. After all seeds finish, picks the
seed with the highest selected_score and copies its outputs into <out>/best/
so the plotter consumes <out>/best/ as canonical.

Outputs:
  <out>/seed<S>/trail.jsonl       -- per-seed trail (one event per line)
  <out>/seed<S>/run_meta.json     -- per-seed args + selected_score
  <out>/seed<S>/best.png          -- per-seed selected image
  <out>/best/trail.jsonl          -- winner's trail (copied from best seed)
  <out>/best/run_meta.json        -- winner's meta
  <out>/best/best.png             -- winner's image
  <out>/summary.json              -- per-seed scores + winner

Example (sid 4-step, 8 seeds prescreen-style on a single A6000/A100):
  python collect_mcts_trail.py \
      --prompt "a photograph of an astronaut riding a horse" \
      --seeds 42 43 44 45 46 47 48 49 \
      --n_sims 30 --backend sid \
      --out /tmp/mcts_trail
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import sampling_unified_sd35 as su


def _build_args(cli, seed: int) -> argparse.Namespace:
    argv = [
        "--search_method", "mcts",
        "--prompt", cli.prompt,
        "--seed", str(seed),
        "--n_sims", str(cli.n_sims),
        "--ucb_c", str(cli.ucb_c),
        "--backend", cli.backend,
        "--steps", str(cli.steps),
        "--baseline_cfg", str(cli.baseline_cfg),
        "--cfg_scales", *[str(c) for c in cli.cfg_scales],
        "--correction_strengths", *[str(c) for c in cli.correction_strengths],
        "--reward_backend", cli.reward_backend,
        "--out_dir", cli.out,
        "--no_qwen",
        "--height", str(cli.height),
        "--width", str(cli.width),
    ]
    args = su.parse_args(argv)
    args = su.normalize_paths(args)
    return args


def _run_one_seed(cli, seed: int, ctx, reward_model, emb, action_vocab,
                  out_seed_dir: Path) -> dict:
    out_seed_dir.mkdir(parents=True, exist_ok=True)
    trail_path = out_seed_dir / "trail.jsonl"
    trail_fp = open(trail_path, "w", buffering=1)

    def _logger(event: dict) -> None:
        trail_fp.write(json.dumps(event) + "\n")

    su.MCTS_TRAIL_LOGGER = _logger
    su.MCTSNode._id_counter = 0  # restart node ids for each tree

    args = _build_args(cli, seed)
    print(f"[collect] seed={seed} starting MCTS (n_sims={cli.n_sims})")
    result = su.run_mcts(args, ctx, emb, reward_model, cli.prompt, [cli.prompt], seed)
    trail_fp.close()
    su.MCTS_TRAIL_LOGGER = None

    n_events = sum(1 for _ in open(trail_path))
    try:
        result.image.save(out_seed_dir / "best.png")
    except Exception as e:
        print(f"[collect] WARN seed={seed} couldn't save best image: {e}")

    meta = {
        "prompt": cli.prompt,
        "seed": int(seed),
        "n_sims": int(cli.n_sims),
        "ucb_c": float(cli.ucb_c),
        "backend": cli.backend,
        "steps": int(cli.steps),
        "baseline_cfg": float(cli.baseline_cfg),
        "cfg_scales": [float(c) for c in cli.cfg_scales],
        "correction_strengths": [float(c) for c in cli.correction_strengths],
        "reward_backend": cli.reward_backend,
        "selected_score": float(getattr(result, "score", 0.0)),
        "selected_actions": [list(a) for a in getattr(result, "actions", [])],
        "n_events": int(n_events),
        "action_vocab": action_vocab,
    }
    (out_seed_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[collect] seed={seed} DONE  selected_score={meta['selected_score']:.4f}  events={n_events}")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect MCTS trails across N seeds, keep the winning tree."
    )
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46, 47, 48, 49],
                        help="Seeds to run MCTS over (default: 8, mirrors bon_mcts N_SEEDS).")
    parser.add_argument("--n_sims", type=int, default=30)
    parser.add_argument("--ucb_c", type=float, default=1.0)
    parser.add_argument("--backend", type=str, default="sid",
                        choices=["sid", "senseflow_large", "sd35_base"])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--baseline_cfg", type=float, default=1.0)
    parser.add_argument("--cfg_scales", type=float, nargs="+",
                        default=[1.0, 1.5, 2.0, 2.5])
    parser.add_argument("--correction_strengths", type=float, nargs="+", default=[0.0])
    parser.add_argument("--reward_backend", type=str, default="imagereward")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--out", required=True, type=str)
    cli = parser.parse_args()

    out = Path(cli.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[collect] backend={cli.backend} steps={cli.steps} reward={cli.reward_backend}")
    print(f"[collect] prompt: {cli.prompt!r}")
    print(f"[collect] seeds={cli.seeds}  n_sims={cli.n_sims}  ucb_c={cli.ucb_c}")
    print(f"[collect] cfg_scales={cli.cfg_scales}  correction_strengths={cli.correction_strengths}")

    # Load pipeline + reward ONCE, reuse across all seeds.
    boot_args = _build_args(cli, cli.seeds[0])
    ctx = su.load_pipeline(boot_args)
    reward_model = su.load_reward_model(boot_args, ctx.device)
    emb = su.encode_variants(ctx, [cli.prompt])
    n_variants = len(emb.cond_text)
    action_vocab = []
    for vi in range(n_variants):
        for cfg in cli.cfg_scales:
            for cs in cli.correction_strengths:
                action_vocab.append({"variant_idx": vi, "cfg": float(cfg), "cs": float(cs)})

    # Run MCTS per seed.
    per_seed: list[dict] = []
    for seed in cli.seeds:
        seed_dir = out / f"seed{seed}"
        try:
            meta = _run_one_seed(cli, int(seed), ctx, reward_model, emb, action_vocab, seed_dir)
            per_seed.append(meta)
        except Exception as e:
            print(f"[collect] FAIL seed={seed}: {e}")
            per_seed.append({"seed": int(seed), "selected_score": float("-inf"), "error": str(e)})

    # Pick winner.
    valid = [m for m in per_seed if m.get("selected_score", float("-inf")) != float("-inf")]
    if not valid:
        print("[collect] ERROR no successful seed")
        return 1
    winner = max(valid, key=lambda m: m["selected_score"])
    print(f"[collect] WINNER seed={winner['seed']}  selected_score={winner['selected_score']:.4f}")

    # Copy winner into <out>/best/ for the plotter's canonical input.
    best_dir = out / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(out / f"seed{winner['seed']}", best_dir)
    print(f"[collect] best/ copied from seed{winner['seed']} → {best_dir}")

    summary = {
        "prompt": cli.prompt,
        "seeds": cli.seeds,
        "winner_seed": int(winner["seed"]),
        "winner_score": float(winner["selected_score"]),
        "per_seed": [
            {"seed": int(m.get("seed", -1)),
             "selected_score": float(m.get("selected_score", float("-inf"))),
             "error": m.get("error")}
            for m in per_seed
        ],
        "n_sims": int(cli.n_sims),
        "backend": cli.backend,
        "ucb_c": float(cli.ucb_c),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[collect] summary → {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
