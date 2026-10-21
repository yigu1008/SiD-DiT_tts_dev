#!/usr/bin/env python3
"""Entry point: actdiff (bon_mcts prescreen + run_mcts_lookahead) with an optional Jev decision engine attached.

Same monkey-patch pattern the repo already uses (sd35_ddp_experiment_bon_mcts patches sd35_ddp_experiment):
this wraps ONE more layer to parse `--jev_decider ...` flags and attach the decider object as `args.jev_decider`,
which `run_mcts_lookahead` consults through its opt-in hooks. With `--jev_decider none` the attribute is absent
and the sampler's behaviour is byte-for-byte the shipped actdiff.
"""
from __future__ import annotations
import argparse, os, sys
SID = "/home/ygu/project/SiD-DiT_tts_dev"; HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SID); sys.path.insert(0, HERE)
import sd35_ddp_experiment_bon_mcts as bm        # noqa: E402
import jev_actdiff_decider as jd                   # noqa: E402


def _parse_jev_flags(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--jev_decider", default="none", choices=["none", "rank3only", "jevbank", "randbank", "jevprobe", "randprobe"])
    p.add_argument("--jev_server", default="http://127.0.0.1:5161"); p.add_argument("--vlm_server", default="http://127.0.0.1:5151")
    p.add_argument("--reward_server3", default="http://127.0.0.1:5142", help="3-backend server for rank3 and probe scoring")
    p.add_argument("--decider_log", default=None); p.add_argument("--probe_k", type=int, default=3)
    p.add_argument("--decider_min_traces", type=int, default=2); p.add_argument("--decider_seed", type=int, default=0)
    return p.parse_known_args(argv)


def main():
    jev_args, rest = _parse_jev_flags(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    orig = bm.base.parse_args          # bm.main() will wrap this again with its own patch; we wrap underneath

    def patched():
        args = orig()
        if jev_args.jev_decider != "none":
            kw = dict(reward_server=jev_args.reward_server3, vlm_server=jev_args.vlm_server, jev_server=jev_args.jev_server,
                      log_path=jev_args.decider_log or os.path.join(args.out_dir, "decider_log.jsonl"),
                      seed=jev_args.decider_seed, min_traces=jev_args.decider_min_traces)
            if jev_args.jev_decider.endswith("probe"): kw["probe_k"] = jev_args.probe_k
            args.jev_decider = jd.build(jev_args.jev_decider, **kw)
            print(f"[actdiff_jev] decider={jev_args.jev_decider} log={kw['log_path']}", flush=True)
        else:
            print("[actdiff_jev] decider=none (shipped actdiff behaviour)", flush=True)
        return args

    bm.base.parse_args = patched
    try: bm.main()
    finally: bm.base.parse_args = orig


if __name__ == "__main__":
    main()
