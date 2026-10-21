#!/usr/bin/env python3
"""Reimplementation of the run-sheet helper (original not present on this server).

Behaviour follows A6000_TODAY.md exactly where specified:
  cohort : deterministic SHA256 ordering of prompt IDs -> first n (so 25 nests in 50)
  budget : minutes = 1.30*n*(static+pair+judge+rerun)/60 + setup + 60
  paired : paired difference A-B per prompt, 10,000 prompt-bootstrap resamples,
           refuses duplicate or missing pairs, 95% percentile interval
Assumption stated: the SHA256 key is the UTF-8 prompt_id string itself.
Existing output files are never overwritten.
"""
import argparse, csv, hashlib, os, random, statistics, sys

def _read_ids(p): return [l.strip() for l in open(p) if l.strip()]

def cohort(a):
    ids = _read_ids(a.ids)
    if len(set(ids)) != len(ids): sys.exit("duplicate ids in input")
    order = sorted(ids, key=lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())
    if os.path.exists(a.out): sys.exit(f"refusing to overwrite {a.out}")
    open(a.out, "w").write("\n".join(order[: a.n]) + "\n")
    print(f"wrote {a.n} of {len(ids)} ids to {a.out} (sha256-ordered; nested prefixes)")

def budget(a):
    per = a.static_seconds + a.pair_seconds + a.judge_seconds + a.rerun_seconds
    for n in (25, 50, 99):
        m = 1.30 * n * per / 60 + a.setup_minutes + 60
        print(f"n={n:>3}: {m:6.0f} min forecast  ({'fits' if m <= a.minutes else 'DOES NOT FIT'} in {a.minutes} min)")

def paired(a):
    ids = set(_read_ids(a.ids))
    rows = [r for r in csv.DictReader(open(a.scores)) if r["metric"] == a.metric and r["prompt_id"] in ids and r["arm"] in (a.a, a.b)]
    by = {}
    for r in rows:
        k = (r["prompt_id"], r["seed"], r["arm"])
        if k in by: sys.exit(f"duplicate record for {k}")
        by[k] = float(r["score"])
    seeds = sorted({k[1] for k in by})
    diffs, missing = [], []
    for pid in sorted(ids):
        for s in seeds:
            ka, kb = (pid, s, a.a), (pid, s, a.b)
            if ka in by and kb in by: diffs.append(by[ka] - by[kb])
            else: missing.append((pid, s, a.a if ka not in by else a.b))
    if missing: sys.exit(f"missing paired records ({len(missing)}), e.g. {missing[:5]} -- refusing to drop them silently")
    if not diffs: sys.exit("no paired records")
    rng = random.Random(a.seed); n = len(diffs); mean = statistics.fmean(diffs)
    boots = sorted(statistics.fmean(rng.choices(diffs, k=n)) for _ in range(a.n_boot))
    lo, hi = boots[int(0.025 * a.n_boot)], boots[int(0.975 * a.n_boot) - 1]
    verdict = "inconclusive (interval crosses 0)" if lo <= 0 <= hi else ("A > B" if lo > 0 else "A < B")
    print(f"{a.a} - {a.b} on {a.metric}: n={n} mean={mean:+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}]  {verdict}")

p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
c = sp.add_parser("cohort"); c.add_argument("--ids", required=True); c.add_argument("--n", type=int, required=True); c.add_argument("--out", required=True); c.set_defaults(f=cohort)
b = sp.add_parser("budget"); [b.add_argument(f"--{x}", type=float, default=0.0) for x in ("minutes", "static-seconds", "pair-seconds", "judge-seconds", "rerun-seconds", "setup-minutes")]; b.set_defaults(f=budget)
q = sp.add_parser("paired"); [q.add_argument(f"--{x}", required=True) for x in ("scores", "ids", "a", "b", "metric")]; q.add_argument("--n-boot", type=int, default=10000); q.add_argument("--seed", type=int, default=0); q.set_defaults(f=paired)
a = p.parse_args(); a.f(a)
