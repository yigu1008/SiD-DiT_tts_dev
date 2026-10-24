# START HERE — Jev / VLM decider study on base SD3.5-Large

If you are a fresh Claude Code session that was told "pull my repo and continue", do these in order:

1. **Recall the project memory** (12 notes: findings, pitfalls, cluster rules). From the directory you started `claude` in:
   `bash experiments/vlm_decider_sd35base/memory/install_memory.sh` — then re-read `MEMORY_EXPORT.md` once so this
   session has it even before the next restart.
2. **Read `RESUME.md`** — especially §7 (state at hand-off: what ran, what is partial, what is pending) and §6 (pitfalls
   already paid for: `--n_variants` is a hard cap that silently truncates wider banks; never `pgrep -f` a pattern in
   your own command line; tag output dirs with seconds+PID; match NFE before believing any comparison; share seed
   pools or use n>=25).
3. **Read `RECOMMENDATION.md`** — the operating recipe the study converged on (best-of-k initial noise, rank-sum of
   HPSv3+PickScore+ImageReward, no tree, no decision model), and the budget table for choosing k.
4. **Skim the last three sections of `PLAN.md`** — the review of the Jev-as-policy traces, the actdiff hooks, and the
   two experiments that are still pending.
5. **To run the pending experiments:** on Azure ML, `DEPLOY.md` (one `amlt run`). On a GPU box you control:
   remap paths with `node_remap.sh` (see RESUME.md §3), start the three servers, then `scripts/run_jevact3_flex.sh`
   followed by `scripts/run_actdiff_jev.sh`. Both self-check and wait for 3 (or 2) clean GPUs; see DEPLOY.md.

What is here: `scripts/` (every arm, probe, launcher, the diversity-first prompt proposer, the actdiff decider module
and entry wrapper), `config/` (prompt ids, the v2 wording bank), `deps/` (prompt map, shipped bank, judge scripts),
`results/` (scores.csv for 74 judged arms + per-arm records/decisions; no images), `memory/` (the notes + installer).

Standing rule from the owner: on the shared A6000 box, never place anything on a GPU another user is using; check
`nvidia-smi --query-compute-apps` owners immediately before every launch.
