---
name: unattended-chain-monitoring
description: Lessons from a 13h silent stall of a multi-stage GPU chain; how to launch and monitor long runs here
metadata:
  type: feedback
---

On 2026-09-22 a chained run (BoN scoring -> MCTS smoke -> MCTS full) stalled silently for 13 hours: the launcher used a bare `wait` while long-lived reward servers were its background children, so it never returned; my wake-ups only triggered on a *verdict string*, so a stall produced no signal. Separately, `local G=$1 S=$3 PF=".../$S.txt"` on one line expanded `$S` from the OUTER scope before the local took effect.

**Why:** the user checks in expecting progress; a silent stall wastes a night of 8 GPUs and their trust.

**How to apply:**
- In launchers that start servers, `wait "${WORKER_PIDS[@]}"`, never bare `wait`; kill servers by recorded PID (exec'd, not via a subshell) in an EXIT trap.
- Every background monitor needs a *stall* branch (no progress for N min) and a hard deadline, not just a success/failure grep.
- Smoke-test the exact code path the full run uses (the smoke used a different prompt-file branch and hid the bug).
- Never `pkill -f <pattern>` from a command line that itself contains the pattern; use `[p]attern` or PIDs.
- Reward-model serving for MCTS here: `reward_orthogonality/run_mcts.sh` (2x hpsv3+IR servers in `reward` env, 1x VQA in `vqascore_reward` with cwd=SiD repo, loopback aggregator on :5200, `REWARD_SERVER_URL` -> workers load no reward models). See [[image-reward-env-pitfalls]].

**Orphaned `start_heartbeat` subshells (found 2026-10-19).** `SiD-DiT_tts_dev/_heartbeat.sh` backgrounds
`( while true; do echo tick; sleep N; done ) &`. If the parent script dies (e.g. the ryan-SIGTERM incidents, see
[[a6000-shared-gpu-kills-2026-10-16]]), that subshell is reparented to init and loops forever. 16 of them were alive
129-141 days, from `run_actdiff_render_a6000.sh`, `run_limited_noise.sh`, `run_qual_exp1_budget_sid.sh`,
`run_qual_exp2_axes_sid.sh`. They hold no GPU and ~2.5 MB each, but in `ps` they look like live runs because the
backgrounded subshell inherits the parent script's argv.
**How to tell a heartbeat orphan from a live run:** its only child is `sleep N` and it has ZERO python descendants.
Check that before killing, then kill by explicit PID (never a `pgrep -f` pattern — that self-matches this session's
own shell, which killed my shell ~4x in this project).
**True zombies cannot be killed**: a `<defunct>` entry is already dead and only its parent can reap it. The ygu ones
here belong to `codex app-server-daemon` (pid 167105) and `harbor view jobs` (pid 2673681) - the user's own tooling,
so leave them and report rather than killing the parent.
