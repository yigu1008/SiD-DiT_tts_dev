---
name: a6000-shared-gpu-kills-2026-10-16
description: "On the shared A6000 box, user ryan's multi-GPU launches (miniworld-dpdp, 2026-10-16) coincided with SIGTERM to ygu processes on the GPUs they took; GPU 1 was the only one left alone. Servers died with empty logs."
metadata: 
  node_type: memory
  type: project
  originSessionId: cb928cd6-a9aa-4c91-bce4-ed781fe2d2d9
  modified: 2026-10-16T20:08:39.174Z
---

2026-10-16 19:51–20:05: every ygu process on GPUs 0,2,3,4,5,6,7 received SIGTERM (exit 143) within ~1 min of
ryan's `miniworld-dpdp` workers appearing on that GPU (16 x 0.5 GB spawn workers + 2.4 GB main per GPU, 99% util).
Killed: ImageReward server (GPU 1, 19:51 — ryan not there afterwards), Jev-Omni server x3 (GPU 6), Qwen2.5-VL server
(GPU 3), stock-MCTS probe (GPU 2), rand/vlm/jev-prune probes (GPUs 4,5,7). GPU 1 was not touched after 19:51.
A SIGTERM'd python server leaves NO traceback in its log; launchers now append logs, run `python -X faulthandler`,
and echo `EXIT code=` (see [[vlm-decider-sd35base-2026-10-16]] scripts/*_server.sh).

**Why:** the user's standing rule is "my job must not interfere with ryan's"; ryan evidently clears GPUs before
launching, so anything I put on a GPU ryan later wants will be killed, and my idle servers holding 16–24 GB on
ryan's GPUs could OOM their job.
**How to apply:** before launching, check `nvidia-smi --query-compute-apps` owners and keep off GPUs with ryan's
workers; prefer GPU 1 (or whichever is ryan-free at the time); keep arms resumable per prompt and servers retried
(done); never assume a free-GPU reading older than a minute (same lesson as jev-imagegen/PLAN.md).
