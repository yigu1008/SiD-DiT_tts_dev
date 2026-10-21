---
name: sid-method-timing-memory-2026-09-25
description: "Wall-clock and GPU-memory profile of all 25 SiD suite search methods at default compute (4 prompts, 2 passes); memory is identical across methods, timing spans 170x; noise_inject default is not budget-matched"
metadata: 
  node_type: memory
  type: project
  originSessionId: cb928cd6-a9aa-4c91-bce4-ed781fe2d2d9
  modified: 2026-09-25T05:49:57.220Z
---

Measured 2026-09-24/25 in `/data/ygu/runs/timing_mem_2026-09-24/` (TIMING_REPORT.md, timing_summary*.tsv, per_prompt_stats_p2.json, logs/gpu_poll.csv). One RTX 6000 Ada (GPU 5) + 2.3 GB ImageReward server; one method per process; prompts q0117/q0174/q0042/q0184, seed 42; suite defaults + the user's SiD pins (MCTS_KEY_STEP_COUNT=4, bon_mcts 16/4, UCB 1.0 for the ours family).

**Facts that change how the paper table should be written:**
1. **Memory is not a per-method quantity here.** Every method is sequential over trajectories: peak allocated during search is 18.1-18.2 GB for all 25, within 0.1 GB of one baseline trajectory (rollout + 1024^2 VAE decode + reward call). Largest search overhead = 61 MiB (beam). nvidia-smi process peak = 31.6 GB for every method, set by the T5-XXL encode phase and held by the caching allocator. Report ONE memory figure for the family.
2. **Timing per prompt** (s): baseline 10, dynamic_cfg_x0 15, greedy_prompt 26, bon 47, smc/fksteering ~50, bon_actdiff_* ~65, das 68, mcts_improved 52, dts 77-83, sop 88, unified_root 89, mcts 96, mcts_all 102, sop_actdiff_full 106, bon_mcts(_full) 165-168, greedy 166, ga 477, beam 507, noise_inject 1688. Per-prompt CV 0.2-9.4%; 11 methods run twice agree to median 2.4% -> 4 prompts suffices for timing.
3. **noise_inject at suite default = 3744 NFE/prompt** (8 seeds x 9 policies x 7 CFG candidates = 504 full rollouts, because `--noise_inject_cfg` unset -> searches all cfg_scales). Not budget-matched to anything; fix CFG for a 72-rollout version.

**Why:** nvidia-smi per-process memory cannot rank algorithms on this stack (allocator keeps the load-time reservation); only in-process `torch.cuda.max_memory_allocated` with resets per phase separates encode / baseline / search. The env-gated trace (`SID_MEM_TRACE=1`, 25 lines in `sd35_ddp_experiment.py`, diff in `config/mem_trace_patch.diff`, uncommitted) does this and is inherited by every runner that calls `base.main()`.

**How to apply:** for any future cost table, run one method per process, use the trace, and quote peak-search memory as a single family figure. See [[sid-mcts-followup-2026-09-24]] for the ablation this accompanies.
