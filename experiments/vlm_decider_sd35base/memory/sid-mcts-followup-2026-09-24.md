---
name: sid-mcts-followup-2026-09-24
description: Same-day static-vs-MCTS, fixed-bank, and VQA-ablation follow-up on the 99-prompt SiD diagnostic (n=25); algorithm-identity findings about the implemented MCTS; Random-IR run at n=99 on 09-25
metadata:
  type: project
---

Run on 2026-09-24 per the user's A6000_SERVER_HANDOFF.md / A6000_TODAY.md protocol. Everything under `/data/ygu/reward_orthogonality/followup_2026-09-24/` (REPORT.md, selected_scores.csv, scripts/, main/, traces/, config/). Cohort: 25 prompts, SHA256-ordered nested subset of the 99 (`prompt_ids_25.txt`), decided from pilot timing before any comparative result; one GPU worker (GPU 5) on the shared 8-GPU box.

**Algorithm identity of the stock `run_mcts` (sampling_unified_sd35.py), verified from source + a trail trace:** UCB1 (mean + 1.41*sqrt(ln N/n)), untried-first random; NO prior term, NO PUCT, NO "feedback prior" code path. Bank = variants x CFG (28 for 4 variants) at steps 0,3 and a disjoint union with 8 anchored noise-injection actions at steps 1,2 -- hardwired on for steps<=4 (`use_integrated_noise_actions = total_steps <= 4`; disable via `--noise_inject_gamma_bank 0.0`). Rollouts are complete trajectories; return = best of {exploit replay, best rollout} and the best rollout wins ~90-95% of prompts. Trace (q0117): 28 untried root expansions + 12 root-only UCB picks, depth never > 1 -> effectively a round-robin-seeded best-of-40. Per-step re-noise uses the unseeded global torch RNG -> reference runs are NOT bit-reproducible (same config rerun: 1.255 vs 1.525).

**Results (n=25, paired 10k prompt-bootstrap, unadjusted 95% CI):**
- ir_mcts - static_ir: IR +0.078 [+0.037,+0.119]*, HPSv3 -0.685 [-1.167,-0.227]*; PickScore/HPSv2/Qwen-alignment inconclusive.
- Fixed-bank reference (noise actions off via `--noise_inject_gamma_bank 0.0`, same 28 pairs at all steps, finished 22:32): ir_mcts_fixedbank - static_ir IR +0.054 [+0.005,+0.111]* (gain survives without noise actions), HPSv3 -0.446 [-0.981,+0.085] n.s.; fixedbank - noise-bank ir_mcts inconclusive on every metric. Still confounded with 40 vs 28 trajectories under the unseeded re-noise stream; Random-IR (the trajectory-count control) was deferred by the budget rule (forecast 143 min vs 84 min left), so that confound is open.
- Fixed-bank run cost 137 s/prompt (vs 100 static, 184 pair with HPSv3 server) and mean NFE 281.
- triple_mcts - pair_mcts (coefficient-preserving VQA removal, (gH+gI)/3): Qwen-alignment +0.003 [-0.033,+0.034] inconclusive; HPSv3 -0.314* (pair reallocates to HPSv3); others inconclusive. No detectable alignment contribution of the VQA term at n=25.
- Live-vs-offline consistency exact (max|diff| 0.00000) for both new arms.

**Expansion to n=99 (2026-09-25, user-directed; 3 shared-GPU workers, GPU 5 untouched; `main/contrast_n99_*.txt`, `main/contrast_n74_*.txt`, REPORT.md §6):**
- Table A holds: ir_mcts - static_ir IR +0.055 [+0.036,+0.074]*; the n=25 HPSv3 cost shrinks to -0.22 n.s. Fixed-bank: IR +0.036*, HPSv3 -0.40*.
- **Table B flips to significant**: triple - pair on Qwen alignment +0.043 [+0.011,+0.078]* (0.57 within-prompt SD; 58 better / 32 worse), no significant cost elsewhere. The 25-cohort was unrepresentative (+0.003 there vs +0.056 on the 74 new). Still includes the scale effect; one seed.
- Random-IR ran (99/99, 28 per-step-random trajectories, matched to static's 28): IR -0.013 n.s. vs static, HPSv3 -0.40*, PickScore -0.15*. UCB1 over the same bank beats it +0.048* IR. Residual confound: MCTS arms use 40 trajectories vs 28 -> a `--n_traj 40` Random-IR would close it.
- Lesson: n=25 zero-crossing on a 0-1 judge with within-prompt SD 0.075 was underpowered for a 0.04 effect; the n=99 rerun cost ~12 GPU-h shared.

**Multi-reward config prepared 2026-09-26 (not launched):** `config/multireward_ablation_plan.md`. Objectives mr2 = `composite_hpsv3_ir` (0.5·norm HPSv3[9,14] + 0.5·norm IR[-3,3]) and mr3 = `composite_3` ((IR+HPSv3+PickScore[17,23])/3), `HPSV3_SCORE_STYLE=raw`. Generalized arms `static_arm.py` / `random_arm.py` (store raw components + objective check vs reward_model.score, pilot diff 0.0), `mcts_arm.py` ARMS mr2_mcts/mr3_mcts(+_fixedbank), launchers `mr_server.sh` / `mr_queue.sh` / `mr_finish.sh`. Pilot on q0117: static_mr3 155 s/prompt (208 NFE), mr3_mcts 202 s (265 NFE); a 3-backend server = 23 GB and must sit on its own GPU. Compact set (static+mcts) for both objectives ≈ 10 h wall on 2 sampler GPUs + 2 server GPUs. `export_scores.py` now also accepts `selected_score`.

**Operational lessons:** pipeline load peaks ~27 GB (all encoders + transformer on GPU) before any offload path; co-locating an 18 GB HPSv3 server requires `OFFLOAD_TRANSFORMER_DURING_ENCODE=1` + only one reward server resident; IR-only arms and the HPSv3 arm must alternate servers on a single 48 GB GPU. New backend `composite_hpsv3_ir_pair` added to reward_unified.py. Item 5 MI factorial data is NOT on this server (amlt blob output) -> narrow the MI claim. See [[reward-orthogonality-result-2026-09-22]], [[image-reward-env-pitfalls]].
