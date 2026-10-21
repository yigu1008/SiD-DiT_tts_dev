---
name: reward-choice-beats-reward-selection-2026-10-19
description: "On base SD3.5 key-step MCTS, simply choosing WHICH reward to search on moves the ir/hpsv3 frontier cleanly (hpsv3 +0.879, held-out metrics unchanged); a Jev per-node reward selector adds nothing beyond the per-prompt choice, and the per-prompt oracle headroom is only ~0.01."
metadata:
  node_type: memory
  type: project
---

2026-10-19, n=9 prompts, 4 arms all at exactly 616 NFE (`/data/ygu/runs/vlm_decider_sd35base_2026-10-16`,
`scripts/jev_reward_select.py`, PLAN.md section "Multi-reward"):

**What works, and needs no decision model:** searching on HPSv3 instead of ImageReward gives hpsv3
**+0.879 [+0.549, +1.249]** for ir −0.182, with every held-out metric unchanged (hpsv2 +0.001,
qwen_alignment −0.002, vqascore −0.021, all CIs crossing 0). One flag, zero VLM/Jev calls.

**What does not pay:** Jev picking the reward per node. The decision is genuine — each reward was given a
FOCUSED AREA (imagereward=CONTENT, hpsv3=APPEARANCE, pickscore=KIND/FRAMING) matched against end-of-trace VLM
complaints, and a probe scored **45/45** with the choice flipping for 9/9 prompts when the evidence changed
(vs 0/8 flips for the CFG-policy question). But a `root_mixture` baseline — just the per-PROMPT root choice,
no within-tree switching — matched or beat the full arm on 5 of 6 metrics. Per-prompt oracle headroom over the
best fixed arm is only **+0.011 ir / +0.014 hpsv3**, the same order as the branch-point oracle (+0.006).
Jev's root choice was responsive but not predictive: 4/9 agreement with which fixed arm actually won on hpsv3.

**Design note worth keeping (the user's correction):** no normalization is needed to mix rewards. Score every
rollout under all backends and accumulate the score VECTOR at each node (one running sum per reward per action);
a node reads only its chosen column, so its statistics stay in one reward's native units and re-deciding costs
nothing. Avoids the arbitrary `_COMPOSITE_*_RANGE` min-max and the non-stationary backups it causes.

**Why the motivating headroom estimate was wrong:** +1.836 hpsv3 / −0.077 ir computed as a max over 34
heterogeneous judged arms in `scores.csv` was an extreme order statistic. The within-prompt ir~hpsv3 rank
correlation of **+0.014** (near-independent, per [[reward-orthogonality-result-2026-09-22]]) is real and is why
reward choice matters at all — but it does not imply large reachable headroom from choosing adaptively.

**How to apply:** for frontier movement on this pipeline, set the search reward per run and stop there. See
[[vlm-decider-sd35base-2026-10-16]] for the operating point (blind key-step MCTS, ~10 sims, truncate rollouts at
step 16). Before building any new decider here, measure the oracle headroom of the decision first — three
decisions in this study (CFG/wording pruning, branch-point placement, reward selection) all turned out real but
worth ~0.01.
