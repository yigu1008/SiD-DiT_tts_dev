# Operating recipe for base SD3.5-Large (as of 2026-10-19)

## The choice
**Draw k initial latents at the base prompt and CFG 4.5, then pick the winner by the RANK-SUM of HPSv3 + PickScore
+ ImageReward over those k candidates.**
No tree search, no prompt rewriting, no VLM, no Jev.

```bash
R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16
# 1. reward server: needs all three of imagereward+hpsv3+pickscore for the rank3 selector (~22 GB, one GPU)
bash $R/scripts/mr3_server.sh <GPU_A> 5142
# 2. the arm.  --m 0 = pure best-of-k;  --m 4 = the two-stage variant (same result, ~6% cheaper)
CUDA_VISIBLE_DEVICES=<GPU_B> OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python $R/scripts/two_stage.py --k <K> --m 0 \
    --ids <IDS> --run_dir <OUT> --reward_server http://127.0.0.1:5142
```
**NO decision model is involved.** `two_stage.py` contains zero references to a Jev or VLM server -- no argument, no
call site -- so there is nothing to disable and no 24 GB Jev server to provision. Verified empirically: `ts_k11m0`
and `ts_k8m4` recorded **0 Jev calls and 0 VLM calls** over 9 prompts (the steering arm `nx_jev`, for contrast,
made 276 and 83).
*(Historical note on naming: `jev_noise_budget.py --policy never` is an equivalent path and also makes zero model
calls, but that file exists for the adaptive-budget EXPERIMENT where `--policy jev` was one of four policies. Do not
use it as the production entry point -- the name implies a decision model that the recipe does not use.)*

## Where a decision model sits in this recipe: NOWHERE
MCTS and every Jev steering role are measured dead -- wording/CFG pruning never beat blind; the PUCT soft prior was
worse; per-node reward choice lost -0.275 hpsv3 to a fixed reward; saturation/abandonment lost -0.275 to a free
partial-reward threshold; pool expansion was a constant (15/15). Not from incompetence (Jev scored 45/45 on
reward-focus matching, 7/7 and 3/3 on perception) but because the decisions had no headroom: oracle ceilings of
+0.006 (branch placement), +0.014 (reward choice), +0.413 for abandonment of which a free threshold already takes
+0.140. The one role still under test is Jev as a SELECTOR over the k finished candidates (`sel_jev`), which is a
matching question rather than a policy one; until that lands, the recipe is reward-only.

## Choosing k (56 NFE per generation)
| k | NFE | hpsv3 | gain vs k=1 | marginal gain per extra generation |
|---|---|---|---|---|
| 1 | 56 | 10.374 | — | — |
| **2** | **112** | **11.051** | **+0.677** | **+0.677** |
| 4 | 224 | 11.530 | +1.156 | +0.240 |
| **8** | **448** | **11.874** | **+1.500** | **+0.086** |
| 11 | 616 | 11.999 | +1.626 | +0.042 |
| 16 | 896 | 12.153 | +1.779 | +0.031 |
| 32 | 1792 | 12.418 | +2.045 | +0.014 |

* **k = 2 (112 NFE)** is the single best value for money: it matches full 10-sim key-step MCTS (12.005 at 616 NFE)
  on both ir and hpsv3 at **1/5.5 the compute**.
* **k = 8 (448 NFE)** is the recommended default: 73% of the attainable gain, and the marginal return has fallen 8x
  from k=2. Past k=8 each extra generation buys <0.09 hpsv3.
* k > 16 only if compute is free.

## The selector: rank-sum of all three, NOT a single reward (measured 2026-10-19 21:21)
Five selectors over the SAME 11 candidate noises per prompt, 616 NFE, judged. `()` = inside that selector's objective:
| selector | ir | hpsv3 | pickscore | hpsv2* | qwen_align* | vqascore* | held-out rank |
|---|---|---|---|---|---|---|---|
| **hpsv3+pickscore+ir (`rank3`)** | (1.4505) | (11.2309) | (23.5948) | **0.3026** | **0.8368** | **0.8618** | **1.00** |
| hpsv3+pickscore | 1.2451 | (11.6011) | (23.4266) | 0.3011 | 0.8325 | 0.8596 | 2.00 |
| hpsv3 only | 1.1618 | (11.9189) | 23.1278 | 0.2978 | 0.8306 | 0.8425 | 3.33 |
| pickscore only | 1.3086 | 10.6524 | (23.6675) | 0.3007 | 0.7919 | 0.8423 | 3.67 |
| ir only | (1.5162) | 10.4446 | 23.0222 | 0.2950 | 0.7427 | 0.8119 | 5.00 |

* adding PickScore to HPSv3 wins a held-out metric outright: **vqascore +0.0171 [+0.0050, +0.0304]** (excludes 0);
* adding ImageReward as a third rank-vote is best on all three held-out metrics (vs the pair: qwen +0.0043
  [+0.0000, +0.0100], vqascore +0.0022 [−0.0000, +0.0058] — both at the CI boundary, so rank3 vs the pair is a
  near-tie; both clearly beat any single reward);
* **ImageReward ALONE remains the worst choice** (qwen 0.7427, vqascore 0.8119) — the hacking result stands. The
  lesson is that IR is harmful as the sole objective, useful as one vote of three.
* the three models are only ~0.3 mutually correlated across noise candidates (hps~ps +0.353, hps~ir +0.308,
  ps~ir +0.388), so the rank-sum is a genuine ensemble.
* **ranks are scale-free within the candidate set, so no min-max normalisation is used anywhere.**
With all three inside the selector the held-out set is hpsv2, qwen_alignment and VQAScore (a different model family).

## What NOT to spend budget on (all measured, all at matched NFE)
| discarded | why |
|---|---|
| key-step MCTS over (wording x CFG) | best-of-2 noise matches 10-sim MCTS at 1/5.5 the cost |
| prompt rewriting / wording bank | isolated, the faithful bank costs −0.193 hpsv3; the shipped bank is 46% corrupt; neither beats plain noise |
| action search on a screened noise | gain collapses from +1.549 to +0.219 once screening picked a good noise — the axes are substitutes |
| VLM (Qwen2.5-VL) pruning | never beat blind at matched budget |
| Jev-Omni decisions (CFG, reward choice, saturation, pool expansion) | every variant tied or lost; the expand decision was a constant (15/15) |
| adaptive abandon / expand | oracle ceiling +0.413, free reward screen captures +0.140 (CI crosses 0 at n=9), Jev loses to that free screen |

## Honest limits of this recommendation
1. **n = 9 prompts.** Two uniform best-of-11 runs differing only in which seeds they drew differ by
   **+0.527 hpsv3 [−0.138, +1.498]** — draw luck exceeds every policy effect measured. The *shape* of the best-of-k
   curve is solid (bootstrapped over a 32-noise pool); absolute levels are not.
2. **CFG 4.5 and the base prompt were fixed, not tuned,** in every noise arm. A CFG sweep at fixed noise was never run
   under this recipe.
3. **Unverified at n >= 25.** Repeat before publishing; every script is parameterised for it.
4. **One cheap open item:** a free partial-HPSv3 screen at step 12 (never before step 12 — rho is +0.04 at step 4,
   +0.61 at step 12) gained +0.088 [−0.124, +0.367] here and +0.140 in simulation, against an oracle ceiling of
   +0.413. Worth confirming at n >= 25 with shared seed pools; it needs no model calls.
