---
name: initial-noise-dominates-action-search-2026-10-19
description: "On base SD3.5, the initial latent carries ~5x more reward variance than the (wording x CFG) action axis the whole MCTS study searched; best-of-11 noise selected by HPSv3 beats every 616-NFE search arm and is the first thing to dominate blind MCTS on a held-out metric. Select on HPSv3, never ImageReward."
metadata:
  node_type: memory
  type: project
---

2026-10-19, `/data/ygu/runs/vlm_decider_sd35base_2026-10-16`, `scripts/noise_probe.py` + `noise_table.py`,
n=9 prompts x 16 initial latents at a fixed anchor action (wording A, CFG 4.5). PLAN.md section
"THE AXIS THE WHOLE STUDY HELD FIXED".

**The blind spot:** every arm in the 2-day study built its start latent as `make_latents(ctx, seed=42, ...)` —
one noise sample for every arm and every prompt. All search was (wording x CFG) at fixed z0.

**Key arithmetic:** 28 Euler steps = 56 NFE per complete generation, so the 616-NFE operating point every arm
was matched to buys exactly **11 full generations**. Best-of-11 noise is an exactly budget-matched, search-free
competitor. Always check this ratio before crediting a search method.

**Measured:** mean per-prompt range across 16 noises at fixed action = ir 1.055, **hpsv3 4.523**, pickscore 1.546.
The largest effect the entire action-search apparatus produced was hpsv3 +0.879. Seed 42 ranked mid-pack
(7.3-8.8 of 16), so this is not a bad-seed artifact. Judged, at matched 616 NFE, `ns_bo11_hps` vs blind MCTS:
**hpsv3 +1.761 [+0.963, +2.712]**, pickscore +0.237, ir −0.075 (ns), vqascore +0.002, qwen −0.049 (ns) —
it dominates every other arm on the (vqascore, hpsv3) frontier, the first held-out win in the study.
**Best-of-2 noise (112 NFE) matches 10-sim blind MCTS (616 NFE) on both ir and hpsv3** — 5.5x cheaper.

**Selection metric is critical:** `ns_bo11_ir` (same images, selected by ImageReward instead) significantly LOSES
the held-out Qwen alignment judge (−0.081 [−0.162, −0.004]) and is dominated by all five other arms on both
held-out frontiers. Same hacking signature as [[reward-choice-beats-reward-selection-2026-10-19]].

**How to apply:** on this pipeline, spend inference budget on initial-noise draws selected by HPSv3 before any
tree search, prompt rewriting or VLM/Jev decision layer, and do NOT put a decision model in the allocation loop.
Still open and cheap: confirm the free partial-HPSv3 step-12 screen at n >= 25 with shared seed pools.
Supersedes the operating-point advice in [[vlm-decider-sd35base-2026-10-16]].

**Adaptive allocation (abandon saturated noises / expand the pool), measured 2026-10-19 06:30, n=9, 616-NFE cap,
`scripts/jev_noise_budget.py`:** the MECHANISM carries real information — abandoning at a matched rate but
UNINFORMED costs hpsv3 **−1.298 [−2.168, −0.568]**. But the realisable gain over uniform best-of-11 from a FREE
partial-HPSv3 screen at step 12 is only +0.088 [−0.124, +0.367] (CI crosses 0; offline simulation over 32 noises
predicted +0.140, oracle ceiling +0.413). Jev **loses to that free threshold**: hpsv3 −0.275 [−0.609, −0.008] while
gaining ir +0.077 [+0.011, +0.154] — the IR-for-HPSv3 trade again. Jev's expand decision was a CONSTANT (NEW_NOISE
15/15); the constant is the correct policy but hard-coding it is free.
Across-noise predictability of the finished hpsv3 from the x0 preview: rho = +0.04 (step 4), +0.35 (8), **+0.61 (12)**,
+0.67 (16), +0.91 (24) — so never screen before step 12.

**CRITICAL for any future noise experiment:** two uniform best-of-11 runs differing ONLY in which seeds they drew
differ by **+0.527 hpsv3 [−0.138, +1.498]**. Draw luck exceeds every policy effect measured. Share seed pools across
arms (paired), or run n >= 25 prompts; never compare absolute best-of-k levels across different seed pools.

**Joint (noise x action) allocation, measured 2026-10-19 15:30 (`scripts/joint_arm.py`, fixed schedules, no decider):
the two axes are SUBSTITUTES, not additive.** Action search at a fixed UNSELECTED noise is worth **+1.549** hpsv3
(9/9 prompts; the earlier "+0.88" was the wrong contrast — `rs_hps − rs_blind`, two search arms with different
objectives). But the gain ON THE SCREENED-FOR-BEST noise decays with screening effort: **+0.510 (screen 4), +0.235
(8), +0.219 (16)**. Both axes buy escape from a bad layout, so screening consumes the action axis's headroom.
Best joint schedule (`jt_s8k3`: screen 8 to step 12, keep 3, 10 action variants) vs noise-only at the same budget:
hpsv3 **−0.002 [−0.341, +0.241]** — a dead tie, though it has the study's best held-out VQAScore (0.8513). Every
allocation at 616 NFE lands ~12.0; only all-actions-one-noise is clearly worse (11.699).
**Implication: the ceiling is set by the BUDGET, not the split. Reallocating between noise and actions gains nothing;
going higher needs more NFE or a genuinely new axis.**

**Selector settled (2026-10-19 21:21):** over the same candidate noises, a **rank-sum of HPSv3 + PickScore +
ImageReward** is best on all three held-out metrics (hpsv2 0.3026, qwen 0.8368, vqascore 0.8618). Adding PickScore
to HPSv3 alone wins **vqascore +0.0171 [+0.0050, +0.0304]**; adding IR as a third vote adds a little more (boundary
CIs). **IR alone is still the worst selector** (qwen 0.7427, vqascore 0.8119) — IR is harmful as the SOLE objective,
useful as one vote of three. The rewards are only ~0.3 mutually correlated across candidates, so this is a real
ensemble. **Ranks within the candidate set are scale-free, so no normalisation is needed** (same reasoning that
removed min-max from the multi-reward tree). Use `jev_noise_budget.py --policy never --select rank3`.
