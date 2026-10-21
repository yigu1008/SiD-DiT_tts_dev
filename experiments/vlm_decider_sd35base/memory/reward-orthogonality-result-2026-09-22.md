---
name: reward-orthogonality-result-2026-09-22
description: HPSv3/ImageReward/VQAScore within-prompt correlations measured 2026-09-22; HPSv3 vs VQAScore is ~independent
metadata: 
  node_type: memory
  type: project
  originSessionId: cb928cd6-a9aa-4c91-bce4-ed781fe2d2d9
  modified: 2026-09-22T03:02:31.464Z
---

Measured 2026-09-22 in `/data/ygu/reward_orthogonality/` (200 prompts x 4 samples x 4 backends = 3200 images; FLUX-schnell, SD3.5-large base, SiD-DiT-SD3.5-large, SenseFlow; native sampling per backend; prompts = 100 GenAI-Bench + 100 HPSv2).

**Within-prompt** Pearson (each reward mean-centered inside its prompt group of 4 — the quantity a GRPO advantage sees):
- HPSv3 vs ImageReward: 0.32 [0.25, 0.38]
- ImageReward vs VQAScore: 0.28 [0.18, 0.37]
- HPSv3 vs VQAScore: **0.11 [0.05, 0.17]**, and **-0.001 on SiD** — pairwise concordance 0.51-0.54 vs 0.50 chance.

Two facts that shape any reward-mixture work:
1. Only **10-22% of each reward's variance is within-prompt**; VQAScore is lowest (0.10-0.14, within-prompt SD ~0.09 on a saturated 0-1 scale). It is a nearly-independent but *weak* axis. Normalize per-reward within the prompt group before mixing, or HPSv3 (SD ~1.0-1.5) dominates by scale alone.
2. Orthogonality is **backend-dependent** (HPSv3 vs VQAScore: 0.17 on FLUX -> 0.00 on SiD), so mixture weights tuned on one backend will not transfer.

Pooled (uncentered) correlations are misleading here — they are inflated by prompt difficulty and disagree with the within-prompt numbers in both directions. Always report the centered version.

**Best-of-N hacking test (16 samples/prompt, FLUX + SiD, held-out judges = PickScore + HPSv2; paired bootstrap CIs over 200 prompts in `analysis/hacking_bon_paired_ci.csv`):**
- HPSv3-only selection buys NO alignment on SiD: VQAScore of its best-of-16 pick is -0.06 [-0.22, +0.08] (n.s.; NOT significantly negative -- an earlier "actively degrades" reading was an overclaim) and only +0.21 on FLUX. The 3-way mix buys +0.87/+0.91 SD of VQA instead (+0.93*/+0.70* vs HPSv3-only).
- The 3-way z-mean (HPSv3+IR+VQA) costs NOTHING significant on the preference judges vs HPSv3-only (FLUX: -0.03/-0.07 n.s.; SiD: PickScore +0.15*, HPSv2 +0.08 n.s.) while giving up -0.34*/-0.47* SD of raw HPSv3 -- i.e. the extra HPSv3 that single-reward pressure wins is not real quality by the judges' lights.
- The mix dominates ImageReward-only and VQAScore-only on both held-out judges (all *). VQAScore-only saturates on preference judges (+0.09 -> +0.03 on SiD).
- ImageReward is the stabilizer: HPSv3+VQA WITHOUT it hurts HPSv2 on SiD (-0.24*); with it, no harm.
- Adding VQA to HPSv3+IR costs a little preference score on FLUX (-0.14*/-0.12*) for +0.62* alignment; on SiD no cost. It is a trade, not a free lunch.
- Caveat: both held-out judges are preference models; an alignment held-out (GenEval/VLM judge) is still missing. MCTS-arm results (prompt-rewrite x CFG on SiD) land in `analysis/report_hacking.txt`.

**MCTS hacking test (finished 2026-09-22 21:56; SiD, prompt-rewrite x CFG search, n_sims=40, 4 arms x 99 GenAI-Bench prompts, identical seed-42 baseline per prompt; `analysis/hacking_mcts_*.csv`; deltas in SiD within-prompt SD units):**
- Single-reward MCTS REVERSES other rewards: hpsv3-arm drives ImageReward -0.40 and PickScore -0.43* (68% of prompts worse); vqascore-arm drives HPSv3 -0.73, PickScore -0.69*, HPSv2 -0.36* (both held-out judges worse on 49% of prompts); imagereward-arm drives HPSv3 -0.56.
- Composite (HPSv3+IR+VQA) arm: HPSv3 +1.63, VQA +1.20, IR +0.08, held-out PickScore -0.10 n.s. / HPSv2 +0.15 n.s. -- the only arm with no significant held-out decline and no training-reward reversal. Paired vs hpsv3-arm: PickScore +0.34*, IR +0.48*, VQA +0.63*, at a cost of HPSv3 -0.39*. Vs vqascore-arm: better on everything but VQA (PickScore +0.60*, HPSv2 +0.51*).
- Honest framing: NO arm improved the held-out judges under MCTS; the composite holds them flat while single rewards degrade them. Mitigation = prevents degradation.
- The fixed-range composite is HPSv3-dominated (per within-prompt SD: 65% HPSv3 / 16% IR / 19% VQA) and still mitigates; a within-prompt-SD-normalized composite is the obvious next arm.
- Rewrite-selection audit did NOT discriminate arms (task-changing rewrite used in 29-34% of prompts for every arm, incl. VQAScore): the hacking is in image-level trade-offs, not in choosing constraint-dropping rewrites; VQAScore does not reliably penalize them.
- Pipeline validated: search-time scores == offline rescoring (r=1.0000 HPSv3/IR, 0.9993 VQA).

**Balanced-composite arm (2026-09-23 01:24; same MCTS setup, composite ranges = 5x each reward's SiD within-prompt SD so per-SD weights are equal; env COMPOSITE_*_LO/HI, no code change):**
- Constituents now rise evenly: HPSv3 +1.21, IR +0.62, VQA +1.58 (vs original composite +1.63/+0.08/+1.20). Paired balanced-minus-original: HPSv3 -0.42*, IR +0.54*, VQA +0.38*; held-out PickScore -0.13 n.s., HPSv2 +0.16 n.s.
- Held-out: PickScore -0.23 n.s. [-0.52,+0.04], HPSv2 +0.31* -- the only composite with a significantly POSITIVE held-out delta, but its PickScore edge over hpsv3-only is n.s. (+0.21) where the original composite's was * (+0.34). Net: weighting is a knob for WHICH constituents gain, not for hacking -- mitigation (no reversal, no significant held-out decline) holds under both weightings.
- Rewrite-selection audit unchanged (28% task-changing picks; no discrimination).
- Lesson: `build_mcts_manifest.py` originally keyed metadata on (shard, slug) against the 4-way split; a new split silently mislabeled prompt_ids and collapsed 46 rows in dedup. Fixed to key on prompt text. Verified original 4 arms unaffected (0 mismatches).

**Alignment held-out judge (2026-09-23 03:20; zero-shot Qwen2-VL-7B-Instruct P(yes) "does this image depict the description?", `score_vlm_judge.py`, 6 img/s/GPU; `analysis/report_vlm_judge.txt`):**
- It measures alignment: within-prompt r with VQAScore 0.41 (FLUX)/0.30 (SiD) vs HPSv3 0.17/0.19.
- MCTS: composite +1.02* and balanced +1.03* on the judge; vqascore-only +0.85*; hpsv3-only +0.47 n.s.; imagereward-only +0.35 n.s. Composite - hpsv3 = +0.56*; composite - vqascore = +0.17 n.s.
- BoN16: all3 z-mean beats hpsv3-only on the judge by +0.31* (FLUX) / +0.23* (SiD).
- Net: the composite is never significantly worse than the best single reward on ANY independent judge (matches VQA-only on alignment, beats it on both preference judges; beats HPSv3-only on alignment and PickScore). This closes the "held-out judges are preference-only" caveat. Lineage caveat: HPSv3 is fine-tuned from the same Qwen2-VL-7B base.
- Doc: https://claude.ai/code/artifact/1fab9d2f-3c2c-400b-8a20-981da7cee9fb (summary, sections 1-6).

**Rewrite bank** (`shared_rewrites_cache.json`, 200x3): ~24% unusable (dropped negation 38%, comparative 44%, numeral 17%; `/no_think` and "Original prompt:" echoed by Qwen2.5-3B in 26%); styles are aesthetic-only. Lint + cleaned bank + v2 prompt in `reward_orthogonality/rewrites/`. The lint flags let MCTS choices be audited for task-changing rewrites (a built-in hacking channel, since reward is judged vs the ORIGINAL prompt).

Artifacts: `analysis/report.txt`, per-table CSVs, `merged_scores.csv` (3200 rows, all three scores + prompt metadata). See [[image-reward-env-pitfalls]] for how to re-run the scoring.
