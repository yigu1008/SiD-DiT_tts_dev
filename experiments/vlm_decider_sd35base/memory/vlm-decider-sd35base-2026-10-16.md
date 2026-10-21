---
name: vlm-decider-sd35base-2026-10-16
description: "New project step (started 2026-10-16): VLM judge (Qwen2.5-VL-7B) makes per-key-step MCTS decisions on BASE SD3.5-Large (28-step Euler); run dir, servers, arms, what is held out, probe timings"
metadata: 
  node_type: memory
  type: project
  originSessionId: cb928cd6-a9aa-4c91-bce4-ed781fe2d2d9
  modified: 2026-10-16T19:47:49.646Z
---

Next step after the SiD search-vs-fixed ablation ([[sid-mcts-followup-2026-09-24]]): a VLM "judge" that
makes decisions inside MCTS, on base SD3.5-Large (28 steps, real per-step MDP), collecting (preview,
action set, VLM comment, choice, outcome) records as a dataset.

**Run dir:** `/data/ygu/runs/vlm_decider_sd35base_2026-10-16/` (scripts/, probe/, logs/, state/, config/).
- `scripts/base_arm.py --arm baseline|mcts` : stock arms on `--backend sd35_base` (bf16, Euler, cfg bank 3.5..7.0,
  baseline_cfg 4.5, 28-action bank = 4 variants x 7 CFG from the shared rewrite cache, key steps 0,4,8,12,16,20,24,27,
  UCB1 c=1.41, ImageReward objective). Noise-injection actions are auto-off for steps>4.
- `scripts/vlm_mcts.py --arm vlm_prune|rand_prune` : per-node candidate set pruned at node creation. vlm_prune asks the
  VLM server `/decide` (factorized: rank variants with shuffled letter labels + rank CFG levels; keep top-2 variants x
  top-3 CFGs + baseline anchor); rand_prune keeps 6 random actions (control for branching factor). Rollouts sample from
  the leaf's kept set. Decisions + 512px previews logged to `decisions.jsonl` / `previews/`.
- `scripts/vlm_server.py` (+ `vlm_server.sh <GPU> [PORT=5151]`): Qwen2.5-VL-7B-Instruct, sdpa, bf16, ~17 GB, sid_dit env,
  greedy decoding, 3-6 s per decision. `/generate` generic, `/decide` structured.
- IR server pattern: `reward_server.py --backends imagereward` in `reward` env, 2.3 GB (port 5141 used in probe).

**Held-out:** the existing alignment judge is Qwen2-VL-7B (`/data/ygu/reward_orthogonality/score_vlm_judge.py`), so the
decider is a DIFFERENT checkpoint (Qwen2.5-VL) but same family: report it with a caveat; HPSv3/PickScore/VQAScore are the
clean held-out judges. VLM must judge against the ORIGINAL prompt only (rewrites steer its comments otherwise).

**Weights (all in /data/ygu/.cache/hub):** stabilityai/stable-diffusion-3.5-large (26 GiB, downloaded 2026-10-16 with the
user's HF token), Qwen/Qwen2.5-VL-7B-Instruct (16 GB). UnifiedReward-qwen-7b NOT downloaded (needs sid_dit env, not reward env).
VQAScore cache moved: `SiD-DiT_tts_dev/hf_cache` is now a symlink to `/data/ygu/.cache/t2v_hf_cache` (26 GB off root).

**Probe timings (q0000, 1024^2, RTX 6000 Ada):** baseline 49 s/prompt incl. encode (56 NFE, peak 25.7 GiB);
stock MCTS ~21 s per simulation (one 28-step CFG rollout) -> ~15 min/prompt at 40 sims. Fill in vlm_prune/rand_prune.

**Why:** reviewer critique "the per-step MDP is never tested" cannot be answered on 4-step SiD (depth-1 in practice);
base SD3.5 with key-step branching is where a VLM policy/pruner can matter. Dataset of decisions feeds the CoT-SFT line.
**How to apply:** new arms go in this run dir, never modify stock `run_mcts`; keep terminal objective = IR so the
search-vs-fixed story transfers; 40-sim stock MCTS on base is expensive (~15 min/prompt) so budget-match by NFE.

**Probe results (2026-10-16 evening, q0000, 40 sims; PROBE.md):** stock `run_mcts` default for steps>4 silently adds a
sparse-noise refine pass (+~1000 NFE) → `base_arm.py --arm mcts` disables it, `mcts_refine` keeps it. NFE depends on
tree depth (prefix reuse): flat stock tree ~2200 NFE/40 sims, pruned trees ~2000. Costs/prompt: mcts ~18 min, rand 14,
vlm 17, jev 18. n=1: stock+refine IR +0.25 but HPSv3 −3.8/PickScore −0.9 (over-saturated); VLM/Jev-pruned IR +0.16/+0.11
with HPSv3 +0.5/+0.7; random prune IR +0.14, HPSv3 −1.0. Jev picks original wording 40/41 nodes; CFG choice weak.
"Jev" = Jev-Omni decision model (akhilaaa3/Jev-Omni, user's jev-cub/jev-imagegen projects), NOT a generic judge.
Cohort run (25 prompts, 5 arms, ~28 GPU-h) prepared, NOT launched; shared-GPU situation see [[a6000-shared-gpu-kills-2026-10-16]].

**Arm 6 `opt_prompt` (2026-10-16 22:20, user-approved "deliberate discrete prompt optimizer"):** offline per-prompt loop, checklist
(VLM) -> sample -> per-element present/wrong/missing readout -> K typed edits (emphasize/reorder/clarify/drop_clause/revert) ->
Jev faithfulness guard (yes/no) -> Jev choice incl. keep; output = most-complete round; NO reward in the loop. Scripts
`opt_prompt_arm.py`, `run_optprompt_probe.sh <N>` (waits for 2 ryan-free GPUs), `make_optprompt_sheet.py` (eye-check sheets).
Next: eye-check checklist accuracy on 5 prompts before trusting it; online variant (edits at key steps 0,4) only after that.

**opt_prompt results (2026-10-17 night):** v1 readout too lenient (never edits). v2/v3 (observe-then-grade + Jev per-element verify, guard that
allows implied contrasts @0.7, no-repeat, 896px) fixed 3/5 hard prompts for real: birds tail contrast 1/6->6/6, train toys non-adjacency 0/4->4/4,
bracelets 0/4->3/4 (eye-checked on sheets). Dresses/gemstones stuck because the proposer leaked observed attributes -> v4 = original-vocabulary
proposer + lexical filter. Pitfalls hit: editing a running bash launcher corrupts it (rc=127); ryan's GPU workers come and go within minutes.
**opt_prompt final (2026-10-17 02:40):** v5 = retention-only lexical filter + Jev guard for additions + sampled retries is the keeper
(scripts/opt_prompt_arm.py). Held-out vs original wording on hard prompts: birds VQA +0.16/Pick +0.61, trains HPSv3 +0.76/VQA +0.12,
bracelets HPSv3 +4.25/Pick +0.92; dresses/gemstones not fixable by wording at seed 42. v4 (strict vocabulary filter) hurt. OPTPROMPT.md has all tables.
Arm names must be unique per run dir (judge collides otherwise) — now defaults to run_dir basename.

**opt_prompt COHORT n=25 (2026-10-17 10:43, main/opt_prompt, contrast_n25_opt_prompt_minus_r0.txt):** no reward in the loop, so
ALL metrics are held out. IR +0.077 [+0.001,+0.167]*, qwen_alignment +0.035 [+0.003,+0.077]* (same-family caveat: Qwen2.5-VL
readout vs Qwen2-VL judge), HPSv3/PickScore/HPSv2/VQAScore all inconclusive; VQAScore +0.0015 = no movement. Self-graded
checklist completeness 0.61 -> 0.74. **Reading: the loop improves what it measures itself; independent compositional evidence is
weak.** 16/25 prompts changed wording, 9 improved completeness, 267 s/prompt. Before scaling, fix the readout-vs-VQAScore gap.

**Arm 7 `jev_online` (2026-10-17, user asked for the online variant despite the unvalidated-readout caveat):**
scripts/jev_online.py + run_online.sh. = jev_prune but at depth<=1 (steps 0,4) each node grades the checklist on its own
preview, proposes K=3 typed edits of its current wording, filters (lexical retention + Jev guard >=0.7) and ENCODES the
survivors mid-trajectory into the prompt's embedding bank (cap 12). Deeper nodes only re-rank. Contrast jev_online -
jev_prune isolates online wording generation vs the 4 cached wordings. Mid-trajectory encode needs the transformer
park/unpark swap (~6 s/event), hence depth<=1 only.

**jev_online FIRST RESULT (2026-10-17, n=3, budget-matched 2032 NFE): NEGATIVE.** online minus fixed-bank jev_prune:
ir −0.137, hpsv3 −0.773 (both intervals exclude 0), everything else n.s. and none favouring online. Likely mechanism:
keep stays top-2 wordings, so a 12-wording bank = noisier 12-way Jev ranking than a 4-way one (dilution), and the cap
bound after 2-3 nodes. Sweep running (wide/deep/k5/root) on the 3 dev prompts; disjoint test6 set reserved for the winner.
Don't quote n=3 bootstrap intervals as evidence - direction only.

**Decision state (2026-10-18, user asked to "keep the states for jev" + "build smarter prompt base"):** scripts/state_store.py
= PromptState (per-prompt: per-wording visit counts + mean/best terminal reward, action path, realised checklist deltas of
each edit) + WordingBase (cross-prompt append-only JSONL: (element_type, operator) -> how often it flipped that element to
present; rendered as a prior into the proposer and Jev's state). Flag --use_state on jev_online.py; --freeze_base for
held-out evaluation (MUST freeze, else contamination). Caveat: PromptState surfaces reward-derived values, so a state-enabled
arm is no longer reward-free - never give it to the opt_prompt arm, whose selling point is that no reward enters the loop.

**SWEEP VERDICT (2026-10-18, n=3 dev, judged, SWEEP.md): online wording generation does NOT beat the fixed 4-wording bank**
in any of 5 configs (root-only = parity; wide/k5/deep/original all worse on clean held-out metrics). Original arm's loss was
the keep gate (top-2 of 12 = noisy ranking); widening -> parity only. deep (generate at step 8) is worst + ~90% costlier.
**Key cost finding: sw_root (4 wordings at the root, no image, no readouts) matches the control with 2 VLM calls/prompt vs
the control's 122** -> the per-node preview readouts buy nothing either. Report as a negative ablation result.

**LADDER VERDICT (2026-10-18, n=3 dev, judged) — THE headline:** blind MCTS hacks ImageReward, Jev-guided search does not.
blind mcts ir 1.541 / hpsv3 9.594; jev_prune ir 1.502 / hpsv3 10.806; sw_root ir 1.485 / hpsv3 10.887 (best); no-search
baseline ir 1.142 / hpsv3 10.271. **Blind search is worse on HPSv3 than NO SEARCH while gaining +0.40 ir.** Jev arms beat
baseline on both axes. Never judge these arms on ImageReward alone - it is the search objective and it reverses the ordering.
Best config = sw_root (root-only wording generation, 2 VLM calls/prompt vs 122). Confound: blind had 8% more NFE; bm44 rerun queued.

**STATE A/B judged (2026-10-18):** per-prompt memory gives ir +0.004 but hpsv3 +0.300 (held-out). sw_beststate is the best
arm overall: vs blind MCTS ir −0.018, hpsv3 +1.452, pickscore +0.363; highest hpsv3 of any arm (11.046 vs blind 9.594 vs
no-search 10.271). Cross-prompt base contributed NOTHING - it never populated (attribution needs parent AND child readouts;
fixed via --readout_depth, still untested). Pattern across the whole study: the decision machinery doesn't find better
optima, it stops the search from trading perceptual quality for ImageReward.

**CORRECTION (2026-10-18): "Jev prevents reward hacking" was a BUDGET ARTIFACT.** Same arm at 40 vs 44 sims: ir +0.016 but
hpsv3 −0.951 (q0027 −1.60, q0041 −1.26). At matched ~2200 NFE: blind hpsv3 9.594 / Jev 9.854 / Jev-root 9.834 — all BELOW
the no-search baseline 10.271; Jev advantage shrinks +1.21 -> +0.26. The earlier headline compared Jev@2032 NFE with
blind@2200 and attributed a budget effect to a mechanism. **hpsv3 degrades with search budget for EVERY arm.** The right
experiment is a frontier: sweep n_sims {10,20,40,80} per arm, plot hpsv3 vs ir; only a higher curve proves a mechanism.

**BUDGET FRONTIER (2026-10-18), blind MCTS on base SD3.5 — reframes everything:** sims 0/10/20/40/80 ->
ir 1.142/1.488/1.494/1.541/1.541, hpsv3 10.271/**10.744**/9.748/9.594/9.594. (a) **Search SATURATES by 40 sims**: at 80 sims
images are byte-identical to 40 (same sha256 + action paths) for 4120 vs 2200 NFE. (b) **Knee at 10 sims**: 97% of the ir
gain with hpsv3 ABOVE the no-search baseline. The hpsv3 cliff is between 10 and 20 sims. **Every arm in this study ran at
40 sims = 4x past saturation and past the cliff** — all arm-vs-arm comparisons were at the worst trade-off point.
Use ~10 sims as the operating point; re-ask every comparison there.

**BOTH FRONTIERS (2026-10-18) — CAPSTONE: the Jev machinery does NOT buy a better frontier.** blind@10 (616 NFE, 0 VLM
calls, 238 s) ir 1.488 / hpsv3 10.744 DOMINATES jev@10 and jev@20 on both axes and TIES jev@40 (2032 NFE, 122 VLM calls,
866 s; ir 1.502 / hpsv3 10.806) at ~1/3 the cost. Recommendation: run blind key-step MCTS at ~10 sims. Caveat: jev curve
is non-monotonic (hpsv3 10.806@40 -> 9.854@44 -> 9.058@80) so nearby-point ordering is NOT established at n=3
(hpsv3 spans 8.8-13.1 across the 3 prompts). To overturn: same frontier at n>=25 AND a Jev point strictly above the blind curve.

**TRUNCATION PROBE (2026-10-18, main/truncation):** rollouts = 83% of search NFE. Spearman of x0-preview IR vs final IR:
step4 **−0.735 (ANTI-correlated)**, step8 −0.234, step12 +0.479, step16 +0.838 (regret 0.085), step20 +0.901, step24 +0.921.
**Truncate rollouts at step 16** -> ~36% total NFE saving; stacked with 10 sims: 2200 -> 616 -> ~400 NFE (5.5x).
**Never score an early x0 preview with ImageReward** - it is worse than guessing before step ~12 (audit the
noise-injection path at steps 1-2). Open question where the VLM could still win: can Jev rank partial trajectories at
step 8 where IR anti-correlates? That would prune whole rollouts.

**HOW JEV DECIDES / why it loses at low budget (2026-10-18):** per node it ranks 4 wordings + 7 CFGs (order-averaged),
keep = top-2 wordings x top-3 CFGs + anchor (~7 of 28 actions). **It ranks CFG 7.0 first at 33/33 nodes** regardless of
prompt/step/preview - a constant prior, not a decision; kept set skews high (7.0 x66, 6.0 x52 vs 4.0 x8). CFG top-prob
0.705 vs order-spread 0.288. Label probe (12 nodes, faithful state): current 12/12 7.0, **neutral bare numbers 10/12**,
reversed 6/12 - so it is the MODEL's prior, not my option wording. High CFG = the oversaturation mode, so at 10 sims the
prior is the whole policy and the arm loses on BOTH ir and hpsv3; at 40 sims UCB1 overrules it. **Fix: drop Jev from the
CFG decision, let UCB1 use the full bank; keep Jev for wording only, or use its probs as a soft PUCT prior.**

**JOINT (wording,CFG) SELECTION (2026-10-18):** user was right that the factorised design assumes separability - verified:
best CFG differs across wordings at 8/8 nodes, factorised set misses ~46% of the joint top-6, and the degenerate CFG-7.0
prior is an artifact of the MARGINAL question (joint spreads 3.5-7.0). Implemented `vlm_mcts.py --joint`. **But at 10 sims
it changes nothing**: joint−factorised ir +0.014, hpsv3 −0.037, vqascore −0.036; both still far below blind (ir 1.488,
hpsv3 10.744). **So the low-budget deficit comes from RESTRICTING the action set at all, not from how it is chosen** -
blind tries 10 of 28 root actions, Jev tries 7. **Stop hard-pruning; use Jev as a soft PUCT prior or only where IR is blind.**

**PUCT SOFT PRIOR (2026-10-18) — softening makes it WORSE, so the prior (not the interface) is the problem.**
@10 sims ir: blind 1.488 > jev-joint-prune 1.384 > jev-factorised-prune 1.370 > **jev-PUCT 1.219** > no-search 1.142.
Monotone in how much influence Jev has (PUCT = action set + selection + rollouts). Failure shapes: depth-4 tunnelling on
one prompt, correlated prior-sampled rollouts on the others. Nuance: PUCT keeps hpsv3 10.661 (> both pruned arms) - it
preserves quality but fails to find good images. Untested: prior tempering / root Dirichlet noise (anti-concentration).

**WHY JEV DOESN'T HELP (2026-10-18, the root cause):** NOT plumbing, NOT state starvation — a QUESTION-TYPE mismatch.
State ablation on the CFG question: withholding the image flips 0/8 choices (TV 0.069), swapping in another node's
readout 1/8, adding the current CFG to the state 1/8 — it says 7.0 whether the preview was made at 4.5 or at 7.0.
BUT image sanity: "sharp or blurry?" 7/7 correct (P=0.999 on step-4 previews), "which description matches?" 3/3 (P≈1.0).
**So Jev answers PERCEPTION questions from evidence and POLICY questions ("which CFG should I use?") from a lexical
prior.** DESIGN RULE: never ask Jev what setting to use; ask which OUTCOME is better (decode candidate children's
previews, ask which better matches the prompt) - a matching question, in distribution. This also explains why
per-prompt memory helped (+0.30 hpsv3): it converted the decision into evidence rather than control prediction.

**RETRIEVAL-GROUNDED JEV (2026-10-18, user's design) — first interface that does NOT damage the search.**
Show Jev the FINAL images nearby configs already produced (captioned with wording/CFG + end-of-trace VLM assessment +
reward) and ask which panel best matches the prompt. scripts/jev_retrieval.py. @10 sims, 616 NFE (= blind):
mean ir 1.487 vs blind 1.489; on q0027 (where every other Jev arm collapsed to 0.83-0.86) it matches blind at 1.074.
Judged vs blind: ir −0.001, pickscore +0.026, vqascore +0.023, hpsv3 −0.424; 271s vs 238s + 18 model calls.
**Damage gone, gain absent.** CRITICAL impl detail: retrieve one panel per DISTINCT wording - ranking by score returned
the same wording at adjacent CFGs (near-identical images, degenerate comparison). Next: same mechanism in the early
window (<step 12) where IR anti-correlates - the only measured capability gap left.

**BRANCH-POINT SELECTION CLOSED (2026-10-18):** 6 key-step schedules, blind MCTS @10 sims, all 616 NFE (placement
changes tree shape, not cost): spread8 (default) 1.488 > two 1.474 > spread4 1.459 > late4 1.457 > mid4 1.444 >
early4 1.439. **Per-prompt oracle 1.494 vs best fixed 1.488 = +0.006 headroom** -> a branch-point decider is not worth
building; keep spread8. Per-prompt crossover exists (early4 +0.017 on q0011, −0.124 on q0027) but is worth nothing.

**Prompt-proposer audit + two upstream bugs (2026-10-19).** The shipped wording cache
(`genai200_v1/shared_rewrites_cache.json`) is badly damaged: of 600 non-base variants, **46% corrupt the prompt**
(numeral multiset changes or a content word is dropped), **20% carry generator artifacts** (`/no_think`,
`Original prompt:`), and 35 of 117 variants on negation prompts lose the negation. Root cause:
`precompute_sd35_rewrites.py:156` appends Qwen3's `/no_think` to the USER message and `clean_generation` stripped
only `<think>` spans. Fixed in `sanitize_rewrite_text` (sampling_unified_sd35.py) + the sibling copies in
`sampling_unified.py`, `zimage_tts.py`, and at the source in precompute: 80/773 wordings carried an artifact before,
**0/737 after** (watch for `/no_think黄昏` — fused CJK defeats `\b`, needs an ASCII-boundary assertion).
**But judged at n=9 the fix changes nothing measurable** (hpsv3 +0.015, vqascore −0.0007): a correctness win only,
and it revises no earlier result.

**And a constraint-PRESERVING replacement bank was a net LOSS: hpsv3 −0.212 [−0.344, −0.096], with NO gain on the
faithfulness judges** (vqascore −0.009, qwen +0.006, both inconclusive). Cause is diversity, not fidelity: the
validator's 56% yield left 2.67 distinct wordings vs 4.00, and prompts down to 2 lost −0.270 vs −0.132 for 3-4.
**General lesson: in reward-guided search, proposal DIVERSITY beats proposal CORRECTNESS** — selection already
filters a bad variant (75% of generations used a non-base wording; the corrupt ones simply lost), but it cannot
invent a proposal that is missing. If revisiting, raise the validator's yield, not its strictness.

**Re-run after fixing the budget-starvation bug (2026-10-19 21:05).** With a phase-4 guard that spends leftover
budget on extra initial noises, the clean-bank arm goes from losing to dominating both shipped-bank arms on all six
metrics (qwen +0.054 [+0.009, +0.125]). **But the gain is noise, not wording**: isolating wording-derived attempts
only, the faithful bank is still **−0.193 [−0.330, −0.078]** hpsv3, and the returned image came from a fill noise on
3 of the 4 prompts that gained. A smaller wording bank wins only because it leaves budget unspent and that budget
buys noise draws. All bank variants remain dominated by plain best-of-11 noise (hpsv3 −1.638 [−2.662, −0.738]).
Lesson: when an arm can exhaust its action pool, ALWAYS add a budget guard and record the fill count, or the
comparison silently becomes an allocation comparison instead of the one you intended.

**actdiff hooks (2026-10-21).** `run_mcts_lookahead` in `SiD-DiT_tts_dev/sampling_unified_sd35_lookahead_reweighting.py`
now has opt-in decider hooks (`args.jev_decider`: bind / adjust_candidates / action_prior / backup_score, phase marker
"tree"/"rollout"); None = shipped behaviour. Deciders live in the run dir (`jev_actdiff_decider.py`), entry wrapper
`actdiff_jev_main.py`, converter `actdiff_to_records.py` (images at `images/p{idx:05d}_mcts.png`, no path key).
Prescreen is 8 seeds -> top-2 in `sd35_ddp_experiment_bon_mcts._run_bon_mcts`. Any sd35 arm must size `--n_variants`
from the bank or the loader silently truncates to 4 wordings. Probes must be gated to the tree phase.

**Pushed to GitHub (2026-10-21):** branch `jev-decider-sd35base` on `git@github.com:yigu1008/SiD-DiT_tts_dev.git`,
commit 9098a69 (from main 800384f). The whole run dir's code/config/docs/results (no images) now lives in the repo at
`experiments/vlm_decider_sd35base/` with `RESUME.md` (dependencies, path-remap table, what was queued). The branch
also carries the user's pre-existing uncommitted WIP in reward_unified.py / sd35_ddp_experiment.py /
sampling_flux_unified.py (labelled in the commit message). Two local launchers (`run_jevact3.sh`, `run_actdiff_jev.sh`)
were still waiting for GPUs on the A6000 box when pushed -- kill them if the work resumes elsewhere to avoid
duplicate runs.

**Deploy path (2026-10-21):** the user's cluster is Azure ML Singularity via AMLT (`amlt run amlt/<job>.yaml`), target
`GenAI-Shared-UKSouth2`, image `acpt-torch2.7.1-py3.10-cuda12.6`, blob mount `/mnt/data` (HF cache
`/mnt/data/v-yigu/model_cache/hf_cache`), `hf_token` passed as `-p` (never embed). No cluster is reachable from the
A6000 box (no amlt/ssh). `amlt/jev_decider_sd35base.yaml` + `experiments/vlm_decider_sd35base/{DEPLOY.md,node_remap.sh,deps/}`
on branch `jev-decider-sd35base` run both queued stages. `jev_omni.py`/`head.pt` ship INSIDE the Jev-Omni HF snapshot
(`jev_server.py` pins `/data/ygu/hf_cache/hub`, a different cache root from `/data/ygu/.cache/hub` -- both remapped).
HPSv3 weights are pre-resolved from `MizzenAI/HPSv3`; VQAScore needs `zhiqiulin/clip-flant5-xxl` (22 GB) prefetched
before `HF_HUB_OFFLINE=1`. AMLT yaml gotcha: an unquoted `key: value` inside a command line parses as a dict.

**HAND-OFF 2026-10-21 22:15 — the user moved this work to another cluster.** Everything is on branch
`jev-decider-sd35base` (repo docs RESUME.md §7 + DEPLOY.md are the hand-off; THIS memory does not travel). On the
A6000 box the jevact launcher ran 46 min (19:05-19:53) then was SIGTERM'd by ryan's relaunch (rc=143, servers
`Terminated`): `ja_jevact_v2w` 7/9, `ja_randact_v2w` 2/9 remain under `main/` locally (not in the repo). Local
launchers were killed at hand-off so nothing fires twice; re-arm with `resume_jevact3.sh` then `run_actdiff_jev.sh`.
If this box is used again: check whether the other cluster already finished these arms before re-arming.
