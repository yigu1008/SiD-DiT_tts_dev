# Claude Code memory export — SiD-DiT / Jev decider study (exported 2026-10-21)

Index:
- [CoT SFT program state, 2026-09-22](cot-sft-program-state-2026-09-22.md) — paused with all arms negative; read DIAGNOSIS.md before proposing new arms
- [Reward orthogonality result, 2026-09-22](reward-orthogonality-result-2026-09-22.md) — HPSv3 vs VQAScore is ~independent within-prompt (0.11, 0.00 on SiD); all three rewards have only 10-22% within-prompt variance
- [Image reward env pitfalls](image-reward-env-pitfalls.md) — HPSv3+ImageReward run in `reward` (not sid_dit); t2v_metrics' relative hf_cache path will refill /data
- [Unattended chain monitoring](unattended-chain-monitoring.md) — bare `wait` + verdict-only monitors caused a 13h silent stall; stall branch + deadline in every monitor
- [SiD MCTS follow-up 2026-09-24](sid-mcts-followup-2026-09-24.md) — implemented MCTS is UCB1 depth-1 best-of-40 with noise actions; static-vs-MCTS, fixed-bank, Random-IR and VQA-ablation contrasts at n=25 and n=99 (VQA term significant at 99)
- [SiD method timing and memory 2026-09-25](sid-method-timing-memory-2026-09-25.md) — all 25 methods: memory identical (18.2 GB search peak, 31.6 GB process), timing 10 s to 1688 s per prompt; noise_inject default not budget-matched
- [AI writing check server setup](ai-writing-check-server-setup.md) — Binoculars/Falcon-7B detector in env `binoculars` (torch must be cu126, not cu13), weights in ~/.cache/huggingface, run/serve commands, root-disk pitfall
- [VLM decider on base SD3.5, 2026-10-16](vlm-decider-sd35base-2026-10-16.md) — Qwen2.5-VL prunes key-step MCTS on base SD3.5-Large; run dir, servers, arms, probe timings, held-out caveat
- [A6000 shared-GPU kills, 2026-10-16](a6000-shared-gpu-kills-2026-10-16.md) — ryan's multi-GPU launches SIGTERM ygu processes on the GPUs they take; check owners before launching, GPU 1 was safe
- [Reward choice beats reward selection, 2026-10-19](reward-choice-beats-reward-selection-2026-10-19.md) — pick the search reward per run (hpsv3 +0.879, held-out flat); per-node Jev reward selection adds nothing, oracle headroom ~0.01
- [Initial noise dominates action search, 2026-10-19](initial-noise-dominates-action-search-2026-10-19.md) — z0 carries ~5x the reward variance of wording/CFG; best-of-11 noise on HPSv3 beats every search arm, best-of-2 matches MCTS at 1/5.5 NFE

---

## a6000-shared-gpu-kills-2026-10-16.md

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

---

## ai-writing-check-server-setup.md

---
name: ai-writing-check-server-setup
description: "Binoculars AI-writing detector deployed on the GPU server 2026-09-26; env path, model cache, GPU, disk pitfalls, and how to run/serve it"
metadata: 
  node_type: memory
  type: reference
  originSessionId: cb928cd6-a9aa-4c91-bce4-ed781fe2d2d9
  modified: 2026-09-26T03:20:59.906Z
---

Deployed 2026-09-26 for the user's `ai-writing-check` skill (Binoculars, Falcon-7B + Falcon-7B-instruct).

- **Checker:** `~/ai-writing-check/check.py` (copy of the skill's script; skill source unpacked at `/data/ygu/ai_writing_check/ai-writing-check/`). Report format: list of `{summary{thresholds likely_ai 0.854, borderline 0.902}, chunks[{idx,pages,words,score,verdict,text}]}`; lower score = more AI-like.
- **Env:** `/home/ygu/miniconda3/envs/binoculars/bin/python` (Python 3.11, torch 2.14.0+cu126, transformers 5.17, accelerate 1.15, pymupdf 1.28). Plain `pip install torch` gives a **CUDA-13 build that the driver (575.64 / CUDA 12.9) cannot run** — must install from `https://download.pytorch.org/whl/cu126`. transformers 5 prints a harmless `torch_dtype` deprecation warning.
- **Weights:** `~/.cache/huggingface/hub/models--tiiuae--falcon-7b{,-instruct}` (safetensors only, 14 GB each). Run with `HF_HOME=/home/ygu/.cache/huggingface HF_HUB_OFFLINE=1`. Both models load in ~13 s from NVMe; ~28 GB GPU.
- **Run:** `CUDA_VISIBLE_DEVICES=5 HF_HOME=/home/ygu/.cache/huggingface HF_HUB_OFFLINE=1 <env>/python ~/ai-writing-check/check.py paper.pdf --json --out report.json`. Service for the Mac panel: same env, `check.py --serve 8765` in tmux, reached via `ssh -N -L 8765:127.0.0.1:8765 gpu`; stop it when done (holds 28 GB).
- **Smoke test passed** on `~/ai-writing-check/smoke/test.txt` (2 chunks, both human-like, 13 s).

**First real run (2026-09-26, `actdiff_iclr2027.pdf`, 47 pp):** `check.py` drops everything from the References heading on, so the appendix (p.13-47) must be scored from a page-split copy (`pymupdf` insert_pdf) and its page numbers re-offset; references span p.10-12, so find the appendix start by heading text, not by assuming the page after References. Outputs in `~/ai-writing-check/mirror/`: `actdiff_report.json` (main, 39 chunks), `actdiff_appendix_report.json` (102 chunks, appendix-pdf page numbers), `actdiff_report_full.json` (both, original page numbers, panel-ready), `actdiff_iclr2027_marked.pdf` (highlights by verdict + margin notes + legend). Result: 0% likely-AI, one borderline chunk that is a table header. **Running a batch `check.py` while the `--serve` process holds GPU 5 OOMs** — use another idle GPU (GPU 3 worked) or the service's `/score`. Service: tmux session `binoculars`, port 8765, ~27 GB on GPU 5; stop with `tmux kill-session -t binoculars`.

**Disk pitfall:** the root partition (`/`, holds home and miniconda) was at 3 GB free after the 27 GB download; /data is ~17 GB free. On 2026-09-26 the user approved deleting `/tmp/sid_reward_cache`, `/tmp/ollama.tar.zst`, `~/.cache/dnnlib`, and `/data/ygu/reward_orthogonality/images` (phase-1 images; scores kept in `merged_scores.csv`). `/tmp/xeh94gxI` and `/tmp/BlVGmAuS` are ygu-owned dirs whose contents belong to other users and cannot be removed. `pgrep -f` on a path that appears in the calling shell's own command line self-matches — use a `ps ... | grep -v 'bash -c'` filter instead. See [[image-reward-env-pitfalls]].

---

## cot-sft-program-state-2026-09-22.md

---
name: cot-sft-program-state-2026-09-22
description: Where the strong-to-weak SFT program stood when the user paused it on 2026-09-22 (all arms negative; what remains; where records live)
metadata: 
  node_type: memory
  type: project
  originSessionId: bafccd3a-99c6-4610-8c23-ab7f781db088
  modified: 2026-09-22T00:01:12.671Z
---

As of 2026-09-22 the strong→weak SFT program in ~/project/CoT_SFT_dev is paused with nothing running.
Every arm measured (rows A–V in DIAGNOSIS.md, one VERIFY.md section each) is at or below base for
post-trained Qwen3 thinking students (0.6B, 1.7B, 4B): R1 traces, same-family 8B traces, self-distillation,
mixtures, zone selection, trace edits, and four token-loss weightings. Mechanism: token weighting sets the
closing rate (44→97% across masks) but not correctness; the only positive was Qwen3-1.7B-Base (no thinking prior)
with budget. Eval prompt suffix mismatch understates the regression (~2.5 pts), protocol kept.

**Why:** the user asked to pause and move to something else; the next session should not re-derive this.

**How to apply:** read DIAGNOSIS.md (top table + "Is it just that the model is too weak?") before proposing
new arms. Runner scripts are in scripts/runners/ (copied from the session scratchpad). Open offer not taken:
a positive control (Qwen3-0.6B-Base on s1K) for the write-up. Related: [[s2w-standing-preferences]].

---

## image-reward-env-pitfalls.md

---
name: image-reward-env-pitfalls
description: "Which conda env runs which reward model, and the two traps that silently break or fill the disk"
metadata: 
  node_type: memory
  type: reference
  originSessionId: cb928cd6-a9aa-4c91-bce4-ed781fe2d2d9
  modified: 2026-09-22T03:02:44.002Z
---

Reward scoring on this box (verified 2026-09-22):

- **HPSv3 + ImageReward -> `reward` env** (transformers 4.45.2, trl 0.12.2, hpsv3 1.0.0). NOT `sid_dit`. hpsv3's checkpoint uses the pre-4.52 Qwen2-VL key layout (`visual.*`, `model.layers.*`) and its forward pass calls `model.embed_tokens`; transformers 4.52 renamed both, so under `sid_dit` the checkpoint fails a strict load and — even if keys are remapped — the forward pass still breaks. `reward_server.py` was written for the `reward` env for exactly this reason.
- **VQAScore (clip-flant5-xxl) -> `vqascore_reward` env** (t2v_metrics pins transformers 4.49). Needs `$CONDA_PREFIX/bin` on PATH for its ffmpeg check.

**Why:** these are separate envs on purpose; forcing all three into one env costs hours and produces a scorer that silently drops a model.

**How to apply:**
1. `t2v_metrics` hardcodes `HF_CACHE_DIR = "./hf_cache/"` — a **relative** path that ignores `HF_HOME` and `HF_HUB_CACHE`. Launched from anywhere but `/home/ygu/project/SiD-DiT_tts_dev`, it re-downloads all 22GB of clip-flant5-xxl into `./hf_cache` (this filled `/data` to 0 bytes once). Always pass `cache_dir="/home/ygu/project/SiD-DiT_tts_dev/hf_cache"` explicitly AND chdir there.
2. `UnifiedRewardScorer(backend="composite_hpsv3_ir")` loads both models; call `_score_hpsv3` / `_score_imagereward` separately to keep raw scores instead of the blend. Assert on `scorer.available` and `scorer.state.hpsv3_impl == "official"` — the loader otherwise falls back silently (and `imscore` isn't installed, so HPSv3 just disappears).
3. There is a PyPI package literally named `apex` installed in `sid_dit` (a Pyramid web toolkit, not NVIDIA apex) that breaks `transformers.trainer`. Not present in `reward`.

Working scripts: `/data/ygu/reward_orthogonality/{score_local,score_vqa}.py`. Results in [[reward-orthogonality-result-2026-09-22]].

---

## initial-noise-dominates-action-search-2026-10-19.md

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

---

## reward-choice-beats-reward-selection-2026-10-19.md

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

---

## reward-orthogonality-result-2026-09-22.md

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

---

## sid-mcts-followup-2026-09-24.md

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

---

## sid-method-timing-memory-2026-09-25.md

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

---

## unattended-chain-monitoring.md

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

---

## vlm-decider-sd35base-2026-10-16.md

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
