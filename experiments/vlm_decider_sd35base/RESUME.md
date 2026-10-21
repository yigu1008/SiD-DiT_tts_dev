# Resuming the Jev / VLM decider study on another cluster

Everything in this directory was developed against base SD3.5-Large on a shared 8xA6000 box and copied here verbatim
from `/data/ygu/runs/vlm_decider_sd35base_2026-10-16/`. `PLAN.md` is the full narrative (every correction recorded beside
the claim it replaced); `RECOMMENDATION.md` is the operating recipe. Read `PLAN.md`'s last three sections first.

## 1. Repo changes this branch carries (vs `main` @ 800384f)
| file | change | mine? |
|---|---|---|
| `sampling_unified_sd35.py` | `sanitize_rewrite_text` strips `/no_think`, `,no_think`, `Original prompt:`, instruction leaks, language drift (80/773 cached wordings reached the encoders with artifacts; now 0) | mine, on top of pre-existing edits |
| `precompute_sd35_rewrites.py` | root cause: `clean_generation` had the same `<think>`-only defect; delegates to the hardened sanitizer | mine |
| `sampling_unified.py`, `zimage_tts.py` | same sanitizer defect in their own copies | mine |
| `sampling_unified_sd35_lookahead_reweighting.py` | **opt-in decider hooks** in `run_mcts_lookahead` (`args.jev_decider`: bind / adjust_candidates / action_prior / backup_score, `phase` marker). None => byte-identical to shipped actdiff | mine |
| `reward_unified.py` | pre-existing WIP: `composite_hpsv3_ir_vqa` backend | not mine (included for reproducibility) |
| `sd35_ddp_experiment.py` | pre-existing WIP: opt-in `SID_MEM_TRACE` memory instrumentation | not mine |
| `sampling_flux_unified.py` | pre-existing WIP: one-line choices update | not mine |

## 2. Non-git dependencies (must exist on the new cluster)
* **Models** (HF cache was `/data/ygu/.cache/hub`, offline mode): `stabilityai/stable-diffusion-3.5-large`,
  `Qwen/Qwen2.5-VL-7B-Instruct`, `akhilaaa3/Jev-Omni` @ `5addda86ddee081a68fb067477ea100c221b8917`,
  `ImageReward-v1.0`, HPSv3, `yuvalkirstain/PickScore_v1`; judges also need Qwen2-VL (alignment) and `clip-flant5-xxl` (VQAScore).
* **Conda envs** (names are hard-coded in the launchers): `sid_dit` (sampler + Qwen2.5-VL server), `reward` (reward
  server: HPSv3+ImageReward+PickScore — NOT sid_dit), `jev` (Jev-Omni server), `vqascore_reward` (VQAScore judge).
* **Data**: `prompt_map.csv` (prompt_id -> text), `genai200_v1/shared_rewrites_cache.json` (the shipped wording bank),
  judge scripts `score_local.py`, `score_vlm_judge.py`, `score_vqa.py`, `build_followup_manifest.py`,
  `export_scores.py`, `same_day_analysis.py` (from `/data/ygu/reward_orthogonality/...`, not in this repo).
* **Results that travel here**: `results/scores.csv` (all judged rows), `results/main/<arm>/records.jsonl` and
  `decisions.jsonl`. Images (~GBs) do NOT travel; `records.jsonl` keeps their sha256 for verification.

## 3. Path remap (every script hard-codes these; sed them or export equivalents)
| old prefix | meaning |
|---|---|
| `/data/ygu/runs/vlm_decider_sd35base_2026-10-16` | the run dir -> this directory's working copy |
| `/home/ygu/project/SiD-DiT_tts_dev` | this repo |
| `/home/ygu/miniconda3/envs/{sid_dit,reward,jev,vqascore_reward}` | the four envs |
| `/data/ygu/.cache/hub` | HF cache |
| `/data/ygu/reward_orthogonality/mcts/prompts/prompt_map.csv` | prompt map |
| `/data/ygu/human_eval_genai40_v1/sid_vqascore_algorithm_sweep/genai200_v1/shared_rewrites_cache.json` | shipped bank |
| `/data/ygu/reward_orthogonality/followup_2026-09-24` | judge/export scripts |
Quick remap: `grep -rlE "/(data|home)/ygu/" scripts | xargs sed -i -e 's#/data/ygu/runs/vlm_decider_sd35base_2026-10-16#<RUN_DIR>#g' ...`

## 4. Servers (ports are hard-coded; keep them)
`mr3_server.sh <GPU> 5142` (imagereward+hpsv3+pickscore, ~22 GB) · `vlm_server.sh <GPU> 5151` (~17 GB) ·
`jev_server.sh <GPU> 5161` (~24 GB; VLM+Jev fit one 48 GB GPU) · `keepalive3.sh` watchdog.

## 5. What was queued but never ran (GPUs were fully occupied by another user)
1. `run_jevact3.sh` — corrected Jev-as-policy 2x2 (bank width actually delivered; sheet order permuted):
   `ja_jevact_v2w`, `ja_randact_v2w`, then `ec_v2w` (early/late control on the real 9-10-wording bank).
2. `run_actdiff_jev.sh` — the two actdiff proposals: `ad_base`, `ad_rank3`, `ad_jevbank`/`ad_randbank`,
   `ad_jevprobe`/`ad_randprobe`; converter `actdiff_to_records.py`; then `judge_base.sh`.
   Reading rules: #2 is `ad_jevprobe` vs `ad_randprobe` (identical probes & NFE); `ad_jevprobe` vs `ad_base` needs an
   NFE-matched base (scale `--n_sims` to the measured probe NFE) before any claim.
Both launchers self-check (`actdiff_jev_main.py --help`, hooks present) and wait for >=4 free GPUs.

## 6. Pitfalls already paid for (do not rediscover)
* `--n_variants` is a HARD CAP in `generate_variants` (keeps n+1 wordings): size it from the bank or any wider bank is
  silently truncated to 4 (this voided one experiment).
* Never `pgrep -f`/`pkill -f` a pattern that appears in your own command line (killed the controlling shell 5x).
* Tag output dirs with seconds+PID (two judges launched in one minute shared a directory).
* Match NFE before believing any comparison; three "wins" here were budget artifacts.
* Two best-of-11 runs differing only in seed pool differ by +0.527 hpsv3 — share seed pools or use n>=25.

## 7. State at hand-off (2026-10-21 22:15, work moved to another cluster)
* On the A6000 box the jevact launcher DID run for 46 min (19:05-19:53) before being SIGTERM'd by another user's
  launch: `ja_jevact_v2w` 7/9, `ja_randact_v2w` 2/9 prompts, all 616 NFE, bank width confirmed delivered
  (wordings 0-8 used), sheet order permuted. Those partial records/images stay on the A6000 box and are NOT in this
  repo: on a new cluster start both arms clean (the judge needs images for all 9 prompts in one place).
* Local launchers were killed at hand-off so nothing fires twice. To re-arm on the A6000 box:
  `nohup bash scripts/resume_jevact3.sh &` then `nohup bash scripts/run_actdiff_jev.sh &`.
* Launcher behaviour now: `resume_jevact3.sh` loops `run_jevact3.sh` until records are complete (a SIGTERM costs only
  the in-flight prompt; restarts the reward server if killed; honest completion status); `run_actdiff_jev.sh` is
  serialized behind jevact by RECORD COUNTS (not a log line) and takes the highest four GPUs when all eight free.
* The AMLT job (`amlt/jev_decider_sd35base.yaml`, see DEPLOY.md) runs the same two stages on a fresh node.
