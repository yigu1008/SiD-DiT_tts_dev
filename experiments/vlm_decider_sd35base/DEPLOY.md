# Deploying the queued experiments on Azure ML (AMLT)

The A6000 box is fully occupied by another user, so the two queued experiments never ran there. This is how to run
them on the Singularity cluster the rest of the repo already uses. No cluster is reachable from the A6000 box itself
(no `amlt`, no SSH hosts) -- submit from wherever you normally run `amlt`.

## Submit
```bash
git fetch && git checkout jev-decider-sd35base
amlt run amlt/jev_decider_sd35base.yaml -p hf_token=$HF_TOKEN -p run_tag=jev_v1
```
Useful overrides: `-p run_vqascore=0` (skip the 22 GB CLIP-FlanT5-XXL judge; VQAScore rows are then absent),
`-p stages=actdiff` or `-p stages=jevact` (one stage only; default runs both in order).

## What the job does (one node, 8x80 GB)
1. Pins the sampler env (`ptca`: torch 2.7.1, transformers 4.52.4, qwen-vl-utils) exactly as `sd35_base_dts_imagereward.yaml`.
2. Builds the `reward` env via `setup_reward_env.sh` (HPSv3 + ImageReward + PickScore) and preloads their weights.
3. Builds a `jev` env (python 3.11, torch 2.11 cu128 -> cu126 fallback, transformers 5.17 for Gemma-4) and, if
   `run_vqascore=1`, a `vqascore` env (python 3.10, torch 2.5.1, transformers 4.49, t2v_metrics 3.0).
4. Prefetches while online: SD3.5-Large, Qwen2.5-VL-7B-Instruct, Jev-Omni @ `5addda86` (23 GB; its `jev_omni.py` and
   `head.pt` ship inside the snapshot and are verified present), MizzenAI/HPSv3, PickScore_v1, and clip-flant5-xxl
   (22 GB) if requested. Then `HF_HUB_OFFLINE=1`.
5. Copies `scripts/ config/ deps/` to `RUN_DIR=/mnt/data/v-yigu/jev_decider/{run_tag}` on the blob mount and runs
   `node_remap.sh` there, rewriting all 12 hard-coded A6000 path prefixes (run dir, repo, 4 conda envs, 2 HF cache
   roots, prompt map, rewrite cache, judge scripts) to the node's paths. `deps/` carries the data and judge scripts
   that are not otherwise in the repo (`prompt_map.csv`, the shipped wording bank, `score_local/vlm_judge/vqa.py`,
   `build_followup_manifest/export_scores/same_day_analysis/static_arm.py`).
6. Runs `run_jevact3.sh` then `run_actdiff_jev.sh`. Both self-check first and wait for free GPUs (4 and 3) -- immediate
   on a fresh node. GPU layout: reward server 22 GB, Qwen2.5-VL 17 GB + Jev-Omni 24 GB (share a GPU), samplers 19 GB.
7. Judging appends to `RUN_DIR/scores.csv` (also copied to `scores_final.csv`).

## Outputs (persist on the blob after the job)
`/mnt/data/v-yigu/jev_decider/{run_tag}/main/<arm>/records.jsonl|decisions.jsonl|decider_log.jsonl`, images,
`logs/*.log`, `scores.csv`. Arms: `ja_jevact_v2w`, `ja_randact_v2w`, `ec_v2w`; `ad_base`, `ad_rank3`, `ad_jevbank`,
`ad_randbank`, `ad_jevprobe`, `ad_randprobe`. Read them with `scripts/rsel_table.py --arms ...` (after copying
`scores.csv` next to it) or the per-arm analyses in `PLAN.md`.

## Known risks
* **Driver vs cu128.** Jev-Omni was validated on torch 2.11.0+cu128 (driver 575 locally). If the node driver is older
  the cu126 fallback installs; Gemma-4 should load either way, but it is the one untested combination.
* **Prefetch time.** ~45 GB of new weights (Jev 23 + FlanT5 22) on top of SD3.5; budget 20-40 min before any GPU work.
* **The hooks have never executed on a GPU with a decider attached.** Static checks only. First failure, if any, will
  be in `logs/ad_*.log` or `logs/ja_*.log`; the `none`-decider arm (`ad_base`) is byte-identical to shipped actdiff
  and is the control for "did the plumbing itself break anything".
* **NFE matching.** `ad_jevprobe` spends extra NFE on probes; compare it to `ad_randprobe` (identical probes), and
  only to `ad_base` after an NFE-matched base run (raise `--n_sims` to the probe NFE recorded in `records.jsonl`).
* Two launchers are still armed on the A6000 box waiting for GPUs; kill them (`pkill -f run_jevact3.sh`, by PID, not
  from a shell whose command line contains the pattern) if this job runs, or results land twice.

## Running on TACC Lonestar6 (gpu-a100, 3x A100-40GB per node; no AMLT)
Everything lives under `$SCRATCH/jev_decider` (= `RUN_DIR`), with `$SCRATCH/jev_decider/env.sh` as the single source of paths:
```
$SCRATCH/jev_decider/            RUN_DIR: scripts/ config/ deps/ = remapped copy of this directory (sync_from_repo.sh after every pull)
  env.sh                         HF_HOME, IMAGEREWARD_CACHE, HPS_ROOT, the four env prefixes, caches off $HOME (its quota is full)
  main/ logs/ state/ scores.csv  run outputs (main/ns_ladder is copied from results/, early_control.py reads it)
  setup/setup_envs.sh            builds $SCRATCH/envs/{reward,jev,vqascore} with $SCRATCH/miniforge3 and completes sid_dit
  scripts/run_ls6.sh|.sbatch     the launcher: fixed GPU placement, all arms sequential on one sampler GPU, resumable
$SCRATCH/envs/{sid_dit,reward,jev,vqascore}      $SCRATCH/hf_cache/hub (SD3.5/HPSv2/CLIP-H are symlinks into $WORK/model_cache)
```
GPU placement on 40 GB cards: GPU0 = mr3 (~22 GB) + Qwen2.5-VL (~16 GB), GPU1 = Jev-Omni (~24 GB), GPU2 = the one sampler.
`STAGES=smoke bash $RUN_DIR/scripts/run_ls6.sh` runs one jevact prompt against all three servers and logs the per-GPU peak
(`logs/gpu_mem.log`) -- do this once before the 48 h job. Then `sbatch $RUN_DIR/scripts/run_ls6.sbatch` (stages jevact ec actdiff
judge; ~25-30 h sequential). Partitions `gpu-a100` only (dev/small partitions have the same 3-GPU nodes but short limits).

## Running on a GPU box you control (no AMLT)
Everything the AMLT job does in software is reproducible by hand; the yaml is the authoritative list of versions.
1. `git clone -b jev-decider-sd35base git@github.com:yigu1008/SiD-DiT_tts_dev.git` and `pip install -r requirements.txt`
   into the sampler env (torch 2.7.1, transformers 4.52.4, qwen-vl-utils; see yaml lines 69-75). This env is "sid_dit".
2. Build the `reward` env with `setup_reward_env.sh` (HPSv3 + ImageReward + PickScore; yaml lines 84-89) and the `jev`
   env (python 3.11, torch 2.11 cu128 or cu126, transformers 5.17; yaml lines 94-96). Optional `vqascore` env (line 100).
3. Prefetch weights into one HF cache (yaml lines 103-105): SD3.5-Large, Qwen2.5-VL-7B-Instruct, Jev-Omni@5addda86,
   MizzenAI/HPSv3, PickScore_v1, optionally clip-flant5-xxl. The HF token is needed for SD3.5 only.
4. `cp -r experiments/vlm_decider_sd35base $RUN_DIR` and remap paths inside it:
   `RUN_DIR=... REPO_DIR=... HF_CACHE=... SIDDIT_PY_PREFIX=... REWARD_PY_PREFIX=... JEV_PY_PREFIX=... VQA_PY_PREFIX=... bash $RUN_DIR/node_remap.sh`
5. Launch (each starts its own servers and is resumable):
   * `nohup bash $RUN_DIR/scripts/run_jevact3_flex.sh &` -- needs 3 clean GPUs for the Jev-policy arms (reward 22 GB,
     Qwen2.5-VL 17 GB + Jev-Omni 24 GB on one card, sampler 26 GB) and 2 for the early-control arm; uses more if free.
   * `nohup bash $RUN_DIR/scripts/run_actdiff_jev.sh &` -- waits for jevact to complete (by records) and 3 clean GPUs.
   On 80 GB cards all of this fits on 2 GPUs; the scripts still count whole free cards, so give them 3-4.
6. Results: `$RUN_DIR/main/<arm>/records.jsonl`, judged scores in `$RUN_DIR/scores.csv`; read with `scripts/rsel_table.py`.
   Start the three arms clean on the new box (do not copy the partial A6000 records; they were produced under a kill).
