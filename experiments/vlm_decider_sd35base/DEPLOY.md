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
6. Runs `run_jevact3.sh` then `run_actdiff_jev.sh`. Both self-check first and wait for >=4 free GPUs -- immediate
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
