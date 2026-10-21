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
