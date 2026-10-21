#!/usr/bin/env python3
"""Base SD3.5-Large (28-step Euler ODE) arms with immediate, resumable per-prompt records.

Arms:
  baseline : variant 0 (original prompt), baseline CFG, no search          (28 transformer steps)
  mcts     : stock run_mcts over the fixed bank (variants x CFG), explicit key steps,
             UCB1, uniform-random rollouts, best-rollout return, ImageReward objective.
Settings follow the repo's base launcher (run_sd35_base_bon_mcts_local.sh): cfg bank
3.5 4.0 4.5 5.0 5.5 6.0 7.0, baseline_cfg 4.5, steps 28, 1024^2, bf16; and the follow-up
ablation's search surface: n_variants 3 (shared Qwen rewrite cache), 28-action bank,
ucb_c 1.41, seed 42.  Noise-injection actions are off automatically for steps > 4.
Records: search score, actions, NFE, seconds, torch peak memory (GiB).
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path

SID = "/home/ygu/project/SiD-DiT_tts_dev"
CACHE = "/data/ygu/human_eval_genai40_v1/sid_vqascore_algorithm_sweep/genai200_v1/shared_rewrites_cache.json"
PMAP = "/data/ygu/reward_orthogonality/mcts/prompts/prompt_map.csv"
CFG_BANK = ["3.5", "4.0", "4.5", "5.0", "5.5", "6.0", "7.0"]; BASELINE_CFG = "4.5"

def _install_retry(su, log_prefix):
    import time as _t
    _orig = su.score_image
    def _retry(rm_, prompt_, img_, _tries=90, _sleep=10):
        for i in range(_tries):
            try: return _orig(rm_, prompt_, img_)
            except Exception as e:
                print(f"{log_prefix} reward call failed ({type(e).__name__}: {str(e)[:80]}); retry {i+1}/{_tries} in {_sleep}s", flush=True); _t.sleep(_sleep)
        raise RuntimeError("reward server unreachable for %d s" % (_tries * _sleep))
    su.score_image = _retry

def _install_stepcounter(su):
    """Independent NFE check: count denoising steps run through su._run_segment (and transformer_step calls)."""
    st = {"steps": 0, "calls": 0}
    _seg = su._run_segment
    def _seg_counted(args, ctx, emb, rm, prompt, lat, dx, action, sched, s0, s1, **kw):
        st["steps"] += int(s1) - int(s0); return _seg(args, ctx, emb, rm, prompt, lat, dx, action, sched, s0, s1, **kw)
    su._run_segment = _seg_counted
    return st

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["baseline", "mcts", "mcts_refine"]); p.add_argument("--ids", required=True); p.add_argument("--run_dir", required=True)
    # --arm selects BEHAVIOUR; the name written to records/scores.csv defaults to the run_dir basename so it
    # cannot collide with another arm in the shared scores.csv (the generic "mcts" collided with 7 rows from
    # an earlier probe, which --skip_existing would then have silently kept instead of the new ones).
    p.add_argument("--arm_name", default=None, help="name recorded in records.jsonl/scores.csv (default: run_dir basename)")
    p.add_argument("--limit", type=int, default=0); p.add_argument("--seed", type=int, default=42); p.add_argument("--n_sims", type=int, default=40)
    p.add_argument("--key_steps", default="0,4,8,12,16,20,24,27"); p.add_argument("--steps", type=int, default=28)
    p.add_argument("--reward_backend", default="imagereward"); p.add_argument("--reward_server", required=True); p.add_argument("--trace", action="store_true")
    a = p.parse_args()
    a.arm_name = a.arm_name or Path(a.run_dir).name
    os.environ.update({"HF_HUB_CACHE": "/data/ygu/.cache/hub", "HF_HOME": "/data/ygu/.cache/hub", "HF_HUB_OFFLINE": "1",
                       "OFFLOAD_TEXT_ENCODER_AFTER_ENCODE": "1", "REWARD_SERVER_URL": a.reward_server, "HPSV3_SCORE_STYLE": "raw"})
    sys.path.insert(0, SID); os.chdir(SID)
    import torch
    import sampling_unified_sd35 as su
    _install_retry(su, f"[{a.arm}]"); STEPS = _install_stepcounter(su)
    sargs = su.normalize_paths(su.parse_args([
        "--backend", "sd35_base", "--search_method", "mcts", "--reward_backend", a.reward_backend, "--rewrites_file", CACHE, "--no_qwen",
        "--n_variants", "3", "--n_sims", str(a.n_sims), "--ucb_c", "1.41", "--steps", str(a.steps), "--seed", str(a.seed),
        "--width", "1024", "--height", "1024", "--out_dir", a.run_dir, "--cfg_scales", *CFG_BANK, "--baseline_cfg", BASELINE_CFG,
        "--keystep_strategy", "explicit", "--mcts_key_steps", a.key_steps]))
    sargs.mcts_sparse_noise_refine = (a.arm == "mcts_refine")  # stock default True for steps>4: re-runs the selected path with noise-injection policies (~+1000 NFE); "mcts" = basic search only
    assert getattr(sargs, "euler_sampler", False) and sargs.model_id == "stabilityai/stable-diffusion-3.5-large", (sargs.euler_sampler, sargs.model_id)
    run_dir = Path(a.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    if a.limit: ids = ids[: a.limit]
    pmap = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(PMAP))}
    cache = json.load(open(sargs.rewrites_file))
    recp = run_dir / "records.jsonl"
    done = {json.loads(l)["prompt_id"] for l in open(recp)} if recp.exists() else set()
    todo = [i for i in ids if i not in done]
    print(f"[{a.arm}] {len(todo)} prompts to do ({len(done)} done)", flush=True)
    if not todo: return
    t_load = time.time(); ctx = su.load_pipeline(sargs); reward_model = su.load_reward_model(sargs, ctx.device)
    print(f"[{a.arm}] models loaded in {time.time()-t_load:.0f}s; resident {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
    (run_dir / "config.json").write_text(json.dumps({
        "arm": a.arm, "backend": "sd35_base", "model_id": sargs.model_id, "dtype": str(sargs.dtype), "euler_sampler": True,
        "steps": sargs.steps, "key_steps": a.key_steps if a.arm != "baseline" else None, "reward_backend": a.reward_backend, "reward": reward_model.describe(),
        "seed": a.seed, "n_sims": a.n_sims if a.arm != "baseline" else None, "ucb_c": sargs.ucb_c, "cfg_scales": sargs.cfg_scales, "baseline_cfg": sargs.baseline_cfg,
        "n_variants": sargs.n_variants, "width": sargs.width, "height": sargs.height,
        "rewrites_sha256": hashlib.sha256(open(sargs.rewrites_file, "rb").read()).hexdigest(),
        "selection": "UCB1 untried-first random; rollout uniform-random between key steps; return best rollout vs exploit replay",
        "noise_actions": "off (steps > 4)", "sparse_noise_refine": bool(sargs.mcts_sparse_noise_refine)}, indent=1))
    traced = False
    with torch.no_grad(), recp.open("a") as rec:
        for pid in todo:
            prompt = pmap[pid]; t0 = time.time(); ctx.nfe = 0; ctx.correction_nfe = 0; STEPS['steps'] = 0; torch.cuda.reset_peak_memory_stats()
            import numpy as _np; _np.random.seed(a.seed + 7919 * (ids.index(pid) + 1))  # stock run_mcts draws actions from the global numpy RNG; seed it per prompt for reproducibility (same scheme as vlm_mcts.py)
            tf = None
            if a.trace and not traced and a.arm != "baseline":
                tf = open(run_dir / f"{pid}_trail.jsonl", "w"); su.MCTS_TRAIL_LOGGER = lambda ev, _f=tf: (_f.write(json.dumps(ev) + "\n"), _f.flush())
            variants = su.generate_variants(sargs, prompt, cache)
            emb = su.encode_variants(ctx, variants); t_enc = time.time() - t0
            if a.arm == "baseline":
                img, score = su.run_baseline(sargs, ctx, emb, reward_model, prompt, a.seed, cfg_scale=float(sargs.baseline_cfg))
                actions = [[0, float(sargs.baseline_cfg), 0.0]]; diag = None
            else:
                res = su.run_mcts(sargs, ctx, emb, reward_model, prompt, variants, a.seed)
                img, score = res.image, float(res.score); actions = [[int(v), float(c), float(s)] for v, c, s in res.actions]; diag = res.diagnostics
            if tf: tf.close(); su.MCTS_TRAIL_LOGGER = None; traced = True
            out = run_dir / f"{pid}_{a.arm_name}.png"; img.save(out)
            rec.write(json.dumps({"prompt_id": pid, "prompt": prompt, "variants": variants, "arm": a.arm_name, "behaviour": a.arm, "seed": a.seed, "search_score": score,
                                  "actions": actions, "diagnostics": diag, "image": str(out), "image_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                                  "nfe_transformer": int(ctx.nfe), "segment_steps": int(STEPS["steps"]), "seconds": round(time.time() - t0, 1), "seconds_encode": round(t_enc, 1),
                                  "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}) + "\n"); rec.flush()
            print(f"[{a.arm}] {pid} score={score:.4f} nfe={ctx.nfe} steps={STEPS['steps']} {time.time()-t0:.0f}s peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)

if __name__ == "__main__":
    main()
