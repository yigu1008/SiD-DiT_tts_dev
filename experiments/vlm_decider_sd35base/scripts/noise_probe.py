#!/usr/bin/env python3
"""The axis this study never varied: the INITIAL LATENT.

Every arm run so far -- baseline, mcts, mcts_refine, all the vlm/jev/retrieval/reward-selection arms -- built its
start latent with `make_latents(ctx, seed=42, ...)`. One noise sample, identical across every arm and every prompt;
`config.json` records `noise_actions: off (steps > 4)`. So the entire search space explored was (wording x CFG) at
fixed z0.

At 28 Euler steps one complete generation costs 56 NFE, so the 616-NFE operating point every arm was matched to buys
exactly **11 full generations**. "Best of 11 initial noises at the anchor config" is therefore an exactly
budget-matched, search-free competitor to the whole apparatus -- never run until now.

This script generates N initial noises per prompt at the ANCHOR action (wording A, CFG = baseline_cfg), scores each
under all three search rewards, and saves every image. Everything downstream is post-hoc on that data:
  * headroom of the noise axis = spread across seeds at fixed (wording, CFG);
  * the budget-matched arm = best of the first 11 seeds under a chosen reward (616 NFE, by construction);
  * best-of-k curves for any k <= N.
Seed 42 is generated FIRST so its image is directly comparable to every existing arm.
Selecting a seed by reward is exactly the hacking-prone operation, so the chosen images must be judged held-out.
"""
from __future__ import annotations
import argparse, base64, csv, hashlib, io, json, os, sys, time, urllib.request
from pathlib import Path
import numpy as np

SID = "/home/ygu/project/SiD-DiT_tts_dev"
CACHE = "/data/ygu/human_eval_genai40_v1/sid_vqascore_algorithm_sweep/genai200_v1/shared_rewrites_cache.json"
PMAP = "/data/ygu/reward_orthogonality/mcts/prompts/prompt_map.csv"
CFG_BANK = ["3.5", "4.0", "4.5", "5.0", "5.5", "6.0", "7.0"]; BASELINE_CFG = "4.5"
SEARCH_REWARDS = ["imagereward", "hpsv3", "pickscore"]


def post(url, body, timeout=600, tries=60, sleep=10):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as e:
            print(f"[post] {url} failed ({type(e).__name__}: {str(e)[:70]}); retry {i+1}/{tries}", flush=True); time.sleep(sleep)
    raise RuntimeError(f"{url} unreachable")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default=None); p.add_argument("--ids", required=True); p.add_argument("--run_dir", required=True)
    p.add_argument("--n_seeds", type=int, default=16, help="initial latents per prompt; seed 42 is always first")
    p.add_argument("--seed_base", type=int, default=1000, help="seeds after 42 are seed_base, seed_base+1, ...")
    p.add_argument("--wording", type=int, default=0); p.add_argument("--cfg", type=float, default=float(BASELINE_CFG))
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--shard_index", type=int, default=0); p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--reward_server", required=True)
    a = p.parse_args()
    a.arm = a.arm or Path(a.run_dir).name
    os.environ.update({"HF_HUB_CACHE": "/data/ygu/.cache/hub", "HF_HOME": "/data/ygu/.cache/hub", "HF_HUB_OFFLINE": "1",
                       "OFFLOAD_TEXT_ENCODER_AFTER_ENCODE": "1", "REWARD_SERVER_URL": a.reward_server, "HPSV3_SCORE_STYLE": "raw"})
    sys.path.insert(0, SID); os.chdir(SID)
    import torch
    import sampling_unified_sd35 as su
    rh = json.loads(urllib.request.urlopen(a.reward_server + "/health", timeout=10).read())
    missing = [b for b in SEARCH_REWARDS if b not in rh.get("backends", [])]
    assert not missing, f"reward server lacks {missing}; has {rh.get('backends')}"
    sargs = su.normalize_paths(su.parse_args([
        "--backend", "sd35_base", "--search_method", "mcts", "--reward_backend", "imagereward", "--rewrites_file", CACHE, "--no_qwen",
        "--n_variants", "3", "--steps", str(a.steps), "--seed", "42", "--width", "1024", "--height", "1024",
        "--out_dir", a.run_dir, "--cfg_scales", *CFG_BANK, "--baseline_cfg", BASELINE_CFG,
        "--keystep_strategy", "explicit", "--mcts_key_steps", "0,4,8,12,16,20,24,27"]))
    assert getattr(sargs, "euler_sampler", False) and sargs.model_id == "stabilityai/stable-diffusion-3.5-large"
    run_dir = Path(a.run_dir); (run_dir / "images").mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    ids = [x for i, x in enumerate(ids) if i % a.num_shards == a.shard_index]
    pmap = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(PMAP))}; cache = json.load(open(sargs.rewrites_file))
    seeds = [42] + [a.seed_base + i for i in range(a.n_seeds - 1)]
    recp = run_dir / f"records_shard{a.shard_index}.jsonl"
    done = {(json.loads(l)["prompt_id"], json.loads(l)["noise_seed"]) for l in open(recp)} if recp.exists() else set()
    print(f"[{a.arm}] shard {a.shard_index}/{a.num_shards}: {len(ids)} prompts x {len(seeds)} seeds "
          f"({len(done)} already done); anchor = wording {a.wording} @ CFG {a.cfg:g}", flush=True)
    ctx = su.load_pipeline(sargs); rm = su.load_reward_model(sargs, ctx.device)
    key_steps = [0]; n_key = 1
    (run_dir / f"config_shard{a.shard_index}.json").write_text(json.dumps({
        "arm": a.arm, "backend": "sd35_base", "steps": a.steps, "seeds": seeds, "anchor_wording": a.wording,
        "anchor_cfg": a.cfg, "nfe_per_generation": 2 * a.steps, "budget_matched_k": 616 // (2 * a.steps),
        "search_rewards": SEARCH_REWARDS, "reward_server": rh,
        "design": "ONE action (anchor wording, anchor CFG), N different initial latents, no search. Isolates the "
                  "initial-noise axis that every other arm in this study held fixed at seed 42. 616 NFE = 11 "
                  "generations, so best-of-11 here is exactly budget-matched to the 10-sim MCTS arms."}, indent=1))

    def score_vec(prompt, img):
        buf = io.BytesIO(); img.save(buf, format="PNG"); enc = base64.b64encode(buf.getvalue()).decode()
        return {b: float(post(a.reward_server + "/score", {"prompt": prompt, "image_b64": enc, "backend": b})["score"])
                for b in SEARCH_REWARDS}

    action = (a.wording, float(a.cfg), 0.0)
    with torch.no_grad(), recp.open("a") as rec:
        for pid in ids:
            prompt = pmap[pid]
            variants = su.generate_variants(sargs, prompt, cache); emb = su.encode_variants(ctx, variants)
            sched = su.step_schedule(ctx.device, emb.cond_text[0].dtype, a.steps, getattr(sargs, "sigmas", None), euler=True)
            fresh = su._resolve_mcts_fresh_noise_steps(sargs, a.steps, key_steps=key_steps)
            assert a.wording < len(emb.cond_text), (a.wording, len(emb.cond_text))
            for sd in seeds:
                if (pid, sd) in done: continue
                t0 = time.time(); ctx.nfe = 0
                lat0 = su.make_latents(ctx, sd, sargs.height, sargs.width, emb.cond_text[0].dtype); dx0 = torch.zeros_like(lat0)
                lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat0, dx0, action, sched, 0, a.steps,
                                          noise_explore_steps=fresh, eps_bank=None)
                img = su.decode_to_pil(ctx, su._final_decode_tensor(lat, dx, True))
                vec = score_vec(prompt, img)
                out = run_dir / "images" / f"{pid}_s{sd}.png"; img.save(out)
                rec.write(json.dumps({"prompt_id": pid, "prompt": prompt, "arm": a.arm, "noise_seed": sd, "seed": 42,
                                      "wording": a.wording, "cfg": float(a.cfg), "vec": vec, "image": str(out),
                                      "image_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                                      "nfe_transformer": int(ctx.nfe), "seconds": round(time.time() - t0, 1)}) + "\n"); rec.flush()
                print(f"  {pid} seed {sd:5d}  ir={vec['imagereward']:+.4f} hps={vec['hpsv3']:.4f} ps={vec['pickscore']:.4f}"
                      f"  nfe={ctx.nfe} {time.time()-t0:.0f}s", flush=True)
    print(f"[{a.arm}] shard {a.shard_index} done", flush=True)


if __name__ == "__main__":
    main()
