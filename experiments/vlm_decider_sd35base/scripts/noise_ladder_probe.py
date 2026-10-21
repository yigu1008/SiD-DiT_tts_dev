#!/usr/bin/env python3
"""Can a noise be judged BEFORE finishing it?  The precondition for adaptive noise allocation.

The proposed policy is: start an initial noise, and at checkpoints decide whether it has saturated -- if so abandon it
and draw the next noise, else keep investing -- all under a fixed NFE budget. Whether ANY such policy (Jev's or an
oracle's) can beat uniform best-of-k depends entirely on two things this script measures:

  1. PREDICTABILITY: across initial noises, does the x0 preview at step t rank the noises the same way the FINISHED
     images are ranked?  (The truncation probe found IR at step 4 ANTI-correlates with final reward, -0.735, within a
     single noise across actions; this measures it ACROSS noises, which is the quantity the policy depends on.)
  2. REMAINING CURVE: best-of-k out to a larger k. Screening cheaply then finishing the survivors buys you a LARGER k
     for the same budget -- worthless if the best-of-k curve has already flattened (at 16 it had: +0.008 from k=11).

One full generation per noise (56 NFE), scoring the x0 preview at each checkpoint along the way, so the cost is the
same as the plain noise probe and every policy can then be simulated offline at any budget.
"""
from __future__ import annotations
import argparse, base64, csv, hashlib, io, json, os, sys, time, urllib.request
from pathlib import Path

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
    p.add_argument("--ids", required=True); p.add_argument("--run_dir", required=True)
    p.add_argument("--n_seeds", type=int, default=32); p.add_argument("--seed_base", type=int, default=2000)
    p.add_argument("--checkpoints", default="4,8,12,16,20,24")
    p.add_argument("--wording", type=int, default=0); p.add_argument("--cfg", type=float, default=float(BASELINE_CFG))
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--shard_index", type=int, default=0); p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--save_previews", action="store_true", help="also write the checkpoint preview PNGs (for the Jev arm's sheets)")
    p.add_argument("--reward_server", required=True)
    a = p.parse_args()
    os.environ.update({"HF_HUB_CACHE": "/data/ygu/.cache/hub", "HF_HOME": "/data/ygu/.cache/hub", "HF_HUB_OFFLINE": "1",
                       "OFFLOAD_TEXT_ENCODER_AFTER_ENCODE": "1", "REWARD_SERVER_URL": a.reward_server, "HPSV3_SCORE_STYLE": "raw"})
    sys.path.insert(0, SID); os.chdir(SID)
    import torch
    import sampling_unified_sd35 as su
    rh = json.loads(urllib.request.urlopen(a.reward_server + "/health", timeout=10).read())
    assert not [b for b in SEARCH_REWARDS if b not in rh.get("backends", [])], rh
    sargs = su.normalize_paths(su.parse_args([
        "--backend", "sd35_base", "--search_method", "mcts", "--reward_backend", "imagereward", "--rewrites_file", CACHE, "--no_qwen",
        "--n_variants", "3", "--steps", str(a.steps), "--seed", "42", "--width", "1024", "--height", "1024",
        "--out_dir", a.run_dir, "--cfg_scales", *CFG_BANK, "--baseline_cfg", BASELINE_CFG,
        "--keystep_strategy", "explicit", "--mcts_key_steps", "0,4,8,12,16,20,24,27"]))
    assert getattr(sargs, "euler_sampler", False) and sargs.model_id == "stabilityai/stable-diffusion-3.5-large"
    run_dir = Path(a.run_dir); (run_dir / "images").mkdir(parents=True, exist_ok=True)
    if a.save_previews: (run_dir / "previews").mkdir(exist_ok=True)
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    ids = [x for i, x in enumerate(ids) if i % a.num_shards == a.shard_index]
    pmap = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(PMAP))}; cache = json.load(open(sargs.rewrites_file))
    seeds = [42] + [a.seed_base + i for i in range(a.n_seeds - 1)]
    cps = sorted(int(x) for x in a.checkpoints.split(",") if x.strip().isdigit())
    recp = run_dir / f"records_shard{a.shard_index}.jsonl"
    done = {(json.loads(l)["prompt_id"], json.loads(l)["noise_seed"]) for l in open(recp)} if recp.exists() else set()
    print(f"[ladder] shard {a.shard_index}/{a.num_shards}: {len(ids)} prompts x {len(seeds)} seeds, checkpoints {cps} "
          f"({len(done)} done); anchor wording {a.wording} @ CFG {a.cfg:g}", flush=True)
    ctx = su.load_pipeline(sargs); rm = su.load_reward_model(sargs, ctx.device)
    (run_dir / f"config_shard{a.shard_index}.json").write_text(json.dumps({
        "arm": "noise_ladder_probe", "steps": a.steps, "seeds": seeds, "checkpoints": cps, "anchor_wording": a.wording,
        "anchor_cfg": a.cfg, "nfe_per_step": 2, "nfe_full_generation": 2 * a.steps, "search_rewards": SEARCH_REWARDS,
        "design": "one full generation per initial noise at the anchor action, with the x0 preview scored at each "
                  "checkpoint. Enables offline simulation of any screen-and-finish / early-abandon policy at any NFE "
                  "budget, plus the across-noise predictability of final quality from partial state."}, indent=1))

    def score_pil(prompt, img):
        buf = io.BytesIO(); img.save(buf, format="PNG"); enc = base64.b64encode(buf.getvalue()).decode()
        return {b: float(post(a.reward_server + "/score", {"prompt": prompt, "image_b64": enc, "backend": b})["score"])
                for b in SEARCH_REWARDS}

    action = (a.wording, float(a.cfg), 0.0)
    with torch.no_grad(), recp.open("a") as rec:
        for pid in ids:
            prompt = pmap[pid]
            variants = su.generate_variants(sargs, prompt, cache); emb = su.encode_variants(ctx, variants)
            sched = su.step_schedule(ctx.device, emb.cond_text[0].dtype, a.steps, getattr(sargs, "sigmas", None), euler=True)
            fresh = su._resolve_mcts_fresh_noise_steps(sargs, a.steps, key_steps=[0])
            for sd in seeds:
                if (pid, sd) in done: continue
                t0 = time.time(); ctx.nfe = 0
                lat = su.make_latents(ctx, sd, sargs.height, sargs.width, emb.cond_text[0].dtype); dx = torch.zeros_like(lat)
                bounds = [0] + cps + [a.steps]; partial = {}
                for i in range(len(bounds) - 1):
                    s0, s1 = bounds[i], bounds[i + 1]
                    lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, action, sched, s0, s1,
                                              noise_explore_steps=fresh, eps_bank=None)
                    if s1 in cps:
                        prev = su.decode_to_pil(ctx, dx)              # x0 estimate at this step
                        partial[str(s1)] = {"nfe": int(ctx.nfe), **score_pil(prompt, prev)}
                        if a.save_previews: prev.save(run_dir / "previews" / f"{pid}_s{sd}_t{s1}.png")
                img = su.decode_to_pil(ctx, su._final_decode_tensor(lat, dx, True))
                fin = score_pil(prompt, img)
                out = run_dir / "images" / f"{pid}_s{sd}.png"; img.save(out)
                rec.write(json.dumps({"prompt_id": pid, "prompt": prompt, "arm": "noise_ladder_probe", "noise_seed": sd,
                                      "seed": 42, "wording": a.wording, "cfg": float(a.cfg), "checkpoints": cps,
                                      "partial": partial, "final": fin, "vec": fin, "image": str(out),
                                      "image_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                                      "nfe_transformer": int(ctx.nfe), "seconds": round(time.time() - t0, 1)}) + "\n"); rec.flush()
                print(f"  {pid} seed {sd:5d} final ir={fin['imagereward']:+.3f} hps={fin['hpsv3']:.3f} | "
                      + " ".join(f"t{c}:hps={partial[str(c)]['hpsv3']:.2f}" for c in cps)
                      + f"  nfe={ctx.nfe} {time.time()-t0:.0f}s", flush=True)
    print(f"[ladder] shard {a.shard_index} done", flush=True)


if __name__ == "__main__":
    main()
