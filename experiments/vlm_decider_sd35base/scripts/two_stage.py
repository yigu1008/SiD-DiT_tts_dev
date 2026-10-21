#!/usr/bin/env python3
"""Two-stage recipe: draw k noises, then LATE-refine the winner.

From the early/late control probe (2026-10-19 21:45), stratified by how good the starting noise is:
  * on a BAD noise, early-window control is worth +2.979 hpsv3 and late only +0.647 -- but early-control best-of-6
    is still **-1.384 [-2.215, -0.511]** WORSE than spending the same NFE on 6 fresh noises. Repairing a bad noise
    loses to discarding it.
  * on a GOOD noise, early control HURTS (-0.550) while **late control gains +0.551 [+0.361, +0.766] at only
    32 NFE per variant** -- cheaper than a 56-NFE noise draw and additive to a candidate that is already good.
So: use the noise axis to FIND a good layout, then spend a little late-window action budget REFINING it, and never
touch the early window of a layout that is already working.

Stage 1  draw k noises at the anchor action, full 28 steps (56 NFE each), caching each one's step-`split` state.
Stage 2  pick the winner by rank-sum of hpsv3+pickscore+imagereward (ranks are scale-free -- no normalisation),
         then run m distinct (wording, CFG) actions on [split, 28) from that winner's cached state (32 NFE each).
Return the rank-sum best over everything generated.  Total = k*56 + m*32.
"""
from __future__ import annotations
import argparse, base64, csv, hashlib, io, json, os, sys, time, urllib.request
from pathlib import Path
import numpy as np

SID = "/home/ygu/project/SiD-DiT_tts_dev"
CACHE = "/data/ygu/human_eval_genai40_v1/sid_vqascore_algorithm_sweep/genai200_v1/shared_rewrites_cache.json"
PMAP = "/data/ygu/reward_orthogonality/mcts/prompts/prompt_map.csv"
CFG_BANK = ["3.5", "4.0", "4.5", "5.0", "5.5", "6.0", "7.0"]; BASELINE_CFG = "4.5"
SR = ["imagereward", "hpsv3", "pickscore"]


def post(url, body, timeout=600, tries=60, sleep=10):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as e:
            print(f"[post] {url} failed ({type(e).__name__}: {str(e)[:70]}); retry {i+1}/{tries}", flush=True); time.sleep(sleep)
    raise RuntimeError(f"{url} unreachable")


def rank_best(cands):
    """rank-sum of all three rewards; ranks are scale-free within the candidate set."""
    tot = {id(c): 0 for c in cands}
    for m in SR:
        for pos, c in enumerate(sorted(cands, key=lambda x: x["vec"][m])): tot[id(c)] += pos
    return max(cands, key=lambda c: tot[id(c)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", required=True); p.add_argument("--run_dir", required=True)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--steps", type=int, default=28)
    p.add_argument("--k", type=int, default=8, help="stage-1 noise draws (56 NFE each)")
    p.add_argument("--m", type=int, default=4, help="stage-2 late-window action variants (32 NFE each)")
    p.add_argument("--split", type=int, default=12)
    p.add_argument("--reward_server", required=True)
    a = p.parse_args(); arm = Path(a.run_dir).name
    os.environ.update({"HF_HUB_CACHE": "/data/ygu/.cache/hub", "HF_HOME": "/data/ygu/.cache/hub", "HF_HUB_OFFLINE": "1",
                       "OFFLOAD_TEXT_ENCODER_AFTER_ENCODE": "1", "REWARD_SERVER_URL": a.reward_server, "HPSV3_SCORE_STYLE": "raw"})
    sys.path.insert(0, SID); os.chdir(SID)
    import torch
    import sampling_unified_sd35 as su
    rh = json.loads(urllib.request.urlopen(a.reward_server + "/health", timeout=10).read())
    assert not [b for b in SR if b not in rh.get("backends", [])], rh
    sargs = su.normalize_paths(su.parse_args([
        "--backend", "sd35_base", "--search_method", "mcts", "--reward_backend", "imagereward", "--rewrites_file", CACHE, "--no_qwen",
        "--n_variants", "3", "--steps", str(a.steps), "--seed", str(a.seed), "--width", "1024", "--height", "1024",
        "--out_dir", a.run_dir, "--cfg_scales", *CFG_BANK, "--baseline_cfg", BASELINE_CFG,
        "--keystep_strategy", "explicit", "--mcts_key_steps", "0,4,8,12,16,20,24,27"]))
    run_dir = Path(a.run_dir); (run_dir / "img").mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    pmap = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(PMAP))}; cache = json.load(open(sargs.rewrites_file))
    recp = run_dir / "records.jsonl"
    done = {json.loads(l)["prompt_id"] for l in open(recp)} if recp.exists() else set()
    todo = [i for i in ids if i not in done]
    total = a.k * 2 * a.steps + a.m * 2 * (a.steps - a.split)
    print(f"[{arm}] k={a.k} noises x{2*a.steps} + m={a.m} late variants x{2*(a.steps-a.split)} = {total} NFE; "
          f"{len(todo)} prompts", flush=True)
    if not todo: return
    ctx = su.load_pipeline(sargs); rm = su.load_reward_model(sargs, ctx.device)
    anchor = (0, float(sargs.baseline_cfg), 0.0)
    (run_dir / "config.json").write_text(json.dumps({
        "arm": arm, "k": a.k, "m": a.m, "split": a.split, "nfe_total": total, "select": "rank3",
        "reward_server": rh, "design": "stage 1 = k full noise draws at the anchor (cache step-split state); "
        "stage 2 = m late-window (wording, CFG) variants from the rank3 winner's cached state. Never touches the "
        "early window of a good layout; early control was measured to HURT a good noise (-0.550) while late "
        "control helps (+0.551 at 32 NFE)."}, indent=1))

    def score_vec(prompt, img):
        buf = io.BytesIO(); img.save(buf, format="PNG"); enc = base64.b64encode(buf.getvalue()).decode()
        return {b: float(post(a.reward_server + "/score", {"prompt": prompt, "image_b64": enc, "backend": b})["score"]) for b in SR}

    with torch.no_grad(), recp.open("a") as rec:
        for pid in todo:
            prompt = pmap[pid]; t0 = time.time(); ctx.nfe = 0; torch.cuda.reset_peak_memory_stats()
            rng = np.random.RandomState(a.seed + 7919 * (ids.index(pid) + 1))
            variants = su.generate_variants(sargs, prompt, cache); emb = su.encode_variants(ctx, variants)
            sched = su.step_schedule(ctx.device, emb.cond_text[0].dtype, a.steps, getattr(sargs, "sigmas", None), euler=True)
            fresh = su._resolve_mcts_fresh_noise_steps(sargs, a.steps, key_steps=[0])
            bank = [(vi, float(c), 0.0) for vi in range(len(emb.cond_text)) for c in sargs.cfg_scales if (vi, float(c), 0.0) != anchor]
            rng.shuffle(bank)
            seeds = ([42] + [5000 + 100 * ids.index(pid) + i for i in range(60)])[: a.k]
            cands = []
            for sd in seeds:                                                   # STAGE 1
                lat = su.make_latents(ctx, sd, sargs.height, sargs.width, emb.cond_text[0].dtype); dx = torch.zeros_like(lat)
                lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, anchor, sched, 0, a.split, noise_explore_steps=fresh, eps_bank=None)
                st = (lat.clone(), dx.clone())
                lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, anchor, sched, a.split, a.steps, noise_explore_steps=fresh, eps_bank=None)
                img = su.decode_to_pil(ctx, su._final_decode_tensor(lat, dx, True)); vec = score_vec(prompt, img)
                ip = run_dir / "img" / f"{pid}_s{sd}_anchor.png"; img.save(ip)
                cands.append({"stage": 1, "seed": sd, "action": [int(anchor[0]), float(anchor[1]), 0.0],
                              "vec": vec, "image": str(ip), "state": st})
                print(f"    s1 seed {sd:5d} hps={vec['hpsv3']:.3f} ir={vec['imagereward']:+.3f} ps={vec['pickscore']:.3f}", flush=True)
            win = rank_best(cands)
            print(f"    -> rank3 winner: seed {win['seed']} (hps={win['vec']['hpsv3']:.3f}); late-refining it", flush=True)
            for vi in range(a.m):                                              # STAGE 2
                act = bank[vi]
                lat, dx = (t.clone() for t in win["state"])
                lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, act, sched, a.split, a.steps, noise_explore_steps=fresh, eps_bank=None)
                img = su.decode_to_pil(ctx, su._final_decode_tensor(lat, dx, True)); vec = score_vec(prompt, img)
                ip = run_dir / "img" / f"{pid}_s{win['seed']}_late{vi}.png"; img.save(ip)
                cands.append({"stage": 2, "seed": win["seed"], "action": [int(act[0]), float(act[1]), 0.0],
                              "vec": vec, "image": str(ip), "state": None})
                print(f"    s2 late v{vi} act=(w{int(act[0])},cfg{act[1]:g}) hps={vec['hpsv3']:.3f} "
                      f"ir={vec['imagereward']:+.3f} ps={vec['pickscore']:.3f}", flush=True)
            best = rank_best(cands)
            out = run_dir / f"{pid}_{arm}.png"
            from PIL import Image as _I; _I.open(best["image"]).convert("RGB").save(out)
            rec.write(json.dumps({"prompt_id": pid, "prompt": prompt, "variants": variants, "arm": arm, "seed": a.seed,
                                  "k": a.k, "m": a.m, "split": a.split, "selected_on": "rank3",
                                  "search_score": best["vec"]["hpsv3"], "search_vec": best["vec"],
                                  "selected_stage": best["stage"], "selected_seed": best["seed"],
                                  "actions": [best["action"]], "stage1_winner_seed": win["seed"],
                                  "stage1_winner_vec": win["vec"],
                                  "late_gain_hpsv3": best["vec"]["hpsv3"] - win["vec"]["hpsv3"],
                                  "all_cands": [{"stage": c["stage"], "seed": c["seed"], "action": c["action"], "vec": c["vec"]} for c in cands],
                                  "image": str(out), "image_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                                  "nfe_transformer": int(ctx.nfe), "seconds": round(time.time() - t0, 1)}) + "\n"); rec.flush()
            print(f"[{arm}] {pid} best from stage {best['stage']} hps={best['vec']['hpsv3']:.4f} "
                  f"(stage1 winner {win['vec']['hpsv3']:.4f}, late gain {best['vec']['hpsv3']-win['vec']['hpsv3']:+.4f}) "
                  f"nfe={ctx.nfe} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
