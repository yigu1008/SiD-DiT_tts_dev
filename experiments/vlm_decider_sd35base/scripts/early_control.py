#!/usr/bin/env python3
"""Where does the (wording, CFG) action actually act, and can it REPAIR a bad initial noise?

Measured context. The initial latent spans ~5.04 hpsv3 of per-prompt range; action search is worth +1.549 on an
unselected noise but only **+0.219** once cheap screening has already picked a good one (corr(anchor quality,
action gain) = -0.488). That pattern says action search is mostly *repairing bad layouts* -- the same thing a
fresh noise draw buys. This arm tests that directly, stratified by how good the starting noise is.

Budget fact that makes the comparison fair and sharp: at 28 Euler steps, 2 NFE/step,
  * EARLY control  = action A on steps [0,12), anchor on [12,28)  -> 24 + 32 = **56 NFE**, the exact price of a
    fresh noise draw. So early control must beat redrawing the noise to be worth anything.
  * LATE control   = anchor on [0,12), action A on [12,28)        -> the [0,12) prefix is computed ONCE and reused,
    so each variant costs only **32 NFE**.
  * FULL control   = action A on all 28 steps                     -> 56 NFE (what every previous arm did).
Per prompt the script takes the BEST and WORST initial noise (by finished hpsv3 from the 32-noise ladder probe) and
runs all three control modes on each, so "does early control repair a bad noise?" and "does it add anything to a
good one?" are answered separately.
"""
from __future__ import annotations
import argparse, base64, csv, io, json, os, sys, time, urllib.request
from pathlib import Path
import numpy as np

SID = "/home/ygu/project/SiD-DiT_tts_dev"
CACHE = "/data/ygu/human_eval_genai40_v1/sid_vqascore_algorithm_sweep/genai200_v1/shared_rewrites_cache.json"
PMAP = "/data/ygu/reward_orthogonality/mcts/prompts/prompt_map.csv"
LADDER = "/data/ygu/runs/vlm_decider_sd35base_2026-10-16/main/ns_ladder/records_all.jsonl"
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


def _bank_n_variants(path, floor=3):
    """--n_variants is a hard cap in generate_variants (keeps n_variants+1 wordings). Size it from the bank so a
    wider bank is actually delivered; the shipped 4-wording cache still yields 4."""
    import json as _j
    try: d = _j.load(open(path))
    except Exception: return floor
    w = max((len(v) for v in d.values() if isinstance(v, list)), default=floor + 1)
    return max(floor, int(w) - 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", required=True); p.add_argument("--run_dir", required=True)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--steps", type=int, default=28)
    p.add_argument("--split", type=int, default=12, help="boundary between the early and late windows")
    p.add_argument("--n_variants", type=int, default=6, help="distinct (wording, CFG) actions tried per mode")
    p.add_argument("--shard_index", type=int, default=0); p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--rewrites_file", default=CACHE, help="wording bank; default is the shipped cache")
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
        "--backend", "sd35_base", "--search_method", "mcts", "--reward_backend", "imagereward", "--rewrites_file", a.rewrites_file, "--no_qwen",
        "--n_variants", str(_bank_n_variants(a.rewrites_file)), "--steps", str(a.steps), "--seed", str(a.seed), "--width", "1024", "--height", "1024",
        "--out_dir", a.run_dir, "--cfg_scales", *CFG_BANK, "--baseline_cfg", BASELINE_CFG,
        "--keystep_strategy", "explicit", "--mcts_key_steps", "0,4,8,12,16,20,24,27"]))
    run_dir = Path(a.run_dir); (run_dir / "img").mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    ids = [x for i, x in enumerate(ids) if i % a.num_shards == a.shard_index]
    pmap = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(PMAP))}; cache = json.load(open(sargs.rewrites_file))
    lad = {}
    for l in open(LADDER):
        d = json.loads(l); lad.setdefault(d["prompt_id"], {})[d["noise_seed"]] = d["final"]["hpsv3"]
    recp = run_dir / f"records_shard{a.shard_index}.jsonl"
    done = {(json.loads(l)["prompt_id"], json.loads(l)["noise_seed"], json.loads(l)["mode"], json.loads(l)["vi"])
            for l in open(recp)} if recp.exists() else set()
    ctx = su.load_pipeline(sargs); rm = su.load_reward_model(sargs, ctx.device)
    anchor = (0, float(sargs.baseline_cfg), 0.0)
    (run_dir / "config.json").write_text(json.dumps({
        "arm": "early_control", "rewrites_file": a.rewrites_file, "steps": a.steps, "split": a.split, "n_variants": a.n_variants,
        "nfe_early_each": 2 * a.split + 2 * (a.steps - a.split), "nfe_late_each": 2 * (a.steps - a.split),
        "nfe_full_each": 2 * a.steps, "nfe_fresh_noise": 2 * a.steps, "reward_server": rh,
        "design": "best and worst initial noise per prompt (by ladder hpsv3); for each, vary the (wording, CFG) "
                  "action in the EARLY window [0,split) only, the LATE window [split,28) only, or the FULL "
                  "trajectory. Early costs the same as a fresh noise draw (56 NFE), late reuses the prefix (32)."}, indent=1))

    def score_vec(prompt, img):
        buf = io.BytesIO(); img.save(buf, format="PNG"); enc = base64.b64encode(buf.getvalue()).decode()
        return {b: float(post(a.reward_server + "/score", {"prompt": prompt, "image_b64": enc, "backend": b})["score"])
                for b in SEARCH_REWARDS}

    with torch.no_grad(), recp.open("a") as rec:
        for pid in ids:
            prompt = pmap[pid]
            variants = su.generate_variants(sargs, prompt, cache); emb = su.encode_variants(ctx, variants)
            sched = su.step_schedule(ctx.device, emb.cond_text[0].dtype, a.steps, getattr(sargs, "sigmas", None), euler=True)
            fresh = su._resolve_mcts_fresh_noise_steps(sargs, a.steps, key_steps=[0])
            bank = [(vi, float(c), 0.0) for vi in range(len(emb.cond_text)) for c in sargs.cfg_scales if (vi, float(c), 0.0) != anchor]
            rng = np.random.RandomState(a.seed + 7919 * (ids.index(pid) + 1)); rng.shuffle(bank)
            acts = bank[: a.n_variants]
            if pid not in lad: print(f"[early] {pid} not in ladder probe, skipped", flush=True); continue
            order = sorted(lad[pid], key=lambda s: lad[pid][s])
            strata = [("worst", order[0]), ("best", order[-1])]
            med = float(np.median([lad[pid][s] for s in lad[pid]]))
            for lab, sd in strata:
                base_h = lad[pid][sd]
                lat0 = su.make_latents(ctx, sd, sargs.height, sargs.width, emb.cond_text[0].dtype); dx0 = torch.zeros_like(lat0)
                # shared anchor prefix for LATE mode
                latp, dxp = su._run_segment(sargs, ctx, emb, rm, prompt, lat0, dx0, anchor, sched, 0, a.split,
                                            noise_explore_steps=fresh, eps_bank=None)
                for mode in ("early", "late", "full"):
                    for vi, act in enumerate(acts):
                        if (pid, sd, mode, vi) in done: continue
                        ctx.nfe = 0
                        if mode == "early":
                            lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat0, dx0, act, sched, 0, a.split, noise_explore_steps=fresh, eps_bank=None)
                            lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, anchor, sched, a.split, a.steps, noise_explore_steps=fresh, eps_bank=None)
                        elif mode == "late":
                            lat, dx = (t.clone() for t in (latp, dxp))
                            lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, act, sched, a.split, a.steps, noise_explore_steps=fresh, eps_bank=None)
                        else:
                            lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat0, dx0, act, sched, 0, a.steps, noise_explore_steps=fresh, eps_bank=None)
                        img = su.decode_to_pil(ctx, su._final_decode_tensor(lat, dx, True)); vec = score_vec(prompt, img)
                        ip = run_dir / "img" / f"{pid}_{lab}_s{sd}_{mode}_v{vi}.png"; img.save(ip)
                        rec.write(json.dumps({"prompt_id": pid, "noise_seed": sd, "stratum": lab, "mode": mode, "vi": vi,
                                              "action": [int(act[0]), float(act[1]), 0.0], "vec": vec,
                                              "anchor_hpsv3": base_h, "median_noise_hpsv3": med,
                                              "nfe": int(ctx.nfe), "image": str(ip)}) + "\n"); rec.flush()
                        print(f"  {pid} {lab:<5} s{sd:<5} {mode:<5} v{vi} act=(w{int(act[0])},cfg{act[1]:g}) "
                              f"hps={vec['hpsv3']:.3f} (anchor {base_h:.3f}) nfe={ctx.nfe}", flush=True)
    print("[early] shard done", flush=True)


if __name__ == "__main__":
    main()
