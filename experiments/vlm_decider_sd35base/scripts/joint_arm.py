#!/usr/bin/env python3
"""Joint (initial noise x action) allocation at a fixed NFE budget, on FIXED schedules -- no decision model.

Why this arm exists. Every noise measurement in this study used the ANCHOR action, and every action measurement used
seed 42, so each axis was credited with the other held still. Measured separately at a fixed value of the other:
  * action search at a FIXED noise is worth **+1.549 hpsv3** (anchor action -> 616-NFE key-step MCTS, 9/9 prompts positive);
  * the initial latent spans **5.044 hpsv3** of per-prompt range at a FIXED action.
Naively adding them says best-of-5 noise + action search ~ 13.2 vs best-of-11 noise-only 12.0. But
corr(anchor quality, action gain) = **-0.488**: action search gains MORE on a bad noise, so after screening to the
best noise the remaining gain shrinks. That regression is exactly what this arm measures instead of assuming.

Three phases under a hard budget, cost exact at 2 NFE/step (28 steps = 56 NFE; step 12 -> 28 = 32 NFE):
  1. SCREEN   --screen_n initial latents to step 12 at the anchor action (24 NFE each), score the x0 preview.
              Step 12 is the earliest step where the preview ranks noises usefully (rho=+0.61; it is +0.04 at step 4).
  2. KEEP     finish the top --keep noises by partial hpsv3 (32 NFE each) -> baseline finished images.
  3. ACT      spend everything left on ACTION_VARIANTs: branch a kept noise at step 12 with a distinct
              (wording, CFG) from the 28-action bank (32 NFE each), round-robin over the kept noises.
Selection: best finished image by HPSv3 (selecting on ImageReward cost the held-out alignment judge in ns_bo11_ir).
"""
from __future__ import annotations
import argparse, base64, csv, hashlib, io, json, os, sys, time, urllib.request
from pathlib import Path
import numpy as np

SID = "/home/ygu/project/SiD-DiT_tts_dev"
CACHE = "/data/ygu/human_eval_genai40_v1/sid_vqascore_algorithm_sweep/genai200_v1/shared_rewrites_cache.json"
PMAP = "/data/ygu/reward_orthogonality/mcts/prompts/prompt_map.csv"
CFG_BANK = ["3.5", "4.0", "4.5", "5.0", "5.5", "6.0", "7.0"]; BASELINE_CFG = "4.5"
SEARCH_REWARDS = ["imagereward", "hpsv3", "pickscore"]; SEL = "hpsv3"


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
    p.add_argument("--arm", default=None); p.add_argument("--ids", required=True); p.add_argument("--run_dir", required=True)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--budget", type=int, default=616); p.add_argument("--steps", type=int, default=28)
    p.add_argument("--branch_step", type=int, default=12)
    p.add_argument("--screen_n", type=int, default=8); p.add_argument("--keep", type=int, default=1)
    p.add_argument("--reward_server", required=True)
    p.add_argument("--rewrites_file", default=CACHE, help="wording bank; default is the shipped cache")
    a = p.parse_args(); a.arm = a.arm or Path(a.run_dir).name
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
    assert getattr(sargs, "euler_sampler", False) and sargs.model_id == "stabilityai/stable-diffusion-3.5-large"
    run_dir = Path(a.run_dir); (run_dir / "attempts").mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in open(a.ids) if l.strip()]; ids = ids[: a.limit] if a.limit else ids
    pmap = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(PMAP))}; cache = json.load(open(sargs.rewrites_file))
    recp = run_dir / "records.jsonl"
    done = {json.loads(l)["prompt_id"] for l in open(recp)} if recp.exists() else set()
    todo = [i for i in ids if i not in done]
    cs, cf = 2 * a.branch_step, 2 * (a.steps - a.branch_step)
    print(f"[{a.arm}] screen_n={a.screen_n} keep={a.keep} budget={a.budget}; "
          f"screen={a.screen_n*cs} finish={a.keep*cf} -> {(a.budget - a.screen_n*cs - a.keep*cf)//cf} action variants; "
          f"{len(todo)} prompts to do", flush=True)
    assert a.screen_n * cs + a.keep * cf <= a.budget, "schedule does not fit the budget"
    ctx = su.load_pipeline(sargs); rm = su.load_reward_model(sargs, ctx.device)
    (run_dir / "config.json").write_text(json.dumps({
        "arm": a.arm, "budget_nfe": a.budget, "steps": a.steps, "nfe_per_step": 2, "branch_step": a.branch_step,
        "screen_n": a.screen_n, "keep": a.keep, "rewrites_file": a.rewrites_file, "screen_cost_each": cs, "finish_cost_each": cf, "select_on": SEL,
        "n_action_variants": (a.budget - a.screen_n * cs - a.keep * cf) // cf, "reward_server": rh,
        "design": "fixed-schedule joint (noise x action) allocation: screen screen_n noises to step 12, finish the top "
                  "keep by partial hpsv3, spend the remainder on distinct-action branches from step 12. No decider."}, indent=1))

    def score_vec(prompt, img):
        buf = io.BytesIO(); img.save(buf, format="PNG"); enc = base64.b64encode(buf.getvalue()).decode()
        return {b: float(post(a.reward_server + "/score", {"prompt": prompt, "image_b64": enc, "backend": b})["score"])
                for b in SEARCH_REWARDS}

    anchor = (0, float(sargs.baseline_cfg), 0.0)
    with torch.no_grad(), recp.open("a") as rec:
        for pid in todo:
            prompt = pmap[pid]; t0 = time.time(); ctx.nfe = 0; torch.cuda.reset_peak_memory_stats()
            rng = np.random.RandomState(a.seed + 7919 * (ids.index(pid) + 1))
            variants = su.generate_variants(sargs, prompt, cache); emb = su.encode_variants(ctx, variants)
            sched = su.step_schedule(ctx.device, emb.cond_text[0].dtype, a.steps, getattr(sargs, "sigmas", None), euler=True)
            fresh = su._resolve_mcts_fresh_noise_steps(sargs, a.steps, key_steps=[0])
            bank = [(vi, float(c), 0.0) for vi in range(len(emb.cond_text)) for c in sargs.cfg_scales]
            seed_pool = [42] + [5000 + 100 * ids.index(pid) + i for i in range(60)]
            seeds = seed_pool[: a.screen_n]
            spent = 0; screened = []
            for sd in seeds:                                            # PHASE 1: screen
                lat = su.make_latents(ctx, sd, sargs.height, sargs.width, emb.cond_text[0].dtype); dx = torch.zeros_like(lat)
                lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, anchor, sched, 0, a.branch_step,
                                          noise_explore_steps=fresh, eps_bank=None)
                spent += cs
                ph = score_vec(prompt, su.decode_to_pil(ctx, dx))[SEL]
                screened.append({"seed": sd, "partial": ph, "state": (lat.clone(), dx.clone())})
            screened.sort(key=lambda x: -x["partial"]); kept = screened[: a.keep]
            print(f"  {pid} screened {len(screened)} to step {a.branch_step} ({spent} NFE); partial {SEL} "
                  + " ".join(f"{s['seed']}:{s['partial']:.2f}" for s in screened)
                  + f" -> keeping {[k['seed'] for k in kept]}", flush=True)
            finished = []
            def run_from_branch(entry, act, tag):                       # PHASE 2/3 shared
                nonlocal spent
                lat, dx = (t.clone() for t in entry["state"])
                lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, act, sched, a.branch_step, a.steps,
                                          noise_explore_steps=fresh, eps_bank=None)
                spent += cf
                img = su.decode_to_pil(ctx, su._final_decode_tensor(lat, dx, True)); vec = score_vec(prompt, img)
                ap = run_dir / "attempts" / f"{pid}_{tag}_s{entry['seed']}.png"; img.save(ap)
                finished.append({"seed": entry["seed"], "action": [int(act[0]), float(act[1]), 0.0], "vec": vec,
                                 "image": str(ap), "phase": tag.split("_")[0]})
                print(f"    {tag:12s} seed {entry['seed']:5d} act=(w{int(act[0])},cfg{act[1]:g}) "
                      f"hps={vec['hpsv3']:.3f} ir={vec['imagereward']:+.3f} spent={spent}/{a.budget}", flush=True)
            for i, k in enumerate(kept): run_from_branch(k, anchor, f"keep_{i}")
            pool = [x for x in bank if x != anchor]; rng.shuffle(pool); vi = 0   # PHASE 3: action variants
            while spent + cf <= a.budget and vi < len(pool):
                run_from_branch(kept[vi % len(kept)], pool[vi], f"act_{vi}"); vi += 1
            # PHASE 4 (budget guard).  bank = DISTINCT wordings x CFGs, and a sanitised or constraint-filtered
            # wording bank can collapse to 2 distinct wordings, so `pool` is exhausted while budget remains and the
            # arm silently runs on less compute than it was matched to.  That exact bug made pb_clean spend 472 of
            # 616 NFE on 4 of 9 prompts and confounded the wording A/B until it was caught.  Spend any remainder on
            # additional initial noises at the anchor action -- the axis measured to be worth the most anyway.
            extra = 0; si = a.screen_n
            while spent + 2 * a.steps <= a.budget and si < len(seed_pool):
                sd = seed_pool[si]; si += 1; extra += 1
                lat = su.make_latents(ctx, sd, sargs.height, sargs.width, emb.cond_text[0].dtype); dx = torch.zeros_like(lat)
                lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, anchor, sched, 0, a.steps,
                                          noise_explore_steps=fresh, eps_bank=None)
                spent += 2 * a.steps
                img = su.decode_to_pil(ctx, su._final_decode_tensor(lat, dx, True)); vec = score_vec(prompt, img)
                ap = run_dir / "attempts" / f"{pid}_fill{extra}_s{sd}.png"; img.save(ap)
                finished.append({"seed": sd, "action": [int(anchor[0]), float(anchor[1]), 0.0], "vec": vec,
                                 "image": str(ap), "phase": "fill"})
                print(f"    fill_{extra:<8} seed {sd:5d} (action pool exhausted) hps={vec['hpsv3']:.3f} "
                      f"spent={spent}/{a.budget}", flush=True)
            bf = max(finished, key=lambda f: f["vec"][SEL])
            out = run_dir / f"{pid}_{a.arm}.png"
            from PIL import Image as _I; _I.open(bf["image"]).convert("RGB").save(out)
            base = max((f for f in finished if f["phase"] == "keep"), key=lambda f: f["vec"][SEL])
            rec.write(json.dumps({"prompt_id": pid, "prompt": prompt, "variants": variants, "arm": a.arm, "seed": a.seed,
                                  "screen_n": a.screen_n, "keep": a.keep, "rewrites_file": a.rewrites_file, "search_score": float(bf["vec"][SEL]),
                                  "search_vec": bf["vec"], "selected_on": SEL, "selected_seed": bf["seed"],
                                  "selected_phase": bf["phase"], "actions": [bf["action"]],
                                  "anchor_on_kept": base["vec"][SEL], "action_gain_on_kept": bf["vec"][SEL] - base["vec"][SEL],
                                  "n_finished": len(finished), "n_action_variants": vi, "n_fill_noises": extra,
                                  "screened": [{"seed": s["seed"], "partial": s["partial"]} for s in screened],
                                  "all_finished": [{"seed": f["seed"], "action": f["action"], "phase": f["phase"], "vec": f["vec"]} for f in finished],
                                  "image": str(out), "image_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                                  "nfe_transformer": int(spent), "seconds": round(time.time() - t0, 1),
                                  "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}) + "\n"); rec.flush()
            print(f"[{a.arm}] {pid} hps={bf['vec'][SEL]:.4f} ir={bf['vec']['imagereward']:+.4f} from {bf['phase']} "
                  f"seed {bf['seed']} | anchor-on-kept {base['vec'][SEL]:.3f} -> action gain "
                  f"{bf['vec'][SEL]-base['vec'][SEL]:+.3f} | finished={len(finished)} variants={vi} "
                  f"nfe={spent}/{a.budget} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
