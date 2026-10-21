#!/usr/bin/env python3
"""Static arm for an arbitrary search reward: one (rewrite, CFG) pair held fixed for
an entire trajectory, selected by the TERMINAL value of the given reward backend
against the ORIGINAL prompt.  Generalization of static_ir.py (which is left untouched).

Identical to the MCTS reference except for HOW actions are used: same SiD pipeline,
steps, resolution, rewrite bank (n_variants=3 -> 4 variants incl. original), CFG bank,
seed-42 initial latent, same segment runner, same UnifiedRewardScorer backend.

Per candidate we store the RAW component scores (imagereward / hpsv3 / pickscore, whichever
the backend uses) and the composite computed with the scorer's own ranges; on the first
candidate of every prompt the composite is cross-checked against reward_model.score() and
the |diff| is recorded, so the selection rule is verifiably the reference objective.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path

SID = "/home/ygu/project/SiD-DiT_tts_dev"
CACHE = "/data/ygu/human_eval_genai40_v1/sid_vqascore_algorithm_sweep/genai200_v1/shared_rewrites_cache.json"
PMAP = "/data/ygu/reward_orthogonality/mcts/prompts/prompt_map.csv"
COMPONENTS = {"imagereward": ["imagereward"], "composite_hpsv3_ir": ["hpsv3", "imagereward"],
              "composite_3": ["imagereward", "hpsv3", "pickscore"]}

def make_objective(rm, backend):
    """Return f(components_dict) -> composite, using the scorer's own fixed ranges."""
    n = rm._norm
    if backend == "imagereward":
        return lambda c: c["imagereward"]
    if backend == "composite_hpsv3_ir":
        (hl, hh), (il, ih) = rm._COMPOSITE_HPSV3_RANGE, rm._COMPOSITE_IR_RANGE
        return lambda c: 0.5 * n(c["hpsv3"], hl, hh) + 0.5 * n(c["imagereward"], il, ih)
    if backend == "composite_3":
        (hl, hh), (il, ih), (pl, ph) = rm._COMPOSITE_HPSV3_RANGE, rm._COMPOSITE_IR_RANGE, rm._COMPOSITE_PICKSCORE_RANGE
        return lambda c: (n(c["imagereward"], il, ih) + n(c["hpsv3"], hl, hh) + n(c["pickscore"], pl, ph)) / 3.0
    raise SystemExit(f"no component formula for backend {backend}; add it to make_objective")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, help="arm name used in records/filenames, e.g. static_mr3")
    p.add_argument("--reward_backend", required=True, choices=sorted(COMPONENTS))
    p.add_argument("--ids", required=True); p.add_argument("--run_dir", required=True)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reward_server", required=True)
    a = p.parse_args()
    os.environ.update({"HF_HUB_CACHE": "/data/ygu/.cache/hub", "HF_HOME": "/data/ygu/.cache/hub", "HF_HUB_OFFLINE": "1",
                       "OFFLOAD_TEXT_ENCODER_AFTER_ENCODE": "1", "REWARD_SERVER_URL": a.reward_server, "HPSV3_SCORE_STYLE": "raw"})
    sys.path.insert(0, SID); os.chdir(SID)
    import torch, numpy as np
    import sampling_unified_sd35 as su

    sargs = su.normalize_paths(su.parse_args([
        "--backend", "sid", "--search_method", "mcts", "--reward_backend", a.reward_backend,
        "--rewrites_file", CACHE, "--no_qwen", "--n_variants", "3", "--n_sims", "40", "--ucb_c", "1.41",
        "--steps", "4", "--seed", str(a.seed), "--width", "1024", "--height", "1024", "--out_dir", a.run_dir]))
    run_dir = Path(a.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    if a.limit: ids = ids[: a.limit]
    pmap = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(PMAP))}
    cache = json.load(open(sargs.rewrites_file))
    recp = run_dir / "records.jsonl"
    done = {json.loads(l)["prompt_id"] for l in open(recp)} if recp.exists() else set()
    todo = [i for i in ids if i not in done]
    print(f"[{a.arm}] {len(todo)} prompts to do ({len(done)} already done)", flush=True)
    if not todo: return

    ctx = su.load_pipeline(sargs)
    rm = su.load_reward_model(sargs, ctx.device)
    comps = COMPONENTS[a.reward_backend]
    missing = [b for b in comps if b not in rm.available]
    assert not missing, f"reward server lacks {missing}; available={rm.available}"
    score_fn = {"imagereward": rm._score_imagereward, "hpsv3": rm._score_hpsv3, "pickscore": rm._score_pickscore}
    objective = make_objective(rm, a.reward_backend)
    corr = list(getattr(sargs, "correction_strengths", [0.0])); use_euler = getattr(sargs, "euler_sampler", False)
    (run_dir / "config.json").write_text(json.dumps({
        "arm": a.arm, "reward_backend": a.reward_backend, "components": comps, "seed": a.seed, "steps": sargs.steps,
        "cfg_scales": sargs.cfg_scales, "baseline_cfg": sargs.baseline_cfg, "n_variants": sargs.n_variants,
        "width": sargs.width, "height": sargs.height, "dtype": str(sargs.dtype), "model_id": sargs.model_id,
        "rewrites_file": sargs.rewrites_file, "rewrites_sha256": hashlib.sha256(open(sargs.rewrites_file, "rb").read()).hexdigest(),
        "reward": rm.describe(), "ranges": {"hpsv3": list(rm._COMPOSITE_HPSV3_RANGE), "imagereward": list(rm._COMPOSITE_IR_RANGE),
                                            "pickscore": list(rm._COMPOSITE_PICKSCORE_RANGE)}, "hpsv3_score_style": "raw",
        "selection": f"argmax terminal {a.reward_backend} vs ORIGINAL prompt; ties -> lowest candidate index",
        "bank": "variants x cfg_scales x correction_strengths (no noise-injection actions)", "correction_strengths": corr}, indent=1))

    with torch.no_grad(), recp.open("a") as rec:
        for pid in todo:
            prompt = pmap[pid]; t0 = time.time(); ctx.nfe = 0; ctx.correction_nfe = 0
            np.random.seed(a.seed); torch.manual_seed(a.seed)
            variants = su.generate_variants(sargs, prompt, cache)
            emb = su.encode_variants(ctx, variants)
            bank = [(vi, float(cfg), float(cs)) for vi in range(len(emb.cond_text)) for cfg in sargs.cfg_scales for cs in corr]
            latents0 = su.make_latents(ctx, a.seed, sargs.height, sargs.width, emb.cond_text[0].dtype)
            dx0 = torch.zeros_like(latents0)
            sched = su.step_schedule(ctx.device, latents0.dtype, sargs.steps, getattr(sargs, "sigmas", None), euler=use_euler)
            _, t0_4d, _ = sched[0]
            start = su._prepare_latents(latents0, dx0, latents0, t0_4d, 0, use_euler)
            cands = []; best = None; check = None
            for ci, action in enumerate(bank):
                lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, start, dx0, action, sched, 0, int(sargs.steps),
                                          noise_explore_steps=set(), eps_bank=None)
                img = su.decode_to_pil(ctx, su._final_decode_tensor(lat, dx, use_euler))
                c = {b: float(score_fn[b](prompt, img)) for b in comps}
                s = float(objective(c))
                if ci == 0:
                    check = abs(s - float(rm.score(prompt, img)))
                cands.append({"cand": ci, "variant_idx": action[0], "cfg": action[1], "corr": action[2], "score": s, **c})
                if best is None or s > best[1]: best = (ci, s, img)
            ci, s, img = best
            out = run_dir / f"{pid}_{a.arm}.png"; img.save(out)
            rec.write(json.dumps({"prompt_id": pid, "prompt": prompt, "variants": variants, "arm": a.arm, "reward_backend": a.reward_backend,
                                  "seed": a.seed, "n_candidates": len(bank), "selected_cand": ci, "selected_action": list(bank[ci]),
                                  "selected_score": s, "selected_components": {b: cands[ci][b] for b in comps},
                                  "objective_check_absdiff": check, "candidates": cands, "image": str(out),
                                  "image_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                                  "nfe_transformer": int(ctx.nfe), "seconds": round(time.time() - t0, 1)}) + "\n"); rec.flush()
            print(f"[{a.arm}] {pid} best cand {ci} {bank[ci]} score={s:.4f} check={check:.2e} nfe={ctx.nfe} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
