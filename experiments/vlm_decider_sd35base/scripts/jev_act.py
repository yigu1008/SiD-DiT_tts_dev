#!/usr/bin/env python3
"""Jev CHOOSES the action. Not a filter, not a prior -- the policy.

Every earlier Jev arm restricted or biased the action set and then let UCB1 make the actual choice:
  jev_prune  -> kept top-2 wordings x top-3 CFGs, UCB1 chose
  ja_joint   -> kept top-6 of 28, UCB1 chose
  pu_puct    -> Jev's probabilities became a PUCT prior, UCB1 chose
  rt_retr    -> pruned 28 -> 4 (winning wording's 3 nearest CFGs + anchor), UCB1 chose
This arm removes UCB1 from the action decision entirely: the winning panel's EXACT (wording, CFG) is the action
executed for that segment.

The question is still posed retrospectively, which is the whole point of the retrieval format and the only reason
this has a chance: Jev is asked "which of these FINISHED images best matches the prompt?", never "what CFG should
I use?". The state ablation found it answers the former from evidence (45/45 focus-matching, 7/7 sharp-vs-blurry,
3/3 content) and the latter from a lexical prior (0/8 flips when the image was withheld). Direct action choice is
normally a prospective question; phrasing it as imitation of a past outcome keeps it retrospective.

There is no tree. Trajectories are generated SEQUENTIALLY and differ because the retrieval state grows: trajectory
t sees the t-1 finished traces that came before it, so Jev's choices evolve. Budget 616 NFE = 11 trajectories x 56.
Noise is held FIXED at seed 42 so this measures the ACTION policy alone, with the noise axis removed.

--policy jev    Jev picks the action at every key step from the retrieval sheet
--policy random uniform random action per segment (matched control: same budget, same fixed noise, no information)
--policy anchor every segment at the anchor action (the no-action-search floor)
"""
from __future__ import annotations
import argparse, base64, csv, hashlib, io, json, os, sys, time, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import opt_prompt_arm as opa
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


def sheet(panels, px=420):
    s = Image.new("RGB", (px * len(panels), px + 26), "white"); d = ImageDraw.Draw(s)
    for i, (im, cap) in enumerate(panels):
        s.paste(im.resize((px, px)), (i * px, 26)); d.text((i * px + 5, 7), cap[:56], fill="black")
    return s


def b64(img):
    buf = io.BytesIO(); img.save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()


def rank_best(c):
    tot = {id(x): 0 for x in c}
    for m in SR:
        for pos, x in enumerate(sorted(c, key=lambda y: y["vec"][m])): tot[id(x)] += pos
    return max(c, key=lambda x: tot[id(x)])


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
    p.add_argument("--key_steps", default="0,4,8,12,16,20,24,27")
    p.add_argument("--n_traj", type=int, default=11, help="sequential trajectories; 11 x 56 = 616 NFE")
    p.add_argument("--panels", type=int, default=3); p.add_argument("--min_traces", type=int, default=2)
    p.add_argument("--policy", default="jev", choices=["jev", "random", "anchor"])
    p.add_argument("--rewrites_file", default=CACHE,
                   help="wording bank Jev chooses from. The shipped cache gives 4.00 distinct wordings "
                        "(28 actions); config/rewrites_v2_9.json gives 7.33 (49 actions) with zero "
                        "artifacts. Since the action optimum is PROMPT-SPECIFIC (cross-prompt quality "
                        "correlation -0.08 for wording, -0.01 for CFG) a wider per-prompt bank is the "
                        "only way to raise the ceiling Jev is choosing against.")
    p.add_argument("--reward_server", required=True); p.add_argument("--vlm_server", default="http://127.0.0.1:5151")
    p.add_argument("--jev_server", default="http://127.0.0.1:5161")
    a = p.parse_args(); arm = Path(a.run_dir).name
    need = a.policy == "jev"
    os.environ.update({"HF_HUB_CACHE": "/data/ygu/.cache/hub", "HF_HOME": "/data/ygu/.cache/hub", "HF_HUB_OFFLINE": "1",
                       "OFFLOAD_TEXT_ENCODER_AFTER_ENCODE": "1", "REWARD_SERVER_URL": a.reward_server, "HPSV3_SCORE_STYLE": "raw"})
    sys.path.insert(0, SID); os.chdir(SID)
    import torch
    import sampling_unified_sd35 as su
    rh = json.loads(urllib.request.urlopen(a.reward_server + "/health", timeout=10).read())
    assert not [b for b in SR if b not in rh.get("backends", [])], rh
    if need:
        for u in (a.vlm_server, a.jev_server):
            assert json.loads(urllib.request.urlopen(u + "/health", timeout=10).read())["status"] == "ok", u
    sargs = su.normalize_paths(su.parse_args([
        "--backend", "sd35_base", "--search_method", "mcts", "--reward_backend", "imagereward", "--rewrites_file", a.rewrites_file, "--no_qwen",
        "--n_variants", str(_bank_n_variants(a.rewrites_file)), "--steps", str(a.steps), "--seed", str(a.seed), "--width", "1024", "--height", "1024",
        "--out_dir", a.run_dir, "--cfg_scales", *CFG_BANK, "--baseline_cfg", BASELINE_CFG,
        "--keystep_strategy", "explicit", "--mcts_key_steps", a.key_steps]))
    run_dir = Path(a.run_dir); (run_dir / "sheets").mkdir(parents=True, exist_ok=True); (run_dir / "traces").mkdir(exist_ok=True)
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    pmap = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(PMAP))}; cache = json.load(open(sargs.rewrites_file))
    recp = run_dir / "records.jsonl"; decp = run_dir / "decisions.jsonl"
    done = {json.loads(l)["prompt_id"] for l in open(recp)} if recp.exists() else set()
    todo = [i for i in ids if i not in done]
    ks = sorted(set(int(x) for x in a.key_steps.split(",") if x.strip().isdigit()))
    if 0 not in ks: ks = [0] + ks
    nk = len(ks); segs = [(ks[i], ks[i + 1] if i + 1 < nk else a.steps) for i in range(nk)]
    print(f"[{arm}] policy={a.policy} {a.n_traj} trajectories x {2*a.steps} NFE = {a.n_traj*2*a.steps}; "
          f"{nk} key steps -> up to {nk} Jev calls per trajectory; {len(todo)} prompts", flush=True)
    if not todo: return
    ctx = su.load_pipeline(sargs); rm = su.load_reward_model(sargs, ctx.device)
    anchor = (0, float(sargs.baseline_cfg), 0.0)
    (run_dir / "config.json").write_text(json.dumps({
        "arm": arm, "policy": a.policy, "rewrites_file": a.rewrites_file, "n_traj": a.n_traj, "key_steps": ks, "nfe_total": a.n_traj * 2 * a.steps,
        "noise_seed": "fixed 42 (noise axis removed on purpose)", "select": "rank3", "reward_server": rh,
        "design": "Jev EXECUTES the action it picks. The winning retrieval panel's exact (wording, CFG) is run for "
                  "that segment -- no pruning, no PUCT prior, no UCB1 anywhere. Trajectories are sequential and "
                  "differ only because the retrieval state grows."}, indent=1))

    def score_vec(prompt, img):
        buf = io.BytesIO(); img.save(buf, format="PNG"); enc = base64.b64encode(buf.getvalue()).decode()
        return {b: float(post(a.reward_server + "/score", {"prompt": prompt, "image_b64": enc, "backend": b})["score"]) for b in SR}

    with torch.no_grad(), recp.open("a") as rec, decp.open("a") as dec:
        for pid in todo:
            prompt = pmap[pid]; t0 = time.time(); ctx.nfe = 0; torch.cuda.reset_peak_memory_stats()
            rng = np.random.RandomState(a.seed + 7919 * (ids.index(pid) + 1))
            variants = su.generate_variants(sargs, prompt, cache); emb = su.encode_variants(ctx, variants)
            sched = su.step_schedule(ctx.device, emb.cond_text[0].dtype, a.steps, getattr(sargs, "sigmas", None), euler=True)
            fresh = su._resolve_mcts_fresh_noise_steps(sargs, a.steps, key_steps=ks)
            bank = [(vi, float(c), 0.0) for vi in range(len(emb.cond_text)) for c in sargs.cfg_scales]
            lat0 = su.make_latents(ctx, a.seed, sargs.height, sargs.width, emb.cond_text[0].dtype)
            traces = []; cands = []; n_jev = n_vlm = 0; jev_s = vlm_s = 0.0; dec_rows = []
            for t in range(a.n_traj):
                lat, dx = lat0.clone(), torch.zeros_like(lat0); taken = []
                for si, (s0, s1) in enumerate(segs):
                    if a.policy == "anchor": act = anchor
                    elif a.policy == "random": act = bank[rng.randint(len(bank))]
                    elif len(traces) < a.min_traces: act = anchor if t == 0 else bank[rng.randint(len(bank))]
                    else:
                        # BUG FIX (2026-10-29): the old per-wording map registered a trace under EVERY wording it had used, so
                        # one good trajectory filled all panels -- 62% of sheets showed ONE image three times, only 11% were
                        # three distinct images. Now: one panel per distinct trace, best-first, labelled by its dominant wording.
                        hits = []
                        for tr in sorted(traces, key=lambda tr: -tr["score"]):
                            ws = [int(x[0]) for x in tr["actions"]]; w = max(set(ws), key=lambda v: (ws.count(v), -ws.index(v)))
                            cf = float(np.mean([c for v, c, _ in tr["actions"] if int(v) == w]))
                            if all(k[0] != w for k, _ in hits) or len(traces) <= a.panels: hits.append(((w, cf), tr))
                            if len(hits) == a.panels: break
                        # REVIEW FIX: orders=3 permutes the option STRINGS but the image strip was fixed, so "Panel 1" was always
                        # the leftmost (and highest-IR) image: picked 50-55% vs 33% chance. Permute sheet order per call, log it.
                        perm = list(range(len(hits))); np.random.RandomState(6000 + 31 * t + s0).shuffle(perm)
                        hits = [hits[i] for i in perm]
                        if len(hits) < 2: act = bank[rng.randint(len(bank))]
                        else:
                            cur = su.decode_to_pil(ctx, dx) if s0 > 0 else Image.new("RGB", (420, 420), (235, 235, 235))
                            pans = [(cur, f"NOW: step {s0}/{a.steps}")] + [
                                (Image.open(tr["image"]).convert("RGB"), f"[{i+1}] w{k[0]} @ CFG {k[1]:g}") for i, (k, tr) in enumerate(hits)]
                            sh = sheet(pans); sp = run_dir / "sheets" / f"{pid}_t{t}_s{s0}.png"; sh.save(sp)
                            # Option strings carry the per-panel description (Jev-Omni scores option TEXTS; state is context only).
                            opts = [f'Panel {i+1}: wording {chr(65+k[0])} at CFG {k[1]:g}, search reward {tr["score"]:.3f}. '
                                    f'Assessment: {str(tr["feedback"]).strip()[:240]}' for i, (k, tr) in enumerate(hits)]
                            state = (f'ORIGINAL PROMPT (the only target): "{prompt}"\n'
                                     f'Image: leftmost = the sampler\'s CURRENT unfinished preview at step {s0} of {a.steps} (blur is normal); '
                                     f'the numbered panels are FINISHED images made for this prompt with the settings in the options. '
                                     f'The chosen setting runs the NEXT {s1-s0} sampling steps of the current image.')
                            r = post(a.jev_server + "/ask", {"state": state,
                                     "question": "Which numbered panel's FINISHED image best matches the ORIGINAL prompt?",
                                     "options": opts, "image_b64": b64(sh),
                                     "orders": 3, "seed": 6000 + 31 * t + s0, "tag": f"{pid}/act/t{t}/s{s0}"})
                            n_jev += 1; jev_s += r["ms"] / 1000
                            wk = hits[int(r["argmax"])][0]
                            act = (int(wk[0]), float(min(sargs.cfg_scales, key=lambda c: abs(float(c) - wk[1]))), 0.0)
                            dec_rows.append({"prompt_id": pid, "traj": t, "step": s0, "chosen_panel": int(r["argmax"]) + 1,
                                             "action_taken": [act[0], act[1], 0.0], "n_traces": len(traces),
                                             "panel_cfgs": [[k[0], k[1]] for k, _ in hits], "sheet_perm": perm, "jev_probs": r["probs"]})
                    taken.append(act)
                    lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, act, sched, s0, s1, noise_explore_steps=fresh, eps_bank=None)
                img = su.decode_to_pil(ctx, su._final_decode_tensor(lat, dx, True)); vec = score_vec(prompt, img)
                tp = run_dir / "traces" / f"{pid}_t{t}.png"; img.save(tp)
                fb = ""
                if need:
                    g = post(a.vlm_server + "/generate", {"text": f'ORIGINAL PROMPT: "{prompt}"\n\nThis is a FINISHED image. In at most two sentences state what matches the prompt and what is missing, wrong or distorted. Do not suggest settings.',
                                                          "images_b64": [opa.b64(img, 512)], "max_new_tokens": 110})
                    n_vlm += 1; vlm_s += g["seconds"]; fb = g["text"].strip()
                traces.append({"actions": [list(x) for x in taken], "image": str(tp), "score": vec["imagereward"], "feedback": fb})
                cands.append({"traj": t, "actions": [[int(v), float(c), 0.0] for v, c, _ in taken], "vec": vec, "image": str(tp)})
                print(f"    traj {t+1:2d}/{a.n_traj} hps={vec['hpsv3']:.3f} ir={vec['imagereward']:+.3f} "
                      f"ps={vec['pickscore']:.3f} jev={n_jev} wordings={sorted({x[0] for x in taken})}", flush=True)
            best = rank_best(cands)
            out = run_dir / f"{pid}_{arm}.png"; Image.open(best["image"]).convert("RGB").save(out)
            for d_ in dec_rows: dec.write(json.dumps(d_) + "\n")
            dec.flush()
            rec.write(json.dumps({"prompt_id": pid, "prompt": prompt, "variants": variants, "arm": arm, "seed": a.seed,
                                  "policy": a.policy, "n_traj": a.n_traj, "selected_on": "rank3",
                                  "search_score": best["vec"]["hpsv3"], "search_vec": best["vec"],
                                  "actions": best["actions"], "best_traj": best["traj"],
                                  "all_traj": [{"traj": c["traj"], "actions": c["actions"], "vec": c["vec"]} for c in cands],
                                  "n_jev_calls": n_jev, "jev_seconds": round(jev_s, 1), "n_vlm_calls": n_vlm,
                                  "vlm_seconds": round(vlm_s, 1), "image": str(out),
                                  "image_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                                  "nfe_transformer": int(ctx.nfe), "seconds": round(time.time() - t0, 1)}) + "\n"); rec.flush()
            print(f"[{arm}] {pid} best traj {best['traj']} hps={best['vec']['hpsv3']:.4f} ir={best['vec']['imagereward']:+.4f} "
                  f"jev={n_jev} nfe={ctx.nfe} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
