#!/usr/bin/env python3
"""Adaptive noise-budget allocation: abandon a saturated initial noise, and decide every N attempts whether to
EXPAND the noise pool instead of searching actions on the noise already in hand.  Hard NFE budget.

Measured context that fixes the design (PLAN.md, 2026-10-19):
  * the initial latent carries ~4.5 hpsv3 of per-prompt range; the whole (wording x CFG) search is worth ~0.88;
  * across noises, the x0 preview predicts the FINISHED hpsv3 with Spearman +0.042 at step 4, +0.347 at step 8,
    **+0.613 at step 12**, +0.667 at 16 -- so abandon decisions are placed at 12/16/20, never at 4;
  * at 616 NFE the ORACLE abandonment policy beats uniform best-of-11 by +0.413 hpsv3, a FREE partial-reward screen
    gets +0.140 of that, and a RANDOM screen gets -0.0005 (so it is information, not budget reallocation);
  * selection is on HPSv3: selecting noise on ImageReward significantly lost the held-out alignment judge.

Every attempt is one of two kinds, and that is decision B:
  NEW_NOISE      fresh initial latent at the anchor action, grown from step 0   (56 NFE if carried to the end)
  ACTION_VARIANT branch the best noise so far from its cached step-12 state with a different (wording, CFG) (32 NFE)
Decision A can abort either attempt at a checkpoint, refunding the unspent steps to the budget.
Cost is exact: 2 NFE per Euler step.

--policy selects who decides, with everything else identical:
  jev     A and B both by Jev, from image evidence (a matching question, the only kind it answers from evidence)
  reward  A = abandon if this attempt's partial hpsv3 is below the median of earlier attempts AT THE SAME STEP
          (scale-correct, free); B = always NEW_NOISE. This is the +0.140 free screen.
  random  A and B at fixed rates (pass the rates Jev actually used) -- isolates information from rate
  never   A = never abandon, B = always NEW_NOISE -> uniform best-of-k. Reproduces ns_bo11_hps; a machinery check.
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
SEARCH_REWARDS = ["imagereward", "hpsv3", "pickscore"]
SEL = "hpsv3"

ABANDON_OPTS = [
    "This unfinished attempt is on track to end up a BETTER match for the prompt than the finished image - keep spending steps on it.",
    "This unfinished attempt will NOT end up better than the finished image - stop spending steps on it and start again from a different random seed.",
]
EXPAND_OPTS = [
    "Start from a DIFFERENT RANDOM SEED. The finished images share a problem that is baked into the layout they were all born with, so rewording the prompt or changing the guidance strength will not fix it.",
    "KEEP the best layout found so far and vary the wording and guidance strength instead. Its content is nearly right and what remains is refinement.",
]


def post(url, body, timeout=600, tries=60, sleep=10):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as e:
            print(f"[post] {url} failed ({type(e).__name__}: {str(e)[:70]}); retry {i+1}/{tries}", flush=True); time.sleep(sleep)
    raise RuntimeError(f"{url} unreachable")


def sheet(panels, px=448):
    H = px + 26; s = Image.new("RGB", (px * len(panels), H), "white"); d = ImageDraw.Draw(s)
    for i, (im, cap) in enumerate(panels):
        s.paste(im.resize((px, px)), (i * px, 26)); d.text((i * px + 5, 7), cap[:62], fill="black")
    return s


def b64(img):
    buf = io.BytesIO(); img.save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default=None); p.add_argument("--ids", required=True); p.add_argument("--run_dir", required=True)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--budget", type=int, default=616, help="hard NFE cap per prompt (matches every other arm)")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--abandon_steps", default="12,16,20", help="checkpoints where decision A is made (step 4 is useless: rho=+0.04)")
    p.add_argument("--branch_step", type=int, default=12, help="step whose state is cached for ACTION_VARIANT attempts")
    p.add_argument("--expand_every", type=int, default=5, help="decision B is made after every this many finished attempts")
    p.add_argument("--policy", default="jev", choices=["jev", "reward", "random", "never"])
    p.add_argument("--abandon_rate", type=float, default=0.5, help="policy=random: P(abandon) at each checkpoint")
    p.add_argument("--expand_rate", type=float, default=0.5, help="policy=random: P(new noise) at each decision B")
    p.add_argument("--select", default="rank3", choices=["rank3", "rank_hps_ps", "hpsv3", "imagereward", "pickscore"],
                   help="final selection over the finished candidates. rank3 = rank-sum of hpsv3+pickscore+imagereward "
                        "(ranks are scale-free within the candidate set, so no min-max normalisation is needed); "
                        "measured best on every held-out metric. A single reward is worse: ir alone hacks the "
                        "alignment judge, hpsv3 alone loses vqascore -0.017 to the pair.")
    p.add_argument("--reward_server", required=True); p.add_argument("--vlm_server", default="http://127.0.0.1:5151")
    p.add_argument("--jev_server", default="http://127.0.0.1:5161"); p.add_argument("--preview_px", type=int, default=448)
    a = p.parse_args(); a.arm = a.arm or Path(a.run_dir).name
    need_models = a.policy == "jev"
    os.environ.update({"HF_HUB_CACHE": "/data/ygu/.cache/hub", "HF_HOME": "/data/ygu/.cache/hub", "HF_HUB_OFFLINE": "1",
                       "OFFLOAD_TEXT_ENCODER_AFTER_ENCODE": "1", "REWARD_SERVER_URL": a.reward_server, "HPSV3_SCORE_STYLE": "raw"})
    sys.path.insert(0, SID); os.chdir(SID)
    import torch
    import sampling_unified_sd35 as su
    rh = json.loads(urllib.request.urlopen(a.reward_server + "/health", timeout=10).read())
    assert not [b for b in SEARCH_REWARDS if b not in rh.get("backends", [])], rh
    if need_models:
        for url in (a.vlm_server, a.jev_server):
            assert json.loads(urllib.request.urlopen(url + "/health", timeout=10).read()).get("status") == "ok", url
    sargs = su.normalize_paths(su.parse_args([
        "--backend", "sd35_base", "--search_method", "mcts", "--reward_backend", "imagereward", "--rewrites_file", CACHE, "--no_qwen",
        "--n_variants", "3", "--steps", str(a.steps), "--seed", str(a.seed), "--width", "1024", "--height", "1024",
        "--out_dir", a.run_dir, "--cfg_scales", *CFG_BANK, "--baseline_cfg", BASELINE_CFG,
        "--keystep_strategy", "explicit", "--mcts_key_steps", "0,4,8,12,16,20,24,27"]))
    assert getattr(sargs, "euler_sampler", False) and sargs.model_id == "stabilityai/stable-diffusion-3.5-large"
    run_dir = Path(a.run_dir); (run_dir / "sheets").mkdir(parents=True, exist_ok=True); (run_dir / "attempts").mkdir(exist_ok=True)
    ids = [l.strip() for l in open(a.ids) if l.strip()]; ids = ids[: a.limit] if a.limit else ids
    pmap = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(PMAP))}; cache = json.load(open(sargs.rewrites_file))
    recp = run_dir / "records.jsonl"; decp = run_dir / "decisions.jsonl"
    done = {json.loads(l)["prompt_id"] for l in open(recp)} if recp.exists() else set()
    todo = [i for i in ids if i not in done]
    print(f"[{a.arm}] policy={a.policy} budget={a.budget} NFE; {len(todo)} prompts to do ({len(done)} done)", flush=True)
    if not todo: return
    ctx = su.load_pipeline(sargs); rm = su.load_reward_model(sargs, ctx.device)
    ab_steps = sorted(int(x) for x in a.abandon_steps.split(",") if x.strip().isdigit())
    (run_dir / "config.json").write_text(json.dumps({
        "arm": a.arm, "policy": a.policy, "budget_nfe": a.budget, "steps": a.steps, "nfe_per_step": 2,
        "abandon_steps": ab_steps, "branch_step": a.branch_step, "expand_every": a.expand_every,
        "select_on": SEL, "abandon_rate": a.abandon_rate, "expand_rate": a.expand_rate,
        "anchor_cfg": float(sargs.baseline_cfg), "search_rewards": SEARCH_REWARDS, "reward_server": rh,
        "design": "Attempts are NEW_NOISE (fresh initial latent, anchor action, from step 0) or ACTION_VARIANT (branch "
                  "the best noise at step 12 with a different wording/CFG). Decision B picks the kind every "
                  "expand_every finished attempts; decision A can abort an attempt at steps 12/16/20, refunding the "
                  "unspent steps. Hard NFE cap, exact at 2 NFE/step. Selection on HPSv3."}, indent=1))

    def score_vec(prompt, img):
        enc = b64(img)
        return {b: float(post(a.reward_server + "/score", {"prompt": prompt, "image_b64": enc, "backend": b})["score"])
                for b in SEARCH_REWARDS}

    anchor = (0, float(sargs.baseline_cfg), 0.0)
    with torch.no_grad(), recp.open("a") as rec, decp.open("a") as dec:
        for pid in todo:
            prompt = pmap[pid]; t0 = time.time(); ctx.nfe = 0; torch.cuda.reset_peak_memory_stats()
            rng = np.random.RandomState(a.seed + 7919 * (ids.index(pid) + 1))
            variants = su.generate_variants(sargs, prompt, cache); emb = su.encode_variants(ctx, variants)
            sched = su.step_schedule(ctx.device, emb.cond_text[0].dtype, a.steps, getattr(sargs, "sigmas", None), euler=True)
            fresh = su._resolve_mcts_fresh_noise_steps(sargs, a.steps, key_steps=[0])
            bank = [(vi, float(c), 0.0) for vi in range(len(emb.cond_text)) for c in sargs.cfg_scales]
            budget = a.budget; spent = 0
            finished = []            # dicts: kind, seed, action, vec, image, feedback
            partial_hist = {}        # step -> [partial hpsv3 of earlier attempts]
            branch_cache = {}        # seed -> (lat, dx) at branch_step
            dec_rows = []; n_jev = n_vlm = 0; jev_sec = vlm_sec = 0.0
            seed_pool = [42] + [5000 + 100 * ids.index(pid) + i for i in range(60)]
            seed_i = 0; next_kind = "NEW_NOISE"; n_abandon = n_ask_a = n_ask_b = 0; attempts = 0
            used_actions = set()

            def best_fin():
                return max(finished, key=lambda f: f["vec"][SEL]) if finished else None

            def decide_abandon(step, prev_img, bf, att_tag):
                """Decision A. A two-panel comparison: unfinished preview vs the best finished image."""
                nonlocal n_jev, jev_sec, n_ask_a
                n_ask_a += 1
                if a.policy == "never": return False
                if a.policy == "random": return bool(rng.rand() < a.abandon_rate)
                if a.policy == "reward":
                    h = partial_hist.get(step, [])
                    if not h: return False
                    cur = score_vec(prompt, prev_img)[SEL]
                    return cur < float(np.median(h))
                panels = [(prev_img, f"UNFINISHED: step {step}/{a.steps}"), (Image.open(bf["image"]).convert("RGB"), "FINISHED earlier attempt")]
                sh = sheet(panels, a.preview_px)
                sp = run_dir / "sheets" / f"{pid}_{att_tag}_abandon_t{step}.png"; sh.save(sp)
                state = (f'ORIGINAL PROMPT (the only target): "{prompt}"\n'
                         f'Left panel: an UNFINISHED image, {step} of {a.steps} sampling steps done. It is blurry and low in '
                         f'contrast BECAUSE it is unfinished - that is normal and is not a defect. Judge its layout, its '
                         f'content and whether the things the prompt asks for are taking shape.\n'
                         f'Right panel: a FINISHED image from an earlier attempt at the same prompt. A vision model said of '
                         f'it: {bf["feedback"]}\n'
                         f'Finishing the left image costs the remaining {a.steps - step} steps. Those steps sharpen and refine '
                         f'what is already there; they do not add missing objects or rebuild the composition.')
                r = post(a.jev_server + "/ask", {"state": state, "question": "Is the unfinished attempt worth finishing?",
                                                 "options": ABANDON_OPTS, "image_b64": b64(sh), "orders": 3,
                                                 "seed": 3000 + step + 17 * attempts, "tag": f"{pid}/abandon/t{step}"})
                n_jev += 1; jev_sec += r["ms"] / 1000
                ab = int(r["argmax"]) == 1
                dec_rows.append({"prompt_id": pid, "kind": "abandon", "attempt": attempts, "step": step, "spent": spent,
                                 "abandon": ab, "jev_probs": r["probs"], "sheet": str(sp),
                                 "best_so_far": bf["vec"][SEL]})
                return ab

            def decide_expand():
                """Decision B: spend the next attempt on a NEW initial noise, or on wording/CFG around the best one?"""
                nonlocal n_jev, jev_sec, n_ask_b
                n_ask_b += 1
                if a.policy in ("never", "reward"): return "NEW_NOISE"
                if a.policy == "random": return "NEW_NOISE" if rng.rand() < a.expand_rate else "ACTION_VARIANT"
                top = sorted(finished, key=lambda f: -f["vec"][SEL])[:3]
                panels = [(Image.open(f["image"]).convert("RGB"), f"finished #{i+1} (seed {f['seed']})") for i, f in enumerate(top)]
                sh = sheet(panels, a.preview_px)
                sp = run_dir / "sheets" / f"{pid}_a{attempts}_expand.png"; sh.save(sp)
                lines = "\n".join(f'  #{i+1} (random seed {f["seed"]}, wording {chr(65+int(f["action"][0]))} at CFG '
                                  f'{f["action"][1]:g}): {f["feedback"]}' for i, f in enumerate(top))
                state = (f'ORIGINAL PROMPT (the only target): "{prompt}"\n'
                         f'{len(finished)} images have been finished for this prompt. The best are attached, and a vision '
                         f'model reported on each:\n{lines}\n\n'
                         f'Each random seed fixes the overall layout the image grows into; rewording the prompt or changing '
                         f'the guidance strength refines an existing layout but cannot replace it. There is budget left for '
                         f'roughly one or two more attempts. Choose where to spend it.')
                r = post(a.jev_server + "/ask", {"state": state, "question": "Where should the remaining budget go?",
                                                 "options": EXPAND_OPTS, "image_b64": b64(sh), "orders": 3,
                                                 "seed": 4000 + attempts, "tag": f"{pid}/expand/a{attempts}"})
                n_jev += 1; jev_sec += r["ms"] / 1000
                kind = "NEW_NOISE" if int(r["argmax"]) == 0 else "ACTION_VARIANT"
                dec_rows.append({"prompt_id": pid, "kind": "expand", "attempt": attempts, "spent": spent,
                                 "choice": kind, "jev_probs": r["probs"], "sheet": str(sp),
                                 "n_finished": len(finished), "best_so_far": top[0]["vec"][SEL]})
                return kind

            def assess(img):
                nonlocal n_vlm, vlm_sec
                fb = post(a.vlm_server + "/generate", {"text": f'ORIGINAL PROMPT: "{prompt}"\n\nThis is a FINISHED image. In at most two sentences state what matches the prompt and what is missing, wrong or distorted (mention oversaturation, plastic surfaces, flat lighting or artifacts if present). Do not suggest settings.',
                                                       "images_b64": [opa.b64(img, 512)], "max_new_tokens": 110})
                n_vlm += 1; vlm_sec += fb["seconds"]; return fb["text"].strip()

            while True:
                kind = next_kind
                if kind == "ACTION_VARIANT" and not branch_cache: kind = "NEW_NOISE"
                start = 0 if kind == "NEW_NOISE" else a.branch_step
                if spent + 2 * (a.steps - start) > budget:          # cannot finish this kind; try the cheaper kind
                    if kind == "NEW_NOISE" and branch_cache and spent + 2 * (a.steps - a.branch_step) <= budget:
                        kind, start = "ACTION_VARIANT", a.branch_step
                    else: break
                attempts += 1; att_tag = f"a{attempts}"
                if kind == "NEW_NOISE":
                    sd = seed_pool[seed_i]; seed_i += 1; act = anchor
                    lat = su.make_latents(ctx, sd, sargs.height, sargs.width, emb.cond_text[0].dtype); dx = torch.zeros_like(lat)
                else:
                    bf = best_fin(); sd = bf["seed"]
                    cand = [x for x in bank if x not in used_actions and x != anchor]
                    act = cand[rng.randint(len(cand))] if cand else anchor
                    lat, dx = (t.clone() for t in branch_cache[sd])
                used_actions.add(act)
                step = start; aborted = False
                while step < a.steps:
                    nxt = min([s for s in ab_steps + [a.steps] if s > step])
                    cost = 2 * (nxt - step)
                    if spent + cost > budget: aborted = True; break
                    lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, act, sched, step, nxt,
                                              noise_explore_steps=fresh, eps_bank=None)
                    spent += cost; step = nxt
                    if step == a.branch_step and kind == "NEW_NOISE":
                        branch_cache[sd] = (lat.clone(), dx.clone())
                    if step in ab_steps and step < a.steps:
                        prev = su.decode_to_pil(ctx, dx)
                        ph = score_vec(prompt, prev)[SEL]; partial_hist.setdefault(step, []).append(ph)
                        bf = best_fin()
                        if bf is not None and decide_abandon(step, prev, bf, att_tag):
                            n_abandon += 1; aborted = True
                            print(f"    {att_tag} {kind} seed {sd} ABANDONED at step {step} (spent {spent}/{budget})", flush=True)
                            break
                if aborted or step < a.steps: next_kind = "NEW_NOISE"; continue
                img = su.decode_to_pil(ctx, su._final_decode_tensor(lat, dx, True))
                vec = score_vec(prompt, img)
                ap = run_dir / "attempts" / f"{pid}_{att_tag}_s{sd}.png"; img.save(ap)
                fb = assess(img) if need_models else ""
                finished.append({"kind": kind, "seed": sd, "action": [int(act[0]), float(act[1]), 0.0], "vec": vec,
                                 "image": str(ap), "feedback": fb})
                print(f"    {att_tag} {kind:14s} seed {sd:5d} act=(w{int(act[0])},cfg{act[1]:g}) "
                      f"hps={vec['hpsv3']:.3f} ir={vec['imagereward']:+.3f} spent={spent}/{budget} fin={len(finished)}", flush=True)
                next_kind = decide_expand() if (len(finished) % a.expand_every == 0) else "NEW_NOISE"
            def final_pick(cands):
                """Rank-sum selection. Ranks within the finished candidate set need no normalisation."""
                if a.select in SEARCH_REWARDS: return max(cands, key=lambda f: f["vec"][a.select])
                ms = ["hpsv3", "pickscore"] + (["imagereward"] if a.select == "rank3" else [])
                tot = {id(f): 0 for f in cands}
                for mm in ms:
                    for pos, f in enumerate(sorted(cands, key=lambda x: x["vec"][mm])): tot[id(f)] += pos
                return max(cands, key=lambda f: tot[id(f)])
            bf = final_pick(finished) if finished else None
            assert bf is not None, f"{pid}: no attempt finished within {budget} NFE"
            out = run_dir / f"{pid}_{a.arm}.png"; Image.open(bf["image"]).convert("RGB").save(out)
            for d_ in dec_rows: dec.write(json.dumps(d_) + "\n")
            dec.flush()
            kinds = {}
            for f in finished: kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
            rec.write(json.dumps({"prompt_id": pid, "prompt": prompt, "variants": variants, "arm": a.arm, "seed": a.seed,
                                  "policy": a.policy, "search_score": float(bf["vec"][SEL]), "search_vec": bf["vec"],
                                  "selected_on": a.select, "selected_seed": bf["seed"], "selected_kind": bf["kind"],
                                  "actions": [bf["action"]], "n_attempts": attempts, "n_finished": len(finished),
                                  "finished_kinds": kinds, "n_abandoned": n_abandon, "n_abandon_asks": n_ask_a,
                                  "n_expand_asks": n_ask_b, "abandon_rate": round(n_abandon / max(1, n_ask_a), 3),
                                  "expand_new_noise_rate": round(sum(1 for d_ in dec_rows if d_["kind"] == "expand"
                                      and d_.get("choice") == "NEW_NOISE") / max(1, n_ask_b), 3),
                                  "all_finished": [{"kind": f["kind"], "seed": f["seed"], "action": f["action"], "vec": f["vec"]} for f in finished],
                                  "n_jev_calls": n_jev, "jev_seconds": round(jev_sec, 1), "n_vlm_calls": n_vlm,
                                  "vlm_seconds": round(vlm_sec, 1), "image": str(out),
                                  "image_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                                  "nfe_transformer": int(spent), "nfe_ctx": int(ctx.nfe),
                                  "seconds": round(time.time() - t0, 1),
                                  "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}) + "\n"); rec.flush()
            print(f"[{a.arm}] {pid} hps={bf['vec'][SEL]:.4f} ir={bf['vec']['imagereward']:+.4f} from {bf['kind']} seed {bf['seed']} | "
                  f"attempts={attempts} finished={len(finished)} {kinds} abandoned={n_abandon}/{n_ask_a} "
                  f"expand_asks={n_ask_b} nfe={spent}/{budget} jev={n_jev} vlm={n_vlm} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
