#!/usr/bin/env python3
"""VLM-pruned key-step MCTS on base SD3.5-Large (28-step Euler), plus a random-prune control.

Identical to the stock arm (base_arm.py --arm mcts) in everything except the per-node candidate set:
  stock      : every tree node offers the full 28-action bank (variants x CFG); rollouts uniform over the bank.
  vlm_prune  : when a node is CREATED, its x0 preview (node.dx decoded; none at the root) and the ORIGINAL prompt
               are sent to the VLM decider (/decide), which comments and ranks variants and CFG levels; the node's
               candidate set = top-nv variants x top-nc CFGs (k = nv*nc) plus the baseline anchor (variant 0, baseline_cfg).
               Rollouts from a leaf sample uniformly from the LEAF's kept set at every remaining key step.
  rand_prune : same pruning geometry, but the kept set is k uniform-random bank actions (seeded), no VLM.
Search: UCB1 (c), untried-first random, best-of-rollouts return vs exploit replay, ImageReward terminal reward
against the ORIGINAL prompt.  Every VLM decision is logged to decisions.jsonl with the preview PNG (512 px) so the
(state, action set, comment, choice, outcome) tuples form a dataset.
"""
from __future__ import annotations
import argparse, base64, csv, hashlib, io, json, math, os, sys, time, urllib.request
from pathlib import Path
import numpy as np

SID = "/home/ygu/project/SiD-DiT_tts_dev"
CACHE = "/data/ygu/human_eval_genai40_v1/sid_vqascore_algorithm_sweep/genai200_v1/shared_rewrites_cache.json"
PMAP = "/data/ygu/reward_orthogonality/mcts/prompts/prompt_map.csv"
CFG_BANK = ["3.5", "4.0", "4.5", "5.0", "5.5", "6.0", "7.0"]; BASELINE_CFG = "4.5"

def post(url, body, timeout=600, tries=90, sleep=10):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as e:
            print(f"[post] {url} failed ({type(e).__name__}: {str(e)[:80]}); retry {i+1}/{tries} in {sleep}s", flush=True); time.sleep(sleep)
    raise RuntimeError(f"{url} unreachable")

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

def _puct_select(node, actions, prior, c_puct):
    """Q + c_puct * P * sqrt(N) / (1 + n). Untried actions get Q=0 and are ordered by their prior."""
    import math as _m
    N = max(1, int(node.visits))
    best, best_s = None, -float("inf")
    for a in actions:
        n = node.action_visits.get(a, 0)
        q = (node.action_values.get(a, 0.0) / n) if n else 0.0
        s_ = q + c_puct * float(prior.get(a, 1.0 / max(1, len(actions)))) * _m.sqrt(N) / (1.0 + n)
        if s_ > best_s: best, best_s = a, s_
    return best

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["vlm_prune", "rand_prune", "jev_prune"]); p.add_argument("--ids", required=True); p.add_argument("--run_dir", required=True)
    # --arm selects BEHAVIOUR; --arm_name is what gets recorded (default: run_dir basename) so two
    # runs of the same behaviour cannot collide in the shared scores.csv.
    p.add_argument("--arm_name", default=None)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--seed", type=int, default=42); p.add_argument("--n_sims", type=int, default=40)
    p.add_argument("--key_steps", default="0,4,8,12,16,20,24,27"); p.add_argument("--steps", type=int, default=28); p.add_argument("--ucb_c", type=float, default=1.41)
    p.add_argument("--n_variants_keep", type=int, default=2); p.add_argument("--n_cfg_keep", type=int, default=3)
    p.add_argument("--reward_backend", default="imagereward"); p.add_argument("--reward_server", required=True); p.add_argument("--vlm_server", default="http://127.0.0.1:5151")
    p.add_argument("--preview_px", type=int, default=512)
    p.add_argument("--puct", action="store_true",
                   help="SOFT prior instead of hard pruning: keep ALL bank actions and select by "
                        "Q(a) + c_puct*P(a)*sqrt(N)/(1+n(a)), with P = Jev's joint probability over the full bank. "
                        "Rationale: at 10 sims, repairing HOW the kept set is chosen (joint vs factorised) changed "
                        "nothing, so the deficit comes from deleting options at all; a prior reorders exploration "
                        "without discarding coverage.")
    p.add_argument("--c_puct", type=float, default=1.5)
    p.add_argument("--joint", action="store_true",
                   help="ask Jev ONE joint question over all (wording, CFG) pairs instead of two marginal questions. "
                        "Measured on real nodes: the best CFG differs across wordings at 8/8 nodes, the factorised "
                        "top-nv x top-nc set misses ~46%% of the joint top-k, and the marginal CFG question returns a "
                        "degenerate constant (7.0 at 33/33 nodes) that the joint question does not."); p.add_argument("--jev_server", default="http://127.0.0.1:5161"); p.add_argument("--jev_orders", type=int, default=3)
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
        "--n_variants", "3", "--n_sims", str(a.n_sims), "--ucb_c", str(a.ucb_c), "--steps", str(a.steps), "--seed", str(a.seed),
        "--width", "1024", "--height", "1024", "--out_dir", a.run_dir, "--cfg_scales", *CFG_BANK, "--baseline_cfg", BASELINE_CFG,
        "--keystep_strategy", "explicit", "--mcts_key_steps", a.key_steps]))
    assert getattr(sargs, "euler_sampler", False) and sargs.model_id == "stabilityai/stable-diffusion-3.5-large"
    k = a.n_variants_keep * a.n_cfg_keep
    if a.arm in ("vlm_prune", "jev_prune"):
        h = json.loads(urllib.request.urlopen(a.vlm_server + "/health", timeout=10).read()); assert h.get("status") == "ok", h
    if a.arm == "jev_prune":
        h = json.loads(urllib.request.urlopen(a.jev_server + "/health", timeout=10).read()); assert h.get("status") == "ok", h
    run_dir = Path(a.run_dir); (run_dir / "previews").mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    if a.limit: ids = ids[: a.limit]
    pmap = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(PMAP))}; cache = json.load(open(sargs.rewrites_file))
    recp = run_dir / "records.jsonl"; decp = run_dir / "decisions.jsonl"
    done = {json.loads(l)["prompt_id"] for l in open(recp)} if recp.exists() else set()
    todo = [i for i in ids if i not in done]; print(f"[{a.arm}] {len(todo)} prompts to do ({len(done)} done)", flush=True)
    if not todo: return
    ctx = su.load_pipeline(sargs); rm = su.load_reward_model(sargs, ctx.device)
    key_steps = sorted(set(int(x) for x in a.key_steps.split(",") if x.strip().isdigit())); key_steps = ([0] if 0 not in key_steps else []) + key_steps
    n_key = len(key_steps); segs = [(key_steps[i], key_steps[i + 1] if i + 1 < n_key else a.steps) for i in range(n_key)]
    (run_dir / "config.json").write_text(json.dumps({
        "arm": a.arm_name, "behaviour": a.arm, "backend": "sd35_base", "model_id": sargs.model_id, "dtype": str(sargs.dtype), "steps": a.steps, "key_steps": key_steps, "n_sims": a.n_sims, "ucb_c": a.ucb_c,
        "cfg_scales": sargs.cfg_scales, "baseline_cfg": sargs.baseline_cfg, "n_variants": sargs.n_variants, "reward_backend": a.reward_backend, "reward": rm.describe(), "seed": a.seed,
        "prune": {"k": k, "n_variants_keep": a.n_variants_keep, "n_cfg_keep": a.n_cfg_keep, "anchor_added": True, "when": "node creation (root included)",
                  "rollout_policy": "uniform over the leaf's kept set at all remaining key steps"},
        "vlm": (json.loads(urllib.request.urlopen(a.vlm_server + "/health", timeout=10).read()) if a.arm in ("vlm_prune", "jev_prune") else None),
        "jev": (json.loads(urllib.request.urlopen(a.jev_server + "/health", timeout=10).read()) if a.arm == "jev_prune" else None), "jev_orders": a.jev_orders, "preview_px": a.preview_px, "joint": a.joint, "puct": a.puct, "c_puct": a.c_puct,
        "jev_pipeline": "VLM /generate perception-only readout of the preview (no suggestions) -> Jev-Omni choice over the 4 prompt wordings and choice over the 7 CFG levels (state = original prompt + step + readout; image attached when depth>0); keep = top-2 wordings x top-3 CFGs by Jev probability + anchor" if a.arm == "jev_prune" else None,
        "rewrites_sha256": hashlib.sha256(open(sargs.rewrites_file, "rb").read()).hexdigest()}, indent=1))
    use_euler = True; anchor = (0, float(sargs.baseline_cfg), 0.0)
    with torch.no_grad(), recp.open("a") as rec, decp.open("a") as dec:
        for pi, pid in enumerate(todo):
            prompt = pmap[pid]; t0 = time.time(); ctx.nfe = 0; ctx.correction_nfe = 0; STEPS['steps'] = 0; torch.cuda.reset_peak_memory_stats()
            np.random.seed(a.seed + 7919 * (ids.index(pid) + 1))  # tree RNG per prompt (stock uses unseeded global np RNG)
            variants = su.generate_variants(sargs, prompt, cache); emb = su.encode_variants(ctx, variants)
            bank = [(vi, float(c), 0.0) for vi in range(len(emb.cond_text)) for c in sargs.cfg_scales]
            cand_meta = [{"id": i, "variant": variants[vi] if vi < len(variants) else f"variant{vi}", "cfg": c} for i, (vi, c, _) in enumerate(bank)]
            latents0 = su.make_latents(ctx, a.seed, sargs.height, sargs.width, emb.cond_text[0].dtype); dx0 = torch.zeros_like(latents0)
            sched = su.step_schedule(ctx.device, latents0.dtype, a.steps, getattr(sargs, "sigmas", None), euler=use_euler)
            fresh = su._resolve_mcts_fresh_noise_steps(sargs, a.steps, key_steps=key_steps)
            root = su.MCTSNode(0, dx0, latents0); kept: dict[int, list[tuple]] = {}; n_vlm = 0; vlm_sec = 0.0; dec_rows = []; node_prior: dict[int, dict] = {}
            def prune(node, path_actions):
                nonlocal n_vlm, vlm_sec
                if node.node_id in kept: return kept[node.node_id]
                step = key_steps[node.step] if node.step < n_key else a.steps; prev_b64 = None; prev_path = None
                if node.step > 0:
                    img = su.decode_to_pil(ctx, node.dx); im = img.resize((a.preview_px, a.preview_px)); buf = io.BytesIO(); im.save(buf, format="PNG")
                    prev_b64 = base64.b64encode(buf.getvalue()).decode(); prev_path = run_dir / "previews" / f"{pid}_n{node.node_id}_d{node.step}_s{step}.png"; im.save(prev_path)
                if a.arm == "vlm_prune":
                    r = post(a.vlm_server + "/decide", {"prompt": prompt, "step": step, "total_steps": a.steps, "preview_b64": prev_b64, "candidates": cand_meta,
                                                       "k": k, "n_variants_keep": a.n_variants_keep, "n_cfg_keep": a.n_cfg_keep, "salt": node.node_id})
                    keep_ids = list(r["keep"]); n_vlm += 1; vlm_sec += float(r["seconds"])
                elif a.arm == "jev_prune":
                    if prev_b64:
                        rd = post(a.vlm_server + "/generate", {"text": f'ORIGINAL PROMPT: "{prompt}"\n\nThis is a diffusion sampler\'s x0 preview at denoising step {step}/{a.steps} (early previews are blurry; describe content and composition, not sharpness). In at most two sentences, state what in the image already matches the ORIGINAL prompt and what is missing, wrong, or at risk. Do NOT suggest settings.',
                                                               "images_b64": [prev_b64], "max_new_tokens": 120})
                        readout = rd["text"].strip(); vlm_sec += float(rd["seconds"]); n_vlm += 1
                    else:
                        readout = "(no preview yet: sampling has not started)"
                    state = (f'ORIGINAL PROMPT (the only target): "{prompt}"\nDenoising step {step} of {a.steps}.\nVLM readout of the current preview: {readout}\n'
                             f"The sampler will apply the chosen setting for the next denoising steps; the FINAL image must match the ORIGINAL prompt.")
                    if a.joint:
                        # ONE question over all (wording, CFG) pairs. Measured on real nodes: the best CFG differs across
                        # wordings at 8/8, the factorised top-nv x top-nc set misses ~46 % of the joint top-k, and the
                        # isolated CFG question returns a degenerate constant (7.0 at 33/33) that this one does not.
                        jopts = [f'wording {chr(65 + int(vi))}: "{variants[int(vi)] if int(vi) < len(variants) else vi}" | CFG {float(c):g}' for vi, c, _ in bank]
                        assert len(set(jopts)) == len(jopts), "joint option strings must be unique"
                        qj = post(a.jev_server + "/ask", {"state": state, "question": "Which setting should the sampler use for the next steps so the final image best matches the ORIGINAL prompt?",
                                                          "options": jopts, "image_b64": prev_b64, "orders": a.jev_orders, "seed": node.node_id, "tag": f"{pid}/joint/d{node.step}"})
                        order = sorted(range(len(bank)), key=lambda i: -qj["probs"][i])
                        keep_ids = list(range(len(bank))) if a.puct else order[:k]   # PUCT keeps everything; the prior does the work
                        if a.puct: node_prior[node.node_id] = {bank[i]: float(qj["probs"][i]) for i in range(len(bank))}
                        r = {"comment": readout, "parsed": True, "seconds": round(qj["ms"] / 1000, 2),
                             "joint_best_first": order[:10], "joint_probs": qj["probs"], "joint_spread": qj["spread"],
                             "variants_best_first": None, "cfg_best_first": None}
                        n_vlm += 1; vlm_sec += r["seconds"]
                    else:
                        vopts = []; seen_v = {}
                        for v in variants: vopts.append(v if v not in seen_v else f"{v} (variant {len(vopts)})"); seen_v[v] = 1
                        q1 = post(a.jev_server + "/ask", {"state": state, "question": "Which prompt wording should the sampler follow for the next steps so that the final image best matches the ORIGINAL prompt?",
                                                          "options": vopts, "image_b64": prev_b64, "orders": a.jev_orders, "seed": node.node_id, "tag": f"{pid}/variant/d{node.step}"})
                        copts = [f"CFG {c:g}" + (" (lowest guidance: most natural, weakest prompt adherence)" if c == min(sargs.cfg_scales) else " (highest guidance: strongest adherence, most saturation and artifacts)" if c == max(sargs.cfg_scales) else "") for c in sargs.cfg_scales]
                        q2 = post(a.jev_server + "/ask", {"state": state, "question": "Which classifier-free guidance (CFG) strength should the sampler use for the next steps?",
                                                          "options": copts, "image_b64": prev_b64, "orders": a.jev_orders, "seed": node.node_id, "tag": f"{pid}/cfg/d{node.step}"})
                        vr = sorted(range(len(variants)), key=lambda i: -q1["probs"][i]); cr = sorted(range(len(sargs.cfg_scales)), key=lambda i: -q2["probs"][i])
                        keep_ids = [i for i, (vi, c, _) in enumerate(bank) if vi in vr[:a.n_variants_keep] and float(c) in {float(sargs.cfg_scales[j]) for j in cr[:a.n_cfg_keep]}]
                        r = {"comment": readout, "parsed": True, "seconds": round((q1["ms"] + q2["ms"]) / 1000, 2), "variants_best_first": vr,
                             "cfg_best_first": [float(sargs.cfg_scales[j]) for j in cr],
                             "jev_variant_probs": q1["probs"], "jev_variant_spread": q1["spread"], "jev_cfg_probs": q2["probs"], "jev_cfg_spread": q2["spread"]}
                        n_vlm += 2; vlm_sec += r["seconds"]
                else:
                    keep_ids = sorted(np.random.choice(len(bank), size=k, replace=False).tolist()); r = {"comment": None, "parsed": None, "seconds": 0.0}
                acts = [bank[i] for i in keep_ids]
                if anchor not in acts: acts.append(anchor)
                kept[node.node_id] = acts
                dec_rows.append({"prompt_id": pid, "arm": a.arm_name, "node_id": node.node_id, "depth": node.step, "step": step, "path_actions": [list(x) for x in path_actions],
                                 "preview": str(prev_path) if prev_path else None, "comment": r.get("comment"), "variants_best_first": r.get("variants_best_first"),
                                 "cfg_best_first": r.get("cfg_best_first"), "variant_texts_best_first": r.get("variant_texts_best_first"), "jev": {kk: r[kk] for kk in r if kk.startswith("jev_")} or None, "keep_ids": keep_ids, "keep_actions": [list(x) for x in acts], "parsed": r.get("parsed"), "vlm_seconds": r.get("seconds")})
                return acts
            best_score = -float("inf"); best_lat = None; best_path = []
            for sim in range(a.n_sims):
                node = root; path = []; action = None
                while not node.is_leaf(n_key):
                    acts = prune(node, [x for _, x in path])
                    if a.puct:
                        action = _puct_select(node, acts, node_prior.get(node.node_id, {}), a.c_puct)
                        if action not in node.children: break          # expand it below
                        path.append((node, action)); node = node.children[action]; continue
                    untried = node.untried_actions(acts)
                    if untried: action = untried[np.random.randint(len(untried))]; break
                    action = node.best_ucb(acts, a.ucb_c); path.append((node, action)); node = node.children[action]
                if not node.is_leaf(n_key):
                    if action not in node.children:
                        s0, s1 = segs[node.step]
                        lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, node.latents, node.dx, action, sched, s0, s1, noise_explore_steps=fresh, eps_bank=None)
                        node.children[action] = su.MCTSNode(node.step + 1, dx, lat)
                    path.append((node, action)); node = node.children[action]
                lat, dx, ki = node.latents, node.dx, node.step; roll = []
                roll_set = prune(node, [x for _, x in path]) if not node.is_leaf(n_key) else None
                while ki < n_key:
                    if a.puct and roll_set:
                        pr = np.array([node_prior.get(node.node_id, {}).get(x, 1.0) for x in roll_set], dtype=float); pr = pr / pr.sum()
                        ra = roll_set[int(np.random.choice(len(roll_set), p=pr))]
                    else:
                        ra = roll_set[np.random.randint(len(roll_set))]; roll.append(ra); s0, s1 = segs[ki]
                    lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, ra, sched, s0, s1, noise_explore_steps=fresh, eps_bank=None); ki += 1
                img = su.decode_to_pil(ctx, su._final_decode_tensor(lat, dx, use_euler)); sc = su.score_image(rm, prompt, img)
                if sc > best_score: best_score, best_lat, best_path = sc, su._final_decode_tensor(lat, dx, use_euler).clone(), [x for _, x in path] + roll
                for pn, pa in path:
                    pn.visits += 1; pn.action_visits[pa] = pn.action_visits.get(pa, 0) + 1; pn.action_values[pa] = pn.action_values.get(pa, 0.0) + sc
                if (sim + 1) % 10 == 0 or sim == 0: print(f"    sim {sim+1:3d}/{a.n_sims} best={best_score:.4f} nodes={len(kept)} vlm_calls={n_vlm}", flush=True)
            # exploit replay
            node = root; ex = []
            for ki in range(n_key):
                act = node.best_exploit(kept.get(node.node_id, bank))
                if act is None: break
                ex.append(act)
                if act in node.children: node = node.children[act]
                else: break
            lat, dx = latents0, dx0
            for ki, act in enumerate(ex): s0, s1 = segs[ki]; lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, act, sched, s0, s1, noise_explore_steps=fresh, eps_bank=None)
            for ki in range(len(ex), n_key): s0, s1 = segs[ki]; lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, anchor, sched, s0, s1, noise_explore_steps=fresh, eps_bank=None); ex.append(anchor)
            ex_img = su.decode_to_pil(ctx, su._final_decode_tensor(lat, dx, use_euler)); ex_sc = su.score_image(rm, prompt, ex_img)
            if ex_sc >= best_score: final_img, final_sc, final_path, src = ex_img, ex_sc, ex, "exploit"
            else: final_img, final_sc, final_path, src = su.decode_to_pil(ctx, best_lat), best_score, best_path, "best_rollout"
            out = run_dir / f"{pid}_{a.arm_name}.png"; final_img.save(out)
            for d in dec_rows: dec.write(json.dumps(d) + "\n")
            dec.flush()
            depth_hist = {}
            for d in dec_rows: depth_hist[d["depth"]] = depth_hist.get(d["depth"], 0) + 1
            rec.write(json.dumps({"prompt_id": pid, "prompt": prompt, "variants": variants, "arm": a.arm_name, "behaviour": a.arm, "seed": a.seed, "search_score": float(final_sc), "selected_from": src,
                                  "exploit_score": float(ex_sc), "best_rollout_score": float(best_score), "actions": [[int(v), float(c), float(s)] for v, c, s in final_path],
                                  "n_nodes": len(kept), "nodes_per_depth": depth_hist, "n_vlm_calls": n_vlm, "vlm_seconds": round(vlm_sec, 1), "k": k,
                                  "image": str(out), "image_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "nfe_transformer": int(ctx.nfe), "segment_steps": int(STEPS["steps"]),
                                  "seconds": round(time.time() - t0, 1), "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}) + "\n"); rec.flush()
            print(f"[{a.arm}] {pid} score={final_sc:.4f} ({src}) nodes={len(kept)} depth_hist={depth_hist} vlm={n_vlm} calls/{vlm_sec:.0f}s nfe={ctx.nfe} steps={STEPS['steps']} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
