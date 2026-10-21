#!/usr/bin/env python3
"""Online deliberate prompt+CFG search on base SD3.5-Large: Jev-decided key-step MCTS whose WORDING BANK GROWS mid-trajectory.

Difference from `jev_prune` (scripts/vlm_mcts.py) -- and the only difference, so the contrast isolates it:
at the EARLY key steps (depth <= --online_depth, i.e. steps 0 and 4), a newly created node also
  1. grades the ORIGINAL prompt's checklist on its x0 preview (observe-then-grade VLM + per-element Jev verification),
  2. asks the VLM for K typed edits of THAT NODE's current wording aimed at its missing/wrong elements
     (operators emphasize | reorder | clarify | drop_clause | revert; original vocabulary only),
  3. filters them: lexical retention (no element of the original dropped) + Jev faithfulness guard (no content added),
  4. ENCODES the survivors and appends them to this prompt's embedding bank -- so later nodes can also pick them.
Every node (any depth) then asks Jev to rank all wordings in the bank and all CFG levels; its candidate set is
top-nv wordings x top-nc CFGs plus the baseline anchor, exactly as in jev_prune. Deeper nodes never add wordings
(composition is committed by then), they only re-rank what exists.
Search is otherwise untouched: UCB1 c=1.41, untried-first, uniform rollouts over the leaf's kept set, terminal
ImageReward against the ORIGINAL prompt, best-rollout vs exploit-replay return.
"""
from __future__ import annotations
import argparse, base64, csv, hashlib, io, json, os, sys, time, urllib.request
from pathlib import Path
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import opt_prompt_arm as opa          # checklist / readout / propose / guard / lexical_check / post / b64
import state_store as ss             # PromptState (per-prompt memory) + WordingBase (cross-prompt priors)

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
                print(f"{log_prefix} reward call failed ({type(e).__name__}: {str(e)[:80]}); retry {i+1}/{_tries}", flush=True); _t.sleep(_sleep)
        raise RuntimeError("reward server unreachable")
    su.score_image = _retry

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default=None); p.add_argument("--ids", required=True); p.add_argument("--run_dir", required=True)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--seed", type=int, default=42); p.add_argument("--n_sims", type=int, default=40)
    p.add_argument("--key_steps", default="0,4,8,12,16,20,24,27"); p.add_argument("--steps", type=int, default=28); p.add_argument("--ucb_c", type=float, default=1.41)
    p.add_argument("--n_variants_keep", type=int, default=2); p.add_argument("--n_cfg_keep", type=int, default=3)
    p.add_argument("--online_depth", type=int, default=1, help="generate new wordings at nodes of depth <= this (0=root only, 1=root+step4)")
    p.add_argument("--k_edits", type=int, default=3); p.add_argument("--guard_thresh", type=float, default=0.7)
    p.add_argument("--max_wordings", type=int, default=12, help="hard cap on the per-prompt wording bank")
    p.add_argument("--reward_backend", default="imagereward"); p.add_argument("--reward_server", required=True)
    p.add_argument("--vlm_server", default="http://127.0.0.1:5151"); p.add_argument("--jev_server", default="http://127.0.0.1:5161")
    p.add_argument("--preview_px", type=int, default=896)
    p.add_argument("--readout_depth", type=int, default=None,
                   help="grade the checklist on previews at nodes of depth <= this (default: --online_depth). Set it ONE DEEPER than "
                        "--online_depth to enable cross-prompt edit attribution: a parent and its child must BOTH have readouts for "
                        "the base to learn which operator fixed which element type.")
    p.add_argument("--use_state", action="store_true", help="give Jev the per-prompt trajectory memory and the cross-prompt wording base")
    p.add_argument("--wording_base", default=None, help="path to the cross-prompt base JSONL (default <run_dir>/../../state/wording_base.jsonl)")
    p.add_argument("--freeze_base", action="store_true", help="read the base but never write to it (use when evaluating on a held-out split)")
    a = p.parse_args(); a.arm = a.arm or Path(a.run_dir).name
    if a.readout_depth is None: a.readout_depth = a.online_depth
    os.environ.update({"HF_HUB_CACHE": "/data/ygu/.cache/hub", "HF_HOME": "/data/ygu/.cache/hub", "HF_HUB_OFFLINE": "1",
                       "OFFLOAD_TEXT_ENCODER_AFTER_ENCODE": "1", "REWARD_SERVER_URL": a.reward_server, "HPSV3_SCORE_STYLE": "raw"})
    sys.path.insert(0, SID); os.chdir(SID)
    import torch
    import sampling_unified_sd35 as su
    _install_retry(su, f"[{a.arm}]")
    for url in (a.vlm_server, a.jev_server):
        h = json.loads(urllib.request.urlopen(url + "/health", timeout=10).read()); assert h.get("status") == "ok", (url, h)
    sargs = su.normalize_paths(su.parse_args([
        "--backend", "sd35_base", "--search_method", "mcts", "--reward_backend", a.reward_backend, "--rewrites_file", CACHE, "--no_qwen",
        "--n_variants", "3", "--n_sims", str(a.n_sims), "--ucb_c", str(a.ucb_c), "--steps", str(a.steps), "--seed", str(a.seed),
        "--width", "1024", "--height", "1024", "--out_dir", a.run_dir, "--cfg_scales", *CFG_BANK, "--baseline_cfg", BASELINE_CFG,
        "--keystep_strategy", "explicit", "--mcts_key_steps", a.key_steps]))
    assert getattr(sargs, "euler_sampler", False) and sargs.model_id == "stabilityai/stable-diffusion-3.5-large"
    k = a.n_variants_keep * a.n_cfg_keep
    run_dir = Path(a.run_dir); (run_dir / "previews").mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in open(a.ids) if l.strip()]; ids = ids[: a.limit] if a.limit else ids
    pmap = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(PMAP))}; cache = json.load(open(sargs.rewrites_file))
    recp = run_dir / "records.jsonl"; decp = run_dir / "decisions.jsonl"
    done = {json.loads(l)["prompt_id"] for l in open(recp)} if recp.exists() else set()
    todo = [i for i in ids if i not in done]; print(f"[{a.arm}] {len(todo)} prompts to do ({len(done)} done)", flush=True)
    if not todo: return
    ctx = su.load_pipeline(sargs); rm = su.load_reward_model(sargs, ctx.device)
    key_steps = sorted(set(int(x) for x in a.key_steps.split(",") if x.strip().isdigit()))
    if 0 not in key_steps: key_steps = [0] + key_steps
    n_key = len(key_steps); segs = [(key_steps[i], key_steps[i + 1] if i + 1 < n_key else a.steps) for i in range(n_key)]
    (run_dir / "config.json").write_text(json.dumps({
        "arm": a.arm, "backend": "sd35_base", "model_id": sargs.model_id, "dtype": str(sargs.dtype), "steps": a.steps, "key_steps": key_steps,
        "n_sims": a.n_sims, "ucb_c": a.ucb_c, "cfg_scales": sargs.cfg_scales, "baseline_cfg": sargs.baseline_cfg, "seed": a.seed,
        "reward_backend": a.reward_backend, "reward": rm.describe(), "preview_px": a.preview_px,
        "online": {"depth": a.online_depth, "readout_depth": a.readout_depth, "k_edits": a.k_edits, "guard_thresh": a.guard_thresh, "max_wordings": a.max_wordings,
                   "operators": opa.OPERATORS, "filters": "lexical retention (no original element dropped) + Jev faithfulness guard (no content added)",
                   "encode": "survivors encoded mid-trajectory and appended to this prompt's embedding bank"},
        "prune": {"k": k, "n_variants_keep": a.n_variants_keep, "n_cfg_keep": a.n_cfg_keep, "anchor_added": True,
                  "rollout_policy": "uniform over the leaf's kept set at all remaining key steps"},
        "vlm": json.loads(urllib.request.urlopen(a.vlm_server + "/health", timeout=10).read()),
        "jev": json.loads(urllib.request.urlopen(a.jev_server + "/health", timeout=10).read()),
        "rewrites_sha256": hashlib.sha256(open(sargs.rewrites_file, "rb").read()).hexdigest()}, indent=1))
    BASE = None
    if a.use_state:
        bp = a.wording_base or f"{Path(a.run_dir).parent.parent}/state/wording_base.jsonl"
        Path(bp).parent.mkdir(parents=True, exist_ok=True); BASE = ss.WordingBase(bp, frozen=a.freeze_base)
        print(f"[{a.arm}] state ON: wording base {bp} ({len(BASE.rows)} rows, frozen={a.freeze_base})", flush=True)
    use_euler = True; anchor_cfg = float(sargs.baseline_cfg)
    with torch.no_grad(), recp.open("a") as rec, decp.open("a") as dec:
        for pid in todo:
            prompt = pmap[pid]; t0 = time.time(); ctx.nfe = 0; ctx.correction_nfe = 0; torch.cuda.reset_peak_memory_stats()
            np.random.seed(a.seed + 7919 * (ids.index(pid) + 1))
            variants = su.generate_variants(sargs, prompt, cache)      # 4 cached wordings: the same starting bank as jev_prune
            emb = su.encode_variants(ctx, variants)
            wordings = list(variants)                                   # index i of `wordings` == variant index i in emb
            try:
                elements, s_ = opa.extract_checklist(a.vlm_server, prompt)
            except Exception as exc:
                print(f"[{a.arm}] {pid} SKIPPED: checklist unparsable after retries ({str(exc)[:90]})", flush=True); continue
            if not elements:
                print(f"[{a.arm}] {pid} SKIPPED: empty checklist", flush=True); continue
            n_vlm = 1; vlm_sec = s_; n_jev = 0; jev_sec = 0.0; n_enc = 0; enc_sec = 0.0
            print(f"[{a.arm}] {pid} checklist: " + " | ".join(f'{e["id"]}:{e["text"][:38]}' for e in elements), flush=True)
            def bank_actions():
                return [(vi, float(c), 0.0) for vi in range(len(wordings)) for c in sargs.cfg_scales]
            anchor = (0, anchor_cfg, 0.0)
            latents0 = su.make_latents(ctx, a.seed, sargs.height, sargs.width, emb.cond_text[0].dtype); dx0 = torch.zeros_like(latents0)
            sched = su.step_schedule(ctx.device, latents0.dtype, a.steps, getattr(sargs, "sigmas", None), euler=use_euler)
            fresh = su._resolve_mcts_fresh_noise_steps(sargs, a.steps, key_steps=key_steps)
            root = su.MCTSNode(0, dx0, latents0); kept: dict[int, list[tuple]] = {}; node_wording: dict[int, int] = {root.node_id: 0}; dec_rows = []
            node_parent: dict[int, int] = {}; wording_op: dict[int, str] = {}
            PS = ss.PromptState(pid, prompt, wordings) if a.use_state else None
            def prune(node, path_actions):
                nonlocal n_vlm, vlm_sec, n_jev, jev_sec, n_enc, enc_sec, wordings, emb
                if node.node_id in kept: return kept[node.node_id]
                step = key_steps[node.step] if node.step < n_key else a.steps
                prev_b64 = None; prev_path = None; img = None
                if node.step > 0:
                    img = su.decode_to_pil(ctx, node.dx); prev_b64 = opa.b64(img, a.preview_px)
                    prev_path = run_dir / "previews" / f"{pid}_n{node.node_id}_d{node.step}_s{step}.png"; img.resize((512, 512)).save(prev_path)
                cur_wi = node_wording.get(node.node_id, 0); cur_w = wordings[cur_wi]
                statuses = None; obs = ""; new_info = {"proposed": [], "accepted": [], "n_encoded": 0}
                if img is not None and node.step <= a.readout_depth:
                    statuses, s_, obs = opa.readout(a.vlm_server, prompt, elements, img, jev=a.jev_server, seed=a.seed + 13 * node.node_id)
                    n_vlm += 1; n_jev += len(elements); vlm_sec += s_
                elif node.step <= a.online_depth:
                    statuses = [{"id": e["id"], "status": "unknown", "note": "", "vlm_status": "unknown", "jev_p_yes": None} for e in elements]
                if node.step <= a.online_depth and len(wordings) < a.max_wordings:
                    if statuses is None:
                        statuses = [{"id": e["id"], "status": "unknown", "note": "", "vlm_status": "unknown", "jev_p_yes": None} for e in elements]
                    guide = ""
                    if BASE is not None:
                        bad_types = {e["type"] for e, st_ in zip(elements, statuses) if st_["status"] in ("missing", "wrong")}
                        guide = BASE.render(element_types=bad_types or None, max_lines=4)
                    cands, s_ = opa.propose(a.vlm_server, prompt, cur_w, elements, statuses, a.k_edits, guidance=guide); n_vlm += 1; vlm_sec += s_
                    acc = []
                    for c in cands:
                        if not c.get("lexical_ok", True): c["guard_p_yes"] = None; c["passed"] = False; continue
                        if c["wording"] in wordings: c["guard_p_yes"] = None; c["passed"] = False; c["lexical_reason"] = "already in bank"; continue
                        gp, gs, ms = opa.guard(a.jev_server, prompt, c["wording"], seed=a.seed + node.node_id); n_jev += 1; jev_sec += ms / 1000
                        c["guard_p_yes"] = gp; c["guard_spread"] = gs; c["passed"] = gp >= a.guard_thresh
                        if c["passed"] and len(wordings) + len(acc) < a.max_wordings: acc.append(c)
                    new_info["proposed"] = [{kk: c.get(kk) for kk in ("wording", "operator", "lexical_ok", "lexical_reason", "guard_p_yes", "passed")} for c in cands]
                    if acc:
                        te = time.time(); e2 = su.encode_variants(ctx, [c["wording"] for c in acc]); enc_sec += time.time() - te; n_enc += len(acc)
                        for j, c in enumerate(acc):
                            emb.cond_text.append(e2.cond_text[j]); emb.cond_pooled.append(e2.cond_pooled[j]); wordings.append(c["wording"])
                            wording_op[len(wordings) - 1] = c.get("operator", "?")
                        new_info["accepted"] = [c["wording"] for c in acc]; new_info["n_encoded"] = len(acc)
                        print(f"[{a.arm}] {pid} node {node.node_id} d{node.step}: +{len(acc)} wordings (bank {len(wordings)})", flush=True)
                if PS is not None and statuses is not None and statuses[0]["status"] != "unknown":
                    PS.record_status(node.node_id, cur_wi, statuses)
                    par = node_parent.get(node.node_id)
                    if par is not None and PS.node_wording.get(par) is not None and PS.node_wording[par] != cur_wi:
                        eo = PS.edit_outcome(par, node.node_id, wording_op.get(cur_wi, "?"))
                        if eo and BASE is not None:
                            ty = {e["id"]: e["type"] for e in elements}
                            BASE.add(pid, eo["operator"], [ty.get(i, "?") for i in eo["fixed"]], [ty.get(i, "?") for i in eo["broke"]],
                                     wordings[PS.node_wording[par]], cur_w)
                # Jev ranks the whole bank (wordings) and the CFG levels for this node
                vopts = []; seen_v = {}
                for i, w in enumerate(wordings): vopts.append(w if w not in seen_v else f"{w} (#{i})"); seen_v[w] = 1
                state_core = (f'ORIGINAL PROMPT (the only target): "{prompt}"\nDenoising step {step} of {a.steps}.\nCurrent wording at this node: "{cur_w}"\n'
                              + (f"VLM readout of the attached preview: {obs}\n" if obs else "")
                              + ("Checklist status: " + "; ".join(f'{e["text"]}={s["status"]}' for e, s in zip(elements, statuses)) + "\n" if statuses and statuses[0]["status"] != "unknown" else "")
                              + "The sampler will apply the chosen wording and guidance for the next denoising steps; the FINAL image must match the ORIGINAL prompt.")
                if PS is not None:
                    state_core += "\n\n" + PS.render(cur_wi, path_actions)
                    gb = BASE.render(max_lines=4) if BASE is not None else ""
                    if gb: state_core += "\n" + gb
                q1 = opa.post(a.jev_server + "/ask", {"state": state_core, "question": "Which wording should the sampler follow for the next steps so that the final image best matches the ORIGINAL prompt?",
                                                      "options": vopts, "image_b64": prev_b64, "orders": 3, "seed": node.node_id, "tag": f"{pid}/online-variant/d{node.step}"})
                copts = [f"CFG {c:g}" + (" (lowest guidance: most natural, weakest prompt adherence)" if c == min(sargs.cfg_scales) else " (highest guidance: strongest adherence, most saturation and artifacts)" if c == max(sargs.cfg_scales) else "") for c in sargs.cfg_scales]
                q2 = opa.post(a.jev_server + "/ask", {"state": state_core, "question": "Which classifier-free guidance (CFG) strength should the sampler use for the next steps?",
                                                      "options": copts, "image_b64": prev_b64, "orders": 3, "seed": node.node_id, "tag": f"{pid}/online-cfg/d{node.step}"})
                n_jev += 2; jev_sec += (q1["ms"] + q2["ms"]) / 1000
                vr = sorted(range(len(wordings)), key=lambda i: -q1["probs"][i]); cr = sorted(range(len(sargs.cfg_scales)), key=lambda i: -q2["probs"][i])
                keep_v = set(vr[: a.n_variants_keep]); keep_c = {float(sargs.cfg_scales[j]) for j in cr[: a.n_cfg_keep]}
                acts = [(vi, c, 0.0) for vi, c, _ in bank_actions() if vi in keep_v and float(c) in keep_c]
                if anchor not in acts: acts.append(anchor)
                kept[node.node_id] = acts
                dec_rows.append({"prompt_id": pid, "arm": a.arm, "node_id": node.node_id, "depth": node.step, "step": step,
                                 "path_actions": [list(x) for x in path_actions], "current_wording": cur_w, "preview": str(prev_path) if prev_path else None,
                                 "observations": obs or None, "statuses": statuses, "online": new_info, "bank_size": len(wordings),
                                 "jev_wording_probs": q1["probs"], "jev_wording_spread": q1["spread"], "jev_cfg_probs": q2["probs"], "jev_cfg_spread": q2["spread"],
                                 "wordings_best_first": [wordings[i][:120] for i in vr], "cfg_best_first": [float(sargs.cfg_scales[j]) for j in cr],
                                 "keep_actions": [list(x) for x in acts]})
                return acts
            best_score = -float("inf"); best_lat = None; best_path = []
            for sim in range(a.n_sims):
                node = root; path = []; action = None
                while not node.is_leaf(n_key):
                    acts = prune(node, [x for _, x in path]); untried = node.untried_actions(acts)
                    if untried: action = untried[np.random.randint(len(untried))]; break
                    action = node.best_ucb(acts, a.ucb_c); path.append((node, action)); node = node.children[action]
                if not node.is_leaf(n_key):
                    if action not in node.children:
                        s0, s1 = segs[node.step]
                        lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, node.latents, node.dx, action, sched, s0, s1, noise_explore_steps=fresh, eps_bank=None)
                        child = su.MCTSNode(node.step + 1, dx, lat); node.children[action] = child; node_wording[child.node_id] = int(action[0]); node_parent[child.node_id] = node.node_id
                    path.append((node, action)); node = node.children[action]
                lat, dx, ki = node.latents, node.dx, node.step; roll = []
                roll_set = prune(node, [x for _, x in path]) if not node.is_leaf(n_key) else None
                while ki < n_key:
                    ra = roll_set[np.random.randint(len(roll_set))]; roll.append(ra); s0, s1 = segs[ki]
                    lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, ra, sched, s0, s1, noise_explore_steps=fresh, eps_bank=None); ki += 1
                img = su.decode_to_pil(ctx, su._final_decode_tensor(lat, dx, use_euler)); sc = su.score_image(rm, prompt, img)
                if PS is not None: PS.credit([x[0] for _, x in path] + [r[0] for r in roll], sc)
                if sc > best_score: best_score, best_lat, best_path = sc, su._final_decode_tensor(lat, dx, use_euler).clone(), [x for _, x in path] + roll
                for pn, pa in path:
                    pn.visits += 1; pn.action_visits[pa] = pn.action_visits.get(pa, 0) + 1; pn.action_values[pa] = pn.action_values.get(pa, 0.0) + sc
                if (sim + 1) % 10 == 0 or sim == 0:
                    print(f"    sim {sim+1:3d}/{a.n_sims} best={best_score:.4f} nodes={len(kept)} bank={len(wordings)} vlm={n_vlm} jev={n_jev}", flush=True)
            node = root; ex = []
            for ki in range(n_key):
                act = node.best_exploit(kept.get(node.node_id, bank_actions()))
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
            out = run_dir / f"{pid}_{a.arm}.png"; final_img.save(out)
            for d in dec_rows: dec.write(json.dumps(d) + "\n")
            dec.flush()
            depth_hist = {}
            for d in dec_rows: depth_hist[d["depth"]] = depth_hist.get(d["depth"], 0) + 1
            rec.write(json.dumps({"prompt_id": pid, "prompt": prompt, "variants": variants, "wordings": wordings, "elements": elements, "arm": a.arm, "seed": a.seed,
                                  "search_score": float(final_sc), "selected_from": src, "exploit_score": float(ex_sc), "best_rollout_score": float(best_score),
                                  "actions": [[int(v), float(c), float(s)] for v, c, s in final_path],
                                  "selected_wordings": [wordings[int(v)] for v, _, _ in final_path],
                                  "n_nodes": len(kept), "nodes_per_depth": depth_hist, "bank_size": len(wordings), "n_wordings_added": len(wordings) - len(variants),
                                  "n_vlm_calls": n_vlm, "vlm_seconds": round(vlm_sec, 1), "n_jev_calls": n_jev, "jev_seconds": round(jev_sec, 1),
                                  "n_encode_calls": n_enc, "encode_seconds": round(enc_sec, 1), "k": k,
                                  "use_state": a.use_state, "state_edits": (PS.edits if PS is not None else None),
                                  "wording_value": ({str(i): {"n": PS.n[i], "mean": round(PS.s[i] / PS.n[i], 4)} for i in range(len(wordings)) if PS.n[i]} if PS is not None else None),
                                  "image": str(out), "image_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "nfe_transformer": int(ctx.nfe),
                                  "seconds": round(time.time() - t0, 1), "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}) + "\n"); rec.flush()
            print(f"[{a.arm}] {pid} score={final_sc:.4f} ({src}) bank={len(wordings)} (+{len(wordings)-len(variants)}) nodes={len(kept)} vlm={n_vlm}/{vlm_sec:.0f}s jev={n_jev}/{jev_sec:.0f}s enc={n_enc}/{enc_sec:.0f}s nfe={ctx.nfe} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
