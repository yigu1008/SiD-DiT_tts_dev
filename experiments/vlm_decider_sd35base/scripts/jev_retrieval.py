#!/usr/bin/env python3
"""Retrieval-grounded Jev decisions: show it what nearby configs ACTUALLY produced, not what a setting might do.

Why: the state ablation showed Jev's CFG answer is a constant (withholding the image flips 0/8 choices, adding the
current CFG flips 1/8) while its PERCEPTION is excellent (sharp-vs-blurry 7/7 at P=0.999, description matching 3/3 at
P=1.000). So it answers matching questions from evidence and control questions from a lexical prior. This arm asks only
matching questions.

Per decision node:
  1. every completed rollout in this prompt's search is kept in a TRACE: action path, terminal image, ImageReward,
     and a VLM assessment of that final image against the ORIGINAL prompt (one /generate call per finished rollout);
  2. for the node's candidate actions, retrieve the nearest earlier trace (same wording, nearest CFG; tie-break on
     recency) and keep up to --panels distinct configs;
  3. tile the current x0 preview plus those retrieved FINAL images into one labelled contact sheet;
  4. the state names each panel: the (wording, CFG) that produced it, its VLM assessment, its reward;
  5. Jev is asked "which panel's FINAL image best matches the ORIGINAL prompt?" -- pure matching -- and the winning
     panel's config becomes the node's preferred action (kept set = that action plus its CFG neighbours and the anchor).
Before enough traces exist the node falls back to untried-first exploration over the full bank, so early simulations
build the evidence the later ones consume.
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

def post(url, body, timeout=600, tries=60, sleep=10):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as e:
            print(f"[post] {url} failed ({type(e).__name__}: {str(e)[:70]}); retry {i+1}/{tries}", flush=True); time.sleep(sleep)
    raise RuntimeError(f"{url} unreachable")

def contact_sheet(panels, px=448):
    """panels: list of (PIL image, caption). One row, captioned, for a single Jev image input."""
    H = px + 26
    sheet = Image.new("RGB", (px * len(panels), H), "white"); d = ImageDraw.Draw(sheet)
    for i, (im, cap) in enumerate(panels):
        sheet.paste(im.resize((px, px)), (i * px, 26)); d.text((i * px + 5, 7), cap[:62], fill="black")
    return sheet

def b64(img):
    buf = io.BytesIO(); img.save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default=None); p.add_argument("--ids", required=True); p.add_argument("--run_dir", required=True)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--seed", type=int, default=42); p.add_argument("--n_sims", type=int, default=10)
    p.add_argument("--key_steps", default="0,4,8,12,16,20,24,27"); p.add_argument("--steps", type=int, default=28); p.add_argument("--ucb_c", type=float, default=1.41)
    p.add_argument("--panels", type=int, default=3, help="retrieved outcome panels shown alongside the current preview")
    p.add_argument("--min_traces", type=int, default=2, help="below this many finished rollouts the node explores untried-first instead of asking")
    p.add_argument("--n_cfg_keep", type=int, default=3)
    p.add_argument("--reward_server", required=True); p.add_argument("--vlm_server", default="http://127.0.0.1:5151"); p.add_argument("--jev_server", default="http://127.0.0.1:5161")
    p.add_argument("--preview_px", type=int, default=448)
    a = p.parse_args(); a.arm = a.arm or Path(a.run_dir).name
    os.environ.update({"HF_HUB_CACHE": "/data/ygu/.cache/hub", "HF_HOME": "/data/ygu/.cache/hub", "HF_HUB_OFFLINE": "1",
                       "OFFLOAD_TEXT_ENCODER_AFTER_ENCODE": "1", "REWARD_SERVER_URL": a.reward_server, "HPSV3_SCORE_STYLE": "raw"})
    sys.path.insert(0, SID); os.chdir(SID)
    import torch
    import sampling_unified_sd35 as su
    for url in (a.vlm_server, a.jev_server):
        assert json.loads(urllib.request.urlopen(url + "/health", timeout=10).read()).get("status") == "ok", url
    sargs = su.normalize_paths(su.parse_args([
        "--backend", "sd35_base", "--search_method", "mcts", "--reward_backend", "imagereward", "--rewrites_file", CACHE, "--no_qwen",
        "--n_variants", "3", "--n_sims", str(a.n_sims), "--ucb_c", str(a.ucb_c), "--steps", str(a.steps), "--seed", str(a.seed),
        "--width", "1024", "--height", "1024", "--out_dir", a.run_dir, "--cfg_scales", *CFG_BANK, "--baseline_cfg", BASELINE_CFG,
        "--keystep_strategy", "explicit", "--mcts_key_steps", a.key_steps]))
    run_dir = Path(a.run_dir); (run_dir / "sheets").mkdir(parents=True, exist_ok=True); (run_dir / "traces").mkdir(exist_ok=True)
    ids = [l.strip() for l in open(a.ids) if l.strip()]; ids = ids[: a.limit] if a.limit else ids
    pmap = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(PMAP))}; cache = json.load(open(sargs.rewrites_file))
    recp = run_dir / "records.jsonl"; decp = run_dir / "decisions.jsonl"
    done = {json.loads(l)["prompt_id"] for l in open(recp)} if recp.exists() else set()
    todo = [i for i in ids if i not in done]; print(f"[{a.arm}] {len(todo)} prompts to do ({len(done)} done)", flush=True)
    if not todo: return
    ctx = su.load_pipeline(sargs); rm = su.load_reward_model(sargs, ctx.device)
    key_steps = sorted(set(int(x) for x in a.key_steps.split(",") if x.strip().isdigit()))
    if 0 not in key_steps: key_steps = [0] + key_steps
    n_key = len(key_steps); segs = [(key_steps[i], key_steps[i+1] if i+1 < n_key else a.steps) for i in range(n_key)]
    (run_dir / "config.json").write_text(json.dumps({
        "arm": a.arm, "backend": "sd35_base", "steps": a.steps, "key_steps": key_steps, "n_sims": a.n_sims, "ucb_c": a.ucb_c,
        "cfg_scales": sargs.cfg_scales, "baseline_cfg": sargs.baseline_cfg, "seed": a.seed, "panels": a.panels, "min_traces": a.min_traces,
        "design": "retrieval-grounded: Jev is shown the FINAL images of earlier rollouts, one per DISTINCT wording (best-scoring), "
                  "each captioned with its (wording, CFG), its end-of-trace VLM assessment and its reward, and asked which panel's "
                  "final image best matches the ORIGINAL prompt - a matching question, which the ablation showed is the only kind it "
                  "answers from evidence. The winning panel's config becomes the node's preferred action.",
        "reward": rm.describe(), "vlm": json.loads(urllib.request.urlopen(a.vlm_server + "/health", timeout=10).read()),
        "jev": json.loads(urllib.request.urlopen(a.jev_server + "/health", timeout=10).read())}, indent=1))
    use_euler = True; anchor_cfg = float(sargs.baseline_cfg)
    with torch.no_grad(), recp.open("a") as rec, decp.open("a") as dec:
        for pid in todo:
            prompt = pmap[pid]; t0 = time.time(); ctx.nfe = 0; torch.cuda.reset_peak_memory_stats()
            np.random.seed(a.seed + 7919 * (ids.index(pid) + 1))
            variants = su.generate_variants(sargs, prompt, cache); emb = su.encode_variants(ctx, variants)
            bank = [(vi, float(c), 0.0) for vi in range(len(emb.cond_text)) for c in sargs.cfg_scales]
            anchor = (0, anchor_cfg, 0.0)
            lat0 = su.make_latents(ctx, a.seed, sargs.height, sargs.width, emb.cond_text[0].dtype); dx0 = torch.zeros_like(lat0)
            sched = su.step_schedule(ctx.device, lat0.dtype, a.steps, getattr(sargs, "sigmas", None), euler=use_euler)
            fresh = su._resolve_mcts_fresh_noise_steps(sargs, a.steps, key_steps=key_steps)
            root = su.MCTSNode(0, dx0, lat0); kept = {}; dec_rows = []; traces = []   # trace: dict(actions, image, score, feedback)
            n_vlm = n_jev = 0; vlm_sec = jev_sec = 0.0
            def retrieve(cands):
                """Panels must be VISUALLY DIFFERENT alternatives. The first version took the top-3 traces by score,
                which returned the SAME wording at adjacent CFGs - near-identical images, so the comparison carried no
                information. Now: one panel per DISTINCT wording (that wording's best-scoring trace), then, if slots
                remain, traces whose mean CFG is far from everything already shown."""
                by_w = {}
                for t in traces:
                    for w in {int(v) for v, _, _ in t["actions"]}:
                        cf = float(np.mean([c for v, c, _ in t["actions"] if int(v) == w]))
                        if w not in by_w or t["score"] > by_w[w][0]["score"]: by_w[w] = (t, (w, cf))
                picks = [(k, t) for w, (t, k) in sorted(by_w.items(), key=lambda kv: -kv[1][0]["score"])]
                if len(picks) < a.panels:
                    shown = {k for k, _ in picks}
                    for t in sorted(traces, key=lambda x: -x["score"]):
                        w0 = int(t["actions"][0][0]); cf = float(np.mean([c for _, c, _ in t["actions"]]))
                        if any(w0 == kw and abs(cf - kc) < 0.75 for kw, kc in shown): continue
                        picks.append(((w0, cf), t)); shown.add((w0, cf))
                        if len(picks) >= a.panels: break
                return [(k, t, k) for k, t in picks[: a.panels]]

            def decide(node, path_actions):
                nonlocal n_vlm, n_jev, vlm_sec, jev_sec
                if node.node_id in kept: return kept[node.node_id]
                step = key_steps[node.step] if node.step < n_key else a.steps
                cands = list(bank)
                if len(traces) < a.min_traces:                       # not enough evidence yet: explore the full bank
                    kept[node.node_id] = cands
                    dec_rows.append({"prompt_id": pid, "node_id": node.node_id, "depth": node.step, "step": step,
                                     "mode": "explore (insufficient traces)", "n_traces": len(traces)})
                    return cands
                hits = retrieve(cands)
                if len(hits) < 2:
                    kept[node.node_id] = cands
                    dec_rows.append({"prompt_id": pid, "node_id": node.node_id, "depth": node.step, "step": step,
                                     "mode": "explore (no retrieval)", "n_traces": len(traces)})
                    return cands
                cur = su.decode_to_pil(ctx, node.dx) if node.step > 0 else Image.new("RGB", (448, 448), (235, 235, 235))
                panels = [(cur, f"NOW: step {step}/{a.steps} preview")]
                lines = []
                for i, (cfgkey, tr, cand) in enumerate(hits):
                    panels.append((Image.open(tr["image"]).convert("RGB"), f"[{i+1}] wording {chr(65+cfgkey[0])} @ CFG {cfgkey[1]:g}"))
                    lines.append(f'  [{i+1}] wording {chr(65+cfgkey[0])} at CFG {cfgkey[1]:g} -> reward {tr["score"]:.3f}; assessment: {tr["feedback"][:170]}')
                sheet = contact_sheet(panels, a.preview_px)
                sp = run_dir / "sheets" / f"{pid}_n{node.node_id}_d{node.step}.png"; sheet.save(sp)
                state = (f'ORIGINAL PROMPT (the only target): "{prompt}"\n'
                         f'The attached strip: the leftmost panel is the sampler\'s CURRENT x0 preview at step {step} of {a.steps}; '
                         f'the remaining panels are FINAL images already produced in this search, each by the setting named above it.\n'
                         f'What those settings produced:\n' + "\n".join(lines) + "\n"
                         f'Judge the FINAL images only; the current preview is shown for context and is blurry because sampling is unfinished.')
                r = post(a.jev_server + "/ask", {"state": state, "question": "Which numbered panel's FINAL image best matches the ORIGINAL prompt?",
                                                 "options": [f"Panel {i+1}" for i in range(len(hits))], "image_b64": b64(sheet),
                                                 "orders": 3, "seed": node.node_id, "tag": f"{pid}/retrieval/d{node.step}"})
                n_jev += 1; jev_sec += r["ms"] / 1000
                win_v, win_c = hits[int(r["argmax"])][0]
                cfgs = sorted(sargs.cfg_scales, key=lambda c: abs(float(c) - win_c))[: a.n_cfg_keep]
                acts = [(win_v, float(c), 0.0) for c in cfgs]
                if anchor not in acts: acts.append(anchor)
                kept[node.node_id] = acts
                dec_rows.append({"prompt_id": pid, "node_id": node.node_id, "depth": node.step, "step": step, "mode": "retrieval",
                                 "n_traces": len(traces), "sheet": str(sp), "panels": [{"wording": k[0], "cfg": k[1], "score": t["score"],
                                 "feedback": t["feedback"][:220]} for k, t, _ in hits], "jev_probs": r["probs"], "jev_spread": r["spread"],
                                 "chosen_panel": int(r["argmax"]) + 1, "keep_actions": [list(x) for x in acts]})
                return acts
            best_score = -float("inf"); best_lat = None; best_path = []
            for sim in range(a.n_sims):
                node = root; path = []; action = None
                while not node.is_leaf(n_key):
                    acts = decide(node, [x for _, x in path]); untried = node.untried_actions(acts)
                    if untried: action = untried[np.random.randint(len(untried))]; break
                    action = node.best_ucb(acts, a.ucb_c); path.append((node, action)); node = node.children[action]
                if not node.is_leaf(n_key):
                    if action not in node.children:
                        s0, s1 = segs[node.step]
                        lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, node.latents, node.dx, action, sched, s0, s1, noise_explore_steps=fresh, eps_bank=None)
                        node.children[action] = su.MCTSNode(node.step + 1, dx, lat)
                    path.append((node, action)); node = node.children[action]
                lat, dx, ki = node.latents, node.dx, node.step; roll = []
                rs = decide(node, [x for _, x in path]) if not node.is_leaf(n_key) else None
                while ki < n_key:
                    ra = rs[np.random.randint(len(rs))]; roll.append(ra); s0, s1 = segs[ki]
                    lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, ra, sched, s0, s1, noise_explore_steps=fresh, eps_bank=None); ki += 1
                img = su.decode_to_pil(ctx, su._final_decode_tensor(lat, dx, use_euler)); sc = su.score_image(rm, prompt, img)
                # END-OF-TRACE VLM assessment of the finished image, stored with the trace
                tp = run_dir / "traces" / f"{pid}_sim{sim}.png"; img.save(tp)
                fb = post(a.vlm_server + "/generate", {"text": f'ORIGINAL PROMPT: "{prompt}"\n\nThis is a FINISHED image. In at most two sentences state what matches the prompt and what is missing, wrong or distorted (mention oversaturation or artifacts if present). Do not suggest settings.',
                                                       "images_b64": [opa.b64(img, 512)], "max_new_tokens": 110})
                n_vlm += 1; vlm_sec += fb["seconds"]
                traces.append({"actions": [list(x) for x in ([y for _, y in path] + roll)], "image": str(tp), "score": float(sc), "feedback": fb["text"].strip()})
                for pn, pa in path:
                    pn.visits += 1; pn.action_visits[pa] = pn.action_visits.get(pa, 0) + 1; pn.action_values[pa] = pn.action_values.get(pa, 0.0) + sc
                if sc > best_score: best_score, best_lat, best_path = sc, su._final_decode_tensor(lat, dx, use_euler).clone(), [y for _, y in path] + roll
                print(f"    sim {sim+1:3d}/{a.n_sims} best={best_score:.4f} traces={len(traces)} jev={n_jev} vlm={n_vlm}", flush=True)
            node = root; ex = []
            for ki in range(n_key):
                act = node.best_exploit(kept.get(node.node_id, bank))
                if act is None: break
                ex.append(act)
                if act in node.children: node = node.children[act]
                else: break
            lat, dx = lat0, dx0
            for ki, act in enumerate(ex): s0, s1 = segs[ki]; lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, act, sched, s0, s1, noise_explore_steps=fresh, eps_bank=None)
            for ki in range(len(ex), n_key): s0, s1 = segs[ki]; lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, anchor, sched, s0, s1, noise_explore_steps=fresh, eps_bank=None); ex.append(anchor)
            ex_img = su.decode_to_pil(ctx, su._final_decode_tensor(lat, dx, use_euler)); ex_sc = su.score_image(rm, prompt, ex_img)
            final_img, final_sc, final_path, src = (ex_img, ex_sc, ex, "exploit") if ex_sc >= best_score else (su.decode_to_pil(ctx, best_lat), best_score, best_path, "best_rollout")
            out = run_dir / f"{pid}_{a.arm}.png"; final_img.save(out)
            for d_ in dec_rows: dec.write(json.dumps(d_) + "\n")
            dec.flush()
            modes = {}
            for d_ in dec_rows: modes[d_["mode"]] = modes.get(d_["mode"], 0) + 1
            rec.write(json.dumps({"prompt_id": pid, "prompt": prompt, "variants": variants, "arm": a.arm, "seed": a.seed,
                                  "search_score": float(final_sc), "selected_from": src, "actions": [[int(v), float(c), float(s)] for v, c, s in final_path],
                                  "n_nodes": len(kept), "decision_modes": modes, "n_traces": len(traces), "n_jev_calls": n_jev, "jev_seconds": round(jev_sec, 1),
                                  "n_vlm_calls": n_vlm, "vlm_seconds": round(vlm_sec, 1), "image": str(out),
                                  "image_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "nfe_transformer": int(ctx.nfe),
                                  "seconds": round(time.time() - t0, 1), "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}) + "\n"); rec.flush()
            print(f"[{a.arm}] {pid} score={final_sc:.4f} ({src}) modes={modes} traces={len(traces)} jev={n_jev}/{jev_sec:.0f}s vlm={n_vlm}/{vlm_sec:.0f}s nfe={ctx.nfe} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
