#!/usr/bin/env python3
"""Multi-reward key-step MCTS where Jev chooses WHICH REWARD each node follows, by matching the
VLM's observed deficiency to each reward's FOCUSED AREA.

Why no normalization.  The obvious way to mix rewards is to min-max them onto a common scale and
average (what reward_unified's composite_* backends do with _COMPOSITE_*_RANGE).  That is arbitrary
and it makes UCB1 backups non-stationary the moment a node changes objective.  This arm never
compares two rewards numerically.  Instead:

  * every finished rollout is scored under ALL THREE backends and its score VECTOR is accumulated
    at every node on its path (three running sums per action, not one);
  * each node selects one reward and reads ONLY that component for UCB1 and for exploitation.
    A node's statistics are therefore always in a single reward's native units, and re-deciding a
    node's reward costs nothing and corrupts nothing -- the other two columns were being kept anyway;
  * the prompt-level choice (which image to return) uses the reward the ROOT settled on.

How the reward is chosen.  Each reward is given a FOCUSED AREA: the class of deficiency it can
actually see (REWARD_FOCUS below).  The state handed to Jev is the end-of-trace VLM assessments of
the finished images -- free-text statements of what is wrong -- and the question is which deficiency
class is the limiting problem.  That is a MATCHING question over evidence, the only kind the state
ablation showed Jev answers from the image rather than from a lexical prior (CFG-policy questions
flipped 0/8 when the image was withheld; perception questions were 7/7 and 3/3 correct).

VQAScore is deliberately NOT one of the search rewards: it stays a fully held-out judge.
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

# ---------------------------------------------------------------------------
# The focused area of each search reward: what deficiency it is able to see.
# Phrased as a deficiency class so it can be matched against a VLM's free-text
# complaint about a finished image.  Full text is sent as the option string --
# the server rejects non-unique options, and truncating these collapses them.
# ---------------------------------------------------------------------------
REWARD_FOCUS = {
    "imagereward": (
        "CONTENT AND FAITHFULNESS: something the prompt explicitly asks for is absent, duplicated, "
        "miscounted, in the wrong place or relation, or rendered as broken anatomy or unreadable text. "
        "The complaint is about WHAT IS IN the picture."),
    "hpsv3": (
        "APPEARANCE AND CRAFT: everything the prompt asks for is present, but the picture itself looks "
        "bad - oversaturated or garish colour, blown contrast, plastic or waxy surfaces, flat or "
        "inconsistent lighting, mushy texture, visible noise or compression artifacts. "
        "The complaint is about HOW the picture looks, not what it contains."),
    "pickscore": (
        "KIND AND FRAMING: the picture is clean and contains the right things, but it is the wrong sort "
        "of picture for this prompt - wrong genre, medium, period or register, or a subject that is "
        "cropped, too distant, too cluttered or awkwardly composed to read as what was asked for. "
        "The complaint is about WHICH PICTURE was made, not its content or its finish."),
}
SHORT = {"imagereward": "content/faithfulness", "hpsv3": "appearance/craft", "pickscore": "kind/framing"}
SEARCH_REWARDS = ["imagereward", "hpsv3", "pickscore"]


def post(url, body, timeout=600, tries=60, sleep=10):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as e:
            print(f"[post] {url} failed ({type(e).__name__}: {str(e)[:70]}); retry {i+1}/{tries}", flush=True); time.sleep(sleep)
    raise RuntimeError(f"{url} unreachable")


def contact_sheet(panels, px=448):
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
    p.add_argument("--panels", type=int, default=3); p.add_argument("--min_traces", type=int, default=2); p.add_argument("--n_cfg_keep", type=int, default=3)
    p.add_argument("--reward_policy", default="jev", choices=["jev"] + SEARCH_REWARDS,
                   help="jev = Jev matches the VLM-observed deficiency to a reward focus area at each node; "
                        "a backend name = that reward everywhere (matched control, identical machinery)")
    p.add_argument("--default_reward", default="imagereward", choices=SEARCH_REWARDS,
                   help="reward a node uses before any evidence exists")
    p.add_argument("--reward_asks", type=int, default=1, help="max Jev reward-selection calls per node")
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
    rh = json.loads(urllib.request.urlopen(a.reward_server + "/health", timeout=10).read())
    missing = [b for b in SEARCH_REWARDS if b not in rh.get("backends", [])]
    assert not missing, f"reward server lacks backends {missing}; has {rh.get('backends')}"

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
        "cfg_scales": sargs.cfg_scales, "baseline_cfg": sargs.baseline_cfg, "seed": a.seed, "panels": a.panels,
        "min_traces": a.min_traces, "reward_policy": a.reward_policy, "default_reward": a.default_reward,
        "reward_asks": a.reward_asks, "search_rewards": SEARCH_REWARDS, "reward_focus": REWARD_FOCUS,
        "design": "Each reward is given a FOCUSED AREA (a deficiency class it can see). Every rollout is scored under all "
                  "three search rewards and its score VECTOR is accumulated at every node on its path, so each node holds "
                  "three independent running sums per action. A node selects one reward -- Jev matches the end-of-trace VLM "
                  "complaints against the focus statements -- and reads only that component for UCB1 and exploitation. No "
                  "min-max normalization and no cross-reward arithmetic anywhere; a node's statistics are always in one "
                  "reward's native units. The returned image is chosen under the reward the ROOT settled on. VQAScore is "
                  "held out of the search entirely.",
        "reward_server": rh, "vlm": json.loads(urllib.request.urlopen(a.vlm_server + "/health", timeout=10).read()),
        "jev": json.loads(urllib.request.urlopen(a.jev_server + "/health", timeout=10).read())}, indent=1))
    use_euler = True; anchor_cfg = float(sargs.baseline_cfg)

    def score_vec(prompt, img):
        """All three search rewards for one image, native scales, one encode."""
        enc = b64(img); out = {}
        for bk in SEARCH_REWARDS:
            out[bk] = float(post(a.reward_server + "/score", {"prompt": prompt, "image_b64": enc, "backend": bk})["score"])
        return out

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
            root = su.MCTSNode(0, dx0, lat0)
            kept: dict = {}                       # node_id -> action set (cached: NFE must stay comparable)
            acc: dict = {}                        # node_id -> action -> {reward: running sum}; visits in nvis
            nvis: dict = {}; navis: dict = {}     # node_id -> visits ; node_id -> action -> visits
            node_reward: dict = {}; node_asks: dict = {}
            dec_rows = []; traces = []
            n_vlm = n_jev = n_rask = 0; vlm_sec = jev_sec = 0.0

            def retrieve():
                """One panel per DISTINCT wording (that wording's best trace under the node's reward is not
                used here -- panels are evidence, ranked by the default reward only for ordering)."""
                by_w = {}
                for t in traces:
                    for w in {int(v) for v, _, _ in t["actions"]}:
                        cf = float(np.mean([c for v, c, _ in t["actions"] if int(v) == w]))
                        if w not in by_w or t["vec"][a.default_reward] > by_w[w][0]["vec"][a.default_reward]:
                            by_w[w] = (t, (w, cf))
                picks = [(k, t) for w, (t, k) in sorted(by_w.items(), key=lambda kv: -kv[1][0]["vec"][a.default_reward])]
                if len(picks) < a.panels:
                    shown = {k for k, _ in picks}
                    for t in sorted(traces, key=lambda x: -x["vec"][a.default_reward]):
                        w0 = int(t["actions"][0][0]); cf = float(np.mean([c for _, c, _ in t["actions"]]))
                        if any(w0 == kw and abs(cf - kc) < 0.75 for kw, kc in shown): continue
                        picks.append(((w0, cf), t)); shown.add((w0, cf))
                        if len(picks) >= a.panels: break
                return picks[: a.panels]

            def reward_for(node, step):
                """Which reward does this node follow?  Jev matches the VLM's complaints about the finished
                images against each reward's focused area.  Asked at most --reward_asks times per node."""
                nonlocal n_jev, jev_sec, n_rask
                nid = node.node_id
                if a.reward_policy != "jev": return a.reward_policy
                if nid in node_reward and node_asks.get(nid, 0) >= a.reward_asks: return node_reward[nid]
                if len(traces) < a.min_traces: return node_reward.setdefault(nid, a.default_reward)
                ev = []
                for t in sorted(traces, key=lambda x: -x["vec"][a.default_reward])[:5]:
                    w = int(t["actions"][0][0]); cf = float(np.mean([c for _, c, _ in t["actions"]]))
                    ev.append(f'  - image from wording {chr(65+w)} at CFG ~{cf:.1f}: {t["feedback"]}')
                state = (f'ORIGINAL PROMPT (the only target): "{prompt}"\n'
                         f'This search has finished {len(traces)} images for this prompt. A vision model was asked, for each '
                         f'finished image, what matches the prompt and what is missing, wrong or distorted. Its reports:\n'
                         + "\n".join(ev) + "\n\n"
                         f'The search is at step {step} of {a.steps} and must now pick which single quality to optimise for the '
                         f'rest of this prompt. Each option below names one class of deficiency. Choose the class that the '
                         f'reports above show is the LIMITING problem for this prompt - the one most worth fixing next. '
                         f'Do not choose a class the reports give no evidence for.')
                opts = [REWARD_FOCUS[b] for b in SEARCH_REWARDS]
                r = post(a.jev_server + "/ask", {"state": state, "question": "Which class of deficiency is the limiting problem for this prompt?",
                                                 "options": opts, "orders": 3, "seed": 1000 + nid, "tag": f"{pid}/rewardsel/d{node.step}"})
                n_jev += 1; n_rask += 1; jev_sec += r["ms"] / 1000
                pick = SEARCH_REWARDS[int(r["argmax"])]
                node_reward[nid] = pick; node_asks[nid] = node_asks.get(nid, 0) + 1
                dec_rows.append({"prompt_id": pid, "node_id": nid, "depth": node.step, "step": step, "kind": "reward_select",
                                 "n_traces": len(traces), "chosen_reward": pick, "jev_probs": r["probs"], "jev_spread": r["spread"],
                                 "evidence": ev})
                return pick

            def decide(node, rw):
                """Action set for this node: the retrieval contact-sheet question, now asked ON the node's
                chosen focused area rather than on generic prompt match."""
                nonlocal n_jev, jev_sec
                if node.node_id in kept: return kept[node.node_id]
                step = key_steps[node.step] if node.step < n_key else a.steps
                cands = list(bank)
                if len(traces) < a.min_traces:
                    kept[node.node_id] = cands
                    dec_rows.append({"prompt_id": pid, "node_id": node.node_id, "depth": node.step, "step": step,
                                     "kind": "action", "mode": "explore (insufficient traces)", "n_traces": len(traces), "reward": rw})
                    return cands
                hits = retrieve()
                if len(hits) < 2:
                    kept[node.node_id] = cands
                    dec_rows.append({"prompt_id": pid, "node_id": node.node_id, "depth": node.step, "step": step,
                                     "kind": "action", "mode": "explore (no retrieval)", "n_traces": len(traces), "reward": rw})
                    return cands
                cur = su.decode_to_pil(ctx, node.dx) if node.step > 0 else Image.new("RGB", (448, 448), (235, 235, 235))
                panels = [(cur, f"NOW: step {step}/{a.steps} preview")]; lines = []
                for i, (k, tr) in enumerate(hits):
                    panels.append((Image.open(tr["image"]).convert("RGB"), f"[{i+1}] wording {chr(65+k[0])} @ CFG {k[1]:g}"))
                    lines.append(f'  [{i+1}] wording {chr(65+k[0])} at CFG {k[1]:g}; assessment: {tr["feedback"]}')
                sheet = contact_sheet(panels, a.preview_px)
                sp = run_dir / "sheets" / f"{pid}_n{node.node_id}_d{node.step}.png"; sheet.save(sp)
                state = (f'ORIGINAL PROMPT (the only target): "{prompt}"\n'
                         f'The attached strip: the leftmost panel is the sampler\'s CURRENT x0 preview at step {step} of {a.steps}; '
                         f'the remaining panels are FINAL images already produced in this search by the settings named above them.\n'
                         f'What a vision model said about those finished images:\n' + "\n".join(lines) + "\n\n"
                         f'This node is optimising ONE quality, chosen from the evidence:\n  {REWARD_FOCUS[rw]}\n'
                         f'Judge the FINAL images on THAT quality alone. Ignore the current preview except as context; it is '
                         f'blurry because sampling is unfinished.')
                q = f"On that quality alone - avoiding {SHORT[rw]} problems - which numbered panel's FINAL image is best?"
                r = post(a.jev_server + "/ask", {"state": state, "question": q,
                                                 "options": [f"Panel {i+1}" for i in range(len(hits))], "image_b64": b64(sheet),
                                                 "orders": 3, "seed": node.node_id, "tag": f"{pid}/action/{rw}/d{node.step}"})
                n_jev += 1; jev_sec += r["ms"] / 1000
                win_v, win_c = hits[int(r["argmax"])][0]
                cfgs = sorted(sargs.cfg_scales, key=lambda c: abs(float(c) - win_c))[: a.n_cfg_keep]
                acts = [(win_v, float(c), 0.0) for c in cfgs]
                if anchor not in acts: acts.append(anchor)
                kept[node.node_id] = acts
                dec_rows.append({"prompt_id": pid, "node_id": node.node_id, "depth": node.step, "step": step, "kind": "action",
                                 "mode": "retrieval", "reward": rw, "n_traces": len(traces), "sheet": str(sp),
                                 "panels": [{"wording": k[0], "cfg": k[1], "vec": t["vec"], "feedback": t["feedback"][:220]} for k, t in hits],
                                 "jev_probs": r["probs"], "jev_spread": r["spread"], "chosen_panel": int(r["argmax"]) + 1,
                                 "keep_actions": [list(x) for x in acts]})
                return acts

            def ucb_select(node, acts, rw, c):
                """UCB1 read off the node's OWN reward column. All columns were accumulated, so switching
                reward never leaves a node reading values it did not collect."""
                nid = node.node_id; N = max(1, nvis.get(nid, 0)); av = acc.get(nid, {}); nv = navis.get(nid, {})
                best, best_s = None, -float("inf")
                for act in acts:
                    n = nv.get(act, 0)
                    if n == 0: return act
                    s_ = av[act][rw] / n + c * np.sqrt(np.log(N + 1.0) / n)
                    if s_ > best_s: best, best_s = act, s_
                return best if best is not None else acts[0]

            def exploit_select(node, acts, rw):
                nid = node.node_id; av = acc.get(nid, {}); nv = navis.get(nid, {})
                cand = [(av[x][rw] / nv[x], x) for x in acts if nv.get(x, 0) > 0]
                return max(cand)[1] if cand else None

            root_rw = a.default_reward if a.reward_policy == "jev" else a.reward_policy
            best_vec = None; best_lat = None; best_path = []; sims = []
            for sim in range(a.n_sims):
                node = root; path = []; action = None
                while not node.is_leaf(n_key):
                    step = key_steps[node.step] if node.step < n_key else a.steps
                    rw = reward_for(node, step)
                    if node.node_id == root.node_id: root_rw = rw
                    acts = decide(node, rw); untried = [x for x in acts if navis.get(node.node_id, {}).get(x, 0) == 0]
                    if untried: action = untried[np.random.randint(len(untried))]; break
                    action = ucb_select(node, acts, rw, a.ucb_c); path.append((node, action)); node = node.children[action]
                if not node.is_leaf(n_key):
                    if action not in node.children:
                        s0, s1 = segs[node.step]
                        lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, node.latents, node.dx, action, sched, s0, s1, noise_explore_steps=fresh, eps_bank=None)
                        node.children[action] = su.MCTSNode(node.step + 1, dx, lat)
                    path.append((node, action)); node = node.children[action]
                lat, dx, ki = node.latents, node.dx, node.step; roll = []
                if not node.is_leaf(n_key):
                    step = key_steps[node.step] if node.step < n_key else a.steps
                    rs = decide(node, reward_for(node, step))
                else: rs = None
                while ki < n_key:
                    ra = rs[np.random.randint(len(rs))]; roll.append(ra); s0, s1 = segs[ki]
                    lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, ra, sched, s0, s1, noise_explore_steps=fresh, eps_bank=None); ki += 1
                img = su.decode_to_pil(ctx, su._final_decode_tensor(lat, dx, use_euler))
                vec = score_vec(prompt, img)
                tp = run_dir / "traces" / f"{pid}_sim{sim}.png"; img.save(tp)
                fb = post(a.vlm_server + "/generate", {"text": f'ORIGINAL PROMPT: "{prompt}"\n\nThis is a FINISHED image. In at most two sentences state what matches the prompt and what is missing, wrong or distorted (mention oversaturation, plastic surfaces, flat lighting or artifacts if present). Do not suggest settings.',
                                                       "images_b64": [opa.b64(img, 512)], "max_new_tokens": 110})
                n_vlm += 1; vlm_sec += fb["seconds"]
                traces.append({"actions": [list(x) for x in ([y for _, y in path] + roll)], "image": str(tp),
                               "vec": vec, "feedback": fb["text"].strip()})
                # accumulate ALL THREE columns at every node on the path
                for pn, pa in path:
                    nid = pn.node_id; nvis[nid] = nvis.get(nid, 0) + 1
                    navis.setdefault(nid, {}); navis[nid][pa] = navis[nid].get(pa, 0) + 1
                    cell = acc.setdefault(nid, {}).setdefault(pa, {b: 0.0 for b in SEARCH_REWARDS})
                    for b in SEARCH_REWARDS: cell[b] += vec[b]
                sims.append({"sim": sim, "vec": vec, "path": [list(y) for _, y in path] + [list(x) for x in roll],
                             "lat": su._final_decode_tensor(lat, dx, use_euler).clone()})
                print(f"    sim {sim+1:3d}/{a.n_sims} ir={vec['imagereward']:.3f} hps={vec['hpsv3']:.3f} ps={vec['pickscore']:.3f} "
                      f"root_rw={root_rw} traces={len(traces)} jev={n_jev}(rsel {n_rask}) vlm={n_vlm}", flush=True)
            # exploitation path under the ROOT's reward
            node = root; ex = []
            for ki in range(n_key):
                act = exploit_select(node, kept.get(node.node_id, bank), root_rw)
                if act is None: break
                ex.append(act)
                if act in node.children: node = node.children[act]
                else: break
            lat, dx = lat0, dx0
            for ki, act in enumerate(ex): s0, s1 = segs[ki]; lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, act, sched, s0, s1, noise_explore_steps=fresh, eps_bank=None)
            for ki in range(len(ex), n_key): s0, s1 = segs[ki]; lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, anchor, sched, s0, s1, noise_explore_steps=fresh, eps_bank=None); ex.append(anchor)
            ex_img = su.decode_to_pil(ctx, su._final_decode_tensor(lat, dx, use_euler)); ex_vec = score_vec(prompt, ex_img)
            # selection under the root's reward -- a single criterion per prompt, never a mixture
            bs = max(sims, key=lambda s: s["vec"][root_rw])
            if ex_vec[root_rw] >= bs["vec"][root_rw]:
                final_img, final_vec, final_path, src = ex_img, ex_vec, ex, "exploit"
            else:
                final_img, final_vec, final_path, src = su.decode_to_pil(ctx, bs["lat"]), bs["vec"], bs["path"], "best_rollout"
            out = run_dir / f"{pid}_{a.arm}.png"; final_img.save(out)
            for d_ in dec_rows: dec.write(json.dumps(d_) + "\n")
            dec.flush()
            modes = {}; rwcount = {}
            for d_ in dec_rows:
                if d_["kind"] == "action": modes[d_["mode"]] = modes.get(d_["mode"], 0) + 1
                else: rwcount[d_["chosen_reward"]] = rwcount.get(d_["chosen_reward"], 0) + 1
            rec.write(json.dumps({"prompt_id": pid, "prompt": prompt, "variants": variants, "arm": a.arm, "seed": a.seed,
                                  "reward_policy": a.reward_policy, "root_reward": root_rw, "reward_votes": rwcount,
                                  "node_rewards": {str(k): v for k, v in node_reward.items()},
                                  "search_score": float(final_vec[root_rw]), "search_vec": final_vec,
                                  "selected_from": src, "actions": [[int(v), float(c), float(s)] for v, c, s in final_path],
                                  "n_nodes": len(kept), "decision_modes": modes, "n_traces": len(traces),
                                  "n_jev_calls": n_jev, "n_reward_asks": n_rask, "jev_seconds": round(jev_sec, 1),
                                  "n_vlm_calls": n_vlm, "vlm_seconds": round(vlm_sec, 1), "image": str(out),
                                  "image_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "nfe_transformer": int(ctx.nfe),
                                  "seconds": round(time.time() - t0, 1), "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}) + "\n"); rec.flush()
            print(f"[{a.arm}] {pid} root_rw={root_rw} votes={rwcount} ir={final_vec['imagereward']:.4f} "
                  f"hps={final_vec['hpsv3']:.4f} ps={final_vec['pickscore']:.4f} ({src}) modes={modes} "
                  f"jev={n_jev}/{jev_sec:.0f}s vlm={n_vlm}/{vlm_sec:.0f}s nfe={ctx.nfe} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
