#!/usr/bin/env python3
"""Decision engines that plug into actdiff (`run_mcts_lookahead`) through the opt-in `args.jev_decider` hooks.

Two engines, matching the two proposals, plus matched random controls:

  BankDecider   (#1) -- Jev updates the per-node ACTION BANK and the BACKUP REWARD from evidence.
      Every finished rollout is kept as a trace: action path, final image, 3-reward vector, VLM assessment.
      At a node with >= min_traces finished rollouts, Jev is shown a PERMUTED contact sheet of one finished image
      per distinct wording (ordered by rank3, not ImageReward) and asked which best matches the ORIGINAL prompt.
      The winning (wording, CFG) re-centres the node's cfg_bank and restricts admitted wordings to the winner plus
      the anchor; the adaptive width logic of actdiff is kept. Backup value = rank-sum of the three rewards among
      finished rollouts so far (scale-free: no min-max), optionally narrowed to a Jev-chosen focus reward.

  ProbeDecider  (#2) -- per-step PROBE TRACES: for the top-k candidates (by actdiff's own heuristic prior, forced
      to span distinct wordings) run ONE segment from this node, decode each x0 preview, get a VLM critique of
      each, then show Jev the permuted previews + critiques and ask which continuation best serves the ORIGINAL
      prompt. Jev's choice becomes the prior (smoothed one-hot). This is genuinely per-step and per-action: the
      state differs at every node because the probes are run FROM that node. Probe NFE is real and is counted by
      the sampler; the matched control (RandomProbeDecider) runs the SAME probes and picks at random, so the
      comparison isolates Jev's information at identical NFE. Per-candidate probe reward vectors are logged so the
      oracle ceiling (pick the probe whose preview scores best) is computable post hoc.

Lessons baked in from this study: sheet order permuted and logged; candidates ordered by rank3 not IR; bank width
must be delivered by the caller (--n_variants sized to the bank); every decision logged with probabilities.
"""
from __future__ import annotations
import base64, io, json, os, time, urllib.request
import numpy as np
from PIL import Image, ImageDraw

SR = ["imagereward", "hpsv3", "pickscore"]


def post(url, body, timeout=600, tries=40, sleep=8):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as e:
            print(f"[decider] {url} failed ({type(e).__name__}: {str(e)[:60]}); retry {i+1}/{tries}", flush=True); time.sleep(sleep)
    raise RuntimeError(f"{url} unreachable")


def b64(img, px=None):
    if px: img = img.resize((px, px))
    buf = io.BytesIO(); img.save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()


def sheet(panels, px=400):
    s = Image.new("RGB", (px * len(panels), px + 26), "white"); d = ImageDraw.Draw(s)
    for i, (im, cap) in enumerate(panels):
        s.paste(im.resize((px, px)), (i * px, 26)); d.text((i * px + 5, 7), cap[:54], fill="black")
    return s


def ranksum(cands, key):
    """rank-sum of the three rewards over a candidate list; scale-free."""
    tot = {id(c): 0 for c in cands}
    for m in SR:
        for pos, c in enumerate(sorted(cands, key=lambda x: key(x)[m])): tot[id(c)] += pos
    return tot


class _Base:
    def __init__(self, *, reward_server, vlm_server, jev_server, log_path, policy="jev", seed=0, min_traces=2, panels=3):
        self.rs, self.vs, self.js = reward_server, vlm_server, jev_server
        self.policy, self.min_traces, self.panels = policy, min_traces, panels
        self.rng = np.random.RandomState(seed); self.log_path = log_path
        self.traces = []; self.n_jev = self.n_vlm = 0; self.jev_s = self.vlm_s = 0.0; self.calls = 0
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.sheet_dir = os.path.join(os.path.dirname(log_path), "sheets"); os.makedirs(self.sheet_dir, exist_ok=True)

    def bind(self, **kw):
        self.k = kw; self.traces = []     # new prompt / new tree: reset traces
        self.prompt = kw["prompt"]; self.tag = f"{abs(hash(self.prompt)) % 10**6}_{kw['seed']}"

    def score_vec(self, img):
        enc = b64(img)
        return {m: float(post(self.rs + "/score", {"prompt": self.prompt, "image_b64": enc, "backend": m})["score"]) for m in SR}

    def assess(self, img):
        r = post(self.vs + "/generate", {"text": f'ORIGINAL PROMPT: "{self.prompt}"\n\nIn at most two sentences state what in this image matches the prompt and what is missing, wrong or distorted. Do not suggest settings.',
                                          "images_b64": [b64(img, 512)], "max_new_tokens": 100})
        self.n_vlm += 1; self.vlm_s += r["seconds"]; return r["text"].strip()

    def ask(self, state, question, options, img, tag):
        r = post(self.js + "/ask", {"state": state, "question": question, "options": options, "image_b64": b64(img),
                                    "orders": 3, "seed": 7000 + self.calls, "tag": tag})
        self.n_jev += 1; self.jev_s += r["ms"] / 1000; self.calls += 1; return r

    def log(self, row):
        row.update({"prompt": self.prompt, "seed": self.k["seed"], "t": time.time()})
        with open(self.log_path, "a") as f: f.write(json.dumps(row) + "\n")

    # defaults: no effect
    def adjust_candidates(self, node, actions, **meta): return actions
    def wants_prior(self, node): return False
    def action_prior(self, node, candidates, base_prior): return None, None
    def backup_score(self, img, score, path):
        vec = self.score_vec(img)
        self.traces.append({"path": [list(map(float, a)) for a in path], "vec": vec, "img": img, "ir": float(score),
                            "feedback": None})
        return float(score)


class BankDecider(_Base):
    """#1: Jev re-centres the CFG bank, restricts admitted wordings, and the backup is rank3 (not IR alone)."""
    def __init__(self, *, keep_cfg=3, focus_reward=False, **kw):
        super().__init__(**kw); self.keep_cfg = keep_cfg; self.focus_reward = focus_reward; self.node_choice = {}

    def backup_score(self, img, score, path):
        vec = self.score_vec(img); fb = self.assess(img) if self.policy == "jev" else ""
        self.traces.append({"path": [list(map(float, a)) for a in path], "vec": vec, "img": img, "ir": float(score), "feedback": fb})
        # rank-sum among finished rollouts so far -> scale-free backup in [0,1]
        tot = ranksum(self.traces, key=lambda t: t["vec"]); n = len(self.traces)
        val = tot[id(self.traces[-1])] / max(1.0, 3.0 * (n - 1))
        self.log({"kind": "backup", "n_traces": n, "vec": vec, "ir": float(score), "rank3_backup": val})
        return float(val)

    def adjust_candidates(self, node, actions, *, cfg_bank, center, width):
        if len(self.traces) < self.min_traces or self.policy == "none": return actions
        if getattr(self, "phase", "tree") != "tree": return actions
        nid = (int(node.step), id(node))
        if nid not in self.node_choice:
            # one panel per distinct wording, that wording's rank3-best trace
            # BUG FIX (2026-10-29): one panel per distinct TRACE (rank3-best first), labelled by its dominant wording. The old
            # per-wording map put the same trajectory under every wording it had used: 96% of ad_jevbank sheets were one image x3.
            tot = ranksum(self.traces, key=lambda t: t["vec"]); hits = []
            for t in sorted(self.traces, key=lambda t: -tot[id(t)]):
                ws = [int(a[0]) for a in t["path"]]; w = max(set(ws), key=lambda v: (ws.count(v), -ws.index(v)))
                cf = float(np.mean([a[1] for a in t["path"] if int(a[0]) == w]))
                if all(hw != w for hw, _ in hits) or len(self.traces) <= self.panels: hits.append((w, (t, cf)))
                if len(hits) == self.panels: break
            if len(hits) < 2: return actions
            if self.policy == "random":
                w, (t, cf) = hits[self.rng.randint(len(hits))]; probs = None; perm = list(range(len(hits)))
            else:
                perm = list(range(len(hits))); self.rng.shuffle(perm); hits = [hits[i] for i in perm]
                prev = self.k["decode"](self.k["ctx"], node.dx) if node.step > 0 else Image.new("RGB", (400, 400), (235, 235, 235))
                pans = [(prev, f"NOW step {self.k['key_steps'][min(node.step, self.k['n_key']-1)]}")] + \
                       [(t["img"], f"[{i+1}] w{w} @ CFG {cf:g}") for i, (w, (t, cf)) in enumerate(hits)]
                sh = sheet(pans); sp = os.path.join(self.sheet_dir, f"{self.tag}_bank_s{node.step}_{self.calls}.png"); sh.save(sp)
                opts = [f'Panel {i+1}: wording {chr(65+w)} at CFG {cf:g}. Assessment: {str(t["feedback"]).strip()[:240]}' for i, (w, (t, cf)) in enumerate(hits)]
                state = (f'ORIGINAL PROMPT (the only target): "{self.prompt}"\nImage: leftmost = the CURRENT unfinished preview (blur is normal); '
                         f'the numbered panels are FINISHED images from this search made with the settings in the options. '
                         f'The chosen setting defines the wording and guidance neighbourhood searched next from this point.')
                r = self.ask(state, "Which numbered panel's FINISHED image best matches the ORIGINAL prompt?", opts, sh, f"bank/s{node.step}")
                w, (t, cf) = hits[int(r["argmax"])]; probs = r["probs"]
            self.node_choice[nid] = (w, cf)
            self.log({"kind": "bank", "step": int(node.step), "n_traces": len(self.traces), "panels": [[ww, c] for ww, (tt, c) in hits],
                      "perm": perm, "chosen_w": w, "chosen_cfg": cf, "probs": probs, "old_center": center, "old_width": width})
        w, cf = self.node_choice[nid]
        cfgs = sorted(cfg_bank, key=lambda c: abs(float(c) - cf))[: self.keep_cfg]
        anchor_cfg = float(self.k["args"].baseline_cfg)
        keep = [a for a in actions if (int(a[0]) == w and float(a[1]) in cfgs) or (int(a[0]) == 0 and abs(float(a[1]) - anchor_cfg) < 1e-6)]
        return keep if len(keep) >= 2 else actions


class ProbeDecider(_Base):
    """#2: per-step probe traces for the top-k candidates, VLM critique each, Jev chooses -> prior."""
    def __init__(self, *, probe_k=3, smooth=0.15, **kw):
        super().__init__(**kw); self.probe_k = probe_k; self.smooth = smooth; self.decided = {}

    def wants_prior(self, node):
        # probes only during TREE selection; the rollout phase also calls compute_action_prior at every
        # remaining step and probing there would multiply NFE without bound
        return self.policy != "none" and getattr(self, "phase", "tree") == "tree" and node.step < self.k["n_key"]

    def action_prior(self, node, candidates, base_prior):
        key = (int(node.step), id(node))
        if key in self.decided: return self.decided[key]
        k = self.k; n = len(candidates)
        # top-k by actdiff's heuristic prior, forced to span distinct wordings
        order = sorted(range(n), key=lambda i: -float(base_prior[i])); pick = []; seen = set()
        for i in order:
            w = int(candidates[i][0])
            if w in seen: continue
            pick.append(i); seen.add(w)
            if len(pick) >= self.probe_k: break
        for i in order:
            if len(pick) >= self.probe_k: break
            if i not in pick: pick.append(i)
        if len(pick) < 2: return None, None
        ks = k["key_steps"]; s0 = ks[node.step]; s1 = ks[node.step + 1] if node.step + 1 < k["n_key"] else k["steps"]
        probes = []
        for i in pick:                                           # PROBE: one segment from THIS node per candidate
            lat, dx = k["run_segment"](k["args"], k["ctx"], k["emb"], k["reward_model"], self.prompt, node.latents, node.dx,
                                       candidates[i], k["sched"], s0, s1, eps_bank=k["eps_bank"])
            prev = k["decode"](k["ctx"], dx); vec = self.score_vec(prev)
            probes.append({"i": i, "action": [float(x) for x in candidates[i]], "img": prev, "vec": vec})
        if self.policy == "random":
            chosen = self.rng.randint(len(probes)); probs = None; perm = list(range(len(probes))); crit = [""] * len(probes)
        else:
            crit = [self.assess(p["img"]) for p in probes]
            perm = list(range(len(probes))); self.rng.shuffle(perm)
            pans = [(probes[j]["img"], f"[{m+1}] w{int(probes[j]['action'][0])} @ CFG {probes[j]['action'][1]:g}") for m, j in enumerate(perm)]
            sh = sheet(pans); sp = os.path.join(self.sheet_dir, f"{self.tag}_probe_s{node.step}_{self.calls}.png"); sh.save(sp)
            opts = [f'Panel {m+1}: wording {chr(65+int(probes[j]["action"][0]))} at CFG {probes[j]["action"][1]:g}. Critique: {str(crit[j]).strip()[:240]}' for m, j in enumerate(perm)]
            state = (f'ORIGINAL PROMPT (the only target): "{self.prompt}"\nEach numbered panel is an UNFINISHED preview (step {s1} of {k["steps"]}) of what '
                     f'the image becomes if that option\'s setting is used for the NEXT {s1-s0} steps from the SAME current state. Blur is normal; '
                     f'judge layout, content and whether the prompt\'s requirements are taking shape.')
            r = self.ask(state, "Which numbered continuation best serves the ORIGINAL prompt?", opts, sh, f"probe/s{node.step}")
            chosen = perm[int(r["argmax"])]; probs = r["probs"]
        prior = np.full(n, self.smooth / max(1, n), dtype=np.float64)
        prior[probes[chosen]["i"]] += 1.0 - self.smooth; prior /= prior.sum()
        dbg = {"step": int(node.step), "probe_k": len(probes), "probes": [{"action": p["action"], "vec": p["vec"], "critique": c} for p, c in zip(probes, crit)],
               "perm": perm, "chosen_probe": int(chosen), "probs": probs, "probe_nfe": 2 * (s1 - s0) * len(probes)}
        self.log({"kind": "probe", **dbg}); self.decided[key] = (prior, dbg)
        return prior, dbg


def build(kind, **kw):
    kind = kind.lower()
    if kind in ("bank", "jevbank"): return BankDecider(policy="jev", **kw)
    if kind == "randbank": return BankDecider(policy="random", **kw)
    if kind in ("probe", "jevprobe"): return ProbeDecider(policy="jev", **kw)
    if kind == "randprobe": return ProbeDecider(policy="random", **kw)
    if kind == "rank3only": return BankDecider(policy="none", **kw)   # rank3 backup, no Jev, no bank change
    raise ValueError(kind)
