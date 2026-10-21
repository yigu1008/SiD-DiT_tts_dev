#!/usr/bin/env python3
"""Is the (wording, CFG) preference separable, or does CFG depend on the wording?

The arms ask Jev two marginal questions (rank wordings; rank CFGs) and keep top-nv x top-nc — a Cartesian product,
which assumes P(wording, cfg) factorises. This asks Jev the JOINT question over all 28 (wording, CFG) pairs on the
same real previews, order-averaged, and reports:
  * conditional CFG argmax per wording -- if it differs, the factorisation is throwing away real signal;
  * separability: how well the 4x7 joint matrix is approximated by the outer product of its own marginals
    (explained variance of the best rank-1 fit; 1.00 = perfectly separable);
  * overlap between the joint top-k action set and the factorised top-2 x top-3 set the arms actually used;
  * degeneracy check: does the joint ranking just echo the presented option order?
No sampling: previews + Jev server only.
"""
import json, base64, sys, os, csv, collections, statistics as st, urllib.request
import numpy as np
R = "/data/ygu/runs/vlm_decider_sd35base_2026-10-16"
JEV = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5161"
NODES = int(sys.argv[2]) if len(sys.argv) > 2 else 8
CFGS = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]
PMAP = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open("/data/ygu/reward_orthogonality/mcts/prompts/prompt_map.csv"))}

def ask(state, opts, seed, tag, b64=None, orders=3):
    body = {"state": state, "question": "Which setting should the sampler use for the next steps so the final image best matches the ORIGINAL prompt?",
            "options": opts, "orders": orders, "seed": seed, "tag": tag}
    if b64: body["image_b64"] = b64
    req = urllib.request.Request(JEV + "/ask", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())

def rank1_fit(M):
    """Explained variance of the best rank-1 approximation (SVD) — 1.0 means perfectly separable."""
    U, S, Vt = np.linalg.svd(M)
    return float(S[0] ** 2 / (S ** 2).sum())

dec = [json.loads(l) for l in open(f"{R}/main/fr_jev_s10/decisions.jsonl")]
nodes = [d for d in dec if d.get("preview") and os.path.exists(d["preview"])][:NODES]
recs = [json.loads(l) for l in open(f"{R}/main/fr_jev_s10/records.jsonl")]
variants = {r["prompt_id"]: r["variants"] for r in recs}
print(f"joint vs factorised on {len(nodes)} real decision nodes\n")
sep, cond_var, overlap, degen = [], 0, [], 0
for d in nodes:
    pid = d["prompt_id"]; W = variants[pid]; b64 = base64.b64encode(open(d["preview"], "rb").read()).decode()
    state = (f'ORIGINAL PROMPT (the only target): "{PMAP[pid]}"\nDenoising step {d["step"]} of 28.\n'
             f'VLM readout of the current preview: {d.get("comment") or ""}\n'
             f'The sampler will apply the chosen setting for the next denoising steps; the FINAL image must match the ORIGINAL prompt.')
    # full text + index: truncation collapsed variants sharing a prefix and the server's uniqueness guard rejected it
    opts = [f'wording {chr(65+i)}: "{W[i]}" | CFG {c:g}' for i in range(len(W)) for c in CFGS]
    assert len(set(opts)) == len(opts), "option strings must stay unique"
    r = ask(state, opts, seed=d["node_id"], tag=f"joint/d{d['depth']}", b64=b64)
    P = np.array(r["probs"]).reshape(len(W), len(CFGS))
    s = rank1_fit(P); sep.append(s)
    best_cfg = [CFGS[int(np.argmax(P[i]))] for i in range(len(W))]
    if len(set(best_cfg)) > 1: cond_var += 1
    joint_top = {tuple(divmod(i, len(CFGS))) for i in np.argsort(-np.array(r["probs"]))[:6]}
    fac = {(int(a[0]), CFGS.index(float(a[1]))) for a in d["keep_actions"] if float(a[1]) in CFGS}
    overlap.append(len(joint_top & fac) / max(1, len(joint_top)))
    order_corr = float(np.corrcoef(np.argsort(np.argsort(-np.array(r["probs"]))), np.arange(len(opts)))[0, 1])
    if order_corr > 0.8: degen += 1
    print(f"  node {d['node_id']:3d} d{d['depth']} {pid}: best CFG per wording {best_cfg}  separability {s:.3f}  "
          f"overlap with the factorised set {overlap[-1]:.2f}  order-echo r={order_corr:+.2f}")
print(f"\n=== {len(nodes)} nodes ===")
print(f"  nodes where the best CFG DIFFERS across wordings: {cond_var}/{len(nodes)}   <-- if high, the factorisation loses real signal")
print(f"  separability (rank-1 explained variance): mean {st.fmean(sep):.3f}  (1.000 = CFG preference independent of wording)")
print(f"  joint top-6 overlap with the factorised top-2 x top-3 set: mean {st.fmean(overlap):.2f}")
print(f"  degenerate order-echo rankings: {degen}/{len(nodes)}")
