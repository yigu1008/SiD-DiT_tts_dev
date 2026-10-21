#!/usr/bin/env python3
"""The one untested role for Jev: SELECTOR over the k finished candidates, not a decision maker inside search.

Why this and nothing else. Every steering role failed, not through incompetence but because the decisions had no
headroom (branch placement oracle +0.006, reward choice +0.014, abandonment +0.413 of which a free threshold takes
+0.140), and because conditioning competes with the initial latent rather than complementing it. What survived is a
draw-and-select regime in which the sampler is unguided and the SELECTOR is the only lever -- and the single clean
positive result of the study was a selector change (rank3 ensemble, vqascore +0.0171 [+0.0050, +0.0304] over HPSv3).
`rank3` is three reward models voting; none of them was asked whether the image depicts the prompt.
Jev's measured strength is exactly that: comparative matching over images, 45/45 on reward-focus matching and 7/7 /
3/3 on perception, versus 0/8 flips on a policy question when the image was withheld. Selection is a matching
question. So this arm asks Jev to pick the winner, on the SAME candidate sets the reward selectors used, with no
generation at all -- the ladder images already exist.

Tournament over k candidates: Jev compares them in a labelled contact sheet of at most --panels at a time, the
winner of each heat carries forward, so cost is ~ceil(k/panels) calls per prompt (~1-3), not k^2.
"""
from __future__ import annotations
import argparse, base64, csv, io, json, os, sys, urllib.request
from pathlib import Path
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from prompt_proposer import post, PMAP
LADDER = "/data/ygu/runs/vlm_decider_sd35base_2026-10-16/main/ns_ladder/records_all.jsonl"
SR = ["imagereward", "hpsv3", "pickscore"]


def sheet(panels, px=420):
    s = Image.new("RGB", (px * len(panels), px + 26), "white"); d = ImageDraw.Draw(s)
    for i, (im, cap) in enumerate(panels):
        s.paste(im.resize((px, px)), (i * px, 26)); d.text((i * px + 5, 7), cap[:52], fill="black")
    return s


def b64(img):
    buf = io.BytesIO(); img.save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", required=True); p.add_argument("--run_dir", required=True)
    p.add_argument("--k", type=int, default=11); p.add_argument("--panels", type=int, default=4)
    p.add_argument("--jev_server", default="http://127.0.0.1:5161")
    a = p.parse_args(); arm = Path(a.run_dir).name
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    pmap = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(PMAP))}
    lad = {}
    for l in open(LADDER):
        d = json.loads(l); lad.setdefault(d["prompt_id"], {})[d["noise_seed"]] = d
    assert json.loads(urllib.request.urlopen(a.jev_server + "/health", timeout=10).read())["status"] == "ok"
    run_dir = Path(a.run_dir); (run_dir / "sheets").mkdir(parents=True, exist_ok=True)
    recp = run_dir / "records.jsonl"
    done = {json.loads(l)["prompt_id"] for l in open(recp)} if recp.exists() else set()
    with recp.open("a") as rec:
        for pid in [i for i in ids if i not in done]:
            prompt = pmap[pid]
            seeds = [s for s in ([42] + [2000 + i for i in range(31)])[: a.k] if s in lad[pid]]
            field = list(seeds); n_calls = 0; heats = []
            while len(field) > 1:                       # tournament; winner of each heat carries forward
                nxt = []
                for i in range(0, len(field), a.panels):
                    grp = field[i: i + a.panels]
                    if len(grp) == 1: nxt.append(grp[0]); continue
                    panels = [(Image.open(lad[pid][s]["image"]).convert("RGB"), f"[{j+1}]") for j, s in enumerate(grp)]
                    sh = sheet(panels)
                    sp = run_dir / "sheets" / f"{pid}_h{len(heats)}_{i}.png"; sh.save(sp)
                    state = (f'ORIGINAL PROMPT (the only target): "{prompt}"\n'
                             f'The attached strip shows {len(grp)} FINISHED images, all generated for that prompt. '
                             f'They differ only in the random seed they started from. Judge which single image is the '
                             f'best realisation of the prompt: everything the prompt asks for present and correct, '
                             f'rendered without distortion or artifacts.')
                    r = post(a.jev_server + "/ask", {"state": state,
                             "question": "Which numbered image is the best realisation of the prompt?",
                             "options": [f"Image {j+1}" for j in range(len(grp))], "image_b64": b64(sh),
                             "orders": 3, "seed": 9000 + len(heats) * 17 + i, "tag": f"{pid}/select/h{len(heats)}"})
                    n_calls += 1
                    w = grp[int(r["argmax"])]
                    heats.append({"heat": len(heats), "field": grp, "winner": w, "probs": r["probs"]})
                    nxt.append(w)
                field = nxt
            win = field[0]; d = lad[pid][win]
            out = run_dir / f"{pid}_{arm}.png"
            Image.open(d["image"]).convert("RGB").save(out)
            import hashlib
            rec.write(json.dumps({"prompt_id": pid, "prompt": prompt, "arm": arm, "seed": 42, "noise_seed": win,
                                  "selector": "jev_tournament", "k": len(seeds), "n_jev_calls": n_calls,
                                  "search_score": d["final"]["hpsv3"], "search_vec": d["final"],
                                  "actions": [[d["wording"], d["cfg"], 0.0]], "heats": heats,
                                  "image": str(out), "image_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                                  "nfe_transformer": len(seeds) * 56}) + "\n"); rec.flush()
            print(f"[{arm}] {pid} jev picked seed {win} in {n_calls} calls  hps={d['final']['hpsv3']:.3f} "
                  f"ir={d['final']['imagereward']:+.3f} ps={d['final']['pickscore']:.3f}", flush=True)


if __name__ == "__main__":
    main()
