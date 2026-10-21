#!/usr/bin/env python3
"""Is an early x0 preview's reward predictive of the final reward?  If yes, MCTS rollouts can be truncated,
which is where 83 % of the search NFE goes.

For each prompt, run N complete trajectories with random (wording, CFG) actions from the standard bank, decoding
the x0 estimate at every checkpoint step AND at the end, scoring all of them with ImageReward against the ORIGINAL
prompt. Then report, per checkpoint, Spearman and Pearson correlation with the terminal score, and - the number
that actually matters for search - how often picking the best trajectory by the step-k score picks the trajectory
that is also best at the end (top-1 agreement), plus the terminal-reward regret of doing so.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path
import numpy as np

SID = "/home/ygu/project/SiD-DiT_tts_dev"
CACHE = "/data/ygu/human_eval_genai40_v1/sid_vqascore_algorithm_sweep/genai200_v1/shared_rewrites_cache.json"
PMAP = "/data/ygu/reward_orthogonality/mcts/prompts/prompt_map.csv"
CFG_BANK = ["3.5", "4.0", "4.5", "5.0", "5.5", "6.0", "7.0"]

def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", required=True); p.add_argument("--run_dir", required=True)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_traj", type=int, default=8); p.add_argument("--steps", type=int, default=28)
    p.add_argument("--checkpoints", default="4,8,12,16,20,24")
    p.add_argument("--reward_server", required=True)
    a = p.parse_args()
    os.environ.update({"HF_HUB_CACHE": "/data/ygu/.cache/hub", "HF_HOME": "/data/ygu/.cache/hub", "HF_HUB_OFFLINE": "1",
                       "OFFLOAD_TEXT_ENCODER_AFTER_ENCODE": "1", "REWARD_SERVER_URL": a.reward_server, "HPSV3_SCORE_STYLE": "raw"})
    sys.path.insert(0, SID); os.chdir(SID)
    import torch
    import sampling_unified_sd35 as su
    cps = sorted(int(x) for x in a.checkpoints.split(",") if x.strip())
    sargs = su.normalize_paths(su.parse_args([
        "--backend", "sd35_base", "--search_method", "mcts", "--reward_backend", "imagereward", "--rewrites_file", CACHE, "--no_qwen",
        "--n_variants", "3", "--steps", str(a.steps), "--seed", str(a.seed), "--width", "1024", "--height", "1024",
        "--out_dir", a.run_dir, "--cfg_scales", *CFG_BANK, "--baseline_cfg", "4.5"]))
    run_dir = Path(a.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in open(a.ids) if l.strip()]; ids = ids[: a.limit] if a.limit else ids
    pmap = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(PMAP))}; cache = json.load(open(sargs.rewrites_file))
    recp = run_dir / "trajectories.jsonl"
    done = {json.loads(l)["prompt_id"] for l in open(recp)} if recp.exists() else set()
    todo = [i for i in ids if i not in done]
    print(f"[trunc] {len(todo)} prompts to do; {a.n_traj} trajectories each; checkpoints {cps}", flush=True)
    if todo:
        ctx = su.load_pipeline(sargs); rm = su.load_reward_model(sargs, ctx.device)
        use_euler = True
        with torch.no_grad(), recp.open("a") as rec:
            for pid in todo:
                prompt = pmap[pid]; t0 = time.time(); ctx.nfe = 0
                variants = su.generate_variants(sargs, prompt, cache); emb = su.encode_variants(ctx, variants)
                bank = [(vi, float(c), 0.0) for vi in range(len(emb.cond_text)) for c in sargs.cfg_scales]
                rs = np.random.RandomState(a.seed + 17 * ids.index(pid))
                lat0 = su.make_latents(ctx, a.seed, sargs.height, sargs.width, emb.cond_text[0].dtype); dx0 = torch.zeros_like(lat0)
                sched = su.step_schedule(ctx.device, lat0.dtype, a.steps, getattr(sargs, "sigmas", None), euler=use_euler)
                rows = []
                for ti in range(a.n_traj):
                    act = bank[rs.randint(len(bank))]            # one fixed action for the whole trajectory
                    lat, dx = lat0, dx0; scores = {}
                    for k in range(len(cps) + 1):
                        s0 = 0 if k == 0 else cps[k - 1]; s1 = cps[k] if k < len(cps) else a.steps
                        lat, dx = su._run_segment(sargs, ctx, emb, rm, prompt, lat, dx, act, sched, s0, s1, noise_explore_steps=set(), eps_bank=None)
                        img = su.decode_to_pil(ctx, su._final_decode_tensor(lat, dx, use_euler))
                        scores[str(s1)] = float(su.score_image(rm, prompt, img))
                    rows.append({"traj": ti, "action": [int(act[0]), float(act[1]), float(act[2])], "scores": scores})
                    print(f"[trunc] {pid} traj {ti}: " + " ".join(f"s{k}={v:.3f}" for k, v in scores.items()), flush=True)
                rec.write(json.dumps({"prompt_id": pid, "prompt": prompt, "checkpoints": cps + [a.steps],
                                      "trajectories": rows, "nfe": int(ctx.nfe), "seconds": round(time.time() - t0, 1)}) + "\n"); rec.flush()
    # analysis
    data = [json.loads(l) for l in open(recp)]
    cps_all = data[0]["checkpoints"]; fin = str(cps_all[-1])
    print("\n=== Does the step-k preview reward predict the FINAL reward? ===")
    print(f"| checkpoint | rollout steps saved | Spearman | Pearson | top-1 agreement | mean regret |")
    print("|---|---|---|---|---|---|")
    for k in cps_all[:-1]:
        xs, ys, agree, regret = [], [], 0, []
        for d in data:
            sk = [t["scores"][str(k)] for t in d["trajectories"]]; sf = [t["scores"][fin] for t in d["trajectories"]]
            xs += sk; ys += sf
            pick = int(np.argmax(sk)); best = int(np.argmax(sf))
            agree += int(pick == best); regret.append(sf[best] - sf[pick])
        print(f"| step {k}/{cps_all[-1]} | {100*(cps_all[-1]-k)/cps_all[-1]:.0f} % | {spearman(xs, ys):+.3f} | "
              f"{float(np.corrcoef(xs, ys)[0,1]):+.3f} | {agree}/{len(data)} | {float(np.mean(regret)):.3f} |")
    print(f"\n({len(data)} prompts x {len(data[0]['trajectories'])} trajectories; regret = terminal reward lost by "
          f"choosing the trajectory that looked best at step k instead of the one that is actually best.)")

if __name__ == "__main__":
    main()
