#!/usr/bin/env bash
# Held-out fidelity judges on the ec_v2w images (324, 36/prompt): VQAScore (vqascore env, ~22 GB) + Qwen2-VL alignment (sid_dit env).
#   bash judge_ec_v2w.sh <GPU>      -> $R/scores_ec_v2w/{vqa_scores.shard0.csv, vlm_scores.csv}; sample_index = index within prompt
# Needs a free 40 GB card (run after the arms + judge have released the servers).
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; RO=/data/ygu/reward_orthogonality
RW=/home/ygu/miniconda3/envs/reward/bin/python; SD=/home/ygu/miniconda3/envs/sid_dit/bin/python; VQ=/home/ygu/miniconda3/envs/vqascore_reward
source $R/env.sh; GPU=${1:-0}; MAN=$R/manifests_ec_v2w; OUT=$R/scores_ec_v2w; mkdir -p $MAN $OUT
$SD - $R/main/ec_v2w/records_all.jsonl $R/config/prompts_rsel9_index.json $MAN/followup_ec_v2w.jsonl <<'PY'
import json, sys, hashlib, os, collections
recs, idx, out = sys.argv[1:4]; m = json.load(open(idx)); lines = [l.rstrip("\n") for l in open(os.path.join(os.path.dirname(idx), "prompts_rsel9.txt"))]
pmap = {pid: lines[int(i)] for i, pid in m.items()}; k = collections.Counter(); n = 0
with open(out, "w") as f:
    for l in open(recs):
        r = json.loads(l); pid = r["prompt_id"]; k[pid] += 1
        if not os.path.exists(r["image"]): continue
        f.write(json.dumps({"backend": "followup:ec_v2w", "arm": "ec_v2w", "prompt_id": pid, "sample_index": k[pid], "prompt": pmap[pid], "source": "genai_bench", "category": "", "skills": "",
                            "seed": r["noise_seed"], "image": r["image"], "image_sha256": hashlib.sha256(open(r["image"], "rb").read()).hexdigest(),
                            "stratum": r["stratum"], "mode": r["mode"], "vi": r["vi"], "action": r["action"]}) + "\n"); n += 1
print(f"manifest: {n} images -> {out}")
PY
t=$(date +%s); CUDA_VISIBLE_DEVICES=$GPU PATH="$VQ/bin:$PATH" $VQ/bin/python $RO/score_vqa.py --manifest_dir $MAN --out $OUT/vqa_scores.csv --shard_index 0 --num_shards 1 > $R/logs/judge_ec_vqa.log 2>&1; echo "vqascore exit=$? $(( $(date +%s)-t ))s"
t=$(date +%s); CUDA_VISIBLE_DEVICES=$GPU $SD $RO/score_vlm_judge.py --manifest_dirs $MAN --out $OUT/vlm_scores.csv --shard_index 0 --num_shards 1 > $R/logs/judge_ec_vlm.log 2>&1; echo "qwen2-vl judge exit=$? $(( $(date +%s)-t ))s"
ls -la $OUT
