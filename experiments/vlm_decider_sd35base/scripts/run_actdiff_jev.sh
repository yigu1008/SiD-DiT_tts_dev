#!/usr/bin/env bash
# actdiff (bon_mcts prescreen 8->2 + lookahead MCTS) with Jev decision engines. Six arms, same prescreen seeds,
# same v2 bank delivered at full width, rank3 selection where applicable:
#   ad_base      shipped actdiff (IR backup, heuristic prior)           -- the reference
#   ad_rank3     actdiff + rank3 backup only (no Jev)                    -- isolates the reward change
#   ad_jevbank   #1 Jev re-centres CFG bank + admits wordings + rank3    ad_randbank  matched random control
#   ad_jevprobe  #2 per-step probe traces + VLM critique + Jev prior     ad_randprobe matched random (same probes, same NFE)
# Self-checks: wrapper --help must pass; hooks must be present; waits (with retry) for >=4 free GPUs (ryan holds all).
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; SID=/home/ygu/project/SiD-DiT_tts_dev
PY=/home/ygu/miniconda3/envs/sid_dit/bin/python
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/actdiff_driver.log; }
grep -q "jev_decider" $SID/sampling_unified_sd35_lookahead_reweighting.py || { log "GUARD: hooks missing"; exit 1; }
cd $SID && timeout 600 $PY $R/scripts/actdiff_jev_main.py --help > $R/logs/actdiff_help.log 2>&1 || { log "GUARD: wrapper --help failed (see logs/actdiff_help.log)"; exit 1; }
log "self-checks passed"
free_gpus(){ APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv,noheader,nounits | while IFS=, read g u m; do
    n=$(echo "$APPS" | grep -c "${u// /}"); [ "${m// /}" -lt 2000 ] && [ "$n" = 0 ] && echo $g; done; }
DEADLINE=$(( $(date +%s) + 172800 ))
log "waiting for >=4 free GPUs"
while :; do F=($(free_gpus)); [ ${#F[@]} -ge 4 ] && { log "free: ${F[*]}"; break; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { log "STALL 48h; giving up"; exit 1; }; sleep 300; done
sleep 30; F=($(free_gpus)); [ ${#F[@]} -ge 4 ] || { log "taken during settle"; exit 1; }
bash $R/scripts/mr3_server.sh ${F[0]} 5142 >> $R/logs/actdiff_driver.log 2>&1 || { log "mr3 failed"; exit 1; }
bash $R/scripts/vlm_server.sh ${F[1]} 5151 >> $R/logs/actdiff_driver.log 2>&1 || { log "vlm failed"; exit 1; }
bash $R/scripts/jev_server.sh ${F[1]} 5161 >> $R/logs/actdiff_driver.log 2>&1 || { log "jev failed"; exit 1; }
V2=$R/config/rewrites_v2_9.json; NV=$($PY -c "import json;d=json.load(open('$V2'));print(max(len(v) for v in d.values())-1)")
COMMON="--backend sd35_base --prompt_file $R/config/prompts_rsel9.txt --start_index 0 --end_index 9 --modes mcts
  --steps 28 --width 1024 --height 1024 --cfg_scales 3.5 4.0 4.5 5.0 5.5 6.0 7.0 --baseline_cfg 4.5
  --n_variants $NV --no_qwen --rewrites_file $V2 --n_sims 30 --ucb_c 1.0 --keystep_strategy explicit --mcts_key_steps 0,4,8,12,16,20,24,27
  --seed 42 --reward_backend imagereward --save_best_images --max_sequence_length 256
  --bon_mcts_n_seeds 8 --bon_mcts_topk 2 --bon_mcts_sim_alloc split --bon_mcts_min_sims 8 --bon_mcts_refine_method ours_tree
  --lookahead_mode rollout_tree_prior_adaptive_cfg"
run(){ local G=$1 N=$2 D=$3 t=$(date +%s); log "--- $N (decider=$D) gpu$G"; mkdir -p $R/main/$N
  cd $SID && CUDA_VISIBLE_DEVICES=$G OFFLOAD_TRANSFORMER_DURING_ENCODE=1 OFFLOAD_TEXT_ENCODER_AFTER_ENCODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True REWARD_SERVER_URL=http://127.0.0.1:5142 HPSV3_SCORE_STYLE=raw \
    HF_HUB_CACHE=/data/ygu/.cache/hub HF_HOME=/data/ygu/.cache/hub HF_HUB_OFFLINE=1 \
    $PY -u $R/scripts/actdiff_jev_main.py $COMMON --out_dir $R/main/$N --jev_decider $D --probe_k 3 \
      --decider_log $R/main/$N/decider_log.jsonl >> $R/logs/$N.log 2>&1
  log "--- $N end rc=$? in $(( $(date +%s)-t ))s"
  $PY $R/scripts/actdiff_to_records.py --run_dir $R/main/$N --index_map $R/config/prompts_rsel9_index.json --arm $N >> $R/logs/actdiff_driver.log 2>&1; }
# two samplers in parallel (F[2], F[3]); arms paired so each GPU runs three sequentially
( run ${F[2]} ad_base none; run ${F[2]} ad_jevbank jevbank; run ${F[2]} ad_jevprobe jevprobe ) & P1=$!
( run ${F[3]} ad_rank3 rank3only; run ${F[3]} ad_randbank randbank; run ${F[3]} ad_randprobe randprobe ) & P2=$!
wait $P1; wait $P2
kill $(cat $R/state/vlm_server.pid) $(cat $R/state/jev_server.pid) $(cat $R/state/mr3_server.pid) 2>/dev/null; sleep 15
DIRS=""; for n in ad_base ad_rank3 ad_jevbank ad_randbank ad_jevprobe ad_randprobe; do [ -s $R/main/$n/records.jsonl ] && DIRS="$DIRS $R/main/$n"; done
G=($(free_gpus)); [ -n "$DIRS" ] && [ ${#G[@]} -ge 1 ] && bash $R/scripts/judge_base.sh ${G[0]} $R/config/prompt_ids_rsel9.txt $DIRS >> $R/logs/actdiff_driver.log 2>&1
log "ACTDIFF+JEV COMPLETE"
