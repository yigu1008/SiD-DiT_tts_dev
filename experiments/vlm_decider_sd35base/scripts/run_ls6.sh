#!/usr/bin/env bash
# TACC LS6 launcher: one dedicated 3x A100-40GB node, fixed placement, no GPU polling (nobody else is on the node).
#   GPU0  mr3 reward server (~22 GB) + Qwen2.5-VL decider (~16 GB)   -- the only pair that fits a 40 GB card; `smoke` measures it
#   GPU1  Jev-Omni server (~24 GB)
#   GPU2  the single sampler: every arm / shard runs here sequentially (~26 GB peak at encode)
# Stages (STAGES="..."; default all): smoke -> jevact (ja_jevact_v2w, ja_randact_v2w) -> ec (ec_v2w, 3 shards)
#   -> actdiff (6 arms) -> judge (held-out judges on the ja_* and ad_* arms; ec_v2w is read from its own records).
# Resumable: jev_act.py / early_control.py skip finished work; an actdiff arm is skipped once its records.jsonl exists.
# Paths: the /data/ygu... literals below are rewritten by node_remap.sh (sync_from_repo.sh); env.sh supplies the caches/envs.
#   sbatch $RUN_DIR/scripts/run_ls6.sbatch        or interactively:  STAGES=smoke bash $RUN_DIR/scripts/run_ls6.sh
set -uo pipefail; R=/data/ygu/runs/vlm_decider_sd35base_2026-10-16; PY=/home/ygu/miniconda3/envs/sid_dit/bin/python; SID=/home/ygu/project/SiD-DiT_tts_dev
source $R/env.sh
MR3_GPU=${MR3_GPU:-0}; VLM_GPU=${VLM_GPU:-0}; JEV_GPU=${JEV_GPU:-1}; SAMPLER_GPU=${SAMPLER_GPU:-2}
STAGES=${STAGES:-smoke jevact ec actdiff judge}
V2=$R/config/rewrites_v2_9.json; IDS=$R/config/prompt_ids_rsel9.txt; DIR=$R/main/ec_v2w; NS=3
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/logs/ls6_driver.log; }
has(){ case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
nrec(){ wc -l < "$1" 2>/dev/null || echo 0; }
up(){ curl -s --max-time 5 http://127.0.0.1:$1/health | grep -q '"status": "ok"'; }
gpumem(){ nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed "s/^/[$(date '+%F %T')] /" | tee -a $R/logs/gpu_mem.log; }
kill_servers(){ for s in vlm jev mr3 ir; do kill $(cat $R/state/${s}_server.pid 2>/dev/null) 2>/dev/null; done; sleep 10; }
status(){ echo "jevact $(nrec $R/main/ja_jevact_v2w/records.jsonl)/9 randact $(nrec $R/main/ja_randact_v2w/records.jsonl)/9 ec_v2w $(cat $DIR/records_shard*.jsonl 2>/dev/null | wc -l)/324 actdiff $(ls $R/main/ad_*/records.jsonl 2>/dev/null | wc -l)/6"; }
trap 'log "exit: stopping servers"; kill_servers' EXIT
grep -q "sheet_perm" $R/scripts/jev_act.py && grep -q "_bank_n_variants" $R/scripts/jev_act.py || { log "GUARD: review fixes not present in jev_act.py; refusing to launch"; exit 1; }
grep -q "_bank_n_variants" $R/scripts/early_control.py || { log "GUARD: bank fix missing in early_control.py"; exit 1; }
[ -s $R/main/ns_ladder/records_all.jsonl ] || { log "GUARD: main/ns_ladder/records_all.jsonl missing (early_control input; sync_from_repo.sh copies it)"; exit 1; }
# need_servers <with_vlm_jev 0|1>
need_servers(){ up 5142 || bash $R/scripts/mr3_server.sh $MR3_GPU 5142 >> $R/logs/ls6_driver.log 2>&1 || { log "mr3 failed"; return 1; }
  if [ "$1" = 1 ]; then
    up 5151 || bash $R/scripts/vlm_server.sh $VLM_GPU 5151 >> $R/logs/ls6_driver.log 2>&1 || { log "vlm failed"; return 1; }
    up 5161 || bash $R/scripts/jev_server.sh $JEV_GPU 5161 >> $R/logs/ls6_driver.log 2>&1 || { log "jev failed"; return 1; }; fi
  log "servers up (mr3 gpu$MR3_GPU, vlm gpu$VLM_GPU, jev gpu$JEV_GPU):"; gpumem; }
jevact_arm(){ local N=$1 P=$2 I=$3 t=$(date +%s); log "--- $N (policy=$P) gpu$SAMPLER_GPU"
  (cd $SID && CUDA_VISIBLE_DEVICES=$SAMPLER_GPU OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u $R/scripts/jev_act.py --ids $I --run_dir $R/main/$N --policy $P --rewrites_file $V2 \
    --reward_server http://127.0.0.1:5142 --vlm_server http://127.0.0.1:5151 --jev_server http://127.0.0.1:5161 >> $R/logs/$N.log 2>&1)
  log "--- $N end rc=$? in $(( $(date +%s)-t ))s records=$(nrec $R/main/$N/records.jsonl)"; }

log "=== LS6 launcher on $(hostname) stages=[$STAGES] : $(status)"
if has smoke; then   # all three servers + one jevact prompt on the sampler; peak memory per GPU is the answer we need
  need_servers 1 || exit 1; head -1 $IDS > $R/config/prompt_ids_smoke1.txt; rm -rf $R/main/_smoke_jevact
  ( while :; do gpumem > /dev/null; sleep 20; done ) & MON=$!
  jevact_arm _smoke_jevact jev $R/config/prompt_ids_smoke1.txt; kill $MON 2>/dev/null
  log "smoke peak MiB per GPU: $(awk -F'[ ,]+' '{if($4>m[$3])m[$3]=$4} END{for(g in m)printf "gpu%s=%s ",g,m[g]}' $R/logs/gpu_mem.log)"
  [ "$(nrec $R/main/_smoke_jevact/records.jsonl)" -ge 1 ] || { log "SMOKE FAILED (no record; see logs/_smoke_jevact.log)"; exit 1; }; log "SMOKE OK"; fi
if has jevact; then need_servers 1 || exit 1
  for a in ja_jevact_v2w:jev ja_randact_v2w:random; do N=${a%%:*}
    [ "$(nrec $R/main/$N/records.jsonl)" -ge 9 ] && { log "$N complete, skip"; continue; }; jevact_arm $N ${a##*:} $IDS; done; fi
if has ec; then need_servers 0 || exit 1
  if [ "$(cat $DIR/records_shard*.jsonl 2>/dev/null | wc -l)" -ge 324 ]; then log "ec_v2w complete, skip"; else
    for i in $(seq 0 $((NS-1))); do t=$(date +%s); log "--- ec_v2w shard $i/$NS gpu$SAMPLER_GPU"
      (cd $SID && CUDA_VISIBLE_DEVICES=$SAMPLER_GPU OFFLOAD_TRANSFORMER_DURING_ENCODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PY -u $R/scripts/early_control.py --ids $IDS --run_dir $DIR --rewrites_file $V2 --shard_index $i --num_shards $NS \
        --reward_server http://127.0.0.1:5142 >> $R/logs/ecv2w_shard$i.log 2>&1)
      log "--- ec_v2w shard $i end rc=$? in $(( $(date +%s)-t ))s shard_records=$(nrec $DIR/records_shard$i.jsonl)"; done
    cat $DIR/records_shard*.jsonl > $DIR/records_all.jsonl 2>/dev/null; fi; fi
if has actdiff; then   # same arms/args as run_actdiff_jev.sh, one sampler
  grep -q "jev_decider" $SID/sampling_unified_sd35_lookahead_reweighting.py || { log "GUARD: hooks missing"; exit 1; }
  (cd $SID && timeout 600 $PY $R/scripts/actdiff_jev_main.py --help > $R/logs/actdiff_help.log 2>&1) || { log "GUARD: wrapper --help failed (logs/actdiff_help.log)"; exit 1; }
  need_servers 1 || exit 1; NV=$($PY -c "import json;d=json.load(open('$V2'));print(max(len(v) for v in d.values())-1)")
  COMMON="--backend sd35_base --prompt_file $R/config/prompts_rsel9.txt --start_index 0 --end_index 9 --modes mcts
    --steps 28 --width 1024 --height 1024 --cfg_scales 3.5 4.0 4.5 5.0 5.5 6.0 7.0 --baseline_cfg 4.5
    --n_variants $NV --no_qwen --rewrites_file $V2 --n_sims 30 --ucb_c 1.0 --keystep_strategy explicit --mcts_key_steps 0,4,8,12,16,20,24,27
    --seed 42 --reward_backend imagereward --save_best_images --max_sequence_length 256
    --bon_mcts_n_seeds 8 --bon_mcts_topk 2 --bon_mcts_sim_alloc split --bon_mcts_min_sims 8 --bon_mcts_refine_method ours_tree
    --lookahead_mode rollout_tree_prior_adaptive_cfg"
  for a in ad_base:none ad_rank3:rank3only ad_jevbank:jevbank ad_randbank:randbank ad_jevprobe:jevprobe ad_randprobe:randprobe; do N=${a%%:*}; D=${a##*:}
    [ -s $R/main/$N/records.jsonl ] && { log "$N has records, skip"; continue; }; mkdir -p $R/main/$N; t=$(date +%s); log "--- $N (decider=$D) gpu$SAMPLER_GPU"
    (cd $SID && CUDA_VISIBLE_DEVICES=$SAMPLER_GPU OFFLOAD_TRANSFORMER_DURING_ENCODE=1 OFFLOAD_TEXT_ENCODER_AFTER_ENCODE=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True REWARD_SERVER_URL=http://127.0.0.1:5142 HPSV3_SCORE_STYLE=raw HF_HUB_OFFLINE=1 \
      $PY -u $R/scripts/actdiff_jev_main.py $COMMON --out_dir $R/main/$N --jev_decider $D --probe_k 3 --decider_log $R/main/$N/decider_log.jsonl >> $R/logs/$N.log 2>&1)
    log "--- $N end rc=$? in $(( $(date +%s)-t ))s"
    $PY $R/scripts/actdiff_to_records.py --run_dir $R/main/$N --index_map $R/config/prompts_rsel9_index.json --arm $N >> $R/logs/ls6_driver.log 2>&1; done; fi
if has judge; then kill_servers; DIRS=""
  for n in ja_jevact_v2w ja_randact_v2w ad_base ad_rank3 ad_jevbank ad_randbank ad_jevprobe ad_randprobe; do [ -s $R/main/$n/records.jsonl ] && DIRS="$DIRS $R/main/$n"; done
  [ -n "$DIRS" ] && bash $R/scripts/judge_base.sh $MR3_GPU $IDS $DIRS >> $R/logs/ls6_driver.log 2>&1 || log "judge: nothing to judge"; fi
log "=== LS6 launcher done: $(status)"
