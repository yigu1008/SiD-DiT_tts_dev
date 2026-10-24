#!/usr/bin/env bash
# Rewrite the hard-coded A6000-box paths inside this directory's scripts to the paths of the machine we are on.
# Operates IN PLACE on the current checkout/upload (on AMLT the code dir is a fresh per-job copy, so this is safe).
# All 12 prefixes found by `grep -ohE "/(data|home)/ygu/..."` over scripts/*.py, scripts/*.sh, deps/**/*.py.
#
# Required env:  RUN_DIR REPO_DIR HF_CACHE  (HF_CACHE = the dir that contains hub/)
#                SIDDIT_PY_PREFIX REWARD_PY_PREFIX JEV_PY_PREFIX VQA_PY_PREFIX   (conda env prefixes)
# Usage: RUN_DIR=... REPO_DIR=... HF_CACHE=... SIDDIT_PY_PREFIX=... REWARD_PY_PREFIX=... JEV_PY_PREFIX=... VQA_PY_PREFIX=... \
#        bash node_remap.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for v in RUN_DIR REPO_DIR HF_CACHE SIDDIT_PY_PREFIX REWARD_PY_PREFIX JEV_PY_PREFIX VQA_PY_PREFIX; do
  [ -n "${!v:-}" ] || { echo "node_remap: $v is required"; exit 1; }; done
DEPS="$HERE/deps"
FILES=$(grep -rlE "/(data|home)/ygu/" "$HERE/scripts" "$HERE/deps" 2>/dev/null || true)
[ -n "$FILES" ] || { echo "node_remap: nothing to remap"; exit 0; }
n=0
for f in $FILES; do
  sed -i \
    -e "s#/data/ygu/runs/vlm_decider_sd35base_2026-10-16#${RUN_DIR}#g" \
    -e "s#/home/ygu/project/SiD-DiT_tts_dev/hf_cache#${HF_CACHE}/hub#g" \
    -e "s#/home/ygu/project/SiD-DiT_tts_dev#${REPO_DIR}#g" \
    -e "s#/home/ygu/miniconda3/envs/sid_dit#${SIDDIT_PY_PREFIX}#g" \
    -e "s#/home/ygu/miniconda3/envs/reward#${REWARD_PY_PREFIX}#g" \
    -e "s#/home/ygu/miniconda3/envs/jev#${JEV_PY_PREFIX}#g" \
    -e "s#/home/ygu/miniconda3/envs/vqascore_reward#${VQA_PY_PREFIX}#g" \
    -e "s#/data/ygu/\.cache/hub#${HF_CACHE}/hub#g" \
    -e "s#/data/ygu/hf_cache/hub#${HF_CACHE}/hub#g" \
    -e "s#/data/ygu/reward_orthogonality/mcts/prompts/prompt_map.csv#${DEPS}/prompt_map.csv#g" \
    -e "s#/data/ygu/human_eval_genai40_v1/sid_vqascore_algorithm_sweep/genai200_v1/shared_rewrites_cache.json#${DEPS}/shared_rewrites_cache.json#g" \
    -e "s#/data/ygu/reward_orthogonality/followup_2026-09-24/scripts#${DEPS}/followup_scripts#g" \
    -e "s#/data/ygu/reward_orthogonality/followup_2026-09-24#${DEPS}/followup_scripts#g" \
    -e "s#/data/ygu/reward_orthogonality#${DEPS}#g" \
    "$f"; n=$((n+1))
done
left=$(grep -rhoE "/(data|home)/ygu/[^\"' ]*" "$HERE/scripts" "$HERE/deps" 2>/dev/null | sort -u | head -5 || true)
echo "node_remap: rewrote $n files"
[ -z "$left" ] && echo "node_remap: no /data/ygu or /home/ygu paths remain" || { echo "node_remap: WARNING unmapped paths remain:"; echo "$left"; }
