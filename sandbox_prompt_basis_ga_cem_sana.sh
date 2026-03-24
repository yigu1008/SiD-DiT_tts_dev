#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

PROMPT_STYLE="${PROMPT_STYLE:-all}"
PROMPT_DIR="${PROMPT_DIR:-/data/ygu}"
PROMPT_FILE="${PROMPT_FILE:-${PROMPT_DIR}/hpsv2_prompts.txt}"
OUT_DIR="${OUT_DIR:-/data/ygu/sandbox_prompt_basis_ga_cem_sana}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "[sandbox-basis] prompt file not found, exporting HPSv2 prompts first ..."
  OUT_DIR="${PROMPT_DIR}" STYLE="${PROMPT_STYLE}" bash "${SCRIPT_DIR}/get_hpsv2_prompts.sh"
fi

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found after export: ${PROMPT_FILE}" >&2
  exit 1
fi

extra_args=()
if [[ "${RUN_NAIVE_BLEND_ABLATION:-0}" == "1" ]]; then
  extra_args+=(--run_naive_blend_ablation --naive_blend_k "${NAIVE_BLEND_K:-2}")
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/sandbox_prompt_basis_ga_cem_sana.py" \
  --prompt_file "${PROMPT_FILE}" \
  --max_prompts "${MAX_PROMPTS:-0}" \
  --out_dir "${OUT_DIR}" \
  --model_id "${MODEL_ID:-YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow}" \
  --dtype "${DTYPE:-bf16}" \
  --steps "${STEPS:-4}" \
  --width "${WIDTH:-512}" \
  --height "${HEIGHT:-512}" \
  --seed "${SEED:-42}" \
  --guidance_scale "${GUIDANCE_SCALE:-1.0}" \
  --preview_every "${PREVIEW_EVERY:-1}" \
  --reward_type "${REWARD_TYPE:-imagereward}" \
  --reward_device "${REWARD_DEVICE:-cpu}" \
  --image_reward_model "${IMAGE_REWARD_MODEL:-ImageReward-v1.0}" \
  --basis_k "${BASIS_K:-3}" \
  --fixed_family "${FIXED_FAMILY:-nlerp}" \
  --blend_families ${BLEND_FAMILIES:-nlerp slerp} \
  --cem_iters "${CEM_ITERS:-4}" \
  --cem_population "${CEM_POP:-8}" \
  --cem_elite_frac "${CEM_ELITE_FRAC:-0.25}" \
  --cem_init_std "${CEM_INIT_STD:-1.0}" \
  --cem_min_std "${CEM_MIN_STD:-0.05}" \
  --cem_clip "${CEM_CLIP:-3.0}" \
  --ga_population "${GA_POP:-8}" \
  --ga_generations "${GA_GENS:-6}" \
  --ga_elites "${GA_ELITES:-2}" \
  --ga_mutation_prob "${GA_MUTATION_PROB:-0.15}" \
  --ga_selection "${GA_SELECTION:-rank}" \
  --ga_rank_pressure "${GA_RANK_PRESSURE:-1.7}" \
  --ga_tournament_k "${GA_TOURNAMENT_K:-3}" \
  --ga_log_topk "${GA_LOG_TOPK:-3}" \
  --ga_anchor_family "${GA_ANCHOR_FAMILY:-nlerp}" \
  --hybrid_reward_mode "${HYBRID_REWARD_MODE:-paired_rollout}" \
  --hybrid_reward_mix_orig "${HYBRID_REWARD_MIX_ORIG:-1.0}" \
  --hybrid_reward_mix_cond "${HYBRID_REWARD_MIX_COND:-0.5}" \
  --hybrid_reward_mix_delta "${HYBRID_REWARD_MIX_DELTA:-0.5}" \
  --hybrid_cond_reduce "${HYBRID_COND_REDUCE:-max}" \
  --save_first_k "${SAVE_FIRST_K:-10}" \
  --save_images \
  "${extra_args[@]}" \
  "$@"
