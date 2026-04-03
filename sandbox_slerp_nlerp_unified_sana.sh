#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

PROMPT_STYLE="${PROMPT_STYLE:-all}"
PROMPT_DIR="${PROMPT_DIR:-/data/ygu}"
DEFAULT_PROMPT_FILE="${SCRIPT_DIR}/hpsv2_subset.txt"
if [[ -f "${DEFAULT_PROMPT_FILE}" ]]; then
  PROMPT_FILE="${PROMPT_FILE:-${DEFAULT_PROMPT_FILE}}"
else
  PROMPT_FILE="${PROMPT_FILE:-${PROMPT_DIR}/hpsv2_prompts.txt}"
fi
OUT_DIR="${OUT_DIR:-/data/ygu/sandbox_slerp_nlerp_unified_sana}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "[sandbox-slerp-nlerp] prompt file not found, exporting HPSv2 prompts first ..."
  OUT_DIR="${PROMPT_DIR}" STYLE="${PROMPT_STYLE}" bash "${SCRIPT_DIR}/get_hpsv2_prompts.sh"
fi

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found after export: ${PROMPT_FILE}" >&2
  exit 1
fi

ensure_hpsv3_runtime() {
  local backend_lc
  backend_lc="$(echo "${REWARD_TYPE:-unifiedreward}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${backend_lc}" != "hpsv3" && "${backend_lc}" != "auto" ]]; then
    return 0
  fi
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import hpsv3
import omegaconf
import hydra
print(getattr(hpsv3, "__file__", "ok"), getattr(omegaconf, "__version__", "ok"), getattr(hydra, "__version__", "ok"))
PY
  then
    return 0
  fi
  echo "[deps] HPSv3 runtime deps missing. Installing with install_reward_deps.sh ..."
  PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/install_reward_deps.sh"
}

ensure_unifiedreward_runtime() {
  local backend_lc
  backend_lc="$(echo "${REWARD_TYPE:-unifiedreward}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${backend_lc}" != "unifiedreward" && "${backend_lc}" != "unified" && "${backend_lc}" != "auto" ]]; then
    return 0
  fi
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import re
import transformers
import tokenizers
import qwen_vl_utils  # noqa: F401
parts_t = re.findall(r"\d+", getattr(transformers, "__version__", ""))[:3]
ver_t = tuple(int(x) for x in (parts_t + ["0", "0", "0"])[:3])
parts_tok = re.findall(r"\d+", getattr(tokenizers, "__version__", ""))[:3]
ver_tok = tuple(int(x) for x in (parts_tok + ["0", "0", "0"])[:3])
ok = (
    ver_t >= (4, 52, 4)
    and hasattr(transformers, "Qwen2_5_VLForConditionalGeneration")
    and (0, 21, 0) <= ver_tok < (0, 22, 0)
)
if not ok:
    raise SystemExit(1)
print(transformers.__version__)
PY
  then
    return 0
  fi
  echo "[deps] UnifiedReward runtime missing/incompatible. Applying lightweight repair ..."
  if ! "${PYTHON_BIN}" -m pip install --no-cache-dir --no-deps \
    "transformers==4.52.4" "tokenizers==0.21.1" "qwen-vl-utils==0.0.14"; then
    echo "[deps] lightweight repair failed."
    if [[ "${UNIFIEDREWARD_FULL_REPAIR:-0}" == "1" ]]; then
      echo "[deps] running full reward deps installer (UNIFIEDREWARD_FULL_REPAIR=1) ..."
      PYTHON_BIN="${PYTHON_BIN}" FORCE_INSTALL_DEPS=1 bash "${SCRIPT_DIR}/install_reward_deps.sh"
    else
      echo "[deps] full reinstall skipped. Set UNIFIEDREWARD_FULL_REPAIR=1 to allow full repair."
      return 1
    fi
  fi
  if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import re
import transformers
import tokenizers
import qwen_vl_utils  # noqa: F401
parts_t = re.findall(r"\d+", getattr(transformers, "__version__", ""))[:3]
ver_t = tuple(int(x) for x in (parts_t + ["0", "0", "0"])[:3])
parts_tok = re.findall(r"\d+", getattr(tokenizers, "__version__", ""))[:3]
ver_tok = tuple(int(x) for x in (parts_tok + ["0", "0", "0"])[:3])
ok = (
    ver_t >= (4, 52, 4)
    and hasattr(transformers, "Qwen2_5_VLForConditionalGeneration")
    and (0, 21, 0) <= ver_tok < (0, 22, 0)
)
raise SystemExit(0 if ok else 1)
PY
  then
    echo "[deps] UnifiedReward runtime still incompatible after lightweight repair."
    return 1
  fi
}

ensure_hpsv3_runtime
ensure_unifiedreward_runtime

reward_type_lc="$(echo "${REWARD_TYPE:-unifiedreward}" | tr '[:upper:]' '[:lower:]')"

reward_args=(
  --reward_type "${REWARD_TYPE:-unifiedreward}"
  --reward_device "${REWARD_DEVICE:-cpu}"
  --reward_model "${REWARD_MODEL:-CodeGoat24/UnifiedReward-qwen-7b}"
  --image_reward_model "${IMAGE_REWARD_MODEL:-ImageReward-v1.0}"
  --pickscore_model "${PICKSCORE_MODEL:-yuvalkirstain/PickScore_v1}"
  --reward_max_new_tokens "${REWARD_MAX_NEW_TOKENS:-512}"
  --reward_prompt_mode "${REWARD_PROMPT_MODE:-standard}"
)

if [[ -n "${UNIFIEDREWARD_MODEL:-}" ]]; then
  reward_args+=(--unifiedreward_model "${UNIFIEDREWARD_MODEL}")
fi
if [[ -n "${REWARD_API_BASE:-}" ]]; then
  reward_args+=(--reward_api_base "${REWARD_API_BASE}")
fi
if [[ -n "${REWARD_API_KEY:-}" ]]; then
  reward_args+=(--reward_api_key "${REWARD_API_KEY}")
fi
if [[ -n "${REWARD_API_MODEL:-}" ]]; then
  reward_args+=(--reward_api_model "${REWARD_API_MODEL}")
fi
reward_weights_str="${REWARD_WEIGHTS:-1.0 1.0}"
if [[ -n "${reward_weights_str}" ]]; then
  # shellcheck disable=SC2206
  reward_weights_arr=(${reward_weights_str})
  if [[ "${#reward_weights_arr[@]}" -eq 2 ]]; then
    reward_args+=(--reward_weights "${reward_weights_arr[0]}" "${reward_weights_arr[1]}")
  fi
fi

algo_args=()
if [[ "${RUN_SWEEP:-1}" == "0" ]]; then
  algo_args+=(--no-run_sweep)
fi
if [[ "${RUN_GA:-1}" == "0" ]]; then
  algo_args+=(--no-run_ga)
fi
if [[ "${RUN_MCTS:-1}" == "0" ]]; then
  algo_args+=(--no-run_mcts)
fi
if [[ "${RUN_MCTS_WEIGHT_ABLATION:-0}" == "1" ]]; then
  algo_args+=(--run_mcts_weight_ablation)
fi
if [[ "${RUN_MCTS_DESIGN_ABLATION:-1}" == "1" ]]; then
  algo_args+=(--run_mcts_design_ablation)
fi
if [[ "${RUN_MCTS_FAMILY_SPSA_ABLATION:-0}" == "1" ]]; then
  algo_args+=(--run_mcts_family_spsa_ablation)
fi
if [[ "${ENABLE_PROMPT_WEIGHT_SPSA:-1}" == "0" ]]; then
  algo_args+=(--no-enable_prompt_weight_spsa)
fi

family_args=()
if [[ "${reward_type_lc}" == "imagereward" ]]; then
  echo "[sandbox-slerp-nlerp] REWARD_TYPE=imagereward -> forcing --families nlerp"
  family_args=(--families nlerp)
  if [[ "${RUN_MCTS_FAMILY_SPSA_ABLATION:-0}" == "1" ]]; then
    echo "[sandbox-slerp-nlerp] disabling RUN_MCTS_FAMILY_SPSA_ABLATION for imagereward (slerp arm removed)."
    filtered_algo_args=()
    for a in "${algo_args[@]}"; do
      if [[ "${a}" != "--run_mcts_family_spsa_ablation" ]]; then
        filtered_algo_args+=("${a}")
      fi
    done
    algo_args=("${filtered_algo_args[@]}")
  fi
else
  families_str="${FAMILIES:-nlerp slerp}"
  # shellcheck disable=SC2206
  families_arr=(${families_str})
  if [[ "${#families_arr[@]}" -eq 0 ]]; then
    families_arr=(nlerp slerp)
  fi
  family_args=(--families "${families_arr[@]}")
fi

mix_args=()
if [[ -n "${MIX_WEIGHT_VECTORS:-}" ]]; then
  # shellcheck disable=SC2206
  mix_vec_arr=(${MIX_WEIGHT_VECTORS})
  if [[ "${#mix_vec_arr[@]}" -gt 0 ]]; then
    mix_args+=(--mix_weight_vectors "${mix_vec_arr[@]}")
  fi
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/sandbox_slerp_nlerp_unified_sana.py" \
  --prompt_file "${PROMPT_FILE}" \
  --max_prompts "${MAX_PROMPTS:-20}" \
  --out_dir "${OUT_DIR}" \
  --model_id "${MODEL_ID:-YGu1998/SiD-DiT-SANA-0.6B-RectifiedFlow}" \
  --dtype "${DTYPE:-bf16}" \
  --steps "${STEPS:-4}" \
  --width "${WIDTH:-512}" \
  --height "${HEIGHT:-512}" \
  --seed "${SEED:-42}" \
  --guidance_scale "${GUIDANCE_SCALE:-1.0}" \
  --baseline_cfg "${BASELINE_CFG:-1.0}" \
  --cfg_scales ${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5} \
  "${reward_args[@]}" \
  --qwen_device "${QWEN_DEVICE:-auto}" \
  --qwen_timeout_sec "${QWEN_TIMEOUT_SEC:-240}" \
  --interp_labels ${INTERP_LABELS:-balanced subject composition texture} \
  --interp_k "${INTERP_K:-4}" \
  --interp_values ${INTERP_VALUES:-0.0 0.25 0.5 0.75 1.0} \
  "${mix_args[@]}" \
  "${family_args[@]}" \
  --preview_every "${PREVIEW_EVERY:--1}" \
  --ga_population "${GA_POPULATION:-24}" \
  --ga_generations "${GA_GENERATIONS:-8}" \
  --ga_elites "${GA_ELITES:-3}" \
  --ga_mutation_prob "${GA_MUTATION_PROB:-0.10}" \
  --ga_tournament_k "${GA_TOURNAMENT_K:-3}" \
  --ga_selection "${GA_SELECTION:-rank}" \
  --ga_rank_pressure "${GA_RANK_PRESSURE:-1.7}" \
  --ga_crossover "${GA_CROSSOVER:-uniform}" \
  --ga_log_topk "${GA_LOG_TOPK:-3}" \
  --mcts_n_sims "${MCTS_N_SIMS:-50}" \
  --mcts_ucb_c "${MCTS_UCB_C:-1.41}" \
  --mcts_log_every "${MCTS_LOG_EVERY:-10}" \
  --num_prompt_weights "${NUM_PROMPT_WEIGHTS:-4}" \
  --spsa_block_rollouts "${SPSA_BLOCK_ROLLOUTS:-8}" \
  --spsa_c "${SPSA_C:-0.07}" \
  --spsa_eta "${SPSA_ETA:-0.05}" \
  --weight_clip_min "${WEIGHT_CLIP_MIN:-0.5}" \
  --weight_clip_max "${WEIGHT_CLIP_MAX:-2.0}" \
  "${algo_args[@]}" \
  --save_first_k "${SAVE_FIRST_K:-10}" \
  --save_images \
  "$@"
