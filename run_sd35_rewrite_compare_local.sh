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

OUT_DIR="${OUT_DIR:-./sd35_rewrite_compare_out}"
START_INDEX="${START_INDEX:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
END_INDEX="${END_INDEX:-}"
NUM_GPUS="${NUM_GPUS:-$(${PYTHON_BIN} - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "[sd35-rewrite-compare] prompt file not found, exporting HPSv2 prompts first ..."
  OUT_DIR="${PROMPT_DIR}" STYLE="${PROMPT_STYLE}" bash "${SCRIPT_DIR}/get_hpsv2_prompts.sh"
fi
if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT_FILE not found after export: ${PROMPT_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

if (( NUM_PROMPTS > 0 )); then
  if [[ -n "${END_INDEX}" ]]; then
    echo "[sd35-rewrite-compare] both NUM_PROMPTS and END_INDEX are set; using NUM_PROMPTS (END_INDEX ignored)."
  fi
  END_INDEX="$((START_INDEX + NUM_PROMPTS))"
elif [[ -z "${END_INDEX}" ]]; then
  END_INDEX="-1"
fi

CFG_SCALES_STR="${CFG_SCALES:-1.0 1.25 1.5 1.75 2.0 2.25 2.5}"
read -r -a CFG_SCALES_ARR <<< "${CFG_SCALES_STR}"
if [[ "${#CFG_SCALES_ARR[@]}" -eq 0 ]]; then
  echo "Error: CFG_SCALES is empty." >&2
  exit 1
fi

CORRECTION_STRENGTHS_STR="${CORRECTION_STRENGTHS:-0.0}"
read -r -a CORRECTION_STRENGTHS_ARR <<< "${CORRECTION_STRENGTHS_STR}"
if [[ "${#CORRECTION_STRENGTHS_ARR[@]}" -eq 0 ]]; then
  CORRECTION_STRENGTHS_ARR=(0.0)
fi

MODES_STR="${MODES:-base mcts}"
read -r -a MODES_ARR <<< "${MODES_STR}"
if [[ "${#MODES_ARR[@]}" -eq 0 ]]; then
  MODES_ARR=(base mcts)
fi

N_VARIANTS_STANDARD="${N_VARIANTS_STANDARD:-4}"
AXIS_TARGET_SIZE="${AXIS_TARGET_SIZE:-6}"
USE_QWEN="${USE_QWEN:-1}"
COMPARE_MODE="${COMPARE_MODE:-mcts}"
REWARD_MODEL="${REWARD_MODEL:-CodeGoat24/UnifiedReward-qwen-7b}"
UNIFIEDREWARD_MODEL="${UNIFIEDREWARD_MODEL:-${REWARD_MODEL}}"
IMAGE_REWARD_MODEL="${IMAGE_REWARD_MODEL:-ImageReward-v1.0}"
PICKSCORE_MODEL="${PICKSCORE_MODEL:-yuvalkirstain/PickScore_v1}"

qwen_args=()
if [[ "${USE_QWEN}" == "1" ]]; then
  qwen_args+=(--qwen_id "${QWEN_ID:-Qwen/Qwen3-4B}")
  qwen_args+=(--qwen_dtype "${QWEN_DTYPE:-bfloat16}")
  qwen_args+=(--qwen_timeout_sec "${QWEN_TIMEOUT_SEC:-240}")
else
  qwen_args+=(--no_qwen)
fi

extra_reward_args=()
if [[ -n "${REWARD_API_BASE:-}" ]]; then
  extra_reward_args+=(--reward_api_base "${REWARD_API_BASE}")
fi

extra_args=()
if [[ -n "${SIGMAS:-}" ]]; then
  # shellcheck disable=SC2206
  sigmas_arr=(${SIGMAS})
  if [[ "${#sigmas_arr[@]}" -gt 0 ]]; then
    extra_args+=(--sigmas "${sigmas_arr[@]}")
  fi
fi

reward_weights_str="${REWARD_WEIGHTS:-1.0 1.0}"
# shellcheck disable=SC2206
reward_weights_arr=(${reward_weights_str})
if [[ "${#reward_weights_arr[@]}" -ne 2 ]]; then
  echo "Error: REWARD_WEIGHTS must contain exactly 2 values." >&2
  exit 1
fi

STANDARD_OUT="${OUT_DIR}/standard_4rewrite"
AXIS_OUT="${OUT_DIR}/axis_rewrite"
mkdir -p "${STANDARD_OUT}" "${AXIS_OUT}"

echo "[sd35-rewrite-compare] prompt_file=${PROMPT_FILE} range=[${START_INDEX},${END_INDEX}) num_gpus=${NUM_GPUS}"
echo "[sd35-rewrite-compare] modes=[${MODES_ARR[*]}] compare_mode=${COMPARE_MODE}"
echo "[sd35-rewrite-compare] standard_n_variants=${N_VARIANTS_STANDARD} axis_target_size=${AXIS_TARGET_SIZE} use_qwen=${USE_QWEN}"
echo "[sd35-rewrite-compare] cfg_scales=[${CFG_SCALES_ARR[*]}] reward_backend=${REWARD_BACKEND:-imagereward}"

run_case() {
  local label="$1"
  local runner="$2"
  local case_out="$3"
  local n_variants="$4"
  local axis_target_size="$5"
  local rewrites_file="$6"

  local -a case_args=()
  if [[ -n "${rewrites_file}" ]]; then
    case_args+=(--rewrites_file "${rewrites_file}")
  fi
  if [[ "${label}" == "axis" ]]; then
    case_args+=(--axis_target_size "${axis_target_size}")
  fi

  echo "[sd35-rewrite-compare] running ${label} -> ${case_out}"
  torchrun --standalone --nproc_per_node "${NUM_GPUS}" "${runner}" \
    --backend "${SD35_BACKEND:-sid}" \
    --prompt_file "${PROMPT_FILE}" \
    --start_index "${START_INDEX}" \
    --end_index "${END_INDEX}" \
    --gen_batch_size "${GEN_BATCH_SIZE:-1}" \
    --modes "${MODES_ARR[@]}" \
    --steps "${STEPS:-4}" \
    --cfg_scales "${CFG_SCALES_ARR[@]}" \
    --baseline_cfg "${BASELINE_CFG:-1.0}" \
    --n_variants "${n_variants}" \
    --correction_strengths "${CORRECTION_STRENGTHS_ARR[@]}" \
    --n_sims "${N_SIMS:-50}" \
    --ucb_c "${UCB_C:-1.41}" \
    --smc_k "${SMC_K:-8}" \
    --smc_gamma "${SMC_GAMMA:-0.10}" \
    --ess_threshold "${ESS_THRESHOLD:-0.5}" \
    --resample_start_frac "${RESAMPLE_START_FRAC:-0.3}" \
    --smc_cfg_scale "${SMC_CFG_SCALE:-1.25}" \
    --smc_variant_idx "${SMC_VARIANT_IDX:-0}" \
    --seed "${SEED:-42}" \
    --reward_backend "${REWARD_BACKEND:-imagereward}" \
    --reward_model "${REWARD_MODEL}" \
    --unifiedreward_model "${UNIFIEDREWARD_MODEL}" \
    --image_reward_model "${IMAGE_REWARD_MODEL}" \
    --pickscore_model "${PICKSCORE_MODEL}" \
    --reward_weights "${reward_weights_arr[0]}" "${reward_weights_arr[1]}" \
    --reward_api_key "${REWARD_API_KEY:-unifiedreward}" \
    --reward_api_model "${REWARD_API_MODEL:-UnifiedReward-7b-v1.5}" \
    --reward_max_new_tokens "${REWARD_MAX_NEW_TOKENS:-512}" \
    --reward_prompt_mode "${REWARD_PROMPT_MODE:-standard}" \
    --out_dir "${case_out}" \
    "${extra_reward_args[@]}" \
    "${qwen_args[@]}" \
    "${extra_args[@]}" \
    "${case_args[@]}"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_sd35_ddp.py" \
    --log_dir "${case_out}/logs" \
    --out_dir "${case_out}"
}

run_case "standard" "${SCRIPT_DIR}/sd35_ddp_experiment.py" "${STANDARD_OUT}" "${N_VARIANTS_STANDARD}" "${AXIS_TARGET_SIZE}" "${REWRITES_FILE_STANDARD:-}"
run_case "axis" "${SCRIPT_DIR}/sd35_ddp_experiment_axis_rewrite.py" "${AXIS_OUT}" "${N_VARIANTS_STANDARD}" "${AXIS_TARGET_SIZE}" "${REWRITES_FILE_AXIS:-}"

COMPARE_TSV="${OUT_DIR}/final_reward_compare_${COMPARE_MODE}.tsv"
COMPARE_JSON="${OUT_DIR}/final_reward_compare_${COMPARE_MODE}.json"

"${PYTHON_BIN}" - <<'PY' "${STANDARD_OUT}" "${AXIS_OUT}" "${COMPARE_MODE}" "${COMPARE_TSV}" "${COMPARE_JSON}"
import glob
import json
import os
import sys
from statistics import mean

standard_out, axis_out, mode, out_tsv, out_json = sys.argv[1:6]


def load_mode_rows(run_out: str, target_mode: str):
    log_dir = os.path.join(run_out, "logs")
    rows = {}
    prompts = {}
    nfe = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "rank_*.jsonl"))):
        if path.endswith("_rewrite_examples.jsonl"):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("mode") != target_mode:
                    continue
                idx = int(row.get("prompt_index"))
                rows[idx] = float(row.get("score", 0.0))
                prompts[idx] = str(row.get("prompt", ""))
                nfe[idx] = float(row.get("nfe", 0.0))
    return rows, prompts, nfe


std_scores, std_prompts, std_nfe = load_mode_rows(standard_out, mode)
axis_scores, axis_prompts, axis_nfe = load_mode_rows(axis_out, mode)

indices = sorted(set(std_scores) | set(axis_scores))

with open(out_tsv, "w", encoding="utf-8") as f:
    f.write("prompt_index\tmode\tstandard_score\taxis_score\tdelta_axis_minus_standard\tstandard_nfe\taxis_nfe\tprompt\n")
    deltas = []
    for idx in indices:
        ss = std_scores.get(idx)
        aa = axis_scores.get(idx)
        d = None
        if ss is not None and aa is not None:
            d = aa - ss
            deltas.append(d)
        prompt = std_prompts.get(idx, axis_prompts.get(idx, ""))
        f.write(
            f"{idx}\t{mode}\t"
            f"{'' if ss is None else f'{ss:.6f}'}\t"
            f"{'' if aa is None else f'{aa:.6f}'}\t"
            f"{'' if d is None else f'{d:+.6f}'}\t"
            f"{'' if idx not in std_nfe else f'{std_nfe[idx]:.1f}'}\t"
            f"{'' if idx not in axis_nfe else f'{axis_nfe[idx]:.1f}'}\t"
            f"{prompt}\n"
        )

summary = {
    "mode": mode,
    "standard_count": len(std_scores),
    "axis_count": len(axis_scores),
    "paired_count": sum(1 for i in indices if i in std_scores and i in axis_scores),
    "standard_mean": mean(std_scores.values()) if std_scores else None,
    "axis_mean": mean(axis_scores.values()) if axis_scores else None,
}
if summary["standard_mean"] is not None and summary["axis_mean"] is not None:
    summary["delta_axis_minus_standard_mean"] = summary["axis_mean"] - summary["standard_mean"]

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
PY

echo "[sd35-rewrite-compare] standard_out=${STANDARD_OUT}"
echo "[sd35-rewrite-compare] axis_out=${AXIS_OUT}"
echo "[sd35-rewrite-compare] compare_tsv=${COMPARE_TSV}"
echo "[sd35-rewrite-compare] compare_json=${COMPARE_JSON}"
