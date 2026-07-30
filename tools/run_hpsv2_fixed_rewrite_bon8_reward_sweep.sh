#!/usr/bin/env bash
set -euo pipefail

# Fixed-rewrite BoN-8 on one shared HPSv2 prompt subset.
#
# Models:
#   sid, senseflow_large, sd35_base, flux_schnell
# Guidance rewards:
#   imagereward, hpsv3, multi_reward
#
# The multi_reward arm is the existing half-half min-max-normalized
# ImageReward + HPSv3 objective (reward backend: composite_hpsv3_ir).
# One rewrite cache is shared by every model/reward cell. Within a model, all
# three reward arms also share the exact same prompt-index root seed map.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO}/shell_env.sh"

HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
PROMPTS_FILE="${PROMPTS_FILE:-${REPO}/hpsv2_subset.txt}"
STUDY_ROOT="${STUDY_ROOT:-${HUMAN_EVAL_ROOT}/hpsv2_fixed_rewrite_bon8_reward_sweep}"
RUN_ID="${RUN_ID:-v1}"
RUN_ROOT="${STUDY_ROOT}/${RUN_ID}"

BACKENDS="${BACKENDS:-sid senseflow_large sd35_base flux_schnell}"
REWARD_ARMS="${REWARD_ARMS:-imagereward hpsv3 multi_reward}"
BON_N="${BON_N:-8}"
GENERATION_BASE_SEED="${GENERATION_BASE_SEED:-20260730}"
EXPECTED_PROMPT_COUNT="${EXPECTED_PROMPT_COUNT:-200}"

QWEN_ID="${QWEN_ID:-Qwen/Qwen2.5-3B-Instruct}"
QWEN_SYSTEM_PROMPT_FILE="${QWEN_SYSTEM_PROMPT_FILE:-${REPO}/configs/bon8_fixed_rewrite_system_prompt.txt}"
QWEN_PRECOMPUTE_DEVICE="${QWEN_PRECOMPUTE_DEVICE:-auto}"
QWEN_PRECOMPUTE_BATCH_SIZE="${QWEN_PRECOMPUTE_BATCH_SIZE:-4}"
QWEN_PRECOMPUTE_MAX_NEW_TOKENS="${QWEN_PRECOMPUTE_MAX_NEW_TOKENS:-160}"
QWEN_PRECOMPUTE_TEMPERATURE="${QWEN_PRECOMPUTE_TEMPERATURE:-0.6}"
QWEN_PRECOMPUTE_TOP_P="${QWEN_PRECOMPUTE_TOP_P:-0.9}"
REWRITES_OVERWRITE="${REWRITES_OVERWRITE:-0}"

REWARD_ENV_CONDA_BASE="${REWARD_ENV_CONDA_BASE:-/home/ygu/miniconda3}"
STANDARD_REWARD_ENV_NAME="${STANDARD_REWARD_ENV_NAME:-reward}"
REWARD_SERVER_BASE_PORT="${REWARD_SERVER_BASE_PORT:-5200}"
POSTHOC_REWARD_SERVER_PORT="${POSTHOC_REWARD_SERVER_PORT:-5290}"
REWARD_SERVER_MAX_WAIT="${REWARD_SERVER_MAX_WAIT:-1800}"
HEALTH_TIMEOUT_SECS="${HEALTH_TIMEOUT_SECS:-1800}"

POST_EVAL_ONLY="${POST_EVAL_ONLY:-0}"
SKIP_POST_EVAL="${SKIP_POST_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"
FAIL_FAST="${FAIL_FAST:-1}"

if [[ "${RUN_ID}" == *"/"* ]]; then
  echo "Error: RUN_ID cannot contain '/': ${RUN_ID}" >&2
  exit 1
fi
if [[ "${BON_N}" != "8" ]]; then
  echo "Error: this control is fixed to BON_N=8; got ${BON_N}" >&2
  exit 1
fi
if [[ ! -f "${PROMPTS_FILE}" ]]; then
  echo "Error: prompt source not found: ${PROMPTS_FILE}" >&2
  exit 1
fi
if [[ ! -f "${QWEN_SYSTEM_PROMPT_FILE}" ]]; then
  echo "Error: system prompt not found: ${QWEN_SYSTEM_PROMPT_FILE}" >&2
  exit 1
fi
for value in "${POST_EVAL_ONLY}" "${SKIP_POST_EVAL}" "${DRY_RUN}" \
  "${REWRITES_OVERWRITE}" "${FAIL_FAST}"; do
  case "${value}" in 0|1) ;; *)
    echo "Error: boolean controls must be 0 or 1; got ${value}" >&2
    exit 1
  esac
done
for backend in ${BACKENDS}; do
  case "${backend}" in
    sid|senseflow_large|sd35_base|flux_schnell) ;;
    *) echo "Error: unknown backend '${backend}'" >&2; exit 1 ;;
  esac
done
for arm in ${REWARD_ARMS}; do
  case "${arm}" in
    imagereward|hpsv3|multi_reward) ;;
    *) echo "Error: unknown reward arm '${arm}'" >&2; exit 1 ;;
  esac
done

mkdir -p "${RUN_ROOT}/seed_maps"
PROMPTS_TXT="${RUN_ROOT}/prompts.txt"
PROMPTS_CSV="${RUN_ROOT}/prompts.csv"
RAW_REWRITES_FILE="${RAW_REWRITES_FILE:-${RUN_ROOT}/rewrite_cache_with_c0.json}"
REWRITES_FILE="${REWRITES_FILE:-${RUN_ROOT}/fixed_rewrite_cache.json}"
STUDY_MANIFEST="${RUN_ROOT}/study_manifest.json"

PROMPTS_FILE="${PROMPTS_FILE}" \
PROMPTS_TXT="${PROMPTS_TXT}" \
PROMPTS_CSV="${PROMPTS_CSV}" \
RUN_ROOT="${RUN_ROOT}" \
STUDY_MANIFEST="${STUDY_MANIFEST}" \
QWEN_SYSTEM_PROMPT_FILE="${QWEN_SYSTEM_PROMPT_FILE}" \
QWEN_ID="${QWEN_ID}" \
REWRITES_FILE="${REWRITES_FILE}" \
RUN_ID="${RUN_ID}" \
BACKENDS="${BACKENDS}" \
REWARD_ARMS="${REWARD_ARMS}" \
GENERATION_BASE_SEED="${GENERATION_BASE_SEED}" \
EXPECTED_PROMPT_COUNT="${EXPECTED_PROMPT_COUNT}" \
"${PYTHON_BIN}" - <<'PY'
import csv
import hashlib
import json
import os
from pathlib import Path

source = Path(os.environ["PROMPTS_FILE"]).expanduser().resolve()
if source.suffix.lower() == ".csv":
    with source.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows or "prompt" not in source_rows[0]:
        raise SystemExit(f"{source} must contain a prompt column")
    rows = []
    for index, row in enumerate(source_rows):
        prompt = str(row.get("prompt", "")).strip()
        prompt_id = str(row.get("prompt_id", f"hpsv2_{index:04d}")).strip()
        rows.append({"prompt_id": prompt_id, "prompt": prompt})
else:
    prompts = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [
        {"prompt_id": f"hpsv2_{index:04d}", "prompt": prompt}
        for index, prompt in enumerate(prompts)
    ]

if not rows or any(not row["prompt"] or not row["prompt_id"] for row in rows):
    raise SystemExit(f"{source} contains empty prompts or prompt IDs")
if len({row["prompt_id"] for row in rows}) != len(rows):
    raise SystemExit(f"{source} contains duplicate prompt IDs")
if len({row["prompt"] for row in rows}) != len(rows):
    raise SystemExit(f"{source} contains duplicate prompt text")
expected = int(os.environ["EXPECTED_PROMPT_COUNT"])
if expected >= 0 and len(rows) != expected:
    raise SystemExit(f"{source} contains {len(rows)} prompts; expected {expected}")

prompts_txt = Path(os.environ["PROMPTS_TXT"])
prompts_txt.write_text(
    "".join(row["prompt"] + "\n" for row in rows),
    encoding="utf-8",
)
with Path(os.environ["PROMPTS_CSV"]).open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["prompt_id", "prompt"])
    writer.writeheader()
    writer.writerows(rows)

base_seed = int(os.environ["GENERATION_BASE_SEED"])
seed_files = {}
for backend in os.environ["BACKENDS"].split():
    seeds = {}
    for index, row in enumerate(rows):
        material = (
            f"hpsv2_fixed_rewrite_bon8\0{backend}\0"
            f"{row['prompt_id']}\0{base_seed}"
        ).encode()
        value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        seeds[str(index)] = 1 + value % 2_147_483_646
    path = Path(os.environ["RUN_ROOT"]) / "seed_maps" / f"{backend}.json"
    payload = {
        "model_id": backend,
        "algorithm_id": "bon_fixed_rewrite",
        "prompt_count": len(rows),
        "prompt_ids": [row["prompt_id"] for row in rows],
        "generation_base_seed": base_seed,
        "seeds": seeds,
    }
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise SystemExit(f"{path} conflicts with this launch; choose another RUN_ID")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    seed_files[backend] = str(path.resolve())

system_path = Path(os.environ["QWEN_SYSTEM_PROMPT_FILE"]).expanduser().resolve()
system_text = system_path.read_text(encoding="utf-8").strip()
model_names = {
    "sid": "SiD-SD3.5",
    "senseflow_large": "SenseFlow-SD3.5-Large",
    "sd35_base": "SD3.5-Base",
    "flux_schnell": "Flux-Schnell",
}
arm_specs = {
    "imagereward": {
        "reward_backend": "imagereward",
        "server_backends": ["imagereward"],
    },
    "hpsv3": {
        "reward_backend": "hpsv3",
        "server_backends": ["hpsv3"],
    },
    "multi_reward": {
        "reward_backend": "composite_hpsv3_ir",
        "server_backends": ["imagereward", "hpsv3"],
        "definition": (
            "0.5 * minmax(ImageReward; [-3,3]) + "
            "0.5 * minmax(HPSv3; [9,14])"
        ),
    },
}
backends = os.environ["BACKENDS"].split()
arms = os.environ["REWARD_ARMS"].split()
manifest = {
    "study_id": "hpsv2_fixed_rewrite_bon8_reward_sweep",
    "run_id": os.environ["RUN_ID"],
    "prompt_source": str(source),
    "prompt_count": len(rows),
    "prompt_ids": [row["prompt_id"] for row in rows],
    "models": [
        {"model_id": backend, "model_name": model_names[backend]}
        for backend in backends
    ],
    "reward_arms": {arm: arm_specs[arm] for arm in arms},
    "cells": [
        {"model_id": backend, "reward_arm": arm}
        for backend in backends
        for arm in arms
    ],
    "candidate_count": 8,
    "algorithm_id": "bon_fixed_rewrite",
    "fixed_rewrite_cache": str(
        Path(os.environ["REWRITES_FILE"]).expanduser().resolve()
    ),
    "rewrite_count_per_prompt": 1,
    "prompt_rewriter": {
        "model": os.environ["QWEN_ID"],
        "system_prompt_file": str(system_path),
        "system_prompt_sha256": hashlib.sha256(
            system_text.encode("utf-8")
        ).hexdigest(),
    },
    "seed_map_files": seed_files,
    "matched_root_rule": (
        "Within each model, all reward arms use the same prompt-index root "
        "seed map and the same eight deterministic candidate roots."
    ),
    "reward_prompt_invariant": (
        "All online selection and post-evaluation scores use original c0, "
        "never the fixed rewritten generation prompt."
    ),
    "post_evaluation_rewards": [
        "imagereward",
        "hpsv3",
        "pickscore",
        "hpsv2",
    ],
    "nfe_rule": (
        "Read actual search NFE from the SD3.5 rank JSONL or FLUX rank "
        "summary diagnostics; never infer it from nominal settings."
    ),
}
manifest_path = Path(os.environ["STUDY_MANIFEST"])
if manifest_path.is_file():
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing != manifest:
        raise SystemExit(
            f"{manifest_path} conflicts with this launch; choose another RUN_ID"
        )
else:
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
print(f"[sweep] prompts={len(rows)} source={source}")
print(f"[sweep] cells={len(manifest['cells'])} manifest={manifest_path}")
PY

PROMPT_COUNT="$("${PYTHON_BIN}" - <<'PY' "${PROMPTS_TXT}"
import sys
print(sum(bool(line.strip()) for line in open(sys.argv[1], encoding="utf-8")))
PY
)"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] HPSv2 fixed-rewrite BoN-8 reward sweep"
  echo "  prompts: ${PROMPT_COUNT}"
  echo "  models: ${BACKENDS}"
  echo "  reward_arms: ${REWARD_ARMS}"
  echo "  cells: $(wc -w <<<"${BACKENDS}") x $(wc -w <<<"${REWARD_ARMS}")"
  echo "  shared_rewrite_cache: ${REWRITES_FILE}"
  echo "  output: ${RUN_ROOT}"
  exit 0
fi

STANDARD_REWARD_PY="${REWARD_ENV_CONDA_BASE}/envs/${STANDARD_REWARD_ENV_NAME}/bin/python"
if [[ ! -x "${STANDARD_REWARD_PY}" ]]; then
  echo "Error: standard reward Python is not executable: ${STANDARD_REWARD_PY}" >&2
  exit 1
fi

validate_rewrite_cache() {
  REWRITES_FILE="${REWRITES_FILE}" PROMPTS_TXT="${PROMPTS_TXT}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

prompts = [
    line.strip()
    for line in Path(os.environ["PROMPTS_TXT"]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
cache_path = Path(os.environ["REWRITES_FILE"])
cache = json.loads(cache_path.read_text(encoding="utf-8"))
if set(cache) != set(prompts):
    raise SystemExit(
        f"{cache_path} prompt set mismatch: "
        f"missing={len(set(prompts)-set(cache))} extra={len(set(cache)-set(prompts))}"
    )
bad = [
    prompt for prompt in prompts
    if not isinstance(cache[prompt], list)
    or len(cache[prompt]) != 1
    or not str(cache[prompt][0]).strip()
    or str(cache[prompt][0]).strip() == prompt
]
if bad:
    raise SystemExit(
        f"{cache_path} must contain exactly one distinct rewrite per c0; "
        f"bad={len(bad)}"
    )
print(f"[sweep] rewrite invariant OK: {len(prompts)} prompts x 1 rewrite")
PY
}

if [[ "${POST_EVAL_ONLY}" != "1" ]]; then
  if [[ "${REWRITES_OVERWRITE}" == "1" ]]; then
    rm -f "${RAW_REWRITES_FILE}" "${REWRITES_FILE}"
  fi
  if [[ ! -s "${REWRITES_FILE}" ]]; then
    echo "[sweep] precomputing one shared conservative rewrite per c0"
    REWRITE_SYSTEM_OVERRIDE="$(<"${QWEN_SYSTEM_PROMPT_FILE}")" \
    REWRITE_STYLES_OVERRIDE="Produce one conservative rewrite that preserves every semantic requirement. This exact rewrite is shared by all models, reward arms, and eight BoN roots." \
    "${PYTHON_BIN}" -u "${REPO}/precompute_sd35_rewrites.py" \
      --prompt_file "${PROMPTS_TXT}" \
      --rewrites_file "${RAW_REWRITES_FILE}" \
      --start_index 0 \
      --end_index "${PROMPT_COUNT}" \
      --n_variants 1 \
      --qwen_id "${QWEN_ID}" \
      --qwen_dtype "${QWEN_DTYPE:-bfloat16}" \
      --device "${QWEN_PRECOMPUTE_DEVICE}" \
      --batch_size "${QWEN_PRECOMPUTE_BATCH_SIZE}" \
      --max_new_tokens "${QWEN_PRECOMPUTE_MAX_NEW_TOKENS}" \
      --temperature "${QWEN_PRECOMPUTE_TEMPERATURE}" \
      --top_p "${QWEN_PRECOMPUTE_TOP_P}"

    RAW_REWRITES_FILE="${RAW_REWRITES_FILE}" \
    REWRITES_FILE="${REWRITES_FILE}" \
    PROMPTS_TXT="${PROMPTS_TXT}" \
    "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

raw_path = Path(os.environ["RAW_REWRITES_FILE"])
out_path = Path(os.environ["REWRITES_FILE"])
prompts = [
    line.strip()
    for line in Path(os.environ["PROMPTS_TXT"]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
raw = json.loads(raw_path.read_text(encoding="utf-8"))
fixed = {}
bad = []
for c0 in prompts:
    values = raw.get(c0)
    rewrites = []
    if isinstance(values, list):
        for value in values:
            text = str(value).strip()
            if text and text != c0 and text not in rewrites:
                rewrites.append(text)
    if len(rewrites) != 1:
        bad.append({"prompt": c0, "found": len(rewrites)})
    else:
        fixed[c0] = [rewrites[0]]
if bad:
    raise SystemExit(
        f"expected exactly one distinct rewrite for every prompt; "
        f"bad={len(bad)} examples={bad[:3]}"
    )
temporary = out_path.with_name(out_path.name + ".tmp")
temporary.write_text(
    json.dumps(fixed, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
temporary.replace(out_path)
print(f"[sweep] fixed rewrite cache: {out_path} ({len(fixed)} prompts)")
PY
  else
    echo "[sweep] reusing shared rewrite cache: ${REWRITES_FILE}"
  fi
  validate_rewrite_cache
fi

cell_index=0
failed_cells=()
for backend in ${BACKENDS}; do
  case "${backend}" in
    sid)
      suite="${REPO}/hpsv2_sd35_sid_ddp_suite.sh"
      layout="sd35"; steps=4; fixed_cfg=1.0 ;;
    senseflow_large)
      suite="${REPO}/hpsv2_sd35_sid_ddp_suite.sh"
      layout="sd35"; steps=4; fixed_cfg=1.0 ;;
    sd35_base)
      suite="${REPO}/hpsv2_sd35_sid_ddp_suite.sh"
      layout="sd35"; steps=28; fixed_cfg=4.5 ;;
    flux_schnell)
      suite="${REPO}/hpsv2_flux_schnell_ddp_suite.sh"
      layout="flux"; steps=4; fixed_cfg=0.0 ;;
  esac

  for arm in ${REWARD_ARMS}; do
    case "${arm}" in
      imagereward)
        reward_backend="imagereward"
        server_backends="imagereward" ;;
      hpsv3)
        reward_backend="hpsv3"
        server_backends="hpsv3" ;;
      multi_reward)
        reward_backend="composite_hpsv3_ir"
        server_backends="imagereward hpsv3" ;;
    esac

    model_out="${RUN_ROOT}/${backend}/${arm}"
    method_out="${model_out}/run_${RUN_ID}/bon_fixed_rewrite"
    cell_index=$((cell_index + 1))
    if [[ "${POST_EVAL_ONLY}" == "1" ]]; then
      continue
    fi
    if [[ -s "${method_out}/aggregate_ddp.json" ]]; then
      echo "[resume] complete: ${backend}/${arm}"
      continue
    fi

    echo
    echo "================================================================"
    echo "[sweep] model=${backend} reward_arm=${arm}"
    echo "  backend=${reward_backend} prompts=${PROMPT_COUNT} bon_n=8"
    echo "  steps=${steps} fixed_cfg=${fixed_cfg}"
    echo "  out=${model_out}"
    echo "================================================================"

    common_env=(
      "PROMPT_FILE=${PROMPTS_TXT}"
      "METHODS=bon_fixed_rewrite"
      "START_INDEX=0"
      "END_INDEX=${PROMPT_COUNT}"
      "SEED_MAP_FILE=${RUN_ROOT}/seed_maps/${backend}.json"
      "RUN_TS=${RUN_ID}"
      "OUT_ROOT=${model_out}"
      "STEPS=${steps}"
      "BON_N=8"
      "N_VARIANTS=1"
      "BASELINE_CFG=${fixed_cfg}"
      "CFG_SCALES=${fixed_cfg}"
      "CORRECTION_STRENGTHS=0.0"
      "USE_QWEN=0"
      "PRECOMPUTE_REWRITES=0"
      "REWRITES_FILE=${REWRITES_FILE}"
      "REWARD_BACKEND=${reward_backend}"
      "USE_REWARD_SERVER=1"
      "REWARD_SERVER_REQUIRE_ALL=1"
      "REWARD_SERVER_BACKENDS=${server_backends}"
      "REWARD_SERVER_PORT=$((REWARD_SERVER_BASE_PORT + cell_index))"
      "REWARD_SERVER_MAX_WAIT=${REWARD_SERVER_MAX_WAIT}"
      "REWARD_SERVER_SCORE_TIMEOUT=${REWARD_SERVER_SCORE_TIMEOUT:-300}"
      "REWARD_ENV_CONDA_BASE=${REWARD_ENV_CONDA_BASE}"
      "REWARD_ENV_NAME=${STANDARD_REWARD_ENV_NAME}"
      "SAVE_IMAGES=0"
      "SAVE_BEST_IMAGES=1"
      "SAVE_VARIANTS=1"
      "SAVE_FIRST_K=-1"
      "EVAL_BEST_IMAGES=0"
      "COMPOSITE_IR_LO=${COMPOSITE_IR_LO:--3.0}"
      "COMPOSITE_IR_HI=${COMPOSITE_IR_HI:-3.0}"
      "COMPOSITE_HPSV3_LO=${COMPOSITE_HPSV3_LO:-9.0}"
      "COMPOSITE_HPSV3_HI=${COMPOSITE_HPSV3_HI:-14.0}"
    )
    if [[ "${layout}" == "flux" ]]; then
      common_env+=(
        "FLUX_BACKEND=flux"
        "BASELINE_GUIDANCE_SCALE=0.0"
        "MODEL_ID=${FLUX_MODEL_ID:-black-forest-labs/FLUX.1-schnell}"
      )
    else
      common_env+=("SD35_BACKEND=${backend}")
    fi

    if env "${common_env[@]}" bash "${suite}"; then
      echo "[sweep] OK ${backend}/${arm}"
    else
      rc=$?
      echo "[sweep] FAIL ${backend}/${arm} rc=${rc}" >&2
      failed_cells+=("${backend}/${arm}")
      if [[ "${FAIL_FAST}" == "1" ]]; then
        exit "${rc}"
      fi
    fi
  done
done

if (( ${#failed_cells[@]} > 0 )); then
  echo "[sweep] failed cells: ${failed_cells[*]}" >&2
  exit 1
fi

if [[ "${SKIP_POST_EVAL}" != "1" ]]; then
  if [[ -z "${REWARD_CUDA_VISIBLE_DEVICES:-}" ]]; then
    visible="${CUDA_VISIBLE_DEVICES:-0}"
    REWARD_CUDA_VISIBLE_DEVICES="${visible##*,}"
  fi
  for backend in ${BACKENDS}; do
    model_root="${RUN_ROOT}/${backend}"
    if [[ ! -d "${model_root}" ]]; then
      continue
    fi
    if [[ "${backend}" == "flux_schnell" ]]; then
      layout="flux"
    else
      layout="sd35"
    fi
    echo "[sweep] post-evaluating ${backend} winners against c0"
    OUT_ROOT="${model_root}" \
    REWARD_PY="${STANDARD_REWARD_PY}" \
    STANDARD_REWARD_PY="${STANDARD_REWARD_PY}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    REWARD_SERVER_PORT="${POSTHOC_REWARD_SERVER_PORT}" \
    REWARD_CUDA_VISIBLE_DEVICES="${REWARD_CUDA_VISIBLE_DEVICES}" \
    POSTHOC_EVAL_BACKENDS="imagereward hpsv3 pickscore hpsv2" \
    POSTHOC_ALLOW_MISSING_BACKENDS=0 \
    POSTHOC_LAYOUT="${layout}" \
    HEALTH_TIMEOUT_SECS="${HEALTH_TIMEOUT_SECS}" \
    bash "${REPO}/post_eval_extra_rewards.sh"
  done

  "${PYTHON_BIN}" "${REPO}/tools/merge_posthoc_reward_evals.py" \
    --root "${RUN_ROOT}" \
    --backends imagereward hpsv3 pickscore hpsv2 \
    --summary-csv "${RUN_ROOT}/fixed_rewrite_bon8_reward_sweep_summary.csv" \
    --expected-count "${PROMPT_COUNT}" \
    --strict
fi

echo "[sweep] complete: ${RUN_ROOT}"
echo "[sweep] summary: ${RUN_ROOT}/fixed_rewrite_bon8_reward_sweep_summary.csv"
