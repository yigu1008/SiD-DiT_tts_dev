#!/usr/bin/env bash
set -euo pipefail

# Fixed-rewrite, eight-particle BoN control for SiD-SD3.5.
#
# For each original prompt c0:
#   1. Qwen produces exactly one conservative rewrite.
#   2. The same rewrite and CFG=1.0 are fixed across all four denoising steps.
#   3. Eight candidate root seeds are generated.
#   4. Online selection scores every terminal image against c0.
#   5. The winner is post-evaluated against c0 with ImageReward, HPSv3,
#      PickScore, and VQAScore.
#
# The prompt CSV can contain either the 40-prompt human-eval set or the
# 200-prompt GenAI-Bench set. It must contain prompt_id,prompt.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO}/shell_env.sh"

HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
PROMPTS_FILE="${PROMPTS_FILE:-${HUMAN_EVAL_ROOT}/prompts.csv}"
STUDY_ROOT="${STUDY_ROOT:-${HUMAN_EVAL_ROOT}/sid_fixed_rewrite_bon8}"
RUN_ID="${RUN_ID:-v1}"
RUN_ROOT="${STUDY_ROOT}/${RUN_ID}"
SID_OUT_ROOT="${RUN_ROOT}/sid"

SEARCH_REWARD="${SEARCH_REWARD:-vqascore}"
BON_N="${BON_N:-8}"
BASELINE_CFG="${BASELINE_CFG:-1.0}"
GENERATION_BASE_SEED="${GENERATION_BASE_SEED:-12345}"
SOURCE_SEED_MAP_FILE="${SOURCE_SEED_MAP_FILE:-}"

QWEN_ID="${QWEN_ID:-Qwen/Qwen2.5-3B-Instruct}"
QWEN_SYSTEM_PROMPT_FILE="${QWEN_SYSTEM_PROMPT_FILE:-${REPO}/configs/bon8_fixed_rewrite_system_prompt.txt}"
QWEN_PRECOMPUTE_DEVICE="${QWEN_PRECOMPUTE_DEVICE:-auto}"
QWEN_PRECOMPUTE_BATCH_SIZE="${QWEN_PRECOMPUTE_BATCH_SIZE:-4}"
QWEN_PRECOMPUTE_MAX_NEW_TOKENS="${QWEN_PRECOMPUTE_MAX_NEW_TOKENS:-160}"
QWEN_PRECOMPUTE_TEMPERATURE="${QWEN_PRECOMPUTE_TEMPERATURE:-0.6}"
QWEN_PRECOMPUTE_TOP_P="${QWEN_PRECOMPUTE_TOP_P:-0.9}"
REWRITES_OVERWRITE="${REWRITES_OVERWRITE:-0}"

REWARD_ENV_CONDA_BASE="${REWARD_ENV_CONDA_BASE:-/home/ygu/miniconda3}"
VQA_REWARD_ENV_NAME="${VQA_REWARD_ENV_NAME:-vqascore_reward}"
STANDARD_REWARD_ENV_NAME="${STANDARD_REWARD_ENV_NAME:-reward}"
VQASCORE_MODEL="${VQASCORE_MODEL:-clip-flant5-xxl}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5180}"
POSTHOC_REWARD_SERVER_PORT="${POSTHOC_REWARD_SERVER_PORT:-5181}"
REWARD_SERVER_MAX_WAIT="${REWARD_SERVER_MAX_WAIT:-1800}"
HEALTH_TIMEOUT_SECS="${HEALTH_TIMEOUT_SECS:-1800}"

POST_EVAL_ONLY="${POST_EVAL_ONLY:-0}"
SKIP_POST_EVAL="${SKIP_POST_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"

case "${SEARCH_REWARD}" in
  imagereward)
    SEARCH_SERVER_BACKENDS="imagereward"
    ;;
  vqascore)
    SEARCH_SERVER_BACKENDS="vqascore"
    ;;
  composite_ir_vqa)
    SEARCH_SERVER_BACKENDS="imagereward vqascore"
    ;;
  *)
    echo "Error: SEARCH_REWARD must be imagereward, vqascore, or composite_ir_vqa; got ${SEARCH_REWARD}" >&2
    exit 1
    ;;
esac
if [[ "${BON_N}" != "8" ]]; then
  echo "Error: this control is fixed to BON_N=8; got ${BON_N}" >&2
  exit 1
fi
if [[ "${BASELINE_CFG}" != "1.0" && "${BASELINE_CFG}" != "1" ]]; then
  echo "Error: fixed-rewrite BoN-8 uses canonical SiD CFG=1.0; got ${BASELINE_CFG}" >&2
  exit 1
fi
if [[ "${RUN_ID}" == *"/"* ]]; then
  echo "Error: RUN_ID cannot contain '/': ${RUN_ID}" >&2
  exit 1
fi
if [[ ! -f "${PROMPTS_FILE}" ]]; then
  echo "Error: prompt CSV not found: ${PROMPTS_FILE}" >&2
  exit 1
fi
if [[ ! -f "${QWEN_SYSTEM_PROMPT_FILE}" ]]; then
  echo "Error: Qwen system-prompt template not found: ${QWEN_SYSTEM_PROMPT_FILE}" >&2
  exit 1
fi
for value in "${POST_EVAL_ONLY}" "${SKIP_POST_EVAL}" "${DRY_RUN}" "${REWRITES_OVERWRITE}"; do
  case "${value}" in 0|1) ;; *)
    echo "Error: boolean controls must be 0 or 1; got ${value}" >&2
    exit 1
  esac
done

mkdir -p "${RUN_ROOT}"
PROMPTS_TXT="${RUN_ROOT}/prompts.txt"
PROMPTS_SNAPSHOT="${RUN_ROOT}/prompts.csv"
SEED_MAP_FILE="${RUN_ROOT}/root_seed_map.json"
RAW_REWRITES_FILE="${RUN_ROOT}/rewrite_cache_with_c0.json"
REWRITES_FILE="${RUN_ROOT}/fixed_rewrite_cache.json"
STUDY_MANIFEST="${RUN_ROOT}/study_manifest.json"

# If a matching algorithm-sweep map exists for this RUN_ID, use it
# automatically. An explicit SOURCE_SEED_MAP_FILE always wins.
if [[ -z "${SOURCE_SEED_MAP_FILE}" ]]; then
  auto_seed_map="${HUMAN_EVAL_ROOT}/sid_vqascore_algorithm_sweep/${RUN_ID}/shared_root_seed_map.json"
  if [[ -f "${auto_seed_map}" ]]; then
    SOURCE_SEED_MAP_FILE="${auto_seed_map}"
  fi
fi

PROMPTS_FILE="${PROMPTS_FILE}" \
PROMPTS_TXT="${PROMPTS_TXT}" \
PROMPTS_SNAPSHOT="${PROMPTS_SNAPSHOT}" \
SEED_MAP_FILE="${SEED_MAP_FILE}" \
SOURCE_SEED_MAP_FILE="${SOURCE_SEED_MAP_FILE}" \
STUDY_MANIFEST="${STUDY_MANIFEST}" \
QWEN_SYSTEM_PROMPT_FILE="${QWEN_SYSTEM_PROMPT_FILE}" \
QWEN_ID="${QWEN_ID}" \
RUN_ID="${RUN_ID}" \
SEARCH_REWARD="${SEARCH_REWARD}" \
GENERATION_BASE_SEED="${GENERATION_BASE_SEED}" \
BON_N="${BON_N}" \
BASELINE_CFG="${BASELINE_CFG}" \
"${PYTHON_BIN}" - <<'PY'
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

source = Path(os.environ["PROMPTS_FILE"]).expanduser().resolve()
with source.open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
if not rows or not {"prompt_id", "prompt"}.issubset(rows[0]):
    raise SystemExit(f"{source} must contain prompt_id,prompt")
prompt_ids = [str(row["prompt_id"]).strip() for row in rows]
prompts = [str(row["prompt"]).strip() for row in rows]
if len(prompt_ids) != len(set(prompt_ids)):
    raise SystemExit(f"{source} contains duplicate prompt IDs")
if any(not value for value in prompt_ids) or any(not value for value in prompts):
    raise SystemExit(f"{source} contains an empty prompt ID or prompt")
if any("\n" in prompt or "\r" in prompt for prompt in prompts):
    raise SystemExit(f"{source} contains multiline prompts")

Path(os.environ["PROMPTS_TXT"]).write_text(
    "".join(prompt + "\n" for prompt in prompts),
    encoding="utf-8",
)
snapshot = Path(os.environ["PROMPTS_SNAPSHOT"])
if source != snapshot.resolve():
    shutil.copyfile(source, snapshot)

source_seed_path_raw = os.environ.get("SOURCE_SEED_MAP_FILE", "").strip()
source_seed_path = (
    Path(source_seed_path_raw).expanduser().resolve()
    if source_seed_path_raw
    else None
)
seed_source = "derived_for_fixed_rewrite_bon8"
if source_seed_path is not None:
    if not source_seed_path.is_file():
        raise SystemExit(f"SOURCE_SEED_MAP_FILE not found: {source_seed_path}")
    source_payload = json.loads(source_seed_path.read_text(encoding="utf-8"))
    source_seeds = source_payload.get("seeds")
    if not isinstance(source_seeds, dict):
        raise SystemExit(f"{source_seed_path} has no seeds mapping")
    sibling_manifest = source_seed_path.parent / "subset_manifest.json"
    if sibling_manifest.is_file():
        subset_payload = json.loads(sibling_manifest.read_text(encoding="utf-8"))
        subset_rows = subset_payload.get("prompts")
        if isinstance(subset_rows, list):
            subset_ids = [str(row.get("prompt_id", "")) for row in subset_rows]
            if subset_ids != prompt_ids:
                raise SystemExit(
                    f"{source_seed_path} belongs to a different ordered prompt "
                    f"subset according to {sibling_manifest}"
                )
    missing = [str(i) for i in range(len(rows)) if str(i) not in source_seeds]
    if missing:
        raise SystemExit(
            f"{source_seed_path} is missing {len(missing)} prompt indices; "
            f"first={missing[:5]}"
        )
    seeds = {str(i): int(source_seeds[str(i)]) for i in range(len(rows))}
    seed_source = str(source_seed_path)
else:
    base_seed = int(os.environ["GENERATION_BASE_SEED"])
    seeds = {}
    for index, prompt_id in enumerate(prompt_ids):
        payload = (
            f"sid_fixed_rewrite_bon8\0sid\0{prompt_id}\0{base_seed}"
        ).encode()
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        seeds[str(index)] = 1 + value % 2_147_483_646

seed_payload = {
    "model_id": "sid",
    "algorithm_id": "bon_fixed_rewrite",
    "prompt_count": len(rows),
    "prompt_ids": prompt_ids,
    "generation_base_seed": int(os.environ["GENERATION_BASE_SEED"]),
    "seed_source": seed_source,
    "seeds": seeds,
}
seed_path = Path(os.environ["SEED_MAP_FILE"])
if seed_path.is_file():
    if json.loads(seed_path.read_text(encoding="utf-8")) != seed_payload:
        raise SystemExit(f"{seed_path} conflicts with this launch; choose another RUN_ID")
else:
    seed_path.write_text(
        json.dumps(seed_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

system_path = Path(os.environ["QWEN_SYSTEM_PROMPT_FILE"]).expanduser().resolve()
system_text = system_path.read_text(encoding="utf-8").strip()
manifest = {
    "study_id": "sid_fixed_rewrite_bon8",
    "run_id": os.environ["RUN_ID"],
    "model_id": "sid",
    "model_name": "SiD-SD3.5",
    "algorithm_id": "bon_fixed_rewrite",
    "algorithm_label": "Fixed-Rewrite BoN-8",
    "prompt_count": len(rows),
    "prompt_source": str(source),
    "prompt_ids": prompt_ids,
    "seed_map_file": str(seed_path.resolve()),
    "seed_source": seed_source,
    "candidate_count": int(os.environ["BON_N"]),
    "steps_per_candidate": 4,
    "fixed_cfg": float(os.environ["BASELINE_CFG"]),
    "rewrite_count_per_prompt": 1,
    "prompt_rewriter": {
        "model": os.environ["QWEN_ID"],
        "system_prompt_file": str(system_path),
        "system_prompt_sha256": hashlib.sha256(
            system_text.encode("utf-8")
        ).hexdigest(),
        "system_prompt": system_text,
    },
    "trajectory_rule": (
        "One Qwen rewrite is fixed for all four denoising steps and shared by "
        "all eight candidate roots; only candidate seed varies."
    ),
    "search_reward": os.environ["SEARCH_REWARD"],
    "post_evaluation_rewards": [
        "imagereward",
        "hpsv3",
        "pickscore",
        "vqascore",
    ],
    "reward_prompt_invariant": (
        "Online selection and every post-evaluation score use original c0, "
        "never the rewritten generation prompt."
    ),
    "nfe_rule": "Read actual NFE from rank JSONL; never infer it.",
    "suite_diagnostic_baseline": (
        "The shared DDP suite also renders one non-candidate diagnostic image "
        "for delta logging. It is not eligible for BoN selection and is not "
        "included in the BoN rank-row NFE."
    ),
}
manifest_path = Path(os.environ["STUDY_MANIFEST"])
if manifest_path.is_file():
    if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise SystemExit(
            f"{manifest_path} conflicts with this launch; choose another RUN_ID"
        )
else:
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
print(f"[bon8] prompts={len(rows)} source={source}")
print(f"[bon8] seed_source={seed_source}")
print(f"[bon8] manifest={manifest_path}")
PY

PROMPT_COUNT="$("${PYTHON_BIN}" - <<'PY' "${PROMPTS_TXT}"
import sys
print(sum(bool(line.strip()) for line in open(sys.argv[1], encoding="utf-8")))
PY
)"

METHOD_OUT="${SID_OUT_ROOT}/run_${RUN_ID}/bon_fixed_rewrite"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] fixed-rewrite BoN-8"
  echo "  prompts: ${PROMPT_COUNT}"
  echo "  prompt_file: ${PROMPTS_TXT}"
  echo "  system_prompt: ${QWEN_SYSTEM_PROMPT_FILE}"
  echo "  rewrite_cache: ${REWRITES_FILE}"
  echo "  seed_map: ${SEED_MAP_FILE}"
  echo "  method_out: ${METHOD_OUT}"
  echo "  search_reward: ${SEARCH_REWARD}"
  echo "  post_eval: imagereward hpsv3 pickscore vqascore"
  exit 0
fi

VQASCORE_REWARD_PY="${REWARD_ENV_CONDA_BASE}/envs/${VQA_REWARD_ENV_NAME}/bin/python"
STANDARD_REWARD_PY="${REWARD_ENV_CONDA_BASE}/envs/${STANDARD_REWARD_ENV_NAME}/bin/python"
if [[ ! -x "${VQASCORE_REWARD_PY}" ]]; then
  echo "Error: VQAScore reward Python not executable: ${VQASCORE_REWARD_PY}" >&2
  exit 1
fi
if [[ "${SKIP_POST_EVAL}" != "1" && ! -x "${STANDARD_REWARD_PY}" ]]; then
  echo "Error: standard reward Python not executable: ${STANDARD_REWARD_PY}" >&2
  exit 1
fi

if [[ "${POST_EVAL_ONLY}" != "1" ]]; then
  if [[ "${REWRITES_OVERWRITE}" == "1" ]]; then
    rm -f "${RAW_REWRITES_FILE}" "${REWRITES_FILE}"
  fi
  if [[ ! -s "${REWRITES_FILE}" ]]; then
    echo "[bon8] generating exactly one conservative rewrite per c0"
    REWRITE_SYSTEM_OVERRIDE="$(<"${QWEN_SYSTEM_PROMPT_FILE}")" \
    REWRITE_STYLES_OVERRIDE="Produce one conservative rewrite that preserves every semantic requirement. This exact rewrite will be shared by all eight BoN particles." \
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
    if not isinstance(values, list):
        bad.append({"prompt": c0, "reason": "missing_cache_entry"})
        continue
    rewrites = []
    for value in values:
        text = str(value).strip()
        if text and text != c0 and text not in rewrites:
            rewrites.append(text)
    if len(rewrites) != 1:
        bad.append(
            {"prompt": c0, "reason": "expected_one_distinct_rewrite", "found": len(rewrites)}
        )
        continue
    fixed[c0] = [rewrites[0]]
if bad:
    raise SystemExit(
        f"failed to produce exactly one distinct rewrite for {len(bad)} prompts; "
        f"examples={json.dumps(bad[:5], ensure_ascii=False)}. "
        "Set REWRITES_OVERWRITE=1 and rerun."
    )
temporary = out_path.with_name(out_path.name + ".tmp")
temporary.write_text(
    json.dumps(fixed, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
temporary.replace(out_path)
print(f"[bon8] fixed rewrite cache: {out_path} ({len(fixed)} prompts)")
PY
  else
    echo "[bon8] reusing fixed rewrite cache: ${REWRITES_FILE}"
  fi

  REWRITES_FILE="${REWRITES_FILE}" \
  PROMPTS_TXT="${PROMPTS_TXT}" \
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
    missing = [prompt for prompt in prompts if prompt not in cache]
    extra = [prompt for prompt in cache if prompt not in set(prompts)]
    raise SystemExit(
        f"{cache_path} prompt set mismatch: missing={len(missing)} extra={len(extra)}"
    )
bad = [
    prompt
    for prompt in prompts
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
print(f"[bon8] rewrite invariant OK: {len(prompts)} prompts × 1 fixed rewrite")
PY

  if [[ -s "${METHOD_OUT}/aggregate_ddp.json" ]]; then
    echo "[resume] generation already complete: ${METHOD_OUT}"
  else
    echo "[bon8] generating ${PROMPT_COUNT} prompts × 8 particles"
    env \
      "PROMPT_FILE=${PROMPTS_TXT}" \
      "METHODS=bon_fixed_rewrite" \
      "SD35_BACKEND=sid" \
      "START_INDEX=0" \
      "END_INDEX=${PROMPT_COUNT}" \
      "SEED_MAP_FILE=${SEED_MAP_FILE}" \
      "RUN_TS=${RUN_ID}" \
      "OUT_ROOT=${SID_OUT_ROOT}" \
      "STEPS=4" \
      "BON_N=8" \
      "N_VARIANTS=1" \
      "BASELINE_CFG=1.0" \
      "CFG_SCALES=1.0" \
      "CORRECTION_STRENGTHS=0.0" \
      "USE_QWEN=0" \
      "PRECOMPUTE_REWRITES=0" \
      "REWRITES_FILE=${REWRITES_FILE}" \
      "REWARD_BACKEND=${SEARCH_REWARD}" \
      "VQASCORE_MODEL=${VQASCORE_MODEL}" \
      "USE_REWARD_SERVER=1" \
      "REWARD_SERVER_REQUIRE_ALL=1" \
      "REWARD_SERVER_BACKENDS=${SEARCH_SERVER_BACKENDS}" \
      "REWARD_SERVER_PORT=${REWARD_SERVER_PORT}" \
      "REWARD_SERVER_MAX_WAIT=${REWARD_SERVER_MAX_WAIT}" \
      "REWARD_SERVER_SCORE_TIMEOUT=${REWARD_SERVER_SCORE_TIMEOUT:-300}" \
      "REWARD_ENV_CONDA_BASE=${REWARD_ENV_CONDA_BASE}" \
      "REWARD_ENV_NAME=${VQA_REWARD_ENV_NAME}" \
      "SAVE_IMAGES=0" \
      "SAVE_BEST_IMAGES=1" \
      "SAVE_VARIANTS=1" \
      "EVAL_BEST_IMAGES=0" \
      "COMPOSITE_IR_LO=${COMPOSITE_IR_LO:--3.0}" \
      "COMPOSITE_IR_HI=${COMPOSITE_IR_HI:-3.0}" \
      "COMPOSITE_VQASCORE_LO=${COMPOSITE_VQASCORE_LO:-0.0}" \
      "COMPOSITE_VQASCORE_HI=${COMPOSITE_VQASCORE_HI:-1.0}" \
      bash "${REPO}/hpsv2_sd35_sid_ddp_suite.sh"
  fi
fi

if [[ "${SKIP_POST_EVAL}" != "1" ]]; then
  if [[ ! -s "${METHOD_OUT}/aggregate_ddp.json" ]]; then
    echo "Error: cannot post-evaluate; generation is incomplete: ${METHOD_OUT}" >&2
    exit 1
  fi
  if [[ -z "${REWARD_CUDA_VISIBLE_DEVICES:-}" ]]; then
    visible="${CUDA_VISIBLE_DEVICES:-0}"
    REWARD_CUDA_VISIBLE_DEVICES="${visible##*,}"
  fi
  echo "[bon8] post-evaluating selected outputs against c0"
  OUT_ROOT="${RUN_ROOT}" \
  REWARD_PY="${STANDARD_REWARD_PY}" \
  STANDARD_REWARD_PY="${STANDARD_REWARD_PY}" \
  VQASCORE_REWARD_PY="${VQASCORE_REWARD_PY}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  REWARD_SERVER_PORT="${POSTHOC_REWARD_SERVER_PORT}" \
  REWARD_CUDA_VISIBLE_DEVICES="${REWARD_CUDA_VISIBLE_DEVICES}" \
  POSTHOC_EVAL_BACKENDS="imagereward hpsv3 pickscore vqascore" \
  POSTHOC_ALLOW_MISSING_BACKENDS=0 \
  POSTHOC_LAYOUT=sd35 \
  VQASCORE_MODEL="${VQASCORE_MODEL}" \
  HEALTH_TIMEOUT_SECS="${HEALTH_TIMEOUT_SECS}" \
  bash "${REPO}/post_eval_extra_rewards.sh"

  "${PYTHON_BIN}" "${REPO}/tools/merge_posthoc_reward_evals.py" \
    --root "${RUN_ROOT}" \
    --backends imagereward hpsv3 pickscore vqascore \
    --summary-csv "${RUN_ROOT}/fixed_rewrite_bon8_summary.csv" \
    --expected-count "${PROMPT_COUNT}" \
    --strict
fi

echo "[bon8] complete: ${RUN_ROOT}"
echo "[bon8] method output: ${METHOD_OUT}"
echo "[bon8] summary: ${RUN_ROOT}/fixed_rewrite_bon8_summary.csv"
