#!/usr/bin/env bash
set -euo pipefail

# Cheap prompt-search control for the GenAI-Bench 40-prompt study:
#   1. Generate three minimal Qwen rewrites from each original prompt c0.
#   2. Remove c0 from the generation bank.
#   3. Generate BON_N trajectories. Each trajectory uses one rewrite, fixed
#      across all four denoising steps, at the canonical SiD CFG.
#   4. Reward-select against c0.
#   5. Post-evaluate the selected output with raw ImageReward and VQAScore.
#
# Default launch:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 \
#   HUMAN_EVAL_ROOT=/data/ygu/human_eval_genai40_v1 \
#   REWARD_ENV_CONDA_BASE=/home/ygu/miniconda3 \
#   VQA_REWARD_ENV_NAME=vqascore_reward \
#   bash tools/run_sid_rewrite_bon_vqa40.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO}/shell_env.sh"

HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
PROMPTS_CSV="${PROMPTS_FILE:-${HUMAN_EVAL_ROOT}/prompts.csv}"
STUDY_ROOT="${STUDY_ROOT:-${HUMAN_EVAL_ROOT}/sid_rewrite_bon_vqa40}"
RUN_ID="${RUN_ID:-v1}"
SEARCH_REWARD="${SEARCH_REWARD:-vqascore}"
BON_N="${BON_N:-16}"
N_REWRITES="${N_REWRITES:-3}"
VQASCORE_MODEL="${VQASCORE_MODEL:-clip-flant5-xxl}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5140}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen2.5-3B-Instruct}"
QWEN_PRECOMPUTE_BATCH_SIZE="${QWEN_PRECOMPUTE_BATCH_SIZE:-4}"

case "${SEARCH_REWARD}" in
  imagereward|vqascore|composite_ir_vqa) ;;
  *)
    echo "Error: SEARCH_REWARD must be imagereward, vqascore, or composite_ir_vqa; got ${SEARCH_REWARD}" >&2
    exit 1
    ;;
esac
if [[ ! -f "${PROMPTS_CSV}" ]]; then
  echo "Error: prompts CSV not found: ${PROMPTS_CSV}" >&2
  exit 1
fi
if [[ "${RUN_ID}" == *"/"* ]]; then
  echo "Error: RUN_ID cannot contain '/': ${RUN_ID}" >&2
  exit 1
fi
if (( N_REWRITES < 1 )); then
  echo "Error: N_REWRITES must be positive." >&2
  exit 1
fi
if (( QWEN_PRECOMPUTE_BATCH_SIZE < 1 )); then
  echo "Error: QWEN_PRECOMPUTE_BATCH_SIZE must be positive." >&2
  exit 1
fi

resolve_conda_base() {
  if [[ -n "${REWARD_ENV_CONDA_BASE:-}" ]]; then
    printf '%s\n' "${REWARD_ENV_CONDA_BASE}"
    return
  fi
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    cd "$(dirname "${CONDA_EXE}")/.." && pwd
    return
  fi
  local candidate
  for candidate in "${HOME}/miniconda3" "${HOME}/miniforge3" "/opt/conda"; do
    if [[ -x "${candidate}/bin/conda" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  echo "Error: cannot infer Conda base; export REWARD_ENV_CONDA_BASE." >&2
  exit 1
}

REWARD_ENV_CONDA_BASE="$(resolve_conda_base)"
VQA_REWARD_ENV_NAME="${VQA_REWARD_ENV_NAME:-vqascore_reward}"
REWARD_ENV_NAME="${VQA_REWARD_ENV_NAME}"
REWARD_PY="${REWARD_ENV_CONDA_BASE}/envs/${REWARD_ENV_NAME}/bin/python"
if [[ ! -x "${REWARD_PY}" ]]; then
  echo "Error: isolated VQAScore environment not found: ${REWARD_PY}" >&2
  echo "Run: CONDA_BASE=${REWARD_ENV_CONDA_BASE} bash setup_vqascore_reward_env.sh" >&2
  exit 1
fi

mkdir -p "${STUDY_ROOT}"
PROMPTS_TXT="${STUDY_ROOT}/prompts_${RUN_ID}.txt"
SEED_MAP_FILE="${STUDY_ROOT}/shared_seed_map_${RUN_ID}.json"
RAW_REWRITES_FILE="${STUDY_ROOT}/qwen_rewrites_with_c0_${RUN_ID}.json"
REWRITES_FILE="${STUDY_ROOT}/qwen_rewrites_only_${RUN_ID}.json"
STUDY_MANIFEST="${STUDY_ROOT}/study_manifest_${RUN_ID}.json"

PROMPTS_CSV="${PROMPTS_CSV}" \
PROMPTS_TXT="${PROMPTS_TXT}" \
SEED_MAP_FILE="${SEED_MAP_FILE}" \
STUDY_MANIFEST="${STUDY_MANIFEST}" \
RUN_ID="${RUN_ID}" \
SEARCH_REWARD="${SEARCH_REWARD}" \
BON_N="${BON_N}" \
N_REWRITES="${N_REWRITES}" \
VQASCORE_MODEL="${VQASCORE_MODEL}" \
QWEN_ID="${QWEN_ID}" \
QWEN_PRECOMPUTE_BATCH_SIZE="${QWEN_PRECOMPUTE_BATCH_SIZE}" \
"${PYTHON_BIN}" - <<'PY'
import csv
import hashlib
import json
import math
import os
from pathlib import Path

source = Path(os.environ["PROMPTS_CSV"]).expanduser().resolve()
with source.open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 40 or not rows or not {"prompt_id", "prompt"}.issubset(rows[0]):
    raise SystemExit(f"{source} must contain exactly 40 rows with prompt_id and prompt")
expected = [f"p{i:03d}" for i in range(40)]
if [str(row["prompt_id"]) for row in rows] != expected:
    raise SystemExit(f"{source} must contain p000..p039 in order")
prompts = [str(row["prompt"]).strip() for row in rows]
if any(not prompt for prompt in prompts):
    raise SystemExit(f"{source} contains an empty prompt")
if any("\n" in prompt or "\r" in prompt for prompt in prompts):
    raise SystemExit(
        f"{source} contains a multiline prompt, which cannot be represented "
        "by the suite's one-prompt-per-line input"
    )

prompt_txt = Path(os.environ["PROMPTS_TXT"])
prompt_txt.write_text("".join(prompt + "\n" for prompt in prompts), encoding="utf-8")

# Match the focused ActDiff run's root-seed rule so the control uses the same
# prompt-level seed and candidate-root sequence.
base_seed = 12345
seeds = {}
for index, row in enumerate(rows):
    payload = (
        f"sid_bon_mcts_ir_vqa40\0sid\0{row['prompt_id']}\0"
        f"bon_mcts\0{base_seed}"
    ).encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    seeds[str(index)] = 1 + value % 2_147_483_646
seed_payload = {
    "model_id": "sid",
    "algorithm_id": "bon_rewrite_control",
    "matched_to": "sid_bon_mcts_ir_vqa40/bon_mcts",
    "generation_base_seed": base_seed,
    "seeds": seeds,
}
seed_path = Path(os.environ["SEED_MAP_FILE"])
if seed_path.is_file():
    if json.loads(seed_path.read_text(encoding="utf-8")) != seed_payload:
        raise SystemExit(f"{seed_path} conflicts with this launch; choose a new RUN_ID")
else:
    seed_path.write_text(
        json.dumps(seed_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

manifest = {
    "study_id": "sid_rewrite_bon_vqa40",
    "run_id": os.environ["RUN_ID"],
    "model_id": "sid",
    "model_name": "SiD-SD3.5",
    "algorithm_id": "bon_rewrite_control",
    "prompt_count": len(prompts),
    "prompt_source": str(source),
    "seed_map_file": str(seed_path.resolve()),
    "search_reward": os.environ["SEARCH_REWARD"],
    "eval_rewards": ["imagereward", "vqascore"],
    "vqascore_model": os.environ["VQASCORE_MODEL"],
    "candidate_count": int(os.environ["BON_N"]),
    "rewrite_count_per_prompt": int(os.environ["N_REWRITES"]),
    "prompt_rewriter": {
        "model": os.environ["QWEN_ID"],
        "logical_rewrite_outputs": len(prompts) * int(os.environ["N_REWRITES"]),
        "batch_size": int(os.environ["QWEN_PRECOMPUTE_BATCH_SIZE"]),
        "batched_generate_invocations": (
            math.ceil(len(prompts) / int(os.environ["QWEN_PRECOMPUTE_BATCH_SIZE"]))
            * int(os.environ["N_REWRITES"])
        ),
    },
    "trajectory_rule": (
        "Each candidate uses one Qwen rewrite fixed across all four denoising "
        "steps; candidates cycle deterministically over the rewrite-only bank."
    ),
    "cfg_rule": "canonical SiD CFG=1.0 fixed across the trajectory",
    "reward_prompt_invariant": (
        "Online selection and post-evaluation always use original prompt c0."
    ),
    "normalization": {
        "imagereward": [-3.0, 3.0],
        "vqascore": [0.0, 1.0],
    },
}
manifest_path = Path(os.environ["STUDY_MANIFEST"])
if manifest_path.is_file():
    if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise SystemExit(
            f"{manifest_path} conflicts with this launch; choose a new RUN_ID"
        )
else:
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
print(f"[control] prompts: {prompt_txt}")
print(f"[control] seed map: {seed_path}")
print(f"[control] manifest: {manifest_path}")
PY

if [[ ! -s "${REWRITES_FILE}" ]]; then
  echo "[control] generating ${N_REWRITES} Qwen rewrites per prompt"
  "${PYTHON_BIN}" -u "${REPO}/precompute_sd35_rewrites.py" \
    --prompt_file "${PROMPTS_TXT}" \
    --rewrites_file "${RAW_REWRITES_FILE}" \
    --start_index 0 \
    --end_index 40 \
    --n_variants "${N_REWRITES}" \
    --qwen_id "${QWEN_ID}" \
    --qwen_dtype "${QWEN_DTYPE:-bfloat16}" \
    --device "${QWEN_PRECOMPUTE_DEVICE:-auto}" \
    --batch_size "${QWEN_PRECOMPUTE_BATCH_SIZE}" \
    --max_new_tokens "${QWEN_PRECOMPUTE_MAX_NEW_TOKENS:-120}" \
    --temperature "${QWEN_PRECOMPUTE_TEMPERATURE:-0.6}" \
    --top_p "${QWEN_PRECOMPUTE_TOP_P:-0.9}"

  RAW_REWRITES_FILE="${RAW_REWRITES_FILE}" \
  REWRITES_FILE="${REWRITES_FILE}" \
  N_REWRITES="${N_REWRITES}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

raw_path = Path(os.environ["RAW_REWRITES_FILE"])
out_path = Path(os.environ["REWRITES_FILE"])
target = int(os.environ["N_REWRITES"])
raw = json.loads(raw_path.read_text(encoding="utf-8"))
rewrite_only = {}
bad = []
for original, values in raw.items():
    unique = []
    for value in values:
        text = str(value).strip()
        if not text or text == str(original).strip() or text in unique:
            continue
        unique.append(text)
    if len(unique) != target:
        bad.append({"prompt": original, "rewrite_count": len(unique)})
    rewrite_only[str(original)] = unique[:target]
if bad:
    examples = json.dumps(bad[:5], ensure_ascii=False)
    raise SystemExit(
        f"rewrite cache does not contain exactly {target} distinct rewrites "
        f"for {len(bad)} prompts; examples={examples}. Remove {raw_path} and rerun."
    )
temporary = out_path.with_name(out_path.name + ".tmp")
temporary.write_text(
    json.dumps(rewrite_only, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
temporary.replace(out_path)
print(f"[control] rewrite-only cache: {out_path}")
PY
else
  echo "[control] reusing rewrite-only cache: ${REWRITES_FILE}"
fi

echo "[control] launching SiD-SD3.5 prompt-only BoN"
echo "  search_reward: ${SEARCH_REWARD}"
echo "  candidates: ${BON_N}"
echo "  rewrites/prompt: ${N_REWRITES}"
echo "  output: ${STUDY_ROOT}/run_${RUN_ID}/bon_rewrite_control"

env \
  "PROMPT_FILE=${PROMPTS_TXT}" \
  "METHODS=bon_rewrite_control" \
  "SD35_BACKEND=sid" \
  "START_INDEX=0" \
  "END_INDEX=40" \
  "SEED_MAP_FILE=${SEED_MAP_FILE}" \
  "RUN_TS=${RUN_ID}" \
  "OUT_ROOT=${STUDY_ROOT}" \
  "BON_N=${BON_N}" \
  "N_VARIANTS=${N_REWRITES}" \
  "BASELINE_CFG=1.0" \
  "SAVE_IMAGES=0" \
  "SAVE_BEST_IMAGES=1" \
  "SAVE_VARIANTS=1" \
  "EVAL_BEST_IMAGES=1" \
  "EVAL_BACKENDS=imagereward vqascore" \
  "EVAL_ALLOW_MISSING_BACKENDS=0" \
  "EVAL_REWARD_DEVICE=cpu" \
  "USE_REWARD_SERVER=1" \
  "REWARD_SERVER_REQUIRE_ALL=1" \
  "REWARD_SERVER_BACKENDS=imagereward vqascore" \
  "REWARD_SERVER_PORT=${REWARD_SERVER_PORT}" \
  "REWARD_SERVER_MAX_WAIT=${REWARD_SERVER_MAX_WAIT:-1800}" \
  "REWARD_SERVER_SCORE_TIMEOUT=${REWARD_SERVER_SCORE_TIMEOUT:-300}" \
  "REWARD_ENV_CONDA_BASE=${REWARD_ENV_CONDA_BASE}" \
  "REWARD_ENV_NAME=${REWARD_ENV_NAME}" \
  "VQASCORE_MODEL=${VQASCORE_MODEL}" \
  "REWARD_BACKEND=${SEARCH_REWARD}" \
  "USE_QWEN=0" \
  "REWRITES_FILE=${REWRITES_FILE}" \
  "COMPOSITE_IR_LO=${COMPOSITE_IR_LO:--3.0}" \
  "COMPOSITE_IR_HI=${COMPOSITE_IR_HI:-3.0}" \
  "COMPOSITE_VQASCORE_LO=${COMPOSITE_VQASCORE_LO:-0.0}" \
  "COMPOSITE_VQASCORE_HI=${COMPOSITE_VQASCORE_HI:-1.0}" \
  bash "${REPO}/hpsv2_sd35_sid_ddp_suite.sh"

echo "[control] done: ${STUDY_ROOT}/run_${RUN_ID}/bon_rewrite_control"
