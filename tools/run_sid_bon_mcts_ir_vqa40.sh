#!/usr/bin/env bash
set -euo pipefail

# Focused GenAI-Bench study:
#   SiD-SD3.5 + BoN-MCTS on the same 40 prompts/root pools, guided by
#   (1) ImageReward, (2) VQAScore, and (3) normalized 50/50 IR + VQAScore.
#
# Required input:
#   PROMPTS_FILE=/data/ygu/human_eval_genai40_v1/prompts.csv
#
# Typical launch:
#   HUMAN_EVAL_ROOT=/data/ygu/human_eval_genai40_v1 \
#   REWARD_ENV_CONDA_BASE=/home/ygu/miniconda3 \
#   REWARD_ENV_NAME=sid_dit \
#   bash tools/run_sid_bon_mcts_ir_vqa40.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO}/shell_env.sh"

HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
PROMPTS_FILE="${PROMPTS_FILE:-${HUMAN_EVAL_ROOT}/prompts.csv}"
STUDY_ROOT="${STUDY_ROOT:-${HUMAN_EVAL_ROOT}/sid_bon_mcts_ir_vqa40}"
STUDY_ID="${STUDY_ID:-sid_bon_mcts_ir_vqa40}"
RUN_ID="${RUN_ID:-v1}"
GENERATION_BASE_SEED="${GENERATION_BASE_SEED:-12345}"
VQASCORE_MODEL="${VQASCORE_MODEL:-clip-flant5-xxl}"
USE_QWEN="${USE_QWEN:-0}"

N_SIMS="${N_SIMS:-25}"
BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS:-8}"
BON_MCTS_TOPK="${BON_MCTS_TOPK:-2}"
BON_MCTS_SIM_ALLOC="${BON_MCTS_SIM_ALLOC:-split}"
BON_MCTS_MIN_SIMS="${BON_MCTS_MIN_SIMS:-8}"
REWARD_SERVER_BASE_PORT="${REWARD_SERVER_BASE_PORT:-5110}"

if [[ ! -f "${PROMPTS_FILE}" ]]; then
  echo "Error: prepared prompt CSV not found: ${PROMPTS_FILE}" >&2
  exit 1
fi
if [[ "${RUN_ID}" == *"/"* ]]; then
  echo "Error: RUN_ID cannot contain '/': ${RUN_ID}" >&2
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
REWARD_ENV_NAME="${REWARD_ENV_NAME:-sid_dit}"
REWARD_PY="${REWARD_ENV_CONDA_BASE}/envs/${REWARD_ENV_NAME}/bin/python"
if [[ ! -x "${REWARD_PY}" ]]; then
  echo "Error: reward environment Python not found: ${REWARD_PY}" >&2
  echo "Set REWARD_ENV_CONDA_BASE and REWARD_ENV_NAME to the environment containing ImageReward and t2v-metrics." >&2
  exit 1
fi

if ! (
  cd "${REPO}"
  "${REWARD_PY}" - <<'PY'
import importlib.metadata as md

import reward_server

reward_server._inject_wandb_stub()
import ImageReward  # noqa: F401,E402
import t2v_metrics  # noqa: F401

version = md.version("t2v-metrics")
if version.split(".", 1)[0] != "3":
    raise SystemExit(
        f"t2v-metrics==3.x is required for clip-flant5-xxl; found {version}"
    )
print(f"[preflight] ImageReward OK; t2v-metrics={version}")
PY
)
then
  echo "Error: the reward environment is missing the compatible VQAScore runtime." >&2
  echo "Install the legacy CLIP-FlanT5 release, then rerun:" >&2
  echo "  ${REWARD_PY} -m pip install 't2v-metrics==3.0'" >&2
  exit 1
fi

mkdir -p "${STUDY_ROOT}"
SEED_MAP_FILE="${STUDY_ROOT}/shared_bon_mcts_seed_map_${RUN_ID}.json"
STUDY_MANIFEST="${STUDY_ROOT}/study_manifest_${RUN_ID}.json"
PROMPT_SNAPSHOT="${STUDY_ROOT}/prompts_${RUN_ID}.csv"
REWRITES_FILE="${REWRITES_FILE:-${STUDY_ROOT}/shared_rewrites_cache_${RUN_ID}.json}"

PROMPTS_FILE="${PROMPTS_FILE}" \
PROMPT_SNAPSHOT="${PROMPT_SNAPSHOT}" \
SEED_MAP_FILE="${SEED_MAP_FILE}" \
STUDY_MANIFEST="${STUDY_MANIFEST}" \
STUDY_ID="${STUDY_ID}" \
RUN_ID="${RUN_ID}" \
GENERATION_BASE_SEED="${GENERATION_BASE_SEED}" \
N_SIMS="${N_SIMS}" \
BON_MCTS_N_SEEDS="${BON_MCTS_N_SEEDS}" \
BON_MCTS_TOPK="${BON_MCTS_TOPK}" \
BON_MCTS_SIM_ALLOC="${BON_MCTS_SIM_ALLOC}" \
BON_MCTS_MIN_SIMS="${BON_MCTS_MIN_SIMS}" \
VQASCORE_MODEL="${VQASCORE_MODEL}" \
USE_QWEN="${USE_QWEN}" \
REWRITES_FILE="${REWRITES_FILE}" \
"${PYTHON_BIN}" - <<'PY'
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

source = Path(os.environ["PROMPTS_FILE"]).expanduser().resolve()
snapshot = Path(os.environ["PROMPT_SNAPSHOT"]).expanduser().resolve()
with source.open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))

required = {"prompt_id", "prompt"}
if not rows or not required.issubset(rows[0]):
    raise SystemExit(f"{source} must contain columns {sorted(required)}")
expected_ids = [f"p{i:03d}" for i in range(40)]
actual_ids = [str(row["prompt_id"]) for row in rows]
if len(rows) != 40 or actual_ids != expected_ids:
    raise SystemExit(
        f"{source} must contain exactly p000..p039 in order; "
        f"found {len(rows)} rows with ids {actual_ids[:3]}...{actual_ids[-3:]}"
    )
if any(not str(row["prompt"]).strip() for row in rows):
    raise SystemExit(f"{source} contains an empty prompt")

snapshot.parent.mkdir(parents=True, exist_ok=True)
if source != snapshot:
    shutil.copyfile(source, snapshot)

study_id = os.environ["STUDY_ID"]
base_seed = int(os.environ["GENERATION_BASE_SEED"])
seeds = {}
prompt_records = []
for index, row in enumerate(rows):
    prompt_id = str(row["prompt_id"])
    payload = (
        f"{study_id}\0sid\0{prompt_id}\0bon_mcts\0{base_seed}".encode()
    )
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    seed = 1 + value % 2_147_483_646
    seeds[str(index)] = seed
    prompt_records.append(
        {
            "prompt_index": index,
            "prompt_id": prompt_id,
            "prompt": str(row["prompt"]),
            "base_seed": seed,
        }
    )

seed_payload = {
    "study_id": study_id,
    "model_id": "sid",
    "algorithm_id": "bon_mcts",
    "seed_rule": "stable_sha256(study_id,sid,prompt_id,bon_mcts,base_seed)",
    "generation_base_seed": base_seed,
    "seeds": seeds,
}
seed_path = Path(os.environ["SEED_MAP_FILE"])
if seed_path.is_file():
    existing_seed_payload = json.loads(seed_path.read_text(encoding="utf-8"))
    if existing_seed_payload != seed_payload:
        raise SystemExit(
            f"{seed_path} conflicts with this launch; choose a new RUN_ID"
        )
else:
    seed_path.write_text(
        json.dumps(seed_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
manifest = {
    "study_id": study_id,
    "run_id": os.environ["RUN_ID"],
    "model_id": "sid",
    "model_name": "SiD-SD3.5",
    "algorithm_id": "bon_mcts",
    "prompt_count": 40,
    "prompts_file": str(snapshot),
    "prompts_sha256": source_sha256,
    "seed_map_file": str(Path(os.environ["SEED_MAP_FILE"]).resolve()),
    "root_comparison": (
        "All reward arms use the same deterministic candidate-root pool. "
        "BoN-MCTS can select a different winning root in each arm."
    ),
    "prompt_variant_control": {
        "use_qwen": bool(int(os.environ["USE_QWEN"])),
        "shared_rewrites_file": os.environ["REWRITES_FILE"],
        "rule": (
            "All reward arms reuse the same rewrite cache when Qwen is enabled; "
            "otherwise the sampler's deterministic legacy prompt variants are used."
        ),
    },
    "reward_prompt_invariant": (
        "Every online reward call uses the original GenAI-Bench prompt c0, "
        "never a generated prompt variant."
    ),
    "search_budget": {
        "n_sims": int(os.environ["N_SIMS"]),
        "prescreen_seed_count": int(os.environ["BON_MCTS_N_SEEDS"]),
        "prescreen_topk": int(os.environ["BON_MCTS_TOPK"]),
        "simulation_allocation": os.environ["BON_MCTS_SIM_ALLOC"],
        "minimum_sims_per_root": int(os.environ["BON_MCTS_MIN_SIMS"]),
        "actual_nfe": "read from each output JSONL; do not infer",
    },
    "reward_arms": [
        {
            "arm_id": "imagereward",
            "reward_backend": "imagereward",
            "server_backends": ["imagereward"],
        },
        {
            "arm_id": "vqascore",
            "reward_backend": "vqascore",
            "server_backends": ["vqascore"],
            "vqascore_model": os.environ["VQASCORE_MODEL"],
        },
        {
            "arm_id": "ir_vqa_equal",
            "reward_backend": "composite_ir_vqa",
            "server_backends": ["imagereward", "vqascore"],
            "formula": "0.5*minmax(ImageReward,-3,3) + 0.5*minmax(VQAScore,0,1)",
        },
    ],
    "normalization": {
        "imagereward": [-3.0, 3.0],
        "vqascore": [0.0, 1.0],
    },
    "prompts": prompt_records,
}
manifest_path = Path(os.environ["STUDY_MANIFEST"])
if manifest_path.is_file():
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing_manifest != manifest:
        raise SystemExit(
            f"{manifest_path} conflicts with this launch; choose a new RUN_ID"
        )
else:
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
print(f"[study] validated {len(rows)} prompts")
print(f"[study] seed map: {os.environ['SEED_MAP_FILE']}")
print(f"[study] manifest: {os.environ['STUDY_MANIFEST']}")
PY

common_env=(
  "PROMPT_FILE=${PROMPT_SNAPSHOT}"
  "METHODS=bon_mcts"
  "SD35_BACKEND=sid"
  "START_INDEX=0"
  "END_INDEX=40"
  "SEED_MAP_FILE=${SEED_MAP_FILE}"
  "RUN_TS=${RUN_ID}"
  "N_SIMS=${N_SIMS}"
  "BON_MCTS_N_SEEDS=${BON_MCTS_N_SEEDS}"
  "BON_MCTS_TOPK=${BON_MCTS_TOPK}"
  "BON_MCTS_SIM_ALLOC=${BON_MCTS_SIM_ALLOC}"
  "BON_MCTS_MIN_SIMS=${BON_MCTS_MIN_SIMS}"
  "SAVE_IMAGES=0"
  "SAVE_BEST_IMAGES=1"
  "SAVE_VARIANTS=1"
  "EVAL_BEST_IMAGES=0"
  "USE_REWARD_SERVER=1"
  "REWARD_SERVER_REQUIRE_ALL=1"
  "REWARD_SERVER_MAX_WAIT=${REWARD_SERVER_MAX_WAIT:-1800}"
  "REWARD_SERVER_SCORE_TIMEOUT=${REWARD_SERVER_SCORE_TIMEOUT:-300}"
  "REWARD_ENV_CONDA_BASE=${REWARD_ENV_CONDA_BASE}"
  "REWARD_ENV_NAME=${REWARD_ENV_NAME}"
  "VQASCORE_MODEL=${VQASCORE_MODEL}"
  "USE_QWEN=${USE_QWEN}"
  "REWRITES_FILE=${REWRITES_FILE}"
  "REWRITES_OVERWRITE=0"
  "COMPOSITE_IR_LO=${COMPOSITE_IR_LO:--3.0}"
  "COMPOSITE_IR_HI=${COMPOSITE_IR_HI:-3.0}"
  "COMPOSITE_VQASCORE_LO=${COMPOSITE_VQASCORE_LO:-0.0}"
  "COMPOSITE_VQASCORE_HI=${COMPOSITE_VQASCORE_HI:-1.0}"
)

arms=(
  "imagereward|imagereward|imagereward"
  "vqascore|vqascore|vqascore"
  "ir_vqa_equal|composite_ir_vqa|imagereward vqascore"
)

for arm_spec in "${arms[@]}"; do
  IFS='|' read -r arm_id reward_backend server_backends <<< "${arm_spec}"
  arm_root="${STUDY_ROOT}/${arm_id}"
  port="${REWARD_SERVER_BASE_PORT}"
  REWARD_SERVER_BASE_PORT=$(( REWARD_SERVER_BASE_PORT + 1 ))
  echo
  echo "[study] arm=${arm_id} reward=${reward_backend} server_backends=${server_backends}"
  echo "[study] output=${arm_root}/run_${RUN_ID}"
  env \
    "${common_env[@]}" \
    "OUT_ROOT=${arm_root}" \
    "REWARD_BACKEND=${reward_backend}" \
    "REWARD_SERVER_BACKENDS=${server_backends}" \
    "REWARD_SERVER_PORT=${port}" \
    bash "${REPO}/hpsv2_sd35_sid_ddp_suite.sh"
done

echo
echo "[study] all three arms completed: ${STUDY_ROOT}"
