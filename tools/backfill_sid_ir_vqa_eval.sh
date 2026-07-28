#!/usr/bin/env bash
set -euo pipefail

# Add raw ImageReward and VQAScore evaluation files to already-generated
# focused SiD runs without regenerating images.
#
# Example:
#   HUMAN_EVAL_ROOT=/data/ygu/human_eval_genai40_v1 \
#   REWARD_ENV_CONDA_BASE=/home/ygu/miniconda3 \
#   VQA_REWARD_ENV_NAME=vqascore_reward \
#   REWARD_CUDA_VISIBLE_DEVICES=7 \
#   bash tools/backfill_sid_ir_vqa_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO}/shell_env.sh"

HUMAN_EVAL_ROOT="${HUMAN_EVAL_ROOT:-/data/ygu/human_eval_genai40_v1}"
OUT_ROOT="${OUT_ROOT:-${HUMAN_EVAL_ROOT}/sid_bon_mcts_ir_vqa40}"
REWARD_ENV_CONDA_BASE="${REWARD_ENV_CONDA_BASE:-/home/ygu/miniconda3}"
VQA_REWARD_ENV_NAME="${VQA_REWARD_ENV_NAME:-vqascore_reward}"
REWARD_PY="${REWARD_ENV_CONDA_BASE}/envs/${VQA_REWARD_ENV_NAME}/bin/python"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-5150}"

if [[ ! -d "${OUT_ROOT}" ]]; then
  echo "Error: focused run root not found: ${OUT_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${REWARD_PY}" ]]; then
  echo "Error: isolated VQAScore Python not found: ${REWARD_PY}" >&2
  exit 1
fi

OUT_ROOT="${OUT_ROOT}" \
REWARD_PY="${REWARD_PY}" \
PYTHON_BIN="${PYTHON_BIN}" \
REWARD_SERVER_PORT="${REWARD_SERVER_PORT}" \
REWARD_CUDA_VISIBLE_DEVICES="${REWARD_CUDA_VISIBLE_DEVICES:-7}" \
POSTHOC_EVAL_BACKENDS="imagereward vqascore" \
POSTHOC_LAYOUT=sd35 \
VQASCORE_MODEL="${VQASCORE_MODEL:-clip-flant5-xxl}" \
HEALTH_TIMEOUT_SECS="${HEALTH_TIMEOUT_SECS:-1800}" \
bash "${REPO}/post_eval_extra_rewards.sh"

# Merge the two memory-bounded, one-backend passes into the standard combined
# evaluation artifact expected by the reporting tools.
OUT_ROOT="${OUT_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
import math
import os
import statistics
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
merged_count = 0
for ir_path in sorted(root.rglob("best_images_imagereward.json")):
    method_dir = ir_path.parent
    vqa_path = method_dir / "best_images_vqascore.json"
    if not vqa_path.is_file():
        continue
    ir_payload = json.loads(ir_path.read_text(encoding="utf-8"))
    vqa_payload = json.loads(vqa_path.read_text(encoding="utf-8"))
    ir_rows = ir_payload.get("rows", [])
    vqa_rows = vqa_payload.get("rows", [])

    def key(row):
        return (
            int(row.get("prompt_index", -1)),
            str(row.get("slug", "")),
            int(row.get("sample_index", 0)),
        )

    by_key = {key(row): dict(row) for row in ir_rows}
    for row in vqa_rows:
        row_key = key(row)
        if row_key not in by_key:
            raise SystemExit(f"{method_dir}: VQAScore row has no ImageReward match: {row_key}")
        by_key[row_key].setdefault("scores", {}).update(row.get("scores", {}))
    if len(by_key) != len(vqa_rows):
        raise SystemExit(
            f"{method_dir}: ImageReward/VQAScore row count mismatch "
            f"({len(by_key)} vs {len(vqa_rows)})"
        )
    rows = [by_key[row_key] for row_key in sorted(by_key)]
    stats = {}
    for backend in ("imagereward", "vqascore"):
        values = [
            float(row["scores"][backend])
            for row in rows
            if isinstance(row.get("scores", {}).get(backend), (int, float))
            and math.isfinite(float(row["scores"][backend]))
        ]
        stats[backend] = {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    aggregate = {
        "layout": "sd35",
        "method": method_dir.name,
        "method_out": str(method_dir.resolve()),
        "backends_requested": ["imagereward", "vqascore"],
        "num_images_found": len(rows),
        "num_images_scored": len(rows),
        "backend_stats": stats,
        "backfill_source_files": [str(ir_path.resolve()), str(vqa_path.resolve())],
    }
    (method_dir / "best_images_multi_reward.json").write_text(
        json.dumps({"aggregate": aggregate, "rows": rows}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    (method_dir / "best_images_multi_reward_aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    merged_count += 1
print(f"[backfill] wrote combined IR+VQAScore artifacts for {merged_count} method dirs")
PY
