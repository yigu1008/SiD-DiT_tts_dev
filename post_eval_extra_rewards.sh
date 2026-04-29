#!/usr/bin/env bash
# Phase 2 of an HPSv3-server run: re-evaluate every method's saved best images
# with one extra reward backend at a time, restarting the reward server per
# backend so GPU 0 only ever holds a single reward model.
#
# Required env (set by the caller / YAML):
#   OUT_ROOT                : run output root (contains run_<TS>/<method>/...)
#   REWARD_PY               : python in the isolated reward env
#   REWARD_SERVER_PORT      : port the previous server used (we kill+reuse)
#   REWARD_HF_HOME          : HF_HOME for the reward env (local-staged or shared)
#   POSTHOC_EVAL_BACKENDS   : space-sep list of extra backends, e.g.
#                             "imagereward hpsv2 pickscore"
#   IMAGE_REWARD_MODEL      : default "ImageReward-v1.0"
#   PICKSCORE_MODEL         : default "yuvalkirstain/PickScore_v1"
#   PYTHON_BIN              : ptca python (used for evaluate_best_images_multi_reward.py)
#
# Optional:
#   POSTHOC_LAYOUT          : sd35 (default) | flux | sana
#   HEALTH_TIMEOUT_SECS     : seconds to wait for /health (default 1800)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${OUT_ROOT:?OUT_ROOT required}"
: "${REWARD_PY:?REWARD_PY required}"
: "${REWARD_SERVER_PORT:?REWARD_SERVER_PORT required}"
: "${POSTHOC_EVAL_BACKENDS:?POSTHOC_EVAL_BACKENDS required (e.g. 'imagereward hpsv2 pickscore')}"
: "${PYTHON_BIN:?PYTHON_BIN required}"

POSTHOC_LAYOUT="${POSTHOC_LAYOUT:-sd35}"
IMAGE_REWARD_MODEL="${IMAGE_REWARD_MODEL:-ImageReward-v1.0}"
PICKSCORE_MODEL="${PICKSCORE_MODEL:-yuvalkirstain/PickScore_v1}"
HEALTH_TIMEOUT_SECS="${HEALTH_TIMEOUT_SECS:-1800}"
REWARD_HF_HOME="${REWARD_HF_HOME:-${HF_HOME:-/mnt/data/v-yigu/model_cache/hf_cache}}"

REWARD_SERVER_URL="http://localhost:${REWARD_SERVER_PORT}"

# Ensure no leftover server is bound to the port (Phase 1 server should already
# have been killed by the caller's trap, but be defensive).
fuser -k "${REWARD_SERVER_PORT}/tcp" >/dev/null 2>&1 || true
sleep 2

start_server_for_backend() {
  local backend="$1"
  local log_path="$2"
  echo "[posthoc] starting reward server (${backend}) on port ${REWARD_SERVER_PORT}"
  env -u NCCL_P2P_LEVEL -u NCCL_ASYNC_ERROR_HANDLING -u TORCH_NCCL_ASYNC_ERROR_HANDLING \
      CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_HOME="${REWARD_HF_HOME}" \
      "${REWARD_PY}" -u "${SCRIPT_DIR}/reward_server.py" \
        --port "${REWARD_SERVER_PORT}" \
        --device cuda:0 \
        --backends "${backend}" \
        --image_reward_model "${IMAGE_REWARD_MODEL}" \
        --pickscore_model "${PICKSCORE_MODEL}" \
      > "${log_path}" 2>&1 &
  echo $!
}

wait_for_health() {
  local pid="$1"
  local log_path="$2"
  local max_iters=$(( HEALTH_TIMEOUT_SECS / 3 ))
  for i in $(seq 1 "${max_iters}"); do
    if curl -s "${REWARD_SERVER_URL}/health" >/dev/null 2>&1; then
      echo "[posthoc] reward server healthy after $((i*3))s"
      return 0
    fi
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "[posthoc] reward server pid=${pid} died before health" >&2
      tail -n 200 "${log_path}" >&2 || true
      return 1
    fi
    if (( i % 10 == 0 )); then
      echo "[posthoc] [health-wait] i=${i} log tail:" >&2
      tail -n 20 "${log_path}" >&2 || true
    fi
    sleep 3
  done
  echo "[posthoc] reward server failed to become healthy in ${HEALTH_TIMEOUT_SECS}s" >&2
  tail -n 400 "${log_path}" >&2 || true
  return 1
}

# Discover (method_out, method_name) pairs by walking run_*/<method>/aggregate_ddp.json.
mapfile -t METHOD_DIRS < <(find "${OUT_ROOT}" -maxdepth 6 -type f -name aggregate_ddp.json 2>/dev/null | sed 's|/aggregate_ddp.json$||' | sort)
if [[ "${#METHOD_DIRS[@]}" -eq 0 ]]; then
  echo "[posthoc] no method dirs with aggregate_ddp.json under ${OUT_ROOT}; nothing to evaluate" >&2
  exit 0
fi

echo "[posthoc] found ${#METHOD_DIRS[@]} method dirs:"
for d in "${METHOD_DIRS[@]}"; do echo "  - ${d}"; done

eval_one_dir_one_backend() {
  local method_out="$1"
  local backend="$2"
  local method_name
  method_name="$(basename "${method_out}")"
  local out_json="${method_out}/best_images_${backend}.json"
  local out_agg="${method_out}/best_images_${backend}_aggregate.json"
  echo "[posthoc] eval method=${method_name} backend=${backend} dir=${method_out}"
  REWARD_SERVER_URL="${REWARD_SERVER_URL}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_best_images_multi_reward.py" \
      --layout "${POSTHOC_LAYOUT}" \
      --method_out "${method_out}" \
      --method "${method_name}" \
      --backends "${backend}" \
      --reward_device cuda \
      --image_reward_model "${IMAGE_REWARD_MODEL}" \
      --pickscore_model "${PICKSCORE_MODEL}" \
      --out_json "${out_json}" \
      --out_aggregate "${out_agg}" \
      --allow_missing_backends \
    || echo "[posthoc] WARN: eval failed for method=${method_name} backend=${backend}"
}

for backend in ${POSTHOC_EVAL_BACKENDS}; do
  echo "════════════════════════════════════════════════════════════════════"
  echo "[posthoc] backend=${backend}"
  echo "════════════════════════════════════════════════════════════════════"
  log_path="${OUT_ROOT}/reward_server_${backend}.log"
  pid="$(start_server_for_backend "${backend}" "${log_path}")"
  trap 'kill "'"${pid}"'" >/dev/null 2>&1 || true' EXIT
  if ! wait_for_health "${pid}" "${log_path}"; then
    echo "[posthoc] skipping backend=${backend} due to unhealthy server" >&2
    kill "${pid}" >/dev/null 2>&1 || true
    sleep 5
    continue
  fi

  for method_out in "${METHOD_DIRS[@]}"; do
    eval_one_dir_one_backend "${method_out}" "${backend}"
  done

  echo "[posthoc] tearing down server for backend=${backend}"
  kill "${pid}" >/dev/null 2>&1 || true
  for j in $(seq 1 30); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then break; fi
    sleep 1
  done
  fuser -k "${REWARD_SERVER_PORT}/tcp" >/dev/null 2>&1 || true
  sleep 5
  trap - EXIT
done

echo "[posthoc] done."
