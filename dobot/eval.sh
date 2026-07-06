#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

TASK="${TASK:-cook}"
PORT="${PORT:-32155}"
URL="${URL:-http://127.0.0.1:${PORT}}"
TEST="${TEST:-inference}"
IMAGE="${IMAGE:-$ROOT_DIR/dobot/dobot_first_frame.jpg}"
STATE_CSV="${STATE_CSV:-0,0,0,0,0,0,0,0,0,0,0,0,0,0}"
BENCHMARK_REQUESTS="${BENCHMARK_REQUESTS:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
INSTRUCTION="${INSTRUCTION:-}"
T5_EMBEDDINGS_PATH="${T5_EMBEDDINGS_PATH:-}"
FRAME_GRID_OUTPUT="${FRAME_GRID_OUTPUT:-}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/lerobot.yaml}"
REPO_ID="${REPO_ID:-}"
EMBODIMENT_TYPE="${EMBODIMENT_TYPE:-local}"
T5_WAN_PATH="${T5_WAN_PATH:-pretrained_models}"
MAX_EPISODES="${MAX_EPISODES:-}"

case "$TASK" in
  cook)
    DATASET_ROOT_DEFAULT="Dobot/dobot_cook_vegetable_full"
    ;;
  pour)
    DATASET_ROOT_DEFAULT="Dobot/dobot_pour_water_full"
    ;;
  *)
    echo "Unsupported TASK='$TASK'. Use TASK=cook or TASK=pour." >&2
    exit 1
    ;;
esac

DATASET_ROOT="${DATASET_ROOT:-$DATASET_ROOT_DEFAULT}"

CMD=(
  python dobot/client.py
  --url "$URL"
  --test "$TEST"
  --benchmark_requests "$BENCHMARK_REQUESTS"
  --dataset_root "$DATASET_ROOT"
  --dataset_config "$DATASET_CONFIG"
  --embodiment_type "$EMBODIMENT_TYPE"
  --t5_wan_path "$T5_WAN_PATH"
  --num_samples "$NUM_SAMPLES"
)

if [[ -n "$IMAGE" && -f "$IMAGE" ]]; then
  CMD+=(--image "$IMAGE")
fi

if [[ -n "$STATE_CSV" ]]; then
  CMD+=(--state_csv "$STATE_CSV")
fi

if [[ -n "$INSTRUCTION" ]]; then
  CMD+=(--instruction "$INSTRUCTION")
fi

if [[ -n "$T5_EMBEDDINGS_PATH" ]]; then
  CMD+=(--t5_embeddings_path "$T5_EMBEDDINGS_PATH")
fi

if [[ -n "$NUM_INFERENCE_STEPS" ]]; then
  CMD+=(--num_inference_steps "$NUM_INFERENCE_STEPS")
fi

if [[ -n "$FRAME_GRID_OUTPUT" ]]; then
  CMD+=(--return_frame_grid --frame_grid_output "$FRAME_GRID_OUTPUT")
fi

if [[ -n "$REPO_ID" ]]; then
  CMD+=(--repo_id "$REPO_ID")
fi

if [[ -n "$MAX_EPISODES" ]]; then
  CMD+=(--max_episodes "$MAX_EPISODES")
fi

if [[ "${DISABLE_AUTO_FIND_T5_EMBEDDINGS:-0}" == "1" ]]; then
  CMD+=(--disable_auto_find_t5_embeddings)
fi

if [[ "${DISABLE_DATASET_T5_FALLBACK:-0}" == "1" ]]; then
  CMD+=(--disable_dataset_t5_fallback)
fi

echo "Running Dobot client eval with:"
printf '  %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
