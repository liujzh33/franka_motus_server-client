#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/bak:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

TASK="${TASK:-place}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8090}"
DEVICE="${DEVICE:-}"
WAN_PATH="${WAN_PATH:-/data/liujingzhi/Motus/pretrained_models/Wan2.2-TI2V-5B}"
VLM_PATH="${VLM_PATH:-/data/liujingzhi/Motus/pretrained_models/Qwen3-VL-2B-Instruct}"
MODEL_CONFIG="${MODEL_CONFIG:-$ROOT_DIR/configs/robotwin_wan_vlm_mask_stage2_franka.yaml}"
T5_EMBEDDINGS_DIR="${T5_EMBEDDINGS_DIR:-}"

case "$TASK" in
  place)
    DEFAULT_INSTRUCTION_FILE_DEFAULT="$ROOT_DIR/franka/place_objects_instruction.txt"
    DEFAULT_T5_EMBEDDINGS_PATH_DEFAULT="$ROOT_DIR/franka/place_objects_instruction.pt"
    ;;
  stack)
    DEFAULT_INSTRUCTION_FILE_DEFAULT="$ROOT_DIR/franka/stack_bowls_instruction.txt"
    DEFAULT_T5_EMBEDDINGS_PATH_DEFAULT="$ROOT_DIR/franka/stack_bowls_instruction.pt"
    ;;
  *)
    echo "Unsupported TASK='$TASK'. Use TASK=place or TASK=stack." >&2
    exit 1
    ;;
esac

# Training uses data/utils/stat.json (key="franka") for BOTH tasks, not per-task stats.json.
DEFAULT_INSTRUCTION_FILE="${DEFAULT_INSTRUCTION_FILE:-$DEFAULT_INSTRUCTION_FILE_DEFAULT}"
DEFAULT_T5_EMBEDDINGS_PATH="${DEFAULT_T5_EMBEDDINGS_PATH:-$DEFAULT_T5_EMBEDDINGS_PATH_DEFAULT}"
DEFAULT_STATS_PATH="${DEFAULT_STATS_PATH:-$ROOT_DIR/data/utils/stat.json}"

: "${CKPT_DIR:?Set CKPT_DIR to the Franka checkpoint directory before running this script.}"

if [[ ! -f "$MODEL_CONFIG" ]]; then
  echo "MODEL_CONFIG not found: $MODEL_CONFIG" >&2
  exit 1
fi

if [[ ! -d "$WAN_PATH" ]]; then
  echo "WAN_PATH not found: $WAN_PATH" >&2
  exit 1
fi

if [[ ! -d "$VLM_PATH" ]]; then
  echo "VLM_PATH not found: $VLM_PATH" >&2
  exit 1
fi

if [[ ! -f "$DEFAULT_INSTRUCTION_FILE" ]]; then
  echo "DEFAULT_INSTRUCTION_FILE not found: $DEFAULT_INSTRUCTION_FILE" >&2
  exit 1
fi

if [[ ! -f "$DEFAULT_T5_EMBEDDINGS_PATH" && -z "$T5_EMBEDDINGS_DIR" ]]; then
  echo "DEFAULT_T5_EMBEDDINGS_PATH not found: $DEFAULT_T5_EMBEDDINGS_PATH" >&2
  echo "Either provide DEFAULT_T5_EMBEDDINGS_PATH or set T5_EMBEDDINGS_DIR." >&2
  exit 1
fi

if [[ ! -f "$DEFAULT_STATS_PATH" ]]; then
  echo "DEFAULT_STATS_PATH not found: $DEFAULT_STATS_PATH" >&2
  echo "Provide the LeRobot v3.0 stats.json for state/action normalization." >&2
  exit 1
fi

CMD=(
  python franka/server_vlm_mask.py
  --model_config "$MODEL_CONFIG"
  --ckpt_dir "$CKPT_DIR"
  --wan_path "$WAN_PATH"
  --vlm_path "$VLM_PATH"
  --default_instruction_file "$DEFAULT_INSTRUCTION_FILE"
  --host "$HOST"
  --port "$PORT"
)

if [[ -n "$DEVICE" ]]; then
  CMD+=(--device "$DEVICE")
fi

if [[ -n "$T5_EMBEDDINGS_DIR" ]]; then
  CMD+=(--t5_embeddings_dir "$T5_EMBEDDINGS_DIR")
fi

if [[ -f "$DEFAULT_T5_EMBEDDINGS_PATH" ]]; then
  CMD+=(--default_t5_embeddings_path "$DEFAULT_T5_EMBEDDINGS_PATH")
fi

if [[ -f "$DEFAULT_STATS_PATH" ]]; then
  CMD+=(--stats_path "$DEFAULT_STATS_PATH")
fi

echo "Launching Franka server with:"
printf '  %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"