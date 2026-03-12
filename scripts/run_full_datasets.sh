#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/datasets}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/autodl-tmp/datasets/splits}"
LLM_MODE="${LLM_MODE:-api}"
LLM_MODEL="${LLM_MODEL:-DeepSeek-V3.2}"
SEGMENT_MODE="${SEGMENT_MODE:-count}"
SEGMENT_VALUE="${SEGMENT_VALUE:-8}"
CAPTION_FRAMES_PER_SEGMENT="${CAPTION_FRAMES_PER_SEGMENT:-4}"
MAX_HISTORY_SEGMENTS="${MAX_HISTORY_SEGMENTS:-8}"
SPLIT_COUNT="${SPLIT_COUNT:-5}"

COMMON_ARGS=(
  --task run_eval
  --dataset_root "$DATASET_ROOT"
  --llm_mode "$LLM_MODE"
  --llm_model "$LLM_MODEL"
  --segment_mode "$SEGMENT_MODE"
  --segment_value "$SEGMENT_VALUE"
  --caption_frames_per_segment "$CAPTION_FRAMES_PER_SEGMENT"
  --split_root "$SPLIT_ROOT"
  --split_count "$SPLIT_COUNT"
  --enable_memory
  --max_history_segments "$MAX_HISTORY_SEGMENTS"
)

python main.py "${COMMON_ARGS[@]}" --dataset_name summe
python main.py "${COMMON_ARGS[@]}" --dataset_name tvsum