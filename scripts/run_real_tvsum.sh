#!/bin/bash

set -e

cd "$(dirname "$0")/.."

python main.py \
  --task run_eval \
  --dataset_root /root/autodl-tmp/datasets \
  --dataset_name tvsum \
  --video_id EE-bNr36nyA \
  --llm_mode api \
  --llm_model gpt-4o-mini \
  --segment_mode count \
  --segment_value 8 \
  --caption_frames_per_segment 4 \
  --enable_memory \
  --max_history_segments 8