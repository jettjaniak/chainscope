#!/bin/bash

FAITHFULNESS_DIR="d/faithfulness"
SUFFIX="non-ambiguous-hard-2"  # dataset suffix
API="ant-batch"  # API to use

wait_for_batches() {
  local api="$1"
  echo "Checking ${api} batches..."
  while true; do
    if python ./scripts/other/check_batches.py --api "${api}" 2>&1 | grep -q "Pending: 0"; then
      echo "All ${api} batches are completed."
      break
    fi
    echo "Waiting for ${api} batches to complete..."
    sleep 300  # Check every 5 minutes
  done
}

for model_dir in "$FAITHFULNESS_DIR"/*/; do
  model=$(basename "$model_dir")
  echo "▶ Processing model $model"
  python ./scripts/iphr/unfaithfulness_patterns_eval.py submit -v -m "$model" -s "$SUFFIX" --api "$API"
done 

if [ "$API" == "ant-batch" ]; then
  wait_for_batches "$API"
  python ./scripts/iphr/unfaithfulness_patterns_eval.py process
fi
