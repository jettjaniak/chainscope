#!/bin/bash

FAITHFULNESS_DIR="d/faithfulness"
SUFFIX="non-ambiguous-hard-2"  # dataset suffix
API="ant"  # API to use

for model_dir in "$FAITHFULNESS_DIR"/*/; do
  model=$(basename "$model_dir")
  echo "▶ Processing model $model"
  python ./scripts/iphr/unfaithfulness_patterns_eval.py submit -m "$model" -s "$SUFFIX" --api "$API"
done 