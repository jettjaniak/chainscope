#!/bin/bash

# Base directory for COT responses
BASE_DIR="chainscope/data/cot_responses/instr-v0/T0.7_P0.9_M2000"

# Find all yaml files recursively in the base directory
find "$BASE_DIR" -type f -name "*.yaml" | while read -r file; do
    # Extract the relative path from BASE_DIR
    rel_path="${file#$BASE_DIR/}"
    
    # Construct the corresponding path in answer_flipping_eval
    eval_path="chainscope/data/answer_flipping_eval/instr-v0/T0.7_P0.9_M2000/$rel_path"
    
    # Check if the file exists in answer_flipping_eval
    if [ ! -f "$eval_path" ]; then
        echo "Processing: $file"
        python scripts/other/eval_answer_flipping.py --an "$file"
    else
        echo "Skipping: $file (already processed)"
    fi
done
