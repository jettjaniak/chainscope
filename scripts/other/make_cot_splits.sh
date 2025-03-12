#!/bin/bash

# Base directory for COT responses
BASE_DIR="chainscope/data/cot_responses/instr-v0/T0.7_P0.9_M2000"

# Find all yaml files recursively in the base directory
find "$BASE_DIR" -type f -name "*.yaml" | while read -r file; do
    # Extract the relative path from BASE_DIR
    rel_path="${file#$BASE_DIR/}"
    
    # Construct the corresponding path in split_cot_responses
    split_path="chainscope/data/split_cot_responses/instr-v0/T0.7_P0.9_M2000/$rel_path"
    
    # Check if the file exists in split_cot_responses
    if [ ! -f "$split_path" ]; then
        echo "Processing: $file"
        # Run split_cots.py on the file
        ./scripts/other/split_cots.py "$file"
    else
        echo "Skipping: $file (already processed)"
    fi
done