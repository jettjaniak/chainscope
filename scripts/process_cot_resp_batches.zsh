#!/bin/zsh

# Find all batch files in the anthropic_batches directory
for batch_file in chainscope/data/anthropic_batches/**/*.yaml; do
    echo "Processing batch file: $batch_file"
    ./scripts/gen_cots.py process-batch "$batch_file"
done 