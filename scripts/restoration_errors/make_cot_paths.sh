#!/bin/bash
for model_id in GPT4O; do
    echo "Processing model $model_id"

    for dataset_id in gsm8k math mmlu; do
        echo "Processing dataset $dataset_id"
        ./scripts/restoration_errors/gen_cot_paths.py -d "$dataset_id" -m "$model_id" -n 1 --oa --append
    done
done
