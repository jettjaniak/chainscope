#!/bin/bash
for model_id in C3.7S_64K; do
    echo "Processing model $model_id"
    for file in d/questions/**/*.yaml; do
        dataset_id="${file##d/questions/*/}"
        dataset_id="${dataset_id%.yaml}"
        if [[ $dataset_id = wm* ]]; then
            echo "Processing wm dataset $dataset_id"
            ./scripts/iphr/gen_cots.py submit -d "$dataset_id" -m "$model_id" -n 10 -i instr-wm --api ant-batch
        fi
    done
done
