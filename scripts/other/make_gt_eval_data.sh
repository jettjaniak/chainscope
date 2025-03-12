#!/bin/bash
for model_id in C3H C3S C3O C3.5H C3.5S; do
    echo "Processing model $model_id"

    for file in d/questions/**/*.yaml; do
        dataset_id="${file##d/questions/*/}"
        dataset_id="${dataset_id%.yaml}"
        if [[ $dataset_id = *tests* ]]; then
            continue
        fi
        if [[ $dataset_id = animals-speed* || $dataset_id = sea-depths* || $dataset_id = sound-speeds* || $dataset_id = train-speeds* ]]; then
            continue
        fi

        echo "Processing dataset $dataset_id"
        ./scripts/other/gen_gt_eval_data.py -d "$dataset_id" -m "$model_id" -n 10 --an
    done
done
