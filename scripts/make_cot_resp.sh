#!/bin/bash
for file in d/questions/**/*.yaml; do
    for model_id in C3.5H; do
        dataset_id="${file##d/questions/*/}"
        dataset_id="${dataset_id%.yaml}"
        if [[ $dataset_id = *tests* ]]; then
            continue
        fi
        if [[ $dataset_id = animals-speed* || $dataset_id = sea-depths* || $dataset_id = sound-speeds* || $dataset_id = train-speeds* ]]; then
            continue
        fi
        ./scripts/gen_cots.py -d "$dataset_id" -m "$model_id" -n 10 --or 
    done
done
