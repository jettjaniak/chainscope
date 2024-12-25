#!/bin/zsh
for file in d/questions/**/*.yaml; do
    for model_id in G2 L1 L3 P Q0.5 Q1.5 Q3; do
        # :t gets the tail (filename), :r removes extension
        dataset_id=${file:t:r}
        if [[ $dataset_id = *tests* ]]; then
            continue
        fi
        if [[ $dataset_id = aircraft-speeds* || $dataset_id = boiling-points* ]]; then
            continue
        fi
        ./scripts/gen_cots.py -d "$dataset_id" -m "$model_id" -n 10
    done
done
