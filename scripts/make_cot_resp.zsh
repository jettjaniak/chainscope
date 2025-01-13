#!/bin/zsh
for file in d/questions/**/*.yaml; do
    for model_id in GF1.5 C3H; do
        # :t gets the tail (filename), :r removes extension
        dataset_id=${file:t:r}
        if [[ $dataset_id = *tests* ]]; then
            continue
        fi
        if [[ $dataset_id = animals-speed* || $dataset_id = sea-depths* || $dataset_id = sound-speeds* || $dataset_id = train-speeds* ]]; then
            continue
        fi
        ./scripts/gen_cots.py -d "$dataset_id" -m "$model_id" -n 10 --or 
    done
done
