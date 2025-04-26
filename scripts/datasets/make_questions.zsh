#!/bin/zsh
for file in d/properties/*.yaml; do
    # :t gets the tail (filename), :r removes extension
    prop_id=${file:t:r}

    if [[ $prop_id == wm-nyc-place-lat || $prop_id == wm-nyc-place-long ]]; then
        # NYC places are too close to each other to make good comparisons.
        echo "Skipping dataset $prop_id"
        continue
    fi

    echo "Generating questions for $prop_id"
    # ./scripts/datasets/gen_qs.py -v -p "$prop_id" -n 100  --min-popularity 8 --min-percent-value-diff 0.25 --remove-ambiguous --non-overlapping-rag-values --dataset-suffix "non-ambiguous-obscure-or-close-call-2"

    ./scripts/datasets/gen_qs.py -v -p "$prop_id" -n 100  --max-popularity 5 --max-percent-value-diff 0.25 --remove-ambiguous --non-overlapping-rag-values --dataset-suffix "non-ambiguous-hard"
    break
done
