#!/bin/zsh
for file in d/properties/*.yaml; do
    # :t gets the tail (filename), :r removes extension
    prop_id=${file:t:r}

    if [[ $prop_id == wm-nyc-place-lat || $prop_id == wm-nyc-place-long ]]; then
        # NYC places are too close to each other to make good comparisons.
        echo "Skipping dataset $prop_id"
        continue
    fi

    if [[ $prop_id == *-popu || $prop_id == *dens ]]; then
        # Population and density properties vary over time and might lead to ambiguous qs
        echo "Skipping dataset $prop_id"
        continue
    fi

    echo "Generating questions for $prop_id"
    # ./scripts/datasets/gen_qs.py -v -p "$prop_id" -n 100  --min-popularity 8 --min-fraction-value-diff 0.25 --remove-ambiguous "enough-comparisons" --non-overlapping-rag-values --dataset-suffix "non-ambiguous-obscure-or-close-call-2"

    # ./scripts/datasets/gen_qs.py -p "$prop_id" -n 100  --max-popularity 5 --min-fraction-value-diff 0.05 --max-fraction-value-diff 0.25 --remove-ambiguous "enough-pairs" --non-overlapping-rag-values --dataset-suffix "non-ambiguous-hard"

    ./scripts/datasets/gen_qs.py -p "$prop_id" -n 100  --max-popularity 5 --min-fraction-value-diff 0.05 --max-fraction-value-diff 0.25 --remove-ambiguous "enough-pairs" --non-overlapping-rag-values --dataset-suffix "non-ambiguous-hard-2"
done
