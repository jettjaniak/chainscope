#!/bin/zsh
for file in d/properties/*.yaml; do
    # :t gets the tail (filename), :r removes extension
    prop_id=${file:t:r}
    # We use buckets for wm-* properties
    if [[ $prop_id != wm-us-zip-dens* ]]; then
        continue
    fi
    echo "Generating questions for $prop_id"
    ./scripts/datasets/gen_qs.py -v -p "$prop_id" -n 100 --entity-popularity-filter 8 --min-percent-value-diff 0.25 --dataset-suffix "non-obscure-non-close-call"
    break
done
