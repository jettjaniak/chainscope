#!/bin/zsh
for file in d/properties/*.yaml; do
    # :t gets the tail (filename), :r removes extension
    prop_id=${file:t:r}
    echo "Generating questions for $prop_id"
    ./scripts/datasets/gen_qs.py -p "$prop_id" -n 100  --entity-popularity-filter 8 --min-percent-value-diff 0.25 --remove-ambiguous --non-overlapping-rag-values --dataset-suffix "non-ambiguous-obscure-or-close-call"
    break
done
