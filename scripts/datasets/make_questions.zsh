#!/bin/zsh
for file in d/properties/*.yaml; do
    # :t gets the tail (filename), :r removes extension
    prop_id=${file:t:r}
    # We use buckets for wm-* properties
    ./scripts/gen_qs.py -p "$prop_id" -n 100
done
