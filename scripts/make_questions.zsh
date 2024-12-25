#!/bin/zsh
for C in gt lt; do
    for A in YES NO; do
        for file in d/properties/*.yaml; do
            # :t gets the tail (filename), :r removes extension
            prop_id=${file:t:r}
            if [[ $prop_id = aircraft-speeds || $prop_id = boiling-points ]]; then
                continue
            fi
            ./scripts/gen_qs.py -c "$C" -a "$A" -p "$prop_id"
        done
    done
done
