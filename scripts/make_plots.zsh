#!/bin/zsh
for file in d/questions/**/*.yaml; do
    # :t gets the tail (filename), :r removes extension
    dataset_id=${file:t:r}
    if [[ $dataset_id = *tests* ]]; then
        continue
    fi
    ./scripts/plot_direct_vs_cot_acc.py -d "$dataset_id" -i "instr-v0"
done
