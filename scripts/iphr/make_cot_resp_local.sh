#!/bin/bash
for model_id in QwQ; do
    echo "Processing model $model_id"
    for file in d/questions/**/*.yaml; do
        dataset_id="${file##d/questions/*/}"
        dataset_id="${dataset_id%.yaml}"
        if [[ $dataset_id = wm* ]]; then
            echo "Processing wm dataset $dataset_id"
	    if [[ $dataset_id != "wm-book-length_gt_NO_1_6fda02e3" ]]; then
		   poetry run ./scripts/iphr/gen_cots.py submit -d "$dataset_id" -m "$model_id" -n 1 -i instr-wm --api local-tl
	    fi
        fi
    done
done
