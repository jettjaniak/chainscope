#!/bin/zsh

# Parse arguments
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <dataset_prefixes> <model> <api>"
    echo "Example: $0 'wm-person-age wm-book !wm-song' 'openai__gpt-4o' 'openai'"
    exit 1
fi

# Split the first argument on spaces
dataset_prefixes=(${(s: :)1})  # Explicitly split on spaces
model=$2
api=$3

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting synchronous evaluations"
echo "Dataset prefixes: ${dataset_prefixes[@]}"
echo "Model: $model"
echo "API: $api"

for file in chainscope/data/cot_responses/instr-wm/T0.7_P0.9_M2000/**/*.yaml; do
    # Skip if model doesn't match
    # :t gets filename, :r removes extension
    f_model=${file:t:r}
    if [[ $f_model != $model ]]; then
        continue
    fi

    # echo "Checking file: $file"
    
    # :h gets parent directory, :t gets its name
    f_dataset=${file:h:t}
    # echo "  Dataset: $f_dataset"
    # echo "  Model: $f_model"

    # Check exclusion patterns first
    skip=0
    for prefix in ${dataset_prefixes[@]}; do
        # echo "  Checking prefix: $prefix"
        if [[ $prefix == !* ]]; then  # If it starts with !
            prefix=${prefix#!}  # Remove the ! character
            if [[ $f_dataset == $prefix* ]]; then
                # echo "  Skipping: matches exclusion pattern $prefix"
                skip=1
                break
            fi
        fi
    done
    
    # If marked for skipping, continue to next file
    if [[ $skip == 1 ]]; then
        continue
    fi

    # Check if matches any of the inclusion patterns
    matches=0
    for prefix in ${dataset_prefixes[@]}; do
        if [[ $prefix != !* ]] && [[ $f_dataset == $prefix* ]]; then
            # echo "  Matches inclusion pattern $prefix"
            matches=1
            break
        fi
    done

    # Skip if doesn't match any inclusion pattern
    if [[ $matches == 0 ]]; then
        # echo "  Skipping: doesn't match any inclusion pattern"
        continue
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing file: $file"
    ./scripts/iphr/eval_cots.py submit --api "$api" -m anthropic/claude-3.5-sonnet "$file"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All evaluation jobs have finished" 
