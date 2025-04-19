#!/bin/zsh

# Parse arguments
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <dataset_patterns> <model> <api>"
    echo "Example: $0 'wm-* *-age !*-song exact-match' 'openai__gpt-4o' 'openai'"
    echo "Pattern format:"
    echo "  - prefix*    : matches at start (e.g., 'wm-*')"
    echo "  - *suffix    : matches at end (e.g., '*-age')"
    echo "  - exact     : exact match (no *)"
    echo "  - !pattern   : negation (works with any pattern type)"
    exit 1
fi

# Split the first argument on spaces
dataset_patterns=(${(s: :)1})  # Explicitly split on spaces
model=$2
api=$3

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting synchronous evaluations"
echo "Dataset patterns: ${dataset_patterns[@]}"
echo "Model: $model"
echo "API: $api"

for file in chainscope/data/cot_responses/instr-wm/T0.7_P0.9_M2000/**/*.yaml; do
    # Skip if model doesn't match
    # :t gets filename, :r removes extension
    f_model=${file:t:r}
    if [[ $f_model != $model ]]; then
        continue
    fi
    
    # :h gets parent directory, :t gets its name
    f_dataset=${file:h:t}

    # Check exclusion patterns first
    skip=0
    for pattern in ${dataset_patterns[@]}; do
        if [[ $pattern == !* ]]; then  # If it starts with !
            pattern=${pattern#!}  # Remove the ! character
            
            # Handle different pattern types
            if [[ $pattern == *\** ]]; then  # Contains *
                if [[ $pattern == \** ]]; then  # Suffix match
                    if [[ $f_dataset == *${pattern#\*} ]]; then
                        skip=1
                        break
                    fi
                elif [[ $pattern == *\* ]]; then  # Prefix match
                    if [[ $f_dataset == ${pattern%\*}* ]]; then
                        skip=1
                        break
                    fi
                fi
            else  # Exact match
                if [[ $f_dataset == $pattern ]]; then
                    skip=1
                    break
                fi
            fi
        fi
    done
    
    # If marked for skipping, continue to next file
    if [[ $skip == 1 ]]; then
        continue
    fi

    # Check if matches any of the inclusion patterns
    matches=0
    for pattern in ${dataset_patterns[@]}; do
        if [[ $pattern != !* ]]; then  # Not an exclusion pattern
            # Handle different pattern types
            if [[ $pattern == *\** ]]; then  # Contains *
                if [[ $pattern == \** ]]; then  # Suffix match
                    if [[ $f_dataset == *${pattern#\*} ]]; then
                        matches=1
                        break
                    fi
                elif [[ $pattern == *\* ]]; then  # Prefix match
                    if [[ $f_dataset == ${pattern%\*}* ]]; then
                        matches=1
                        break
                    fi
                fi
            else  # Exact match
                if [[ $f_dataset == $pattern ]]; then
                    matches=1
                    break
                fi
            fi
        fi
    done

    # Skip if doesn't match any inclusion pattern
    if [[ $matches == 0 ]]; then
        continue
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing file: $file"
    ./scripts/iphr/eval_cots.py submit --api "$api" -m anthropic/claude-3.5-sonnet "$file"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All evaluation jobs have finished" 
