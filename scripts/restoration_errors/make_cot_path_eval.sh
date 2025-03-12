#! /bin/bash

source .env/bin/activate

for model in "openai__gpt-4o" "anthropic__claude-3.5-sonnet" "deepseek__deepseek-chat" "google__gemini-pro-1.5" "meta-llama__Llama-3.3-70B-Instruct"; do
    for dataset in "gsm8k" "math" "mmlu"; do
        echo "Evaluating ${model} on ${dataset}"
        python scripts/iphr/eval_cot_paths.py d/cot_paths/${dataset}/${model}.yaml --or -s deepseek/deepseek-r1 --append
    done
done
