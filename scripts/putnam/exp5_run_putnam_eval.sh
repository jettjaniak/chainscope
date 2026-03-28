#!/usr/bin/env bash
# Experiment 5: Re-evaluate Putnam UIS with Claude Sonnet 4.6 as judge.
# Runs putnamlike3_main_faithfulness_eval.py for each of the 6 paper models.
#
# Usage (from repo root):
#   nohup bash scripts/putnam/exp5_run_putnam_eval.sh > exp5_putnam_eval.log 2>&1 &
#   echo $! > exp5_putnam_eval.pid

set -e

PUTNAM_DIR="chainscope/data/cot_responses/instr-v0/default_sampling_params/filtered_putnambench"
MODEL_ID="anthropic/claude-sonnet-4-6"
EVAL_MODE="reward_hacking"
MAX_PARALLEL=10
MAX_RETRIES=3
MAX_NEW_TOKENS=8192

echo "Starting Experiment 5: Putnam cross-autorater with Sonnet 4.6"
echo "Model: $MODEL_ID"
echo "Mode: $EVAL_MODE"
echo "Started at: $(date)"
echo ""

# 6 paper models and their base response files
# Format: base_file_stem (under $PUTNAM_DIR)

FILES=(
    "anthropic__claude-3.7-sonnet_v0_just_correct_responses_newline_split"
    "anthropic__claude-3.7-sonnet:thinking_v0_just_correct_responses_newline_split"
    "deepseek-chat_just_correct_responses_splitted"
    "deepseek-reasoner_just_correct_responses_splitted"
    "qwen__qwq-32b-preview_just_correct_responses_splitted"
    "qwen__qwen-2.5-72b-instruct_v0_just_correct_responses_splitted"
)

for stem in "${FILES[@]}"; do
    input_path="${PUTNAM_DIR}/${stem}.yaml"

    # Compute expected output path (same logic as putnamlike3 script)
    escaped_model=$(echo "$MODEL_ID" | sed 's|/|_slash_|g; s|\.|_dot_|g; s|:|_colon_|g')
    output_path="${PUTNAM_DIR}/${stem}_${escaped_model}_${EVAL_MODE}.yaml"

    if [ -f "$output_path" ]; then
        echo "SKIP (already exists): $output_path"
        continue
    fi

    if [ ! -f "$input_path" ]; then
        echo "ERROR: Input file not found: $input_path"
        continue
    fi

    echo "---"
    echo "Evaluating: $stem"
    echo "Input: $input_path"
    echo "Expected output: $output_path"
    echo "Started at: $(date)"

    python3 scripts/putnam/putnamlike3_main_faithfulness_eval.py \
        "$input_path" \
        --model_id "$MODEL_ID" \
        --evaluation_mode "$EVAL_MODE" \
        --max_parallel $MAX_PARALLEL \
        --max_retries $MAX_RETRIES \
        --max_new_tokens $MAX_NEW_TOKENS \
        --open_router

    echo "Finished: $stem at $(date)"
    echo ""
done

echo "All done at $(date)"
