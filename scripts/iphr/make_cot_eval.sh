#!/bin/bash
# Uncomment the following to "fail fast and loudly"
# set -euo pipefail

RESPONSES_DIR="d/cot_responses/instr-wm/T0.7_P0.9_M2000"
PREFIX="" # filter for dataset prefix (e.g. "wm-song-release")
SUFFIX="non-ambiguous-hard-2"      # filter for dataset suffix (e.g. "non-ambiguous-hard-2")
EVAL_MODEL="anthropic/claude-3.7-sonnet"

wait_for_batches() {
  local api="$1"
  echo "Checking ${api} batches..."
  while true; do
    if python ./scripts/other/check_batches.py --api "${api}" 2>&1 | grep -q "Pending: 0"; then
      echo "All ${api} batches are completed."
      break
    fi
    echo "Waiting for ${api} batches to complete..."
    sleep 300  # Check every 5 minutes
  done
}

run() {
  local api="$1"; shift
  local models=("$@")

  for model in "${models[@]}"; do
    echo "▶ Processing model ${model}  (api=${api})"
    
    # Collect all matching response files
    response_files=()
    while IFS= read -r -d '' file; do
      response_files+=("${file}")
    done < <(find "${RESPONSES_DIR}" -type f -wholename "*/${PREFIX}*${SUFFIX}/${model}.yaml" -print0)
    
    if [ ${#response_files[@]} -eq 0 ]; then
      echo "No matching response files found for model ${model}"
      continue
    fi
    
    # Join response files with commas
    response_files_str=$(IFS=,; echo "${response_files[*]}")
    echo "   • found ${#response_files[@]} response files"
    
    ./scripts/iphr/eval_cots.py submit \
        --responses-paths "${response_files_str}" \
        -m "${EVAL_MODEL}" \
        --api "${api}"
  done
}

# -------- configuration blocks --------
run ant-batch "qwen__qwq-32b" "openai__gpt-4o-mini" "google__gemini-2.5-flash-preview" "anthropic__claude-3.5-haiku" "anthropic__claude-3.6-sonnet" "anthropic__claude-3.7-sonnet" "anthropic__claude-3.7-sonnet_1k" "openai__gpt-4o-2024-08-06" "openai__chatgpt-4o-latest" "deepseek__deepseek-chat" "deepseek__deepseek-r1" "google__gemini-pro-1.5" "meta-llama__Llama-3.3-70B-Instruct" # "anthropic__claude-3.7-sonnet_64k"

# Process batches once there are no more pending batches
wait_for_batches "ant-batch"
find d/anthropic_batches/ -name "*.yaml" -exec python ./scripts/iphr/eval_cots.py process-batch {} \;