#!/bin/bash
# Uncomment the following to "fail fast and loudly"
# set -euo pipefail

RESPONSES_DIR="d/cot_responses/instr-wm/T0.7_P0.9_M2000"
SUFFIX="non-ambiguous-hard-2"      # dataset filter
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
    while IFS= read -r -d '' file; do
      echo "   • processing ${file}"
      ./scripts/iphr/eval_cots.py submit \
          "${file}" \
          -m "${EVAL_MODEL}" \
          --api "${api}"
    done < <(find "${RESPONSES_DIR}" -type f -wholename "*${SUFFIX}/${model}.yaml" -print0)
  done
}

# -------- configuration blocks --------
run ant-batch "anthropic__claude-3.5-haiku" # "anthropic__claude-3.6-sonnet" "anthropic__claude-3.7-sonnet" "anthropic__claude-3.7-sonnet_1k" "anthropic__claude-3.7-sonnet_64k" "openai__gpt-4o-2024-08-06" "openai__chatgpt-4o-latest" "deepseek__deepseek-chat" "deepseek__deepseek-r1" "google__gemini-pro-1.5" "meta-llama__Llama-3.1-70B-Instruct"

# Process batches once there are no more pending batches
wait_for_batches "ant-batch"
find d/anthropic_batches/ -name "*.yaml" -exec python ./scripts/iphr/eval_cots.py process-batch {} \;