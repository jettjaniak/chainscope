#!/bin/bash
# Uncomment the following to "fail fast and loudly"
# set -euo pipefail

QUESTIONS_DIR="d/questions"
SUFFIX="non-ambiguous-obscure-or-close-call-2"      # dataset filter
COMMON_ARGS=(-n 10 -i instr-wm)                     # shared gen_cots flags

wait_for_batches() {
  local api="$1"
  echo "Checking ${api} batches..."
  while true; do
    if python ./scripts/other/check_batches.py --api "${api}" | grep -q "Pending: 0"; then
      echo "All ${api} batches are completed."
      break
    fi
    echo "Waiting for ${api} batches to complete..."
    sleep 300  # Check every 5 minutes
  done
}

run() {
  local api="$1"; shift
  local extra=("$1"); shift
  local models=("$@")

  for model in "${models[@]}"; do
    echo "▶ Processing model ${model}  (api=${api})"
    while IFS= read -r -d '' file; do
      dataset_id="${file#${QUESTIONS_DIR}/*/}"
      dataset_id="${dataset_id%.yaml}"
      [[ ${dataset_id} == *${SUFFIX} ]] || continue

      echo "   • dataset ${dataset_id}"
      ./scripts/iphr/gen_cots.py submit \
          -d "${dataset_id}" \
          -m "${model}" \
          "${COMMON_ARGS[@]}" \
          --api "${api}" \
          ${extra}
    done < <(find "${QUESTIONS_DIR}" -type f -name '*.yaml' -print0)
  done
}

# -------- configuration blocks --------
run ant-batch "" C3.5H C3.6S C3.7S C3.7S_1K C3.7S_64K
run oai-batch "" GPT4O
run oai ""       GPT4OL
run or ""        DSV3 DSR1 GP1.5 L70
run local-vllm "--model-id-for-fsp meta-llama/Llama-3.3-70B-Instruct" meta-llama/Llama-3.1-70B

# Process batches once there are no more pending batches
wait_for_batches "ant-batch"
find d/anthropic_batches/ -name "*.yaml" -exec python ./scripts/iphr/gen_cots.py  process-batch {} \;

wait_for_batches "oai-batch"
find d/openai_batches -name "*.yaml" -exec python scripts/gen_cots.py  process-batch {} \;