#!/bin/bash
# Uncomment the following to "fail fast and loudly"
# set -euo pipefail

QUESTIONS_DIR="d/questions"
PREFIX="" # filter for dataset prefix (e.g. "wm-song-release")
SUFFIX="non-ambiguous-hard-2"      # filter for dataset suffix (e.g. "non-ambiguous-hard-2")
COMMON_ARGS=(-i instr-wm)                     # shared gen_cots flags
REGULAR_SAMPLE_ARGS="-n 10"
OVERSAMPLE_ARGS="-n 100 --unfaithful-only"

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
  local extra=("$1"); shift
  local models=("$@")

  for model in "${models[@]}"; do
    echo "▶ Processing model ${model}  (api=${api})"
    
    # Collect all matching dataset IDs
    dataset_ids=()
    while IFS= read -r -d '' file; do
      dataset_id="${file#${QUESTIONS_DIR}/*/}"
      dataset_id="${dataset_id%.yaml}"
      [[ ${dataset_id} == *${PREFIX}* && ${dataset_id} == *${SUFFIX} ]] || continue
      dataset_ids+=("${dataset_id}")
    done < <(find "${QUESTIONS_DIR}" -type f -name '*.yaml' -print0)

    if [ ${#dataset_ids[@]} -eq 0 ]; then
      echo "No matching datasets found"
      continue
    fi

    # Join dataset IDs with commas
    dataset_ids_str=$(IFS=,; echo "${dataset_ids[*]}")
    echo "   • datasets: ${dataset_ids_str}"

    ./scripts/iphr/gen_cots.py submit \
        -d "${dataset_ids_str}" \
        -m "${model}" \
        "${COMMON_ARGS[@]}" \
        --api "${api}" \
        ${extra}
  done
}

run_local() {
  local api="$1"; shift
  local extra=("$1"); shift
  local models=("$@")

  for model in "${models[@]}"; do
    echo "▶ Processing model ${model}  (api=${api})"
    
    # Collect all matching dataset IDs
    dataset_ids=()
    while IFS= read -r -d '' file; do
      dataset_id="${file#${QUESTIONS_DIR}/*/}"
      dataset_id="${dataset_id%.yaml}"
      [[ ${dataset_id} == *${PREFIX}* && ${dataset_id} == *${SUFFIX} ]] || continue
      dataset_ids+=("${dataset_id}")
    done < <(find "${QUESTIONS_DIR}" -type f -name '*.yaml' -print0)

    if [ ${#dataset_ids[@]} -eq 0 ]; then
      echo "No matching datasets found"
      continue
    fi

    # Join dataset IDs with commas
    dataset_ids_str=$(IFS=,; echo "${dataset_ids[*]}")
    echo "   • datasets: ${dataset_ids_str}"

    ./scripts/iphr/gen_cots.py local \
        -d "${dataset_ids_str}" \
        -m "${model}" \
        "${COMMON_ARGS[@]}" \
        --api "${api}" \
        ${extra}
  done
}

# -------- configuration blocks --------
run ant-batch "$REGULAR_SAMPLE_ARGS" C3.5H C3.6S C3.7S C3.7S_1K C3.7S_64K
run oai-batch "$REGULAR_SAMPLE_ARGS" GPT4O GPT4OM
run oai "$REGULAR_SAMPLE_ARGS"       GPT4OL
run or "$REGULAR_SAMPLE_ARGS"        QwQ DSV3 DSR1 GP1.5 L70
run_local vllm "$REGULAR_SAMPLE_ARGS --model-id-for-fsp meta-llama/Llama-3.3-70B-Instruct" meta-llama/Llama-3.1-70B

# Oversample the CoT responses for pairs showing unfaithfulness in some models
run ant-batch "$OVERSAMPLE_ARGS" C3.6S C3.7S_1K C3.7S
run oai-batch "$OVERSAMPLE_ARGS" GPT4O
run oai "$OVERSAMPLE_ARGS"       GPT4OL
run or "$OVERSAMPLE_ARGS"        DSR1

# # Process batches once there are no more pending batches
wait_for_batches "ant-batch"
find d/anthropic_batches/ -name "*.yaml" -exec python ./scripts/iphr/gen_cots.py  process-batch {} \;

wait_for_batches "oai-batch"
find d/openai_batches -name "*.yaml" -exec python ./scripts/iphr/gen_cots.py  process-batch {} \;