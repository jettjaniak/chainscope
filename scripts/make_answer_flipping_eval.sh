#! /bin/bash

source .env/bin/activate

# Process files containing "haiku" with another model to avoid using the same one that generated the responses for the answer flipping eval
find chainscope/data/cot_responses/instr-v0/T0.7_P0.9_M2000 -name "*haiku*.yaml" -exec python scripts/eval_answer_flipping.py --or_model_ids anthropic/claude-3.5-sonnet "{}" \;

# Process all other files with claude-3.5-haiku
find chainscope/data/cot_responses/instr-v0/T0.7_P0.9_M2000 -name "*.yaml" ! -name "*haiku*.yaml" -exec python scripts/eval_answer_flipping.py --or_model_ids anthropic/claude-3.5-haiku "{}" \;
