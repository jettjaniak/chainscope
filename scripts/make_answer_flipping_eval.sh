#! /bin/bash

source .env/bin/activate
find chainscope/data/cot_responses/instr-v0/T0.7_P0.9_M2000 -name "*haiku*.yaml" -exec python scripts/eval_answer_flipping.py "{}" \;
