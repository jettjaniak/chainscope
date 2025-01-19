#! /bin/bash

source .env/bin/activate
find chainscope/data/cot_responses/instr-v0/T0.7_P0.9_M2000 -name "*.yaml" -exec python scripts/eval_cots.py "{}" \;
