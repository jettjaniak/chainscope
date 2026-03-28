#!/usr/bin/env python3
"""Fast resubmit: only loads the 27 missing qwq-32b files (path-based check)."""
import json, sys, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from chainscope import DATA_DIR
from chainscope.cot_eval import evaluate_cot_responses_with_batch
from chainscope.typing import CotResponses

RESP_DIR = DATA_DIR / 'cot_responses' / 'instr-wm' / 'T0.7_P0.9_M2000'
EVAL_DIR = DATA_DIR / 'cot_eval_sonnet46' / 'instr-wm' / 'T0.7_P0.9_M2000'
EVALUATOR = 'anthropic/claude-sonnet-4-6'
MANIFEST = DATA_DIR / 'anthropic_batches' / f'exp4_resubmit_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'

MODELS = ['openai__gpt-4o-mini', 'google__gemini-pro-1.5', 'qwen__qwq-32b']

for model_stem in MODELS:
    resp_files = sorted(RESP_DIR.rglob(f'*non-ambiguous-hard-2*/{model_stem}.yaml'))
    missing = [rp for rp in resp_files
               if not (EVAL_DIR / rp.parent.parent.name / rp.parent.name / rp.name).exists()]
    print(f'{model_stem}: {len(resp_files)} total, {len(missing)} missing', flush=True)
    if not missing:
        continue

    for i, rp in enumerate(missing):
        print(f'  [{i+1}/{len(missing)}] {rp.parent.name}...', end=' ', flush=True)
        cot_responses = CotResponses.load(rp)

        for attempt in range(3):
            try:
                batch_info = evaluate_cot_responses_with_batch(cot_responses, EVALUATOR, existing_eval=None)
                break
            except Exception as e:
                print(f'attempt {attempt+1} failed: {e}', flush=True)
                if attempt < 2:
                    time.sleep(10)
                else:
                    batch_info = None

        if batch_info is None:
            print('FAILED', flush=True)
            continue

        bip = batch_info.save()
        entry = {
            'batch_id': batch_info.batch_id,
            'batch_info_path': str(bip),
            'resp_path': str(rp),
            'model_stem': model_stem,
            'evaluated_model_id': batch_info.evaluated_model_id,
            'dataset_id': batch_info.ds_params.id,
            'submitted_at': datetime.now().isoformat(),
        }
        with open(MANIFEST, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        print(f'OK ({batch_info.batch_id})', flush=True)

print(f'\nManifest: {MANIFEST}', flush=True)
