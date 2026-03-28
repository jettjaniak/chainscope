#!/usr/bin/env python3
"""
Experiment 4: Re-submit only the missing qwq-32b batches (27/348).
Checks cot_eval_sonnet46/ for existing output and only submits what's missing.

Usage:
    cd /Users/ivan/src/chainscope
    python3 scripts/iphr/exp4_resubmit_missing.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chainscope import DATA_DIR
from chainscope.cot_eval import evaluate_cot_responses_with_batch
from chainscope.typing import CotResponses, DatasetParams, SamplingParams

MODELS = [
    "openai__gpt-4o-mini",
    "google__gemini-pro-1.5",
    "qwen__qwq-32b",
]
EVALUATOR_MODEL = "anthropic/claude-sonnet-4-6"
EVAL_DIR = DATA_DIR / "cot_eval_sonnet46"
BASE_RESP_DIR = DATA_DIR / "cot_responses" / "instr-wm" / "T0.7_P0.9_M2000"
SAMPLING_PARAMS = SamplingParams(temperature=0.7, top_p=0.9, max_new_tokens=2000)

MANIFEST_FILE = (
    DATA_DIR
    / "anthropic_batches"
    / f"exp4_resubmit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
)


def output_exists(resp_path: Path, model_id: str) -> bool:
    """Check if the Sonnet 4.6 eval already exists for this response file."""
    cot_responses = CotResponses.load(resp_path)
    ds_params = cot_responses.ds_params
    assert isinstance(ds_params, DatasetParams)
    output_path = (
        EVAL_DIR
        / "instr-wm"
        / SAMPLING_PARAMS.id
        / ds_params.pre_id
        / ds_params.id
        / f"{model_id.replace('/', '__')}.yaml"
    )
    return output_path.exists()


def main():
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    total_submitted = 0
    total_skipped = 0

    for model_stem in MODELS:
        files = sorted(
            BASE_RESP_DIR.rglob(f"*non-ambiguous-hard-2*/{model_stem}.yaml")
        )

        missing = []
        for resp_path in files:
            cot_responses = CotResponses.load(resp_path)
            model_id = cot_responses.model_id
            ds_params = cot_responses.ds_params
            assert isinstance(ds_params, DatasetParams)
            output_path = (
                EVAL_DIR / "instr-wm" / SAMPLING_PARAMS.id
                / ds_params.pre_id / ds_params.id
                / f"{model_id.replace('/', '__')}.yaml"
            )
            if not output_path.exists():
                missing.append(resp_path)

        print(f"{model_stem}: {len(files)} total, {len(missing)} missing")

        if not missing:
            total_skipped += len(files)
            continue

        for resp_path in tqdm(missing, desc=model_stem):
            cot_responses = CotResponses.load(resp_path)

            for attempt in range(3):
                try:
                    batch_info = evaluate_cot_responses_with_batch(
                        cot_responses=cot_responses,
                        evaluator_model_id=EVALUATOR_MODEL,
                        existing_eval=None,
                    )
                    break
                except Exception as e:
                    print(f"  Attempt {attempt+1} failed for {resp_path.parent.name}: {e}")
                    if attempt < 2:
                        import time
                        time.sleep(10)
                    else:
                        print(f"  GIVING UP on {resp_path.parent.name}")
                        batch_info = None

            if batch_info is None:
                total_skipped += 1
                continue

            batch_info_path = batch_info.save()

            entry = {
                "batch_id": batch_info.batch_id,
                "batch_info_path": str(batch_info_path),
                "resp_path": str(resp_path),
                "model_stem": model_stem,
                "evaluated_model_id": batch_info.evaluated_model_id,
                "dataset_id": batch_info.ds_params.id,
                "submitted_at": datetime.now().isoformat(),
            }

            with open(MANIFEST_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")

            total_submitted += 1

    print(f"\nDone. Submitted: {total_submitted}, Skipped: {total_skipped}")
    print(f"Manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
