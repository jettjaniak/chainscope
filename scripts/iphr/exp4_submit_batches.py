#!/usr/bin/env python3
"""
Experiment 4: Submit Anthropic batch jobs to re-evaluate 3 IPHR target models
with Claude Sonnet 4.6 as the judge.

Models: openai/gpt-4o-mini, google/gemini-pro-1.5, qwen/qwq-32b
Dataset: non-ambiguous-hard-2 only
Judge: anthropic/claude-sonnet-4-6

Usage:
    cd /Users/ivan/src/chainscope
    uv run python scripts/iphr/exp4_submit_batches.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chainscope import DATA_DIR
from chainscope.cot_eval import evaluate_cot_responses_with_batch
from chainscope.typing import CotResponses

MODELS = [
    "openai__gpt-4o-mini",
    "google__gemini-pro-1.5",
    "qwen__qwq-32b",
]
EVALUATOR_MODEL = "anthropic/claude-sonnet-4-6"
BASE_RESP_DIR = DATA_DIR / "cot_responses" / "instr-wm" / "T0.7_P0.9_M2000"
MANIFEST_FILE = (
    DATA_DIR
    / "anthropic_batches"
    / f"exp4_cross_autorater_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
)


def main():
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    total_submitted = 0
    total_skipped = 0

    for model_stem in MODELS:
        files = sorted(
            BASE_RESP_DIR.rglob(f"*non-ambiguous-hard-2*/{model_stem}.yaml")
        )
        print(f"\n{model_stem}: found {len(files)} files")

        for resp_path in tqdm(files, desc=model_stem):
            cot_responses = CotResponses.load(resp_path)

            # Pass existing_eval=None to force re-evaluation of all responses,
            # even though Claude 3.7 Sonnet evals already exist for this data.
            batch_info = evaluate_cot_responses_with_batch(
                cot_responses=cot_responses,
                evaluator_model_id=EVALUATOR_MODEL,
                existing_eval=None,
            )

            if batch_info is None:
                print(f"  SKIP (no responses): {resp_path.parent.name}")
                total_skipped += 1
                continue

            # batch_info is already saved to disk by evaluate_cot_responses_with_batch
            # Compute the saved path for the manifest
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

    print(f"\nDone.")
    print(f"  Submitted: {total_submitted}")
    print(f"  Skipped (empty): {total_skipped}")
    print(f"  Manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
