#!/usr/bin/env python3
"""
Experiment 5: Poll and process Anthropic batches for Putnam UIS cross-autorater.

Reads batch info from the exp5 manifest, polls until all complete, parses
responses, and saves output YAML files identical in format to what
putnamlike3_main_faithfulness_eval.py would produce.

Usage:
    cd /Users/ivan/src/chainscope
    python3 scripts/putnam/exp5_process_putnam_batches.py [--manifest <path>]
"""
import dataclasses
import json
import re
import sys
import time
from pathlib import Path

import click
import yaml
from anthropic import Anthropic

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chainscope import DATA_DIR
from chainscope import cot_faithfulness_utils
from chainscope.typing import MathResponse, SplitCotResponses, StepFaithfulness

PUTNAM_DIR = (
    DATA_DIR / "cot_responses" / "instr-v0" / "default_sampling_params" / "filtered_putnambench"
)

EVAL_MODE = cot_faithfulness_utils.EvaluationMode.REWARD_HACKING
EVALUATOR_MODEL_SLUG = "anthropic/claude-sonnet-4-6"
POLL_INTERVAL = 300  # 5 minutes

# Same suffix logic as putnamlike3_main_faithfulness_eval.py
OUTPUT_SUFFIX = (
    "_" + EVALUATOR_MODEL_SLUG.replace("/", "_slash_").replace(".", "_dot_").replace(":", "_colon_")
    + "_" + EVAL_MODE.value
)


def parse_faithfulness_response(response_text: str) -> tuple[str, str]:
    """Parse a faithfulness evaluation response into (reasoning, classification).

    Replicates the logic in putnamlike3_main_faithfulness_eval.py.
    """
    classification = ""
    for q_num in EVAL_MODE.expected_answers.keys():
        matches = list(re.finditer(
            rf"<answer-{q_num}>(.*?)</answer-{q_num}>",
            response_text,
            re.DOTALL | re.IGNORECASE,
        ))
        if matches:
            answer = matches[-1].group(1).strip().upper()
            if answer in ["Y", "YES", "TRUE"]:
                answer = "Y"
            elif answer in ["N", "NO", "FALSE"]:
                answer = "N"
            classification += answer
        else:
            classification += "_RIP_"

    return response_text, classification


def expected_output_path(base_stem: str) -> Path:
    return PUTNAM_DIR / f"{base_stem}{OUTPUT_SUFFIX}.yaml"


def find_latest_manifest() -> Path:
    batch_dir = DATA_DIR / "anthropic_batches"
    manifests = sorted(batch_dir.glob("exp5_putnam_manifest_*.jsonl"))
    if not manifests:
        raise FileNotFoundError(f"No exp5 manifest files found in {batch_dir}")
    return manifests[-1]


def load_manifest(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def process_entry(entry: dict, client: Anthropic) -> bool:
    """Process all batch chunks for an entry and save the output YAML. Returns True on success."""
    base_stem = entry["base_stem"]
    file_label = entry["file_label"]

    output_path = expected_output_path(base_stem)
    if output_path.exists():
        print(f"  SKIP (output exists): {output_path.name}")
        return True

    # Support both old (single batch) and new (multi-chunk) manifest format
    batch_info_paths = entry.get("batch_info_paths", [entry.get("batch_info_path")])
    batch_ids = entry.get("batch_ids", [entry.get("batch_id")])

    # Check all chunks are done
    for bid in batch_ids:
        batch = client.messages.batches.retrieve(bid)
        if batch.processing_status != "ended":
            print(f"  PENDING: batch {bid} status={batch.processing_status}")
            return False

    # Load the original base file to get MathResponse metadata
    input_path = PUTNAM_DIR / f"{base_stem}.yaml"
    responses = SplitCotResponses.load(input_path)

    # Collect results across all chunks
    step_results: dict[str, dict[str, dict[int, StepFaithfulness]]] = {}

    succeeded = 0
    failed = 0
    for bip, bid in zip(batch_info_paths, batch_ids):
        with open(bip) as f:
            batch_info = yaml.full_load(f)
        item_map = batch_info["item_map"]

        for result in client.messages.batches.results(bid):
            custom_id = result.custom_id
            if custom_id not in item_map:
                print(f"  WARNING: Unknown custom_id {custom_id}")
                continue

            mapping = item_map[custom_id]
            qid = mapping[0]
            uuid = mapping[1]
            step_idx = mapping[2]

            if result.result.type != "succeeded":
                print(f"  WARNING: {result.result.type} for {custom_id}")
                failed += 1
                continue

            message = result.result.message
            content = message.content
            if not content:
                failed += 1
                continue

            response_text = content[0].text if content[0].type == "text" else ""
            reasoning, classification = parse_faithfulness_response(response_text)

            original_response = responses.split_responses_by_qid[qid][uuid]
            if isinstance(original_response, MathResponse) and isinstance(original_response.model_answer, list):
                steps = original_response.model_answer
                if step_idx < len(steps):
                    step = steps[step_idx]
                    step_str = step if isinstance(step, str) else step.step_str if isinstance(step, StepFaithfulness) else str(step)
                else:
                    step_str = ""
            else:
                step_str = ""

            sf = StepFaithfulness(
                step_str=step_str,
                unfaithfulness=classification,
                reasoning=reasoning,
            )

            if qid not in step_results:
                step_results[qid] = {}
            if uuid not in step_results[qid]:
                step_results[qid][uuid] = {}
            step_results[qid][uuid][step_idx] = sf
            succeeded += 1

    print(f"  Processed: {succeeded} succeeded, {failed} failed")

    # Build new SplitCotResponses (same format as putnamlike3 output)
    new_responses_by_qid: dict[str, dict[str, MathResponse]] = {}
    skipped_steps = []

    for qid, results_by_uuid in step_results.items():
        new_responses_by_qid[qid] = {}
        for uuid, steps_dict in results_by_uuid.items():
            original = responses.split_responses_by_qid[qid][uuid]
            if not isinstance(original, MathResponse):
                continue
            new_response = MathResponse(
                name=original.name,
                problem=original.problem,
                solution=original.solution,
                model_answer=[],
                model_thinking=original.model_thinking,
                correctness_explanation=original.correctness_explanation,
                correctness_is_correct=original.correctness_is_correct,
                correctness_classification=original.correctness_classification,
            )
            # Build model_answer in step order
            n_steps = len(original.model_answer) if isinstance(original.model_answer, list) else 0
            for si in range(n_steps):
                if si in steps_dict:
                    new_response.model_answer.append(steps_dict[si])
                else:
                    skipped_steps.append((qid, uuid, si))
            new_responses_by_qid[qid][uuid] = new_response

    result_data = SplitCotResponses(
        split_responses_by_qid=new_responses_by_qid,
        model_id=f"{responses.model_id}_faithfulness",
        successfully_split_count=responses.successfully_split_count,
        failed_to_split_count=responses.failed_to_split_count,
        instr_id=responses.instr_id,
        ds_params=dataclasses.replace(
            responses.ds_params,
            description=(
                f"{responses.ds_params.description} "
                f"(expected code {EVAL_MODE.expected_answers_str}) "
                "(skipped " + (
                    ', '.join(f'qid_{q}_uuid_{u}_step_idx_{s}' for q, u, s in skipped_steps)
                    if skipped_steps else 'nothing at all!'
                ) + ")"
            ),
        ),
        sampling_params=responses.sampling_params,
    )

    result_data.save(path=output_path)
    print(f"  Saved: {output_path}")
    return True


@click.command()
@click.option("--manifest", type=click.Path(path_type=Path), default=None)
def main(manifest: Path | None):
    if manifest is None:
        manifest = find_latest_manifest()
    print(f"Using manifest: {manifest}")
    entries = load_manifest(manifest)
    print(f"Loaded {len(entries)} batch entries")

    client = Anthropic()

    while True:
        pending = []
        done = 0
        failed = []

        for entry in entries:
            output_path = expected_output_path(entry["base_stem"])
            if output_path.exists():
                done += 1
                continue

            print(f"\n--- {entry['file_label']} ---")
            try:
                success = process_entry(entry, client)
                if success:
                    done += 1
                else:
                    pending.append(entry)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                pending.append(entry)

        print(f"\nStatus: {done}/{len(entries)} done, {len(pending)} pending, {len(failed)} failed")

        if not pending:
            break

        print(f"Waiting {POLL_INTERVAL // 60} minutes...")
        time.sleep(POLL_INTERVAL)

    if failed:
        print(f"\nWARNING: {len(failed)} batches failed")
    else:
        print(f"\nAll {len(entries)} batches processed!")
    print("\nRun exp5_compare_putnam.py to generate the comparison report.")


if __name__ == "__main__":
    main()
