#!/usr/bin/env python3
"""
Experiment 5: Submit Anthropic batch jobs for Putnam UIS cross-autorater evaluation.

For each of the 6 paper model files, submits one Anthropic batch with one request
per (qid, uuid, step_index). The prompt is the same as putnamlike3_main_faithfulness_eval.py.

Usage:
    cd /Users/ivan/src/chainscope
    python3 scripts/putnam/exp5_submit_putnam_batches.py
"""
import json
import sys
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

import yaml
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chainscope import DATA_DIR
from chainscope import cot_faithfulness_utils
from chainscope.typing import MathResponse, SplitCotResponses, StepFaithfulness

PUTNAM_DIR = (
    DATA_DIR / "cot_responses" / "instr-v0" / "default_sampling_params" / "filtered_putnambench"
)

EVALUATOR_MODEL = "claude-sonnet-4-6"
EVAL_MODE = cot_faithfulness_utils.EvaluationMode.REWARD_HACKING
MAX_TOKENS = 2048

MANIFEST_FILE = (
    DATA_DIR
    / "anthropic_batches"
    / f"exp5_putnam_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
)

# 6 paper model base stems
PAPER_MODEL_FILES = [
    ("claude_nonthinking", "anthropic__claude-3.7-sonnet_v0_just_correct_responses_newline_split"),
    ("claude_thinking",    "anthropic__claude-3.7-sonnet:thinking_v0_just_correct_responses_newline_split"),
    ("deepseek_chat",      "deepseek-chat_just_correct_responses_splitted"),
    ("deepseek_reasoner",  "deepseek-reasoner_just_correct_responses_splitted"),
    ("qwq_32b",            "qwen__qwq-32b-preview_just_correct_responses_splitted"),
    ("qwen_72b",           "qwen__qwen-2.5-72b-instruct_v0_just_correct_responses_splitted"),
]


def build_prompt(response: MathResponse, step: str, step_idx: int) -> str:
    """Build the faithfulness evaluation prompt for one step."""
    steps_list: list[str] = []
    if isinstance(response.model_answer, list):
        for s in response.model_answer:
            if isinstance(s, str):
                steps_list.append(s)
            elif isinstance(s, StepFaithfulness):
                steps_list.append(s.step_str)

    context = ""
    for si, step_content in enumerate(steps_list):
        context += f"<step-{si+1}>\n{step_content}\n</step-{si+1}>\n\n"
    context = context.rstrip()

    prompt = EVAL_MODE.prompt_prefix(ask_for_thinking=False)
    prompt += cot_faithfulness_utils._GENERAL_MIDDLE_BIT_IF_SOLUTION_PRESENT
    prompt += "\n\n" + EVAL_MODE.prompt_questions
    prompt += f"\n\n<problem>\n{response.problem}\n</problem>\n"
    prompt += f"\n<solution>\n{response.solution}\n</solution>\n"
    prompt += f"\n<step-to-evaluate><step-{step_idx+1}>{step}</step-{step_idx+1}></step-to-evaluate>\n\n<all steps>\n{context}\n</all steps>"
    prompt += "\n\n" + EVAL_MODE.prompt_suffix(ask_for_thinking=False)
    return prompt


def submit_file_batch(
    file_label: str,
    base_stem: str,
    client: Anthropic,
) -> dict | None:
    """Submit one Anthropic batch for all steps in a model's response file."""
    input_path = PUTNAM_DIR / f"{base_stem}.yaml"
    if not input_path.exists():
        print(f"  ERROR: File not found: {input_path}")
        return None

    responses = SplitCotResponses.load(input_path)

    # Build all (custom_id → (qid, uuid, step_idx)) and prompts
    item_map: dict[str, tuple[str, str, int]] = {}  # custom_id -> (qid, uuid, step_idx)
    requests = []
    item_count = 0

    for qid, responses_by_uuid in responses.split_responses_by_qid.items():
        for uuid, response in responses_by_uuid.items():
            if not isinstance(response, MathResponse):
                continue
            if not isinstance(response.model_answer, list):
                continue

            steps_list: list[str] = []
            for s in response.model_answer:
                if isinstance(s, str):
                    steps_list.append(s)
                elif isinstance(s, StepFaithfulness):
                    steps_list.append(s.step_str)

            for step_idx, step in enumerate(steps_list):
                if not isinstance(step, str):
                    continue

                custom_id = f"i{item_count:06d}"
                item_map[custom_id] = (qid, uuid, step_idx)
                prompt = build_prompt(response, step, step_idx)

                requests.append(
                    Request(
                        custom_id=custom_id,
                        params=MessageCreateParamsNonStreaming(
                            model=EVALUATOR_MODEL,
                            max_tokens=MAX_TOKENS,
                            temperature=0.0,
                            messages=[{"role": "user", "content": prompt}],
                        ),
                    )
                )
                item_count += 1

    if not requests:
        print(f"  SKIP: No steps found for {file_label}")
        return None

    # Split into chunks to avoid 413 Payload Too Large
    CHUNK_SIZE = 2000
    chunks = [requests[i:i + CHUNK_SIZE] for i in range(0, len(requests), CHUNK_SIZE)]
    print(f"  Total: {len(requests)} step evaluations in {len(chunks)} chunk(s)")

    batch_dir = DATA_DIR / "anthropic_batches" / "exp5_putnam"
    batch_dir.mkdir(parents=True, exist_ok=True)

    batch_ids = []
    batch_info_paths = []
    for ci, chunk in enumerate(chunks):
        print(f"  Submitting chunk {ci+1}/{len(chunks)} ({len(chunk)} requests)...")
        message_batch = client.messages.batches.create(requests=chunk)
        batch_id = message_batch.id
        batch_ids.append(batch_id)
        print(f"    Batch ID: {batch_id}")

        # Build the item_map subset for this chunk
        chunk_item_map = {}
        for req in chunk:
            cid = req["custom_id"]
            chunk_item_map[cid] = item_map[cid]

        batch_info_path = batch_dir / f"{file_label}_chunk{ci}_{batch_id}.yaml"
        batch_info = {
            "batch_id": batch_id,
            "file_label": file_label,
            "base_stem": base_stem,
            "evaluator_model": EVALUATOR_MODEL,
            "eval_mode": EVAL_MODE.value,
            "item_count": len(chunk),
            "chunk_index": ci,
            "total_chunks": len(chunks),
            "item_map": chunk_item_map,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(batch_info_path, "w") as f:
            yaml.dump(batch_info, f)
        batch_info_paths.append(str(batch_info_path))

    return {
        "batch_ids": batch_ids,
        "batch_info_paths": batch_info_paths,
        "file_label": file_label,
        "base_stem": base_stem,
        "item_count": item_count,
        "num_chunks": len(chunks),
        "submitted_at": datetime.now().isoformat(),
    }


def main():
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    client = Anthropic()
    total = 0

    for file_label, base_stem in PAPER_MODEL_FILES:
        print(f"\n--- {file_label} ({base_stem}) ---")
        entry = submit_file_batch(file_label, base_stem, client)
        if entry is None:
            continue
        with open(MANIFEST_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        total += 1
        print(f"  Manifest entry saved")

    print(f"\nDone. {total} batches submitted.")
    print(f"Manifest: {MANIFEST_FILE}")
    print("\nRun exp5_process_putnam_batches.py to process results when complete.")


if __name__ == "__main__":
    main()
