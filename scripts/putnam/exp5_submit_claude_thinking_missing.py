#!/usr/bin/env python3
"""
Submit ONLY the 41 missing Claude thinking responses for Exp5.

Safety checks:
- Loads existing eval file and base file
- Computes exact diff
- Asserts exactly 41 missing responses
- Only builds prompts for missing (qid, uuid) pairs
- Saves manifest for processing

Usage:
    cd /Users/ivan/src/chainscope
    uv run python scripts/putnam/exp5_submit_claude_thinking_missing.py --dry-run
    uv run python scripts/putnam/exp5_submit_claude_thinking_missing.py
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import yaml
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chainscope import DATA_DIR
from chainscope import cot_faithfulness_utils

PUTNAM_DIR = (
    DATA_DIR / "cot_responses" / "instr-v0" / "default_sampling_params" / "filtered_putnambench"
)
EVAL_PATH = PUTNAM_DIR / "anthropic__claude-3.7-sonnet:thinking_v0_just_correct_responses_newline_split_anthropic_slash_claude-sonnet-4-6_reward_hacking.yaml"
BASE_PATH = PUTNAM_DIR / "anthropic__claude-3.7-sonnet:thinking_v0_just_correct_responses_newline_split.yaml"

EVALUATOR_MODEL = "claude-sonnet-4-6"
EVAL_MODE = cot_faithfulness_utils.EvaluationMode.REWARD_HACKING
MAX_TOKENS = 2048
CHUNK_SIZE = 2000


def build_prompt(resp: dict, step_str: str, step_idx: int, all_steps: list[str]) -> str:
    """Build faithfulness eval prompt. Same logic as putnamlike3."""
    context = ""
    for si, sc in enumerate(all_steps):
        context += f"<step-{si+1}>\n{sc}\n</step-{si+1}>\n\n"
    context = context.rstrip()

    prompt = EVAL_MODE.prompt_prefix(ask_for_thinking=False)
    prompt += cot_faithfulness_utils._GENERAL_MIDDLE_BIT_IF_SOLUTION_PRESENT
    prompt += "\n\n" + EVAL_MODE.prompt_questions
    prompt += f"\n\n<problem>\n{resp['problem']}\n</problem>\n"
    prompt += f"\n<solution>\n{resp['solution']}\n</solution>\n"
    prompt += f"\n<step-to-evaluate><step-{step_idx+1}>{step_str}</step-{step_idx+1}></step-to-evaluate>\n\n<all steps>\n{context}\n</all steps>"
    prompt += "\n\n" + EVAL_MODE.prompt_suffix(ask_for_thinking=False)
    return prompt


def get_step_strings(resp: dict) -> list[str]:
    """Extract step strings from a base response."""
    steps = resp.get("model_answer", [])
    result = []
    for s in steps:
        if isinstance(s, str):
            result.append(s)
        elif isinstance(s, dict) and "step_str" in s:
            result.append(s["step_str"])
    return result


@click.command()
@click.option("--dry-run", is_flag=True, help="Print what would be submitted without actually submitting")
def main(dry_run: bool):
    # 1. Load both files
    print("Loading existing eval file...", flush=True)
    with open(EVAL_PATH) as f:
        existing = yaml.safe_load(f)
    print("Loading base file...", flush=True)
    with open(BASE_PATH) as f:
        base = yaml.safe_load(f)

    # 2. Compute exact sets
    existing_keys = set()
    for qid, rbu in existing["split_responses_by_qid"].items():
        for uuid in rbu:
            existing_keys.add((qid, uuid))

    base_keys = set()
    for qid, rbu in base["split_responses_by_qid"].items():
        for uuid in rbu:
            base_keys.add((qid, uuid))

    missing_keys = base_keys - existing_keys
    orphan_keys = existing_keys - base_keys

    print(f"\nBase responses:     {len(base_keys)}")
    print(f"Existing responses: {len(existing_keys)}")
    print(f"Missing responses:  {len(missing_keys)}")
    print(f"Orphan responses:   {len(orphan_keys)}")

    assert len(base_keys) == 114, f"Expected 114 base responses, got {len(base_keys)}"
    assert len(existing_keys) == 73, f"Expected 73 existing responses, got {len(existing_keys)}"
    assert len(missing_keys) == 41, f"Expected 41 missing responses, got {len(missing_keys)}"
    assert len(orphan_keys) == 0, f"Expected 0 orphans, got {len(orphan_keys)}"

    # 3. Build prompts for ONLY the missing responses
    item_map = {}  # custom_id -> (qid, uuid, step_idx)
    requests = []
    item_count = 0

    missing_summary = []
    for qid, uuid in sorted(missing_keys):
        resp = base["split_responses_by_qid"][qid][uuid]
        all_steps = get_step_strings(resp)
        n_steps = len(all_steps)
        missing_summary.append((qid, uuid, n_steps))

        for step_idx, step_str in enumerate(all_steps):
            custom_id = f"m{item_count:06d}"
            item_map[custom_id] = (qid, uuid, step_idx)
            prompt = build_prompt(resp, step_str, step_idx, all_steps)

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

    print(f"\nTotal step evaluations to submit: {item_count}")
    print(f"\nMissing responses detail:")
    for qid, uuid, n in missing_summary:
        print(f"  {qid} ({uuid}): {n} steps")
    total_steps_check = sum(n for _, _, n in missing_summary)
    print(f"\nTotal steps (summed): {total_steps_check}")
    assert total_steps_check == item_count, f"Step count mismatch: {total_steps_check} vs {item_count}"
    assert total_steps_check == 10819, f"Expected 10819 total steps, got {total_steps_check}"

    # 4. Verify NO overlap with existing responses
    for cid, (qid, uuid, _) in item_map.items():
        assert (qid, uuid) not in existing_keys, f"BUG: {qid},{uuid} is in existing eval!"

    print(f"\nAll checks passed.")

    if dry_run:
        print("\n=== DRY RUN - nothing submitted ===")
        chunks = [requests[i:i + CHUNK_SIZE] for i in range(0, len(requests), CHUNK_SIZE)]
        print(f"Would submit {len(chunks)} chunk(s) of up to {CHUNK_SIZE} requests each")
        return

    # 5. Submit in chunks
    print(f"\nSubmitting...", flush=True)
    client = Anthropic()
    manifest_path = (
        DATA_DIR / "anthropic_batches"
        / f"exp5_claude_thinking_missing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    batch_dir = DATA_DIR / "anthropic_batches" / "exp5_putnam"
    batch_dir.mkdir(parents=True, exist_ok=True)

    chunks = [requests[i:i + CHUNK_SIZE] for i in range(0, len(requests), CHUNK_SIZE)]
    print(f"{len(chunks)} chunk(s)")

    batch_ids = []
    batch_info_paths = []
    for ci, chunk in enumerate(chunks):
        print(f"  Chunk {ci+1}/{len(chunks)} ({len(chunk)} requests)...", end=" ", flush=True)
        message_batch = client.messages.batches.create(requests=chunk)
        bid = message_batch.id
        batch_ids.append(bid)
        print(f"OK ({bid})", flush=True)

        # Save chunk item_map
        chunk_item_map = {r["custom_id"]: item_map[r["custom_id"]] for r in chunk}
        info_path = batch_dir / f"claude_thinking_missing_chunk{ci}_{bid}.yaml"
        with open(info_path, "w") as f:
            yaml.dump({
                "batch_id": bid,
                "file_label": "claude_thinking_missing",
                "chunk_index": ci,
                "total_chunks": len(chunks),
                "item_count": len(chunk),
                "item_map": chunk_item_map,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }, f)
        batch_info_paths.append(str(info_path))

    # Save manifest
    entry = {
        "batch_ids": batch_ids,
        "batch_info_paths": batch_info_paths,
        "file_label": "claude_thinking_missing",
        "item_count": item_count,
        "num_chunks": len(chunks),
        "submitted_at": datetime.now().isoformat(),
    }
    with open(manifest_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"\nDone. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
