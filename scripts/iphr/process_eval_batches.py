#!/usr/bin/env python3
"""Poll and process temperature experiment eval batches until all are done.

Checks batch status, processes completed ones, resubmits failed ones,
and loops until all 116 eval files exist for each temperature.
"""

import subprocess
import time
from pathlib import Path

import anthropic
import yaml

client = anthropic.Anthropic()

TEMPS = ["T0.3_P0.9_M2000", "T1.0_P0.9_M2000"]
BASE_BATCH_DIR = Path("chainscope/data/anthropic_batches/instr-wm")
BASE_EVAL_DIR = Path("chainscope/data/cot_eval/instr-wm")
BASE_RESP_DIR = Path("chainscope/data/cot_responses/instr-wm")
TARGET = 116
MODEL = "anthropic/claude-sonnet-4-6"


def get_latest_batches(temp: str) -> dict[str, Path]:
    """Get the latest batch file per dataset for a temperature."""
    batch_dir = BASE_BATCH_DIR / temp
    if not batch_dir.exists():
        return {}
    batch_files = sorted(batch_dir.rglob("*.yaml"))
    latest: dict[str, Path] = {}
    for bf in batch_files:
        ds = bf.parent.name
        latest[ds] = bf
    return latest


def count_evals(temp: str) -> int:
    eval_dir = BASE_EVAL_DIR / temp
    if not eval_dir.exists():
        return 0
    return len(list(eval_dir.rglob("openai__gpt-4o-mini.yaml")))


def process_and_resubmit(temp: str) -> tuple[int, int, int]:
    """Process completed batches, resubmit failed ones. Returns (done, in_progress, resubmitted)."""
    eval_dir = BASE_EVAL_DIR / temp
    batch_dir = BASE_BATCH_DIR / temp
    resp_dir = BASE_RESP_DIR / temp
    latest = get_latest_batches(temp)

    to_process = []
    failed_resp_paths = []
    in_progress = 0

    for ds, bf in sorted(latest.items()):
        rel = bf.relative_to(batch_dir)
        eval_path = eval_dir / rel.parent / "openai__gpt-4o-mini.yaml"
        if eval_path.exists():
            continue
        with open(bf) as f:
            info = yaml.safe_load(f)
        batch = client.beta.messages.batches.retrieve(info["batch_id"])
        if batch.processing_status == "ended":
            # Check if it succeeded or errored
            to_process.append(str(bf))
        elif batch.processing_status == "errored" or getattr(batch, "status", None) == "failed":
            # Find the corresponding response file to resubmit
            resp_path = resp_dir / rel.parent / "openai__gpt-4o-mini.yaml"
            if resp_path.exists():
                failed_resp_paths.append(str(resp_path))
        else:
            in_progress += 1

    # Process completed batches
    for bf_path in to_process:
        subprocess.run(
            ["python3", "scripts/iphr/eval_cots.py", "process-batch", bf_path],
            capture_output=True,
        )

    # Resubmit failed ones
    if failed_resp_paths:
        subprocess.run(
            [
                "python3", "scripts/iphr/eval_cots.py", "submit",
                "-r", ",".join(failed_resp_paths),
                "-m", MODEL,
                "--api", "ant-batch",
            ],
            capture_output=True,
        )

    done = count_evals(temp)
    return done, in_progress, len(failed_resp_paths)


def main():
    iteration = 0
    while True:
        iteration += 1
        all_done = True
        print(f"\n=== Iteration {iteration} ({time.strftime('%H:%M:%S')}) ===")
        for temp in TEMPS:
            done = count_evals(temp)
            if done >= TARGET:
                print(f"  {temp}: {done}/{TARGET} COMPLETE")
                continue
            all_done = False
            done, in_prog, resub = process_and_resubmit(temp)
            print(f"  {temp}: {done}/{TARGET} done, {in_prog} in_progress, {resub} resubmitted")

        if all_done:
            print("\nAll eval batches processed!")
            break

        print("Waiting 5 minutes before next check...")
        time.sleep(300)


if __name__ == "__main__":
    main()
