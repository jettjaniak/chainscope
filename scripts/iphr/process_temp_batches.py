#!/usr/bin/env python3
"""Poll and process temperature experiment batches until all are done.

Checks batch status, processes completed ones, resubmits failed ones,
and loops until all 116 datasets have response files for each temperature.
"""

import subprocess
import time
from pathlib import Path

import yaml
from openai import OpenAI

client = OpenAI()

TEMPS = ["T0.3_P0.9_M2000", "T1.0_P0.9_M2000"]
BASE_BATCH_DIR = Path("chainscope/data/openai_batches/instr-wm")
BASE_RESP_DIR = Path("chainscope/data/cot_responses/instr-wm")
TARGET = 116


def get_latest_batches(temp: str) -> dict[str, Path]:
    """Get the latest batch file per dataset for a temperature."""
    batch_dir = BASE_BATCH_DIR / temp
    batch_files = sorted(batch_dir.rglob("*.yaml"))
    latest: dict[str, Path] = {}
    for bf in batch_files:
        ds = bf.parent.name
        latest[ds] = bf  # sorted alphabetically, last file per dir wins
    return latest


def count_responses(temp: str) -> int:
    resp_dir = BASE_RESP_DIR / temp
    return len(list(resp_dir.rglob("openai__gpt-4o-mini.yaml")))


def process_and_resubmit(temp: str) -> tuple[int, int, int]:
    """Process completed batches, resubmit failed ones. Returns (done, in_progress, resubmitted)."""
    resp_dir = BASE_RESP_DIR / temp
    batch_dir = BASE_BATCH_DIR / temp
    latest = get_latest_batches(temp)

    to_process = []
    failed_datasets = []
    in_progress = 0

    for ds, bf in sorted(latest.items()):
        rel = bf.relative_to(batch_dir)
        resp_path = resp_dir / rel.parent / "openai__gpt-4o-mini.yaml"
        if resp_path.exists():
            continue
        with open(bf) as f:
            info = yaml.safe_load(f)
        batch = client.batches.retrieve(info["batch_id"])
        if batch.status == "completed":
            to_process.append(str(bf))
        elif batch.status == "failed":
            failed_datasets.append(ds)
        else:
            in_progress += 1

    # Process completed
    for bf_path in to_process:
        subprocess.run(
            ["python3", "scripts/iphr/gen_cots.py", "process-batch", bf_path],
            capture_output=True,
        )

    # Resubmit failed
    if failed_datasets:
        t_val = "0.3" if "T0.3" in temp else "1.0"
        subprocess.run(
            [
                "python3", "scripts/iphr/gen_cots.py", "submit",
                "-d", ",".join(failed_datasets),
                "-m", "openai/gpt-4o-mini",
                "-i", "instr-wm",
                "-n", "10",
                "--api", "oai-batch",
                "-t", t_val,
                "-p", "0.9",
                "--max-new-tokens", "2000",
            ],
            capture_output=True,
        )

    done = count_responses(temp)
    return done, in_progress, len(failed_datasets)


def main():
    iteration = 0
    while True:
        iteration += 1
        all_done = True
        print(f"\n=== Iteration {iteration} ({time.strftime('%H:%M:%S')}) ===")
        for temp in TEMPS:
            done = count_responses(temp)
            if done >= TARGET:
                print(f"  {temp}: {done}/{TARGET} COMPLETE")
                continue
            all_done = False
            done, in_prog, resub = process_and_resubmit(temp)
            print(f"  {temp}: {done}/{TARGET} done, {in_prog} in_progress, {resub} resubmitted")

        if all_done:
            print("\nAll batches processed!")
            break

        print("Waiting 5 minutes before next check...")
        time.sleep(300)


if __name__ == "__main__":
    main()
