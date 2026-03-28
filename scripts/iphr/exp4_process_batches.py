#!/usr/bin/env python3
"""
Experiment 4: Poll and process Anthropic batches for the cross-autorater experiment.

Reads batch info from the manifest, polls until all complete, saves results
to cot_eval_sonnet46/ (separate from original cot_eval/ Claude 3.7 results).

Usage:
    cd /Users/ivan/src/chainscope
    uv run python scripts/iphr/exp4_process_batches.py --manifest <manifest.jsonl>

Or to auto-find the latest manifest:
    uv run python scripts/iphr/exp4_process_batches.py
"""
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import anthropic
import click
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chainscope import DATA_DIR
from chainscope.api_utils.anthropic_utils import process_batch_results
from chainscope.cot_eval import create_cot_eval_from_batch_results
from chainscope.typing import AnthropicBatchInfo, CotEval

EVAL_DIR_NAME = "cot_eval_sonnet46"
POLL_INTERVAL = 300  # 5 minutes


def get_output_path(batch_info: AnthropicBatchInfo) -> Path:
    """Compute the output path for a processed eval (in cot_eval_sonnet46/)."""
    model_filename = batch_info.evaluated_model_id.replace("/", "__") + ".yaml"
    return (
        DATA_DIR
        / EVAL_DIR_NAME
        / batch_info.instr_id
        / batch_info.evaluated_sampling_params.id
        / batch_info.ds_params.pre_id
        / batch_info.ds_params.id
        / model_filename
    )


def save_cot_eval_to_sonnet46(cot_eval: CotEval, batch_info: AnthropicBatchInfo) -> Path:
    """Save CotEval to cot_eval_sonnet46/ directory."""
    output_path = get_output_path(batch_info)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cot_eval.to_yaml_file(output_path)
    return output_path


def load_manifest(manifest_path: Path) -> list[dict]:
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def find_latest_manifest() -> Path:
    """Find the most recent exp4 manifest file."""
    batch_dir = DATA_DIR / "anthropic_batches"
    manifests = sorted(batch_dir.glob("exp4_cross_autorater_manifest_*.jsonl"))
    if not manifests:
        raise FileNotFoundError(
            f"No exp4 manifest files found in {batch_dir}. "
            "Run exp4_submit_batches.py first."
        )
    return manifests[-1]


def process_all_batches(manifest_path: Path) -> None:
    entries = load_manifest(manifest_path)
    print(f"Loaded {len(entries)} batch entries from {manifest_path}")

    client = anthropic.Anthropic()
    total = len(entries)

    while True:
        pending = []
        done = 0
        failed = []

        for entry in tqdm(entries, desc="Checking batch statuses"):
            batch_info_path = Path(entry["batch_info_path"])
            batch_info = AnthropicBatchInfo.load(batch_info_path)
            output_path = get_output_path(batch_info)

            # Already processed
            if output_path.exists():
                done += 1
                continue

            # Check status
            try:
                batch = client.messages.batches.retrieve(entry["batch_id"])
            except Exception as e:
                print(f"  ERROR retrieving {entry['batch_id']}: {e}")
                pending.append(entry)
                continue

            if batch.processing_status == "ended":
                # Process it
                try:
                    results = process_batch_results(batch_info)
                    if results:
                        cot_eval = create_cot_eval_from_batch_results(
                            batch_info=batch_info,
                            batch_results=results,
                            existing_eval=None,
                        )
                        saved_path = save_cot_eval_to_sonnet46(cot_eval, batch_info)
                        print(f"  Saved: {saved_path.parent.name}/{saved_path.name}")
                        done += 1
                    else:
                        print(f"  WARNING: No results for batch {entry['batch_id']}")
                        failed.append(entry)
                except Exception as e:
                    print(f"  ERROR processing {entry['batch_id']}: {e}")
                    failed.append(entry)
            elif batch.processing_status in ("errored", "canceled"):
                print(
                    f"  FAILED: {entry['batch_id']} status={batch.processing_status}"
                )
                failed.append(entry)
            else:
                # Still processing
                pending.append(entry)

        print(f"\nStatus: {done}/{total} done, {len(pending)} pending, {len(failed)} failed")

        if not pending:
            break

        print(f"Waiting {POLL_INTERVAL // 60} minutes before next check...")
        time.sleep(POLL_INTERVAL)

    if failed:
        print(f"\nWARNING: {len(failed)} batches failed:")
        for entry in failed:
            print(f"  {entry['batch_id']} ({entry['model_stem']}, {entry['dataset_id']})")
    else:
        print(f"\nAll {total} batches processed successfully!")


@click.command()
@click.option(
    "--manifest",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to manifest JSONL file. Defaults to the latest exp4 manifest.",
)
def main(manifest: Path | None):
    if manifest is None:
        manifest = find_latest_manifest()
        print(f"Using manifest: {manifest}")
    process_all_batches(manifest)


if __name__ == "__main__":
    main()
