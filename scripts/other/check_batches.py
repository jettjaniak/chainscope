#!/usr/bin/env python3

import logging
from pathlib import Path

import click
import openai
from anthropic import Anthropic

from chainscope.api_utils.anthropic_utils import \
    cancel_batch as cancel_an_batch
from chainscope.api_utils.open_ai_utils import cancel_batch as cancel_oa_batch
from chainscope.typing import *


@click.command()
@click.option(
    "--api",
    type=click.Choice(["ant-batch", "oai-batch"]),
    required=True,
    help="API to check batches for",
)
@click.option(
    "--cancel-pending",
    is_flag=True,
    help="Cancel all pending batches",
)
def main(api: str, cancel_pending: bool):
    """Check status of all batches for the specified API."""
    logging.basicConfig(level=logging.INFO)

    # Get the appropriate batch directory based on API
    if api == "ant-batch":
        batch_dir = DATA_DIR / "anthropic_batches"
        client = Anthropic()
    else:  # oai-batch
        batch_dir = DATA_DIR / "openai_batches"
        client = openai.OpenAI()

    if not batch_dir.exists():
        logging.error(f"Batch directory {batch_dir} does not exist")
        return

    # Find all batch files
    batch_files = list(batch_dir.glob("**/*.yaml"))
    if not batch_files:
        logging.error(f"No batch files found in {batch_dir}")
        return

    logging.info(f"Found {len(batch_files)} batch files")

    # Track statistics
    completed = []
    pending = []
    failed = []

    # Check each batch
    for batch_file in batch_files:
        try:
            if api == "ant-batch":
                batch_info = AnthropicBatchInfo.load(batch_file)
                batch = client.messages.batches.retrieve(batch_info.batch_id)
                status = batch.processing_status
                is_completed = status == "ended"
            else:  # oai-batch
                batch_info = OpenAIBatchInfo.load(batch_file)
                batch = client.batches.retrieve(batch_info.batch_id)
                status = batch.status
                is_completed = status == "completed"

            if is_completed:
                completed.append((batch_file.name, status))
            elif status in ["failed", "cancelled", "expired"]:
                failed.append((batch_file.name, status))
            else:
                pending.append((batch_file.name, status, batch_info))

        except Exception as e:
            logging.error(f"Error checking batch {batch_file}: {str(e)}")
            failed.append((batch_file.name, "error"))

    # Print summary
    logging.info("\nBatch Status Summary:")
    logging.info(f"Total batches: {len(batch_files)}")
    logging.info(f"Completed: {len(completed)}")
    logging.info(f"Pending: {len(pending)}")
    logging.info(f"Failed: {len(failed)}")

    if pending:
        logging.info("\nPending Batches:")
        for name, status, _ in pending:
            logging.info(f"  {name}: {status}")

        if cancel_pending:
            logging.info("\nCancelling pending batches...")
            for name, status, batch_info in pending:
                try:
                    if api == "ant-batch":
                        cancel_an_batch(batch_info.batch_id)
                    else:  # oai-batch
                        cancel_oa_batch(batch_info.batch_id)
                    logging.info(f"Successfully cancelled batch: {name}")
                except Exception as e:
                    logging.error(f"Error cancelling batch {name}: {str(e)}")

    if failed:
        logging.info("\nFailed Batches:")
        for name, status in failed:
            logging.info(f"  {name}: {status}")


if __name__ == "__main__":
    main() 