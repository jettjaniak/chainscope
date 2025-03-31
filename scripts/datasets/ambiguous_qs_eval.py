#!/usr/bin/env python3

import itertools
import logging

import click

from chainscope.ambiguous_qs_eval import process_batch, submit_batch
from chainscope.typing import *


@click.group()
def cli() -> None:
    """Evaluate questions for ambiguity using OpenAI's batch API."""
    pass


@cli.command()
@click.option("--evaluator-model-id", default="gpt-4o")
@click.option("--temperature", default=0.7)
@click.option("--top-p", default=0.9)
@click.option("--max-tokens", default=1000)
@click.option(
    "-n", "--num-evals", default=10, help="Number of evaluations per question"
)
@click.option(
    "--test",
    is_flag=True,
    help="Test mode: only process 10 questions from first dataset",
)
@click.option("-v", "--verbose", is_flag=True)
def submit(
    evaluator_model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    num_evals: int,
    test: bool,
    verbose: bool,
) -> None:
    """Submit batches of questions for ambiguity evaluation."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_tokens,
    )

    # Find all question files
    question_files = list(DATA_DIR.glob("questions/*/*.yaml"))
    logging.info(f"Found {len(question_files)} question files")

    if test:
        question_files = question_files[:1]
        logging.info("Test mode: using only first dataset")

    for question_file in question_files:
        try:
            qs_dataset = QsDataset.load_from_path(question_file)

            if test:
                # Take only first 10 questions
                test_questions = dict(
                    itertools.islice(qs_dataset.question_by_qid.items(), 10)
                )
                qs_dataset.question_by_qid = test_questions
                logging.info(f"Test mode: using {len(test_questions)} questions")

            batch_info = submit_batch(
                qs_dataset=qs_dataset,
                evaluator_model_id=evaluator_model_id,
                sampling_params=sampling_params,
                num_evals=num_evals,
            )
            logging.info(f"Submitted batch {batch_info.batch_id} for {question_file}")
            logging.info(f"Batch info saved to {batch_info.save()}")
        except Exception as e:
            logging.error(f"Error processing {question_file}: {e}")


@cli.command()
@click.option("-v", "--verbose", is_flag=True)
def process(verbose: bool) -> None:
    """Process all batches of ambiguity evaluation responses."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Find all batch info files
    batch_files = list(DATA_DIR.glob("openai_batches/**/ambiguity_eval*.yaml"))
    logging.info(f"Found {len(batch_files)} batch files to process")

    for batch_path in batch_files:
        try:
            batch_info = OpenAIBatchInfo.load(batch_path)
            ambiguity_eval = process_batch(batch_info)
            saved_path = ambiguity_eval.save()
            logging.info(f"Processed batch {batch_info.batch_id}")
            logging.info(f"Results saved to {saved_path}")
        except Exception as e:
            logging.error(f"Error processing {batch_path}: {e}")


if __name__ == "__main__":
    cli()
