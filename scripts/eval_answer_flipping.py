#!/usr/bin/env python3

import logging
from pathlib import Path

import click

from chainscope.answer_flipping_eval import evaluate_answer_flipping
from chainscope.typing import *


@click.command()
@click.argument("responses_path", type=click.Path(exists=True))
@click.option(
    "--evaluator_model_ids",
    "-s",
    type=str,
    default="anthropic/claude-3.5-sonnet",
    help="Comma-separated list of models used to evaluate answer flipping in responses (needs to be available on Anthropic). "
    "The first model will be used first, and if it fails, the next model will be used, and so on.",
)
@click.option(
    "--max_retries",
    "-r",
    type=int,
    default=2,
    help="Maximum retries for evaluating answer flipping with the each model",
)
@click.option("-v", "--verbose", is_flag=True)
def main(
    responses_path: str,
    verbose: bool,
    evaluator_model_ids: str,
    max_retries: int,
    max_parallel: int | None,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    cot_responses = CotResponses.load(Path(responses_path))
    cot_eval = evaluate_answer_flipping(
        cot_responses,
        evaluator_model_ids=evaluator_model_ids.split(","),
        max_retries=max_retries,
    )
    path = cot_eval.save()
    logging.warning(f"Saved answer flipping eval to {path}")


if __name__ == "__main__":
    main()
