#!/usr/bin/env python3

import logging

import click

from chainscope.cot_splitting import split_cot_responses
from chainscope.typing import *


@click.command()
@click.argument("responses_path", type=click.Path(exists=True))
@click.option(
    "--or_model_ids",
    "-s",
    type=str,
    default="anthropic/claude-3.5-haiku,openai/gpt-4o",
    help="Comma-separated list of models used to split CoT responses (needs to be available on OpenRouter). "
    "The first model will be used first, and if it fails, the next model will be used, and so on.",
)
@click.option(
    "--max_retries",
    "-r",
    type=int,
    default=1,
    help="Maximum retries for splitting CoT responses with the each model",
)
@click.option(
    "--max_parallel",
    "-p",
    type=int,
    default=None,
    help="Maximum number of parallel requests. If not set, it will use the OpenRouter limits.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Increase verbosity (can be used multiple times)",
)
def main(
    responses_path: str,
    or_model_ids: str,
    max_retries: int,
    verbose: int,
    max_parallel: int | None,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    cot_responses = CotResponses.load(Path(responses_path))
    results = split_cot_responses(
        responses=cot_responses,
        or_model_ids=or_model_ids.split(","),
        max_retries=max_retries,
        max_parallel=max_parallel,
    )
    path = results.save()
    logging.error(f"Saved split CoT responses to {path}")


if __name__ == "__main__":
    main()
