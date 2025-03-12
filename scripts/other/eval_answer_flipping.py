#!/usr/bin/env python3

import logging
from pathlib import Path

import click

from chainscope.answer_flipping_eval import evaluate_answer_flipping
from chainscope.api_utils.api_selector import APIPreferences
from chainscope.typing import *


@click.command()
@click.argument("responses_path", type=click.Path(exists=True))
@click.option(
    "--evaluator_model_id",
    "-s",
    type=str,
    default="anthropic/claude-3.5-haiku",
    help="Model used to evaluate answer flipping in responses.",
)
@click.option(
    "--open-router",
    "--or",
    is_flag=True,
    help="Use OpenRouter API instead of local models for generating the open-ended responses",
)
@click.option(
    "--open-ai",
    "--oa",
    is_flag=True,
    help="Use OpenAI API instead of local models for generating the open-ended responses",
)
@click.option(
    "--anthropic",
    "--an",
    is_flag=True,
    help="Use Anthropic API instead of local models for generating the open-ended responses",
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
    evaluator_model_id: str,
    open_router: bool,
    open_ai: bool,
    anthropic: bool,
    max_retries: int,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    cot_responses = CotResponses.load(Path(responses_path))
    api_preferences = APIPreferences.from_args(
        open_router=open_router,
        open_ai=open_ai,
        anthropic=anthropic,
    )
    cot_eval = evaluate_answer_flipping(
        cot_responses,
        evaluator_model_id=evaluator_model_id,
        max_retries=max_retries,
        api_preferences=api_preferences,
    )
    path = cot_eval.save()
    logging.warning(f"Saved answer flipping eval to {path}")


if __name__ == "__main__":
    main()
