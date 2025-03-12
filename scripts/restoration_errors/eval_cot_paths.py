#!/usr/bin/env python3

import logging

import click

from chainscope.api_utils.api_selector import APIPreferences
from chainscope.cot_paths_eval import evaluate_cot_paths
from chainscope.typing import *
from chainscope.utils import MODELS_MAP


@click.command()
@click.argument("responses_path", type=click.Path(exists=True))
@click.option(
    "--evaluator_model_id",
    "-s",
    type=str,
    default="anthropic/claude-3.5-sonnet",
    help="Model used to evaluate the CoT paths",
)
@click.option(
    "--open-router",
    "--or",
    is_flag=True,
    help="Use OpenRouter API instead of local models",
)
@click.option(
    "--open-ai",
    "--oa",
    is_flag=True,
    help="Use OpenAI API instead of local models",
)
@click.option(
    "--anthropic",
    "--an",
    is_flag=True,
    help="Use Anthropic API instead of local models",
)
@click.option(
    "--deepseek",
    "--ds",
    is_flag=True,
    help="Use DeepSeek API instead of local models",
)
@click.option("-t", "--temperature", type=float, default=0)
@click.option("-p", "--top-p", type=float, default=0.9)
@click.option("--max-new-tokens", type=int, default=15_000)
@click.option(
    "--append",
    is_flag=True,
    help="Append to existing data instead of starting fresh",
)
@click.option(
    "--max-retries",
    "-r",
    type=int,
    default=10,
    help="Maximum number of retries for each request",
)
@click.option("-v", "--verbose", is_flag=True)
def main(
    responses_path: str,
    evaluator_model_id: str,
    open_router: bool,
    open_ai: bool,
    anthropic: bool,
    deepseek: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    append: bool,
    max_retries: int,
    verbose: bool,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    cot_paths = CoTPath.load_from_path(Path(responses_path))
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    if sampling_params.temperature != 0.0:
        logging.warning(
            f"Using non-zero temperature {sampling_params.temperature} for CoT paths eval"
        )

    api_preferences = APIPreferences.from_args(
        open_router=open_router,
        open_ai=open_ai,
        anthropic=anthropic,
        deepseek=deepseek,
    )

    existing_cot_paths_eval = None
    if append:
        try:
            existing_cot_paths_eval = CoTPathEval.load(
                cot_paths.model_id, cot_paths.problem_dataset_name
            )
        except FileNotFoundError:
            logging.warning(
                f"No existing CoT path eval found for {cot_paths.model_id} and {cot_paths.problem_dataset_name}, starting fresh"
            )

    evaluator_model_id = MODELS_MAP.get(evaluator_model_id, evaluator_model_id)
    cot_paths_eval = evaluate_cot_paths(
        cot_paths,
        evaluator_model_id=evaluator_model_id,
        sampling_params=sampling_params,
        api_preferences=api_preferences,
        max_retries=max_retries,
        existing_cot_paths_eval=existing_cot_paths_eval,
    )
    result_path = cot_paths_eval.save()
    logging.warning(f"Saved CoT eval to {result_path}")


if __name__ == "__main__":
    main()
