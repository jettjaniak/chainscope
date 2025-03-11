#!/usr/bin/env python3

import logging

import click

from chainscope.api_utils.api_selector import APIPreferences
from chainscope.cot_paths import gen_cot_paths
from chainscope.typing import *
from chainscope.utils import MODELS_MAP


@click.command()
@click.option("-n", "--n-paths", type=int, required=True)
@click.option("-d", "--problem-dataset-name", type=str, required=True)
@click.option("-m", "--model-id", type=str, required=True)
@click.option("-t", "--temperature", type=float, default=0.7)
@click.option("-p", "--top-p", type=float, default=0.9)
@click.option("--max-new-tokens", type=int, default=5_000)
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
@click.option(
    "--append",
    is_flag=True,
    help="Append to existing data instead of starting fresh",
)
@click.option("-v", "--verbose", is_flag=True)
def main(
    n_paths: int,
    problem_dataset_name: str,
    model_id: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    open_router: bool,
    open_ai: bool,
    anthropic: bool,
    deepseek: bool,
    append: bool,
    verbose: bool,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    model_id = MODELS_MAP.get(model_id, model_id)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    api_preferences = APIPreferences.from_args(
        open_router=open_router,
        open_ai=open_ai,
        anthropic=anthropic,
        deepseek=deepseek,
    )

    # Try to load existing paths if append is True
    existing_paths = None
    if append:
        try:
            cot_paths_dir = DATA_DIR / "cot_paths" / problem_dataset_name
            response_path = cot_paths_dir / f"{model_id.replace('/', '__')}.yaml"
            if response_path.exists():
                existing_paths = CoTPath.load_from_path(response_path)
                logging.info(f"Loaded existing paths from {response_path}")
            else:
                logging.warning(
                    f"No existing paths found at {response_path}, starting fresh"
                )
        except Exception as e:
            logging.warning(f"Error loading existing paths: {e}, starting fresh")

    cot_paths = gen_cot_paths(
        model_id=model_id,
        problem_dataset_name=problem_dataset_name,
        sampling_params=sampling_params,
        n_paths=n_paths,
        existing_paths=existing_paths,
        api_preferences=api_preferences,
    )
    cot_paths.save()


if __name__ == "__main__":
    main()
