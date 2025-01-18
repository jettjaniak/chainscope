#!/usr/bin/env python3

import logging

import click

from chainscope.cot_generation import (
    get_all_cot_responses,
    get_all_cot_responses_an,
    get_all_cot_responses_oa,
    get_all_cot_responses_or,
)
from chainscope.typing import *
from chainscope.utils import MODELS_MAP


@click.command()
@click.option("-n", "--n-responses", type=int, required=True)
@click.option("-d", "--dataset-id", type=str, required=True)
@click.option("-m", "--model-id", type=str, required=True)
@click.option("-i", "--instr-id", type=str, default="instr-v0")
@click.option("-t", "--temperature", type=float, default=0.7)
@click.option("-p", "--top-p", type=float, default=0.9)
@click.option("--max-new-tokens", type=int, default=2_000)
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
    "--append",
    is_flag=True,
    help="Append to existing responses instead of starting fresh",
)
@click.option("-v", "--verbose", is_flag=True)
def main(
    n_responses: int,
    dataset_id: str,
    model_id: str,
    instr_id: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    open_router: bool,
    open_ai: bool,
    anthropic: bool,
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

    # Try to load existing responses if append is True
    existing_responses = None
    if append:
        try:
            ds_params = DatasetParams.from_id(dataset_id)
            response_path = (
                DATA_DIR
                / "cot_responses"
                / instr_id
                / sampling_params.id
                / ds_params.pre_id
                / ds_params.id
                / f"{model_id.replace('/', '__')}.yaml"
            )
            if response_path.exists():
                existing_responses = CotResponses.load(response_path)
                logging.info(f"Loaded existing responses from {response_path}")
            else:
                logging.warning(
                    f"No existing responses found at {response_path}, starting fresh"
                )
        except Exception as e:
            logging.warning(f"Error loading existing responses: {e}, starting fresh")

    if open_router:
        get_responses = get_all_cot_responses_or
    elif open_ai:
        get_responses = get_all_cot_responses_oa
    elif anthropic:
        get_responses = get_all_cot_responses_an
    else:
        get_responses = get_all_cot_responses

    cot_responses = get_responses(
        model_id=model_id,
        dataset_id=dataset_id,
        instr_id=instr_id,
        sampling_params=sampling_params,
        n_responses=n_responses,
        question_type="yes-no",
        existing_responses=existing_responses,
    )
    cot_responses.save()


if __name__ == "__main__":
    main()
