#!/usr/bin/env python3

import logging
import os

import click

from chainscope.cot_generation import (
    get_all_cot_responses,
    get_all_cot_responses_an,
    get_all_cot_responses_oa,
    get_all_cot_responses_or,
)
from chainscope.gt_eval import extract_gt_data
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
    "--evaluator_model_id",
    "-e",
    type=str,
    default="openai/gpt-4o",
    help="Model used to extract ground truth from responses.",
)
@click.option(
    "--max_retries",
    "-r",
    type=int,
    default=2,
    help="Maximum retries for extracting ground truth with the each evaluator model",
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
    evaluator_model_id: str,
    max_retries: int,
    verbose: bool,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    model_id = MODELS_MAP.get(model_id, model_id)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    dataset_params = DatasetParams.from_id(dataset_id)

    # Check if output file already exists
    output_directory = (
        DATA_DIR
        / "gt_eval_data"
        / instr_id
        / sampling_params.id
        / dataset_params.pre_id
        / dataset_params.id
    )
    output_path = output_directory / f"{model_id.replace('/', '__')}.yaml"
    if output_path.exists():
        logging.info(
            f"Output file already exists at {output_path}, skipping generation"
        )
        return

    # Save OpenAI API key, because our OpenRouter code overrides it
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

    if open_router:
        get_responses = get_all_cot_responses_or
    elif open_ai:
        get_responses = get_all_cot_responses_oa
    elif anthropic:
        get_responses = get_all_cot_responses_an
    else:
        get_responses = get_all_cot_responses

    open_ended_responses = get_responses(
        model_id=model_id,
        dataset_id=dataset_id,
        instr_id=instr_id,
        sampling_params=sampling_params,
        n_responses=n_responses,
        question_type="open-ended",
    )

    # Restore OpenAI API key
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    gt_eval_data = extract_gt_data(
        open_ended_responses=open_ended_responses,
        evaluator_model_id=evaluator_model_id,
        max_retries=max_retries,
    )

    gt_eval_data.save()


if __name__ == "__main__":
    main()
