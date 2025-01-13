#!/usr/bin/env python3

import click

from chainscope.cot_generation import get_all_cot_responses, get_all_cot_responses_or
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
def main(
    n_responses: int,
    dataset_id: str,
    model_id: str,
    instr_id: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    open_router: bool,
):
    model_id = MODELS_MAP.get(model_id, model_id)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    get_responses = get_all_cot_responses_or if open_router else get_all_cot_responses
    cot_responses = get_responses(
        model_id=model_id,
        dataset_id=dataset_id,
        instr_id=instr_id,
        sampling_params=sampling_params,
        n_responses=n_responses,
    )
    cot_responses.save()


if __name__ == "__main__":
    main()
