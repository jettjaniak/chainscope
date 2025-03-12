#!/usr/bin/env python3

import asyncio
import logging
from pathlib import Path

import click

from chainscope.api_utils.anthropic_utils import (
    process_batch_results as process_anthropic_batch_results,
)
from chainscope.api_utils.anthropic_utils import submit_anthropic_batch
from chainscope.api_utils.common import get_responses_async
from chainscope.api_utils.open_ai_utils import (
    process_batch_results as process_openai_batch_results,
)
from chainscope.api_utils.open_ai_utils import submit_openai_batch
from chainscope.cot_generation import (
    create_batch_of_cot_prompts,
    create_cot_responses,
    get_local_responses,
)
from chainscope.typing import *
from chainscope.utils import MODELS_MAP


@click.group()
def cli():
    """Generate CoT responses using various APIs."""
    pass


@cli.command()
@click.option("-n", "--n-responses", type=int, required=True)
@click.option("-d", "--dataset-id", type=str, required=True)
@click.option("-m", "--model-id", type=str, required=True)
@click.option("-i", "--instr-id", type=str, required=True)
@click.option("-t", "--temperature", type=float, default=0.7)
@click.option("-p", "--top-p", type=float, default=0.9)
@click.option("--max-new-tokens", type=int, default=2_000)
@click.option(
    "--api",
    type=click.Choice(["ant-batch", "oai-batch", "ant", "oai", "or", "ds", "local"]),
    required=True,
    help="API to use for generation",
)
@click.option(
    "--max-retries",
    "-r",
    type=int,
    default=1,
    help="Maximum number of retries for each request",
)
@click.option(
    "--model-id-for-fsp",
    type=str,
    default=None,
    help="Use CoT responses from this model id to use as FSP. Only used if --api is 'local' and generating responses for a base model.",
)
@click.option(
    "--fsp-size",
    type=int,
    default=5,
    help="Size of FSP to use for generation with --model-id-for-fsp",
)
@click.option(
    "--fsp-seed",
    type=int,
    default=42,
    help="Seed for FSP selection",
)
@click.option("--test", is_flag=True)
@click.option("-v", "--verbose", is_flag=True)
def submit(
    n_responses: int,
    dataset_id: str,
    model_id: str,
    instr_id: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    api: str,
    max_retries: int,
    model_id_for_fsp: str | None,
    fsp_size: int,
    fsp_seed: int,
    test: bool,
    verbose: bool,
):
    """Submit CoT generation requests in realtime or using batch APIs."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    model_id = MODELS_MAP.get(model_id, model_id)

    if dataset_id.startswith("wm-"):
        assert instr_id == "instr-wm"

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    # Try to load existing responses
    existing_responses = None
    ds_params = DatasetParams.from_id(dataset_id)
    response_path = ds_params.cot_responses_path(
        instr_id,
        model_id,
        sampling_params,
    )
    if response_path.exists():
        existing_responses = CotResponses.load(response_path)
        logging.warning(f"Loaded existing responses from {response_path}")
    else:
        logging.warning(
            f"No existing responses found at {response_path}, starting fresh"
        )

    instructions = Instructions.load(instr_id)
    question_dataset = QsDataset.load(dataset_id)
    batch_of_cot_prompts = create_batch_of_cot_prompts(
        question_dataset=question_dataset,
        instructions=instructions,
        question_type="yes-no",
        n_responses=n_responses,
        existing_responses=existing_responses,
    )
    if test:
        batch_of_cot_prompts = batch_of_cot_prompts[:10]

    if not batch_of_cot_prompts:
        logging.info("No prompts to process")
        return

    if api in ["ant-batch", "oai-batch"]:
        # Submit batch using appropriate API
        prompt_by_qrid = {
            q_resp_id: prompt for q_resp_id, prompt in batch_of_cot_prompts
        }
        if api == "ant-batch":
            batch_info = submit_anthropic_batch(
                prompt_by_qrid=prompt_by_qrid,
                instr_id=instr_id,
                ds_params=ds_params,
                evaluated_model_id=model_id,
                evaluated_sampling_params=sampling_params,
            )
        else:  # oai-batch
            batch_info = submit_openai_batch(
                prompt_by_qrid=prompt_by_qrid,
                instr_id=instr_id,
                ds_params=ds_params,
                evaluated_model_id=model_id,
                evaluated_sampling_params=sampling_params,
            )
        logging.warning(
            f"Submitted batch {batch_info.batch_id}\nBatch info saved to {batch_info.save()}"
        )
    else:
        # Process in realtime using specified API
        if api == "local":
            results = get_local_responses(
                prompts=batch_of_cot_prompts,
                model_id=model_id,
                instr_id=instr_id,
                ds_params=ds_params,
                sampling_params=sampling_params,
                model_id_for_fsp=model_id_for_fsp,
                fsp_size=fsp_size,
                fsp_seed=fsp_seed,
            )
        else:
            results = asyncio.run(
                get_responses_async(
                    prompts=batch_of_cot_prompts,
                    model_id=model_id,
                    sampling_params=sampling_params,
                    api=api,
                    max_retries=max_retries,
                )
            )
        if results:
            # Create and save CotResponses
            cot_responses = create_cot_responses(
                responses_by_qid=existing_responses.responses_by_qid
                if existing_responses
                else None,
                new_responses=results,
                model_id=model_id,
                instr_id=instr_id,
                ds_params=ds_params,
                sampling_params=sampling_params,
            )
            cot_responses.save()


@cli.command()
@click.argument("batch_path", type=click.Path(exists=True, path_type=Path))
@click.option("-v", "--verbose", is_flag=True)
def process_batch(batch_path: Path, verbose: bool):
    """Process results from a completed batch."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Load batch info and determine type
    if "anthropic_batches" in str(batch_path):
        batch_info = AnthropicBatchInfo.load(batch_path)
        results = process_anthropic_batch_results(batch_info)
    elif "openai_batches" in str(batch_path):
        batch_info = OpenAIBatchInfo.load(batch_path)
        results = process_openai_batch_results(batch_info)
    else:
        raise ValueError("Unknown batch type")

    if results:
        # Create and save CotResponses
        ds_params = batch_info.ds_params
        response_path = ds_params.cot_responses_path(
            batch_info.instr_id,
            batch_info.evaluated_model_id,
            batch_info.evaluated_sampling_params,
        )
        existing_responses = None
        if response_path.exists():
            existing_responses = CotResponses.load(response_path)
            logging.warning(f"Loaded existing responses from {response_path}")

        cot_responses = create_cot_responses(
            responses_by_qid=existing_responses.responses_by_qid
            if existing_responses
            else None,
            new_responses=results,
            model_id=batch_info.evaluated_model_id,
            instr_id=batch_info.instr_id,
            ds_params=ds_params,
            sampling_params=batch_info.evaluated_sampling_params,
        )
        cot_responses.save()


if __name__ == "__main__":
    cli()
