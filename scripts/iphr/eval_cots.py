#!/usr/bin/env python3

import logging
from pathlib import Path

import click
from tqdm import tqdm

from chainscope.api_utils.anthropic_utils import process_batch_results
from chainscope.cot_eval import (create_cot_eval_from_batch_results,
                                 evaluate_cot_responses_realtime,
                                 evaluate_cot_responses_with_batch)
from chainscope.typing import *
from chainscope.utils import MODELS_MAP


@click.group()
def cli():
    """Evaluate CoT responses using various APIs."""
    pass


@cli.command()
@click.option(
    "--responses-paths", 
    "-r", 
    required=True, 
    type=str, 
    help="Comma-separated list of response YAML file paths to evaluate"
)
@click.option("-v", "--verbose", is_flag=True)
@click.option(
    "--llm-model-id",
    "-m",
    type=str,
    help="Model ID to use for LLM-based evaluation. If not provided, will use heuristic evaluation.",
)
@click.option(
    "--api",
    type=click.Choice(["ant-batch", "ant", "oai", "or", "ds"]),
    help="API to use for LLM-based evaluation (required if using --llm-model-id).",
)
@click.option(
    "--test",
    is_flag=True,
    help="Test mode: only evaluate first response from first 10 questions.",
)
def submit(
    responses_paths: str,
    verbose: bool,
    llm_model_id: str | None,
    api: str | None,
    test: bool,
):
    """Submit CoT evaluation requests in realtime or using Anthropic's batch API."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    
    # Process all response files
    path_list = [path.strip() for path in responses_paths.split(",")]
    
    for responses_path in tqdm(path_list, desc="Processing response files"):
        if not Path(responses_path).exists():
            logging.warning(f"Response file not found: {responses_path}")
            continue
            
        logging.info(f"Processing responses file: {responses_path}")
        cot_responses = CotResponses.load(Path(responses_path))

        if test:
            # Limit to first response from first 10 questions
            limited_responses = {}
            for i, (qid, response_by_uuid) in enumerate(
                cot_responses.responses_by_qid.items()
            ):
                if i >= 10:
                    break
                # Take first response only
                first_uuid = next(iter(response_by_uuid))
                limited_responses[qid] = {first_uuid: response_by_uuid[first_uuid]}

            cot_responses = CotResponses(
                responses_by_qid=limited_responses,
                model_id=cot_responses.model_id,
                instr_id=cot_responses.instr_id,
                ds_params=cot_responses.ds_params,
                sampling_params=cot_responses.sampling_params,
            )

        if llm_model_id is not None:
            if api is None:
                raise ValueError("--api is required when using --llm-model-id")
            llm_model_id = MODELS_MAP.get(llm_model_id, llm_model_id)

        # Try to load existing eval if it exists
        existing_eval = None
        if llm_model_id is not None:
            evaluator = llm_model_id
        else:
            evaluator = "heuristic"

        assert isinstance(cot_responses.sampling_params, SamplingParams)
        assert isinstance(cot_responses.ds_params, DatasetParams)
        eval_path = cot_responses.ds_params.cot_eval_path(
            cot_responses.instr_id,
            cot_responses.model_id,
            cot_responses.sampling_params,
        )
        if eval_path.exists():
            existing_eval = CotEval.load(eval_path)
            if existing_eval.evaluator != evaluator:
                logging.warning(
                    f"Evaluator mismatch with existing eval: {existing_eval.evaluator} != {evaluator}"
                )
            logging.info(f"Loaded existing eval from {eval_path}")
        else:
            logging.warning(f"No existing eval found at {eval_path}, starting fresh")

        if api == "ant-batch":
            # Submit batch using Anthropic's batch API
            assert llm_model_id is not None, "llm_model_id is required for batch evaluation"
            batch_info = evaluate_cot_responses_with_batch(
                cot_responses,
                evaluator_model_id=llm_model_id,
                existing_eval=existing_eval,
            )
            if batch_info is not None:
                logging.warning(
                    f"Submitted batch {batch_info.batch_id}\nBatch info saved to {batch_info.save()}"
                )
        else:
            # Process in realtime using specified API
            cot_eval = evaluate_cot_responses_realtime(
                cot_responses=cot_responses,
                llm_model_id=llm_model_id,
                api=api,
                existing_eval=existing_eval,
            )
            path = cot_eval.save()
            logging.warning(f"Saved CoT eval to {path}")


@cli.command()
@click.argument("batch_path", type=click.Path(exists=True, path_type=Path))
@click.option("-v", "--verbose", is_flag=True)
def process_batch(batch_path: Path, verbose: bool):
    """Process results from a completed Anthropic batch."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    batch_info = AnthropicBatchInfo.load(batch_path)
    results = process_batch_results(batch_info)
    if results:
        # Create and save CotEval
        ds_params = batch_info.ds_params
        eval_path = ds_params.cot_eval_path(
            batch_info.instr_id,
            batch_info.evaluated_model_id,
            batch_info.evaluated_sampling_params,
        )
        existing_eval = None
        if eval_path.exists():
            existing_eval = CotEval.load(eval_path)
            logging.warning(f"Loaded existing eval from {eval_path}")

        cot_eval = create_cot_eval_from_batch_results(
            batch_info=batch_info,
            batch_results=results,
            existing_eval=existing_eval,
        )
        cot_eval.save()


if __name__ == "__main__":
    cli()
