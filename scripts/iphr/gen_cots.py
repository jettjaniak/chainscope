#!/usr/bin/env python3

import asyncio
import logging
from pathlib import Path

import click
import pandas as pd
import yaml

from chainscope.api_utils.anthropic_utils import \
    process_batch_results as process_anthropic_batch_results
from chainscope.api_utils.anthropic_utils import submit_anthropic_batch
from chainscope.api_utils.common import get_responses_async
from chainscope.api_utils.open_ai_utils import \
    process_batch_results as process_openai_batch_results
from chainscope.api_utils.open_ai_utils import submit_openai_batch
from chainscope.cot_generation import (create_batch_of_cot_prompts,
                                       create_cot_responses,
                                       get_local_responses_tl,
                                       get_local_responses_vllm)
from chainscope.typing import *
from chainscope.utils import MODELS_MAP


@click.group()
def cli():
    """Generate CoT responses using various APIs."""
    pass


@cli.command()
@click.option("-n", "--n-responses", type=int, required=True)
@click.option("-d", "--dataset-ids", type=str, required=True, help="Comma-separated list of dataset IDs")
@click.option("-m", "--model-id", type=str, required=True)
@click.option("-i", "--instr-id", type=str, required=True)
@click.option("-t", "--temperature", type=float, default=0.7)
@click.option("-p", "--top-p", type=float, default=0.9)
@click.option("--max-new-tokens", type=int, default=2_000)
@click.option(
    "--api",
    type=click.Choice(
        ["ant-batch", "oai-batch", "ant", "oai", "or", "ds"]
    ),
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
@click.option("--test", is_flag=True)
@click.option("-v", "--verbose", is_flag=True)
@click.option(
    "--unfaithful-only",
    is_flag=True,
    help="Only generate CoTs for unfaithful pairs identified in faithfulness YAMLs",
)
def submit(
    n_responses: int,
    dataset_ids: str,
    model_id: str,
    instr_id: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    api: str,
    max_retries: int,
    test: bool,
    verbose: bool,
    unfaithful_only: bool,
):
    """Submit CoT generation requests in realtime or using batch APIs."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    model_id = MODELS_MAP.get(model_id, model_id)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    # Process all datasets
    dataset_id_list = [ds.strip() for ds in dataset_ids.split(",")]
    
    for dataset_id in dataset_id_list:
        if dataset_id.startswith("wm-"):
            assert instr_id == "instr-wm"

        ds_params = DatasetParams.from_id(dataset_id)

        # Check that we have data for unfaithful pairs if requested
        faithfulness_data = None
        if unfaithful_only:
            faithfulness_dir = DATA_DIR / "faithfulness" / model_id.split("/")[-1]
            if not faithfulness_dir.exists():
                logging.warning(f"No faithfulness data found for model {model_id}")
                continue

            if ds_params.suffix is None:
                faithfulness_file_name = f"{ds_params.prop_id}.yaml"
            else:
                faithfulness_file_name = f"{ds_params.prop_id}_{ds_params.suffix}.yaml"
            faithfulness_path = faithfulness_dir / faithfulness_file_name
            if not faithfulness_path.exists():
                logging.warning(f"No faithfulness data found for model {model_id} on dataset {dataset_id}")
                continue

            with open(faithfulness_path, "r") as f:
                faithfulness_data = yaml.safe_load(f)
            if faithfulness_data is None:
                logging.error(f"Unable to load faithfulness data for model {model_id} on dataset {dataset_id}")
                continue

        # Try to load existing responses
        existing_responses = None
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
            if unfaithful_only:
                logging.warning(f"Unfaithful pairs requested but no existing responses found for model {model_id} on dataset {dataset_id}")
                continue

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
            logging.info(f"No prompts to process for dataset {dataset_id}")
            continue

        # Filter for unfaithful pairs if requested
        if faithfulness_data is not None:
            # Collect all unfaithful question IDs
            unfaithful_qids = set()
            logging.info(f"Filtering to {len(faithfulness_data)} unfaithful pairs")
            for qid, qdata in faithfulness_data.items():
                assert "metadata" in qdata and "reversed_q_id" in qdata["metadata"]
                unfaithful_qids.add(qid)
                unfaithful_qids.add(qdata["metadata"]["reversed_q_id"])
            logging.info(f"Found {len(unfaithful_qids)} question IDs after filtering by faithfulness")

            # Filter prompts to only include unfaithful pairs
            batch_of_cot_prompts = [
                (q_resp_id, prompt)
                for q_resp_id, prompt in batch_of_cot_prompts
                if q_resp_id.qid in unfaithful_qids
            ]

        logging.info(f"Number of prompts to process for dataset {dataset_id}: {len(batch_of_cot_prompts)}")

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
@click.option("-n", "--n-responses", type=int, required=True)
@click.option("-d", "--dataset-ids", type=str, required=True, help="Comma-separated list of dataset IDs")
@click.option("-m", "--model-id", type=str, required=True)
@click.option("-i", "--instr-id", type=str, required=True)
@click.option("-t", "--temperature", type=float, default=0.7)
@click.option("-p", "--top-p", type=float, default=0.9)
@click.option("--max-new-tokens", type=int, default=2_000)
@click.option(
    "--api",
    type=click.Choice(["vllm", "ttl"]),
    required=True,
    help="Local API to use for generation",
)
@click.option(
    "--model-id-for-fsp",
    type=str,
    default=None,
    help="Use CoT responses from this model id to use as FSP. Only used if generating responses for a base model.",
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
@click.option(
    "--local-gen-seed",
    type=int,
    default=42,
    help="Seed for local generation",
)
@click.option("--test", is_flag=True)
@click.option("-v", "--verbose", is_flag=True)
@click.option(
    "--unfaithful-only",
    is_flag=True,
    help="Only generate CoTs for unfaithful pairs identified in faithfulness YAMLs",
)
def local(
    n_responses: int,
    dataset_ids: str,
    model_id: str,
    instr_id: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    api: str,
    model_id_for_fsp: str | None,
    fsp_size: int,
    fsp_seed: int,
    local_gen_seed: int,
    test: bool,
    verbose: bool,
    unfaithful_only: bool,
):
    """Generate CoT responses using local models."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    model_id = MODELS_MAP.get(model_id, model_id)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    # Process all datasets
    dataset_id_list = [ds.strip() for ds in dataset_ids.split(",")]
    all_prompts: list[tuple[QuestionResponseId, str]] = []
    dataset_params_list: list[DatasetParams] = []
    existing_responses_list: list[CotResponses | None] = []
    qid_to_dataset: dict[str, str] = {}  # Map question IDs to dataset IDs

    # Check faithfulness data if requested
    faithfulness_data_by_dataset: dict[str, dict] = {}
    if unfaithful_only:
        model_name = model_id.split("/")[-1]
        faithfulness_dir = DATA_DIR / "faithfulness" / model_name
        if not faithfulness_dir.exists():
            logging.warning(f"No faithfulness data found for model {model_id}")
            return

        for dataset_id in dataset_id_list:
            ds_params = DatasetParams.from_id(dataset_id)
            if ds_params.suffix is None:
                faithfulness_file_name = f"{ds_params.prop_id}.yaml"
            else:
                faithfulness_file_name = f"{ds_params.prop_id}_{ds_params.suffix}.yaml"
            faithfulness_path = faithfulness_dir / faithfulness_file_name
            if not faithfulness_path.exists():
                logging.warning(f"No faithfulness data found for model {model_id} on dataset {dataset_id}")
                continue

            with open(faithfulness_path, "r") as f:
                data = yaml.safe_load(f)
            if data is None:
                logging.error(f"Unable to load faithfulness data for model {model_id} on dataset {dataset_id}")
                continue

            faithfulness_data_by_dataset[dataset_id] = data

        if not faithfulness_data_by_dataset:
            logging.error("No faithfulness data found for any dataset")
            return

    for dataset_id in dataset_id_list:
        if dataset_id.startswith("wm-"):
            assert instr_id == "instr-wm"

        ds_params = DatasetParams.from_id(dataset_id)
        dataset_params_list.append(ds_params)

        # Skip if we need faithfulness data but don't have it for this dataset
        if unfaithful_only and dataset_id not in faithfulness_data_by_dataset:
            continue

        # Try to load existing responses
        existing_responses = None
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
            if unfaithful_only:
                raise ValueError(f"Unfaithful pairs requested but no existing responses found for model {model_id} on dataset {dataset_id}")

        existing_responses_list.append(existing_responses)

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

        if batch_of_cot_prompts:
            # Track which dataset each question belongs to
            for q_resp_id, _ in batch_of_cot_prompts:
                qid_to_dataset[q_resp_id.qid] = dataset_id
            all_prompts.extend(batch_of_cot_prompts)

    if not all_prompts:
        logging.info("No prompts to process")
        return

    # Filter for unfaithful pairs if requested
    if unfaithful_only:
        # Collect all unfaithful question IDs
        unfaithful_qids = set()
        for dataset_id, data in faithfulness_data_by_dataset.items():
            for qid, qdata in data.items():
                assert "metadata" in qdata and "reversed_q_id" in qdata["metadata"]
                unfaithful_qids.add(qid)
                unfaithful_qids.add(qdata["metadata"]["reversed_q_id"])

        # Filter prompts to only include unfaithful pairs
        all_prompts = [
            (q_resp_id, prompt)
            for q_resp_id, prompt in all_prompts
            if q_resp_id.qid in unfaithful_qids
        ]
        logging.warning(f"Filtered to {len(all_prompts)} unfaithful pairs")

    # Process using local model
    if api == "vllm":
        results = get_local_responses_vllm(
            prompts=all_prompts,
            model_id=model_id,
            instr_id=instr_id,
            ds_params_list=dataset_params_list,
            sampling_params=sampling_params,
            model_id_for_fsp=model_id_for_fsp,
            fsp_size=fsp_size,
            fsp_seed=fsp_seed,
            qid_to_dataset=qid_to_dataset,
        )
    else:  # ttl
        results = get_local_responses_tl(
            prompts=all_prompts,
            model_id=model_id,
            instr_id=instr_id,
            ds_params_list=dataset_params_list,
            sampling_params=sampling_params,
            model_id_for_fsp=model_id_for_fsp,
            fsp_size=fsp_size,
            fsp_seed=fsp_seed,
            local_gen_seed=local_gen_seed,
            qid_to_dataset=qid_to_dataset,
        )

    if results:
        # Group results by dataset
        results_by_dataset: dict[str, list[tuple[QuestionResponseId, str]]] = {}
        for q_resp_id, response in results:
            dataset_id = qid_to_dataset[q_resp_id.qid]
            if dataset_id not in results_by_dataset:
                results_by_dataset[dataset_id] = []
            results_by_dataset[dataset_id].append((q_resp_id, response))

        # Save responses for each dataset
        for dataset_id, dataset_results in results_by_dataset.items():
            ds_idx = dataset_id_list.index(dataset_id)
            ds_params = dataset_params_list[ds_idx]
            existing_responses = existing_responses_list[ds_idx]

            cot_responses = create_cot_responses(
                responses_by_qid=existing_responses.responses_by_qid
                if existing_responses
                else None,
                new_responses=dataset_results,
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
