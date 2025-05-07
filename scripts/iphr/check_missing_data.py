#!/usr/bin/env python3

"""Check for missing data across models.

This script checks for the presence of 10 responses and evaluations for each question across models.
It can filter by dataset suffix and/or prefix (prop_id).

Example usage:
    python3 scripts/iphr/check_missing_data.py --dataset-suffix "gt_YES_1" --prop-id "aircraft-speeds"
    python3 scripts/iphr/check_missing_data.py --temperature 0.7 --top-p 0.9 --max-new-tokens 2000
"""

import logging
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from beartype import beartype

from chainscope.typing import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def get_datasets(
    instr_id: str,
    dataset_suffix: str | None = None,
    prop_id: str | None = None,
) -> list[DatasetParams]:
    """Get all datasets for checking missing data.

    Args:
        instr_id: Instruction ID to get datasets for
        dataset_suffix: Optional suffix to filter datasets by
        prop_id: Optional property ID to filter datasets by
    """
    datasets = []
    for dataset_dir in (DATA_DIR / "questions").iterdir():
        if not dataset_dir.is_dir():
            continue
        for dataset_file in dataset_dir.glob("*.yaml"):
            # Apply filters before parsing
            if dataset_suffix and not dataset_file.stem.endswith(dataset_suffix):
                continue
            if prop_id and not dataset_file.stem.startswith(prop_id):
                continue
            try:
                ds_params = DatasetParams.from_id(dataset_file.stem)
                datasets.append(ds_params)
            except ValueError:
                continue
    return datasets


def get_available_models(datasets: list[DatasetParams], instr_id: str, sampling_params: SamplingParams) -> set[str]:
    """Get all available models across all datasets.
    
    Args:
        datasets: List of datasets to check
        instr_id: Instruction ID
        sampling_params: Sampling parameters for the response directory
    """
    models = set()
    for ds_params in datasets:
        response_dir = DATA_DIR / "cot_responses" / instr_id / sampling_params.id / ds_params.pre_id / ds_params.id
        if response_dir.exists():
            models.update(p.stem.replace("__", "/") for p in response_dir.glob("*.yaml"))
    return models


@beartype
def load_responses(
    ds_params: DatasetParams,
    instr_id: str,
    model_id: str,
    sampling_params: SamplingParams,
) -> dict[str, dict[str, MathResponse | AtCoderResponse | str]] | None:
    """Load responses for a dataset.
    
    Args:
        ds_params: Dataset parameters
        instr_id: Instruction ID
        model_id: Model ID
        sampling_params: Sampling parameters
        
    Returns:
        Dictionary mapping question IDs to response dictionaries
    """
    response_path = ds_params.cot_responses_path(
        instr_id=instr_id,
        model_id=model_id,
        sampling_params=sampling_params,
    )
    if not response_path.exists():
        return None
        
    cot_responses = CotResponses.load(response_path)
    return cot_responses.responses_by_qid


@beartype
def load_evaluations(
    ds_params: DatasetParams,
    instr_id: str,
    model_id: str,
    sampling_params: SamplingParams,
) -> dict[str, dict[str, CotEvalResult]] | None:
    """Load evaluations for a dataset.
    
    Args:
        ds_params: Dataset parameters
        instr_id: Instruction ID
        model_id: Model ID
        sampling_params: Sampling parameters
        
    Returns:
        Dictionary mapping question IDs to evaluation dictionaries
    """
    eval_path = ds_params.cot_eval_path(
        instr_id=instr_id,
        model_id=model_id,
        sampling_params=sampling_params,
    )
    if not eval_path.exists():
        return None
        
    cot_eval = CotEval.load(eval_path)
    return cot_eval.results_by_qid


@click.command()
@click.option(
    "-s",
    "--dataset-suffix",
    type=str,
    help="Dataset suffix to filter by (e.g., 'non-ambiguous-hard-2')",
)
@click.option(
    "--prop-id",
    type=str,
    help="Property ID to filter by (e.g., 'aircraft-speeds')",
)
@click.option(
    "-t",
    "--temperature",
    type=float,
    default=0.7,
    help="Temperature for sampling (default: 0.7)",
)
@click.option(
    "-p",
    "--top-p",
    type=float,
    default=0.9,
    help="Top-p for sampling (default: 0.9)",
)
@click.option(
    "-m",
    "--max-new-tokens",
    type=int,
    default=2000,
    help="Maximum number of new tokens to generate (default: 2000)",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def main(
    dataset_suffix: Optional[str],
    prop_id: Optional[str],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    verbose: bool,
):
    """Check for missing data across models."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get all datasets with filters applied
    datasets = get_datasets("instr-wm", dataset_suffix, prop_id)

    if not datasets:
        logging.error("No datasets found after filtering")
        return
    
    logging.info(f"Found {len(datasets)} datasets")
    for ds_params in datasets:
        logging.info(f"  {ds_params.id}")

    # Create sampling params from command line arguments
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    # Get all models from all datasets
    models = sorted(get_available_models(datasets, "instr-wm", sampling_params))

    if not models:
        logging.error("No models found in any dataset")
        return

    logging.info(f"Found {len(models)} models")
    for model_id in models:
        logging.info(f"  {model_id}")

    # Check each model and dataset combination
    for model_id in models:
        logging.info(f"Checking model {model_id}")
        for ds_params in sorted(datasets, key=lambda x: x.id):
            try:
                # Load responses and evaluations separately
                responses = load_responses(
                    ds_params=ds_params,
                    instr_id="instr-wm",
                    model_id=model_id,
                    sampling_params=sampling_params,
                )
                if responses is None:
                    logging.warning(f"No responses found for model {model_id} and dataset {ds_params.id}")
                    continue

                evals = load_evaluations(
                    ds_params=ds_params,
                    instr_id="instr-wm",
                    model_id=model_id,
                    sampling_params=sampling_params,
                )
                if evals is None:
                    logging.warning(f"No evaluations found for model {model_id} and dataset {ds_params.id}")
                    continue

                # Check responses
                for qid, response_by_uuid in responses.items():
                    if len(response_by_uuid) != 10:
                        logging.warning(
                            f"Model {model_id} has {len(response_by_uuid)} responses "
                            f"for question {qid} in dataset {ds_params.id}"
                        )

                # Check evaluations
                for qid, eval_by_uuid in evals.items():
                    if len(eval_by_uuid) != 10:
                        logging.warning(
                            f"Model {model_id} has {len(eval_by_uuid)} evaluations "
                            f"for question {qid} in dataset {ds_params.id}"
                        )

            except Exception as e:
                logging.error(
                    f"Error processing model {model_id} dataset {ds_params.id}: {e}"
                )


if __name__ == "__main__":
    main() 