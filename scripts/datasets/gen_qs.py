#!/usr/bin/env python3
import logging

import click

from chainscope.api_utils.api_selector import APIPreferences
from chainscope.questions import gen_qs
from chainscope.typing import *


@click.command()
@click.option(
    "-p",
    "--prop-id",
    type=str,
    required=True,
)
@click.option(
    "-n",
    type=int,
    required=True,
    help="Total number of questions to generate",
)
@click.option(
    "-m",
    "--max-comparisons",
    type=int,
    default=1,
    help="Number of comparisons to make for each question",
)
@click.option(
    "--min-popularity",
    type=int,
    default=None,
    help="How well-known the entities should be in the generated dataset (1-10)",
)
@click.option(
    "--max-popularity",
    type=int,
    default=None,
    help="How unknown the entities should be in the generated dataset (1-10)",
)
@click.option(
    "--min-percent-value-diff",
    type=float,
    default=None,
    help="Minimum percent difference between values to generate (or not) close call comparisons. This is based on the absolute difference between the min and max values for the property.",
)
@click.option(
    "--max-percent-value-diff",
    type=float,
    default=None,
    help="Maximum percent difference between values to generate (or not) close call comparisons. This is based on the absolute difference between the min and max values for the property.",
)
@click.option(
    "--dataset-suffix",
    type=str,
    default=None,
    help="If provided, the suffix to add to the dataset ID when saving the dataset.",
)
@click.option(
    "--min-rag-values-count",
    type=int,
    default=2,
    help="Minimum number of RAG values (inclusive) that each entity should have.",
)
@click.option(
    "--non-overlapping-rag-values",
    is_flag=True,
    default=False,
    help="Whether to ensure that the RAG values for each entity are non-overlapping.",
)
@click.option(
    "--remove-ambiguous",
    is_flag=True,
    default=False,
    help="Whether to remove ambiguous questions from the dataset.",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def main(
    prop_id: str,
    n: int,
    max_comparisons: int,
    min_popularity: int | None,
    max_popularity: int | None,
    min_percent_value_diff: float | None,
    max_percent_value_diff: float | None,
    dataset_suffix: str | None,
    min_rag_values_count: int | None,
    non_overlapping_rag_values: bool,
    remove_ambiguous: bool,
    verbose: bool,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    datasets = gen_qs(
        prop_id=prop_id,
        n=n,
        max_comparisons=max_comparisons,
        min_popularity=min_popularity,
        max_popularity=max_popularity,
        min_percent_value_diff=min_percent_value_diff,
        max_percent_value_diff=max_percent_value_diff,
        dataset_suffix=dataset_suffix,
        remove_ambiguous=remove_ambiguous,
        non_overlapping_rag_values=non_overlapping_rag_values,
        min_rag_values_count=min_rag_values_count,
        evaluator_model_id="gpt-4o-latest",
        evaluator_sampling_params=SamplingParams(
            temperature=0.7,
            max_new_tokens=1500,
            top_p=0.9,
        ),
        api_preferences=APIPreferences(
            open_router=False,
            open_ai=True,
            anthropic=False,
            deepseek=False,
        ),
    )
    for dataset in datasets.values():
        dataset.save()


if __name__ == "__main__":
    main()
