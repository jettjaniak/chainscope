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
    "--min-fraction-value-diff",
    type=float,
    default=None,
    help="Minimum fraction difference between values to generate (or not) close call comparisons. This is based on the absolute difference between the min and max values for the property. Should be between 0 and 1.",
)
@click.option(
    "--max-fraction-value-diff",
    type=float,
    default=None,
    help="Maximum fraction difference between values to generate (or not) close call comparisons. This is based on the absolute difference between the min and max values for the property. Should be between 0 and 1.",
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
    type=click.Choice(["no", "enough-comparisons", "enough-pairs"], case_sensitive=False),
    default="enough-pairs",
    help="Whether to remove ambiguous questions from the dataset. 'no' means no filtering, 'enough-comparisons' evaluate each pair of potential questions until we have max_comparisons for each entity, 'enough-pairs' evaluate each pair of potential entities until they have enough pairs (n)",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def main(
    prop_id: str,
    n: int,
    max_comparisons: int,
    min_popularity: int | None,
    max_popularity: int | None,
    min_fraction_value_diff: float | None,
    max_fraction_value_diff: float | None,
    dataset_suffix: str | None,
    min_rag_values_count: int | None,
    non_overlapping_rag_values: bool,
    remove_ambiguous: Literal["no", "enough-comparisons", "enough-pairs"],
    verbose: bool,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    if min_fraction_value_diff is not None:
        assert 0 <= min_fraction_value_diff <= 1, f"min_fraction_value_diff must be between 0 and 1, got {min_fraction_value_diff}"
    if max_fraction_value_diff is not None:
        assert 0 <= max_fraction_value_diff <= 1, f"max_fraction_value_diff must be between 0 and 1, got {max_fraction_value_diff}"
    if min_popularity is not None:
        assert 1 <= min_popularity <= 10, f"min_popularity must be between 1 and 10, got {min_popularity}"
    if max_popularity is not None:
        assert 1 <= max_popularity <= 10, f"max_popularity must be between 1 and 10, got {max_popularity}"
    datasets = gen_qs(
        prop_id=prop_id,
        n=n,
        max_comparisons=max_comparisons,
        min_popularity=min_popularity,
        max_popularity=max_popularity,
        min_fraction_value_diff=min_fraction_value_diff,
        max_fraction_value_diff=max_fraction_value_diff,
        dataset_suffix=dataset_suffix,
        remove_ambiguous=remove_ambiguous,
        non_overlapping_rag_values=non_overlapping_rag_values,
        min_rag_values_count=min_rag_values_count,
        evaluator_model_id="chatgpt-4o-latest",
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
