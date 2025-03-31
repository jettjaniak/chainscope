#!/usr/bin/env python3
import logging

import click

from chainscope.questions import gen_qs


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
    "--entity-popularity-filter",
    type=int,
    default=None,
    help="How well-known the entities should be in the generated dataset (1-10)",
)
@click.option(
    "--min-percent-value-diff",
    type=float,
    default=None,
    help="Minimum percent difference between values to generate (or not) close call comparisons. This is based on the absolute difference between the min and max values for the property.",
)
@click.option(
    "--dataset-suffix",
    type=str,
    default=None,
    help="If provided, the suffix to add to the dataset ID when saving the dataset.",
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
    entity_popularity_filter: int | None,
    min_percent_value_diff: float | None,
    dataset_suffix: str | None,
    remove_ambiguous: bool,
    verbose: bool,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    datasets = gen_qs(
        prop_id=prop_id,
        n=n,
        max_comparisons=max_comparisons,
        entity_popularity_filter=entity_popularity_filter,
        min_percent_value_diff=min_percent_value_diff,
        dataset_suffix=dataset_suffix,
        remove_ambiguous=remove_ambiguous,
    )
    for dataset in datasets.values():
        dataset.save()


if __name__ == "__main__":
    main()
