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
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def main(
    prop_id: str,
    n: int,
    max_comparisons: int,
    verbose: bool,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    datasets = gen_qs(
        prop_id=prop_id,
        n=n,
        max_comparisons=max_comparisons,
    )
    for dataset in datasets.values():
        dataset.save()


if __name__ == "__main__":
    main()
