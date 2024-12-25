#!/usr/bin/env python3
import logging
from typing import Literal

import click

from chainscope.questions import QsDataset, gen_qs


@click.command()
@click.option(
    "-c",
    "--comparison",
    type=click.Choice(["gt", "lt"]),
    required=True,
)
@click.option(
    "-a",
    "--answer",
    type=click.Choice(["YES", "NO"]),
    required=True,
)
@click.option(
    "-p",
    "--prop-id",
    type=str,
    required=True,
)
@click.option(
    "-m",
    "--max-comparisons",
    type=int,
    default=1,
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def main(
    comparison: Literal["gt", "lt"],
    max_comparisons: int,
    answer: Literal["YES", "NO"],
    prop_id: str,
    verbose: bool,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    qsds: QsDataset = gen_qs(
        answer=answer,
        comparison=comparison,
        max_comparisons=max_comparisons,
        prop_id=prop_id,
    )
    path = qsds.save()
    logging.warning(f"Saved {len(qsds.question_by_qid)} questions to {path}")


if __name__ == "__main__":
    main()
