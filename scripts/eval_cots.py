#!/usr/bin/env python3

import logging

import click

from chainscope.cot_eval import evaluate_cot_responses
from chainscope.typing import *


@click.command()
@click.argument("responses_path", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True)
def main(responses_path: str, verbose: bool):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    cot_responses = CotResponses.load(Path(responses_path))
    cot_eval = evaluate_cot_responses(cot_responses)
    path = cot_eval.save()
    logging.warning(f"Saved CoT eval to {path}")


if __name__ == "__main__":
    main()
