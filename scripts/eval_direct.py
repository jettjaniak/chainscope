#!/usr/bin/env python3

import click

from chainscope.qs_evaluation import evaluate_direct
from chainscope.qs_generation import QsDataset
from chainscope.typing import *
from chainscope.utils import MODELS_MAP, load_model_and_tokenizer


@click.command()
@click.option("-d", "--dataset-id", type=str, required=True)
@click.option("-m", "--model-id", type=str, required=True)
@click.option("-i", "--instr-id", type=str, default="v0")
def main(dataset_id: str, model_id: str, instr_id: str):
    question_dataset = QsDataset.load(dataset_id)
    model_id = MODELS_MAP.get(model_id, model_id)
    model, tokenizer = load_model_and_tokenizer(model_id)
    direct_eval = evaluate_direct(
        model=model,
        tokenizer=tokenizer,
        question_dataset=question_dataset,
        instr_id=instr_id,
    )
    direct_eval.save(dataset_id)


if __name__ == "__main__":
    main()