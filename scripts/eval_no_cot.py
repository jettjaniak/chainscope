#!/usr/bin/env python3
import argparse
import json
import logging
from dataclasses import asdict

from chainscope import DATA_DIR
from chainscope.prompts import load_prompt
from chainscope.qs_evaluation import evaluate_no_cot
from chainscope.qs_generation import Question
from chainscope.utils import is_chat_model, load_model_and_tokenizer, setup_determinism


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate no-CoT")
    parser.add_argument(
        "-d",
        "--dataset-id",
        type=str,
        help="Dataset ID",
    )
    parser.add_argument(
        "-m",
        "--model-id",
        type=str,
        default="google/gemma-2-2b-it",
        help="Model ID",
    )
    parser.add_argument(
        "-p",
        "--prompt-id",
        type=str,
        default="v0",
        help="Prompt ID",
    )
    parser.add_argument("-s", "--seed", type=int, help="Random seed", default=0)
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    qs_dir = DATA_DIR / "qs"
    dataset_path = qs_dir / f"qs_{args.dataset_id}.json"
    assert dataset_path.exists(), f"Dataset {args.dataset_id} not found"
    with open(dataset_path, "r") as f:
        question_dataset: list[Question] = json.load(f)

    assert is_chat_model(args.model_id), "Model must be a chat model"
    model, tokenizer = load_model_and_tokenizer(args.model_id)

    setup_determinism(args.seed)
    prompt = load_prompt(args.prompt_id)

    results = evaluate_no_cot(
        model=model,
        tokenizer=tokenizer,
        question_dataset=question_dataset,
        prompt=prompt,
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        seed=args.seed,
    )

    no_cot_dir = DATA_DIR / "no-cot-accuracy"
    no_cot_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_id.split("/")[-1]
    filename = f"{model_name}_{args.dataset_id}.json"
    with open(no_cot_dir / filename, "w") as f:
        json.dump(asdict(results), f)


if __name__ == "__main__":
    main(parse_args())
