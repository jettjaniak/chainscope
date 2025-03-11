#!/usr/bin/env python3

import random
from pathlib import Path

import click
import yaml
from beartype import beartype

from chainscope.typing import *


@beartype
def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@click.command()
@click.argument("model_name")
@click.option("--seed", default=42, help="Random seed for shuffling")
def main(model_name: str, seed: int):
    base_dir = Path("d")
    old_dir = base_dir / "cot_eval/instr-wm-old-0"
    new_dir = base_dir / "cot_eval/instr-wm"
    responses_dir = base_dir / "cot_responses/instr-wm"

    # List to store all differences
    all_differences = []

    # Walk through all yaml files in new directory
    for new_path in new_dir.rglob(f"**/{model_name}.yaml"):
        # Construct corresponding old path
        rel_path = new_path.relative_to(new_dir)
        old_path = old_dir / rel_path
        responses_path = responses_dir / rel_path

        # Get the base name without the model
        base_name = rel_path.parent.name  # e.g. wm-movie-length_gt_NO_1_a08c203f
        comparison_type = rel_path.parent.parent.name  # e.g. gt_NO_1
        questions_path = base_dir / "questions" / comparison_type / f"{base_name}.yaml"

        if not old_path.exists():
            print(f"No corresponding file found for {rel_path}")
            continue

        new_data = load_yaml(new_path)
        old_data = load_yaml(old_path)
        responses_data = load_yaml(responses_path)
        questions_data = load_yaml(questions_path)["question-by-qid"]

        # Get the results sections
        new_results = new_data["results-by-qid"]
        old_results = old_data["results-by-qid"]

        # Compare results
        for qid in new_results:
            if qid not in old_results:
                continue

            new_result = new_results[qid]
            old_result = old_results[qid]

            # Process all UUIDs for this QID
            for uuid, new_result_data in new_result.items():
                if uuid not in old_result:
                    continue

                new_answer = new_result_data["result"]
                old_answer = old_result[uuid]["result"]

                if new_answer == old_answer:
                    continue

                question_str = questions_data[qid]["q-str"]
                question_str = question_str[question_str.index(":\n") + 2 :].strip()

                difference = {
                    "rel_path": str(rel_path),
                    "qid": qid,
                    "question": question_str,
                    "old_answer": old_answer,
                    "new_answer": new_answer,
                    "final_answer": new_result_data["final-answer"],
                    "explanation_final_answer": new_result_data[
                        "explanation-final-answer"
                    ],
                    "equal_values": new_result_data["equal-values"],
                    "explanation_equal_values": new_result_data[
                        "explanation-equal-values"
                    ],
                    "uuid": uuid,
                    "response": responses_data["responses-by-qid"][qid][uuid],
                }
                all_differences.append(difference)

    # Shuffle the differences
    random.seed(seed)
    random.shuffle(all_differences)

    # Present differences in random order
    for diff in all_differences:
        print(f"\nDifference found in {diff['rel_path']} for QID: {diff['qid']}")
        print(f"Question: {diff['question']}")
        print(f"Old answer: {diff['old_answer']}")
        print(f"New answer: {diff['new_answer']}")
        print("\nNew format details:")
        print(f"final-answer: {diff['final_answer']}")
        print(f"explanation-final-answer: {diff['explanation_final_answer']}")
        print(f"equal-values: {diff['equal_values']}")
        print(f"explanation-equal-values: {diff['explanation_equal_values']}")
        print(f"UUID: {diff['uuid']}")
        print("\nResponse being evaluated:")
        print("-" * 80)
        print(diff["response"])
        print("-" * 80)

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
