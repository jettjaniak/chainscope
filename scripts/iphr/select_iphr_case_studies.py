#!/usr/bin/env python3

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import click
import yaml
from tqdm import tqdm

from chainscope.typing import *


def simplify_question_str(q_str: str) -> str:
    """Remove first line and strip whitespace from question string."""
    lines = q_str.split("\n")
    if len(lines) > 1:
        lines = lines[1:]  # Remove first line
    return "\n".join(lines).strip()


def has_enough_responses(qdata: dict[str, Any], min_responses: int = 3) -> bool:
    """Check if question has enough responses in both directions."""
    n_q1_incorrect = len(qdata["unfaithful_responses"])
    n_q2_correct = len(qdata["metadata"]["reversed_q_correct_responses"])
    return n_q1_incorrect >= min_responses and n_q2_correct >= min_responses


def simplify_data(qdata: dict[str, Any]) -> dict[str, Any]:
    """Extract and rename only the requested fields from the data."""
    meta = qdata["metadata"]
    q1_correct_answer = meta["answer"]
    q2_correct_answer = "YES" if q1_correct_answer == "NO" else "NO"
    return {
        "q1_incorrect_resp": qdata["unfaithful_responses"],
        "q2_correct_resp": meta["reversed_q_correct_responses"],
        "n_q1_incorrect": len(qdata["unfaithful_responses"]),
        "n_q2_correct": len(meta["reversed_q_correct_responses"]),
        "q1_p_correct": meta["p_correct"],
        "q2_p_correct": meta["reversed_q_p_correct"],
        "q1_str": simplify_question_str(meta["q_str"]),
        "q2_str": simplify_question_str(meta["reversed_q_str"]),
        "q1_correct_answer": q1_correct_answer,
        "q2_correct_answer": q2_correct_answer,
        "prop_id": meta["prop_id"],
        "comparison": meta["comparison"],
        "group_p_yes_mean": meta["group_p_yes_mean"],
        "x_name": meta["x_name"],
        "y_name": meta["y_name"],
        "x_value": meta["x_value"],
        "y_value": meta["y_value"],
    }


def process_model_directory(
    input_dir: Path, output_file: Path, min_responses: int, seed: int
) -> tuple[int, int]:
    """Process all property files in a model directory and return number of pairs saved.

    Args:
        input_dir: Directory containing property YAML files
        output_file: Output file to save case studies
        min_responses: Minimum number of responses required in both directions
        seed: Random seed for reproducibility

    Returns:
        tuple of (total number of pairs saved, number of new pairs added)
    """
    random.seed(seed)

    # Load existing data if output file exists
    existing_data = {}
    existing_pairs = set()
    if output_file.exists():
        with open(output_file) as f:
            existing_data = yaml.safe_load(f) or {}
            # Get existing (prop_id, comparison) pairs
            for qdata in existing_data.values():
                existing_pairs.add((qdata["prop_id"], qdata["comparison"]))

    # Group questions by prop_id and comparison
    groups: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)

    # Process each prop_id file in the model directory
    for prop_file in input_dir.glob("*.yaml"):
        with open(prop_file) as f:
            prop_data = yaml.safe_load(f)

        if not prop_data:
            continue

        for qid, qdata in prop_data.items():
            q_meta = qdata["metadata"]
            # Only include questions with enough responses
            if not has_enough_responses(qdata, min_responses):
                continue

            key = (q_meta["prop_id"], q_meta["comparison"])
            # Skip if we already have this pair
            if key in existing_pairs:
                continue

            groups[key].append((qid, qdata))

    # Select one random question from each new group and simplify its data
    len_existing = len(existing_data)
    selected = existing_data
    for group_key, questions in groups.items():
        if not questions:  # Skip groups with no valid questions
            continue
        qid, qdata = random.choice(questions)
        selected[qid] = simplify_data(qdata)

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save selected questions
    with open(output_file, "w") as f:
        yaml.dump(selected, f)

    return len(selected), len(selected) - len_existing


@click.command()
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility")
@click.option(
    "--min-responses",
    type=int,
    default=3,
    help="Minimum number of responses required in both directions",
)
def main(seed: int, min_responses: int) -> None:
    """Process all model directories in DATA_DIR/faithfulness and save to DATA_DIR/case_studies.

    For each model, selects random question pairs that have enough responses
    in both directions.

    Args:
        seed: Random seed for reproducibility
        min_responses: Minimum number of responses required in both directions
    """
    input_dir = DATA_DIR / "faithfulness"
    output_dir = DATA_DIR / "case_studies"

    # Find all model directories
    model_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    for model_dir in tqdm(model_dirs, desc="Processing models"):
        model_name = model_dir.name
        output_file = output_dir / f"{model_name}.yaml"
        n_pairs, n_new_pairs = process_model_directory(
            model_dir, output_file, min_responses, seed
        )
        print(f"{model_name}: {n_pairs} pairs total, {n_new_pairs} new pairs added")


if __name__ == "__main__":
    main()
