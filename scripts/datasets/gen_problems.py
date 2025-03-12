#! /usr/bin/env python3
import json
import logging
import shutil
import tarfile
import tempfile
from pathlib import Path

import click
import requests
from datasets import load_dataset
from tqdm import tqdm

from chainscope.typing import Problem, ProblemDataset

MATH_URL = "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar"


def get_answer_with_reasoning_for_math_problem(qid: str, answer: str) -> str:
    # MATH answers usually show the final answer in a "\boxed{}" LaTeX command at the end of the answer. E.g., "\boxed{\frac{x+2}{7}}". We need to consider only the last appearance of "\boxed{", since some answers have multiple "\boxed{}" commands.
    last_boxed_start = answer.rfind("\\boxed{")
    if last_boxed_start == -1:
        logging.info(
            f"Unable to extract answer without reasoning for MATH problem {qid} because it doesn't have a \\boxed{{}} command. Using the entire answer as the correct answer."
        )
        return answer
    else:
        # Find the matching closing brace by counting braces
        brace_count = 1
        pos = last_boxed_start + len("\\boxed{")
        while brace_count > 0 and pos < len(answer):
            if answer[pos] == "{":
                brace_count += 1
            elif answer[pos] == "}":
                brace_count -= 1
            pos += 1

        if brace_count > 0:
            logging.info(
                f"Unable to extract answer without reasoning for MATH problem {qid} because it has unmatched braces in the \\boxed{{}} command. Using the entire answer as the correct answer."
            )
            return answer
        else:
            return answer[last_boxed_start:pos]


def gen_math_dataset() -> ProblemDataset:
    """Generate ProblemDataset from the MATH dataset."""
    # Download and extract dataset
    logging.info("Downloading MATH dataset...")
    with tempfile.NamedTemporaryFile() as tmp_file:
        response = requests.get(MATH_URL, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192

        with tqdm(total=total_size, unit="iB", unit_scale=True) as pbar:
            for data in response.iter_content(block_size):
                tmp_file.write(data)
                pbar.update(len(data))
        tmp_file.flush()

        logging.info("Extracting archive...")
        with tarfile.open(tmp_file.name) as tar:
            tar.extractall()

    math_dir = Path("MATH")
    problems_by_qid: dict[str, Problem] = {}

    # Process train and test directories
    for split in ["train", "test"]:
        split_dir = math_dir / split
        for category_dir in sorted(split_dir.iterdir()):
            if not category_dir.is_dir():
                continue

            category = category_dir.name
            logging.info(f"Processing {split}/{category}...")

            # Process each problem JSON file
            for i, problem_file in enumerate(tqdm(sorted(category_dir.glob("*.json")))):
                with open(problem_file) as f:
                    data = json.load(f)

                qid = f"math_{split}_{category}_{i}"
                q_str = data["problem"]
                answer = data["solution"]
                answer_without_reasoning = get_answer_with_reasoning_for_math_problem(
                    qid, answer
                )
                problems_by_qid[qid] = Problem(
                    q_str=q_str,
                    answer=answer,
                    answer_without_reasoning=answer_without_reasoning,
                    split=split,
                    category=category,
                )

    # Clean up extracted files
    if math_dir.exists():
        shutil.rmtree(math_dir)

    return ProblemDataset(dataset_name="math", problems_by_qid=problems_by_qid)


def gen_gsm8k_dataset() -> ProblemDataset:
    # Load GSM8K dataset from HuggingFace
    dataset = load_dataset("gsm8k", "main")
    problems_by_qid: dict[str, Problem] = {}

    # Process both train and test splits
    logging.info("Processing GSM8K dataset...")
    for split in ["train", "test"]:
        data = dataset[split]

        # Create problem entries
        for i, item in enumerate(data):
            qid = f"gsm8k_{split}_{i}"
            q_str = item["question"]
            answer = item["answer"]

            # All gsm8k answers end with "####" and then the actual correct answer
            answer_without_reasoning = answer.split("####")[-1].strip()

            problems_by_qid[qid] = Problem(
                q_str=q_str,
                answer=answer,
                answer_without_reasoning=answer_without_reasoning,
                split=split,
                category=None,
            )

    return ProblemDataset(dataset_name="gsm8k", problems_by_qid=problems_by_qid)


def gen_mmmlu_dataset() -> ProblemDataset:
    """Generate ProblemDataset from the MMLU dataset."""
    # Math-related categories from MMLU
    MATH_CATEGORIES = [
        "abstract_algebra",
        "college_mathematics",
        "elementary_mathematics",
        "high_school_mathematics",
        "high_school_statistics",
        "college_physics",  # Including physics as it's heavily math-based
        "high_school_physics",
        "conceptual_physics",
    ]

    problems_by_qid: dict[str, Problem] = {}

    logging.info("Processing MMLU dataset...")
    for category in MATH_CATEGORIES:
        logging.info(f"Processing category: {category}")
        try:
            ds = load_dataset("cais/mmlu", category)

            # Process each split
            for split in ["dev", "validation", "test"]:
                if split not in ds:
                    continue

                logging.info(f"Processing {split} split...")
                for i, item in enumerate(tqdm(ds[split])):
                    # Format question and answer
                    question = item["question"]
                    if "Statement 1" in question:
                        # Add clarification for statement questions
                        question += (
                            "\n\nAssess whether each statement is true or false."
                        )
                        # Format statement-type answers
                        choice = item["choices"][item["answer"]]
                        parts = choice.split(", ")
                        if len(parts) == 2:
                            statement1 = (
                                "true" if parts[0].lower() == "true" else "false"
                            )
                            statement2 = (
                                "true" if parts[1].lower() == "true" else "false"
                            )
                            answer = f"Statement 1 is {statement1}. Statement 2 is {statement2}."
                        else:
                            answer = choice
                    else:
                        # For multiple choice, use the answer directly
                        answer = item["choices"][item["answer"]]

                    # MMLU answers are just the correct answer, no reasoning included.
                    answer_without_reasoning = answer

                    qid = f"mmlu_{category}_{split}_{i}"
                    problems_by_qid[qid] = Problem(
                        q_str=question,
                        answer=answer,
                        answer_without_reasoning=answer_without_reasoning,
                        split=split,
                        category=category,
                    )

        except Exception as e:
            logging.error(f"Error processing category {category}: {str(e)}")
            continue

    return ProblemDataset(dataset_name="mmlu", problems_by_qid=problems_by_qid)


@click.command()
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def main(
    verbose: bool,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    math_dataset = gen_math_dataset()
    gsm8k_dataset = gen_gsm8k_dataset()
    mmmlu_dataset = gen_mmmlu_dataset()

    math_dataset.save()
    gsm8k_dataset.save()
    mmmlu_dataset.save()


if __name__ == "__main__":
    main()
