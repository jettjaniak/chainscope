#!/usr/bin/env python3

import logging
from collections import defaultdict

import click
import yaml

from chainscope.typing import *


@click.command()
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def main(verbose: bool):
    """Check that all models have response files for all datasets and each question has 10 responses."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Get all question files and response files
    question_files = list(DATA_DIR.glob("questions/*/*.yaml"))
    response_files = list(
        DATA_DIR.glob("cot_responses/instr-v0/T0.7_P0.9_M2000/*/*/*.yaml")
    )

    # Get set of all unique model IDs
    all_models = {resp_file.stem for resp_file in response_files}
    logging.info(f"Found {len(all_models)} models: {sorted(all_models)}")

    # Create mapping of dataset path -> list of model response files
    responses_by_dataset = defaultdict(list)
    for resp_file in response_files:
        # Extract dataset path from response file path
        # e.g., gt_NO_1/aircraft-speeds_gt_NO_1_377c39d3 from the full path
        dataset_path = f"{resp_file.parent.parent.name}/{resp_file.parent.name}"
        responses_by_dataset[dataset_path].append(resp_file)

    # Check each question file
    for q_file in question_files:
        # Get dataset path relative to questions dir
        # e.g., gt_NO_1/aircraft-speeds_gt_NO_1_377c39d3
        rel_path = f"{q_file.parent.name}/{q_file.stem}"

        # Skip test datasets and specific datasets
        skip_datasets = {"animals-speed", "sea-depths", "sound-speeds", "train-speeds"}
        if "test" in rel_path.lower() or any(
            skip in rel_path for skip in skip_datasets
        ):
            continue

        # Load questions to get number of questions
        with open(q_file) as f:
            questions = yaml.safe_load(f)

        # Get response files for this dataset
        dataset_responses = responses_by_dataset[rel_path]

        # Check which models are missing files for this dataset
        dataset_models = {resp_file.stem for resp_file in dataset_responses}
        missing_models = all_models - dataset_models
        if missing_models:
            logging.warning(
                f"Dataset {rel_path} missing files for models: {sorted(missing_models)}"
            )

        if not dataset_responses:
            logging.warning(f"No response files found for dataset: {rel_path}")
            continue

        # Check each response file
        for resp_file in dataset_responses:
            model_id = resp_file.stem
            with open(resp_file) as f:
                responses = yaml.safe_load(f)

            # Count responses per question
            for qid, resp_list in responses["responses-by-qid"].items():
                if len(resp_list) != 10:
                    logging.warning(
                        f"Dataset {rel_path}, model {model_id}, question {qid}: "
                        f"expected 10 responses, found {len(resp_list)}"
                    )

            # Check if all questions have responses
            missing_qids = set(questions["question-by-qid"].keys()) - set(
                responses["responses-by-qid"].keys()
            )
            if missing_qids:
                logging.warning(
                    f"Dataset {rel_path}, model {model_id}: "
                    f"missing responses for questions: {missing_qids}"
                )


if __name__ == "__main__":
    main()
